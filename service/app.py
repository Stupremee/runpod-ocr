"""PaddleOCR-VL 1.6 service: the official PaddleX serving app, with the VLM on Runpod.

Routes
  POST /layout-parsing        official PaddleX API, unchanged (one file per request)
  POST /layout-parsing/batch  {"requests": [<layout-parsing request>, ...]}
                              -> {"results": [<layout-parsing response>, ...]}, same order
  POST /warmup                wake the Runpod GPU ahead of a batch; returns once it serves
  GET  /health                official PaddleX health check

Cold starts: the Runpod endpoint scales to zero. Before any parsing, a shared
guard makes sure a GPU worker is actually serving (one tiny probe job, awaited
by every concurrent request), so a batch pays for at most one cold start and no
VLM call times out while a worker boots. Send work as batches (or call /warmup
first) and keep batches within the endpoint's idle timeout of each other.
"""

import asyncio
import os
import time
from pathlib import Path
from typing import Any

import httpx
import uvicorn
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from paddlex import create_pipeline
from paddlex.inference.serving.basic_serving import create_pipeline_app

ENDPOINT_ID = os.environ["PADDLEOCR_VL_ENDPOINT_ID"]
API_KEY = os.environ["RUNPOD_API_KEY"]
# treat the GPU as warm this long after the last VLM traffic; keep below the
# endpoint's idle_timeout (300s, see runpod_ocr/backends.py)
WARM_TTL = float(os.environ.get("WARM_TTL_SECONDS", "240"))
COLD_START_TIMEOUT = float(os.environ.get("COLD_START_TIMEOUT_SECONDS", "900"))
# files of one batch parsed at the same time; their VLM calls share the warm GPU
BATCH_CONCURRENCY = int(os.environ.get("BATCH_CONCURRENCY", "4"))
PARSE_PATHS = ("/layout-parsing", "/layout-parsing/batch", "/warmup")


class RunpodWarmup:
    """Ensures a Runpod worker is serving before VLM traffic starts."""

    def __init__(self) -> None:
        self._base = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"
        self._headers = {"Authorization": f"Bearer {API_KEY}"}
        self._lock = asyncio.Lock()
        self._warm_until = 0.0

    def touch(self) -> None:
        self._warm_until = time.monotonic() + WARM_TTL

    async def ensure(self) -> None:
        if time.monotonic() < self._warm_until:
            return
        async with self._lock:  # concurrent callers share one probe
            if time.monotonic() < self._warm_until:
                return
            started = time.monotonic()
            async with httpx.AsyncClient(headers=self._headers, timeout=30) as client:
                # a job only completes once vLLM is up, unlike worker "ready" states
                job = (await client.post(f"{self._base}/run", json={"input": {"openai_route": "/v1/models"}})).json()
                while True:
                    status = (await client.get(f"{self._base}/status/{job['id']}")).json()
                    if status["status"] == "COMPLETED":
                        break
                    if status["status"] in ("FAILED", "CANCELLED", "TIMED_OUT"):
                        raise RuntimeError(f"Runpod warm-up job {status['status']}: {status.get('error')}")
                    if time.monotonic() - started > COLD_START_TIMEOUT:
                        await client.post(f"{self._base}/cancel/{job['id']}")
                        raise TimeoutError(f"Runpod worker not ready after {COLD_START_TIMEOUT:.0f}s")
                    await asyncio.sleep(2)
            print(f"runpod backend warm after {time.monotonic() - started:.1f}s", flush=True)
            self.touch()


def load_config() -> dict[str, Any]:
    config = yaml.safe_load((Path(__file__).parent / "pipeline.yaml").read_text())
    genai = config["SubModules"]["VLRecognition"]["genai_config"]
    genai["server_url"] = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/openai/v1"
    genai["client_kwargs"]["api_key"] = API_KEY
    return config


def build_app() -> FastAPI:
    config = load_config()
    pipeline = create_pipeline(config=config, device=os.environ.get("DEVICE", "cpu"))
    app = create_pipeline_app(pipeline, config)
    warmup = RunpodWarmup()

    @app.middleware("http")
    async def _warm_backend(request: Request, call_next):
        if request.method == "POST" and request.url.path in PARSE_PATHS:
            try:
                await warmup.ensure()
            except Exception as e:
                # same shape as PaddleX error responses
                return JSONResponse(status_code=503, content={"logId": "", "errorCode": 503, "errorMsg": str(e)})
        response = await call_next(request)
        if request.url.path in PARSE_PATHS:
            warmup.touch()
        return response

    @app.post("/warmup")
    async def _warmup() -> dict[str, str]:
        return {"status": "warm"}

    @app.post("/layout-parsing/batch")
    async def _batch(body: dict[str, list[dict[str, Any]]]) -> dict[str, list[Any]]:
        limit = asyncio.Semaphore(BATCH_CONCURRENCY)
        # each item goes through the official route in-process: identical responses
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://ocr", timeout=None) as client:

            async def parse(item: dict[str, Any]) -> Any:
                async with limit:
                    return (await client.post("/layout-parsing", json=item)).json()

            return {"results": await asyncio.gather(*(parse(item) for item in body["requests"]))}

    return app


if __name__ == "__main__":
    uvicorn.run(build_app(), host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
