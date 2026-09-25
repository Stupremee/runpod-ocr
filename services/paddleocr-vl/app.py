"""PaddleOCR-VL 1.6 service: the official PaddleX serving app, with the VLM on Runpod.

Routes
  POST /layout-parsing        official PaddleX API, plus two optional fields:
                                tableFormat: "html" (default) | "markdown"
                                layoutText:  false (default) | true -> adds
                                  layoutText (page text laid out like the page)
                                  to every layoutParsingResults item
  POST /layout-parsing/batch  {"requests": [<layout-parsing request>, ...]}
                              -> {"results": [<layout-parsing response>, ...]}, same order
  POST /warmup                wake the Runpod GPU ahead of a batch; returns once it serves
  *                           every other PaddleX route (/health, /restructure-pages, ...)

Cold starts: the Runpod endpoint scales to zero. Before any parsing, a shared
guard makes sure a GPU worker is actually serving (one tiny probe job, awaited
by every concurrent request), so a batch pays for at most one cold start and no
VLM call times out while a worker boots. Send work as batches (or call /warmup
first) and keep batches within the endpoint's idle timeout of each other.
"""

import asyncio
import contextlib
import os
import time
from pathlib import Path
from typing import Any

import httpx
import uvicorn
import yaml
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from paddlex import create_pipeline
from paddlex.inference.serving.basic_serving import create_pipeline_app

from formatting import layout_text, tables_to_markdown

ENDPOINT_ID = os.environ["PADDLEOCR_VL_ENDPOINT_ID"]
API_KEY = os.environ["RUNPOD_API_KEY"]
# treat the GPU as warm this long after the last VLM traffic; keep below the
# endpoint's idle_timeout (300s, see runpod_ocr/backends.py)
WARM_TTL = float(os.environ.get("WARM_TTL_SECONDS", "240"))
COLD_START_TIMEOUT = float(os.environ.get("COLD_START_TIMEOUT_SECONDS", "900"))
# files of one batch parsed at the same time; their VLM calls share the warm GPU
BATCH_CONCURRENCY = int(os.environ.get("BATCH_CONCURRENCY", "4"))
TABLE_FORMATS = ("html", "markdown")


def error(status: int, message: str) -> JSONResponse:
    """Same shape as PaddleX error responses."""
    return JSONResponse(status_code=status, content={"logId": "", "errorCode": status, "errorMsg": message})


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


def postprocess(body: dict[str, Any], table_format: str, with_layout_text: bool) -> dict[str, Any]:
    """Apply the optional tableFormat/layoutText to a successful PaddleX response."""
    for page in (body.get("result") or {}).get("layoutParsingResults", []):
        pruned = page["prunedResult"]
        if with_layout_text:
            page["layoutText"] = layout_text(pruned)
        if table_format == "markdown":
            page["markdown"]["text"] = tables_to_markdown(page["markdown"]["text"])
            for block in pruned["parsing_res_list"]:
                if block["block_label"] == "table":
                    block["block_content"] = tables_to_markdown(block["block_content"]).strip()
    return body


def build_app() -> FastAPI:
    config = load_config()
    pipeline = create_pipeline(config=config, device=os.environ.get("DEVICE", "cpu"))
    paddle = create_pipeline_app(pipeline, config)  # official app, unmodified
    warmup = RunpodWarmup()

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI):
        async with paddle.router.lifespan_context(paddle):  # loads the pipeline wrapper
            yield

    app = FastAPI(title="PaddleOCR-VL 1.6", lifespan=lifespan)
    paddle_client = httpx.AsyncClient(transport=httpx.ASGITransport(app=paddle), base_url="http://paddle", timeout=None)

    async def parse(request: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        request = dict(request)
        table_format = request.pop("tableFormat", "html")
        with_layout_text = request.pop("layoutText", False)
        if table_format not in TABLE_FORMATS or not isinstance(with_layout_text, bool):
            return 422, {"logId": "", "errorCode": 422,
                         "errorMsg": f"tableFormat must be one of {TABLE_FORMATS}; layoutText must be a boolean"}
        response = await paddle_client.post("/layout-parsing", json=request)
        body = response.json()
        if response.status_code == 200:
            body = postprocess(body, table_format, with_layout_text)
        warmup.touch()
        return response.status_code, body

    @app.post("/layout-parsing")
    async def _layout_parsing(request: dict[str, Any]) -> JSONResponse:
        try:
            await warmup.ensure()
        except Exception as e:
            return error(503, str(e))
        status, body = await parse(request)
        return JSONResponse(status_code=status, content=body)

    @app.post("/layout-parsing/batch")
    async def _batch(body: dict[str, list[dict[str, Any]]]) -> JSONResponse:
        try:
            await warmup.ensure()  # once for the whole batch
        except Exception as e:
            return error(503, str(e))
        limit = asyncio.Semaphore(BATCH_CONCURRENCY)

        async def one(request: dict[str, Any]) -> dict[str, Any]:
            async with limit:
                return (await parse(request))[1]

        return JSONResponse({"results": await asyncio.gather(*(one(r) for r in body["requests"]))})

    @app.post("/warmup")
    async def _warmup() -> JSONResponse:
        try:
            await warmup.ensure()
        except Exception as e:
            return error(503, str(e))
        return JSONResponse({"status": "warm"})

    app.mount("/", paddle)  # every other official route
    return app


if __name__ == "__main__":
    uvicorn.run(build_app(), host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
