"""PaddleOCR-VL 1.6 service: the official PaddleX pipeline with the VLM on Runpod,
behind an async job API.

Routes
  POST /jobs, /batches, GET|DELETE /jobs/{id}, /batches/{id}
                        async processing with webhooks (see api.py)
  POST /layout-parsing  official PaddleX route, synchronous, plus the
                        tableFormat / layoutText options (see formatting.py)
  POST /warmup          wake the Runpod GPU ahead of time
  *                     every other PaddleX route (/health, ...)

OCR_BACKEND=emulated runs everything except GPU inference (see emulated.py).

Cold starts: the Runpod endpoint scales to zero. Before parsing, a shared guard
makes sure a GPU worker is actually serving (one tiny probe job that every
concurrent caller awaits), so a burst of jobs pays for at most one cold start
and no VLM call times out while a worker boots.
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
from paddlex.inference.serving.schemas.paddleocr_vl import InferRequest
from pydantic import ValidationError

import emulated
from api import create_router
from formatting import layout_text, tables_to_markdown
from jobs import JobStore
from runner import JobRunner
from webhooks import WebhookSender

# "runpod" (default) or "emulated": a local stub instead of the GPU, for testing
BACKEND = os.environ.get("OCR_BACKEND", "runpod")
EMULATED = BACKEND == "emulated"
ENDPOINT_ID = "" if EMULATED else os.environ["PADDLEOCR_VL_ENDPOINT_ID"]
API_KEY = "" if EMULATED else os.environ["RUNPOD_API_KEY"]
PORT = int(os.environ.get("PORT", "8080"))
MODEL_NAME = "PaddlePaddle/PaddleOCR-VL-1.6"
# treat the GPU as warm this long after the last VLM traffic; keep below the
# endpoint's idle_timeout (300s, see runpod_ocr/backends.py)
WARM_TTL = float(os.environ.get("WARM_TTL_SECONDS", "240"))
COLD_START_TIMEOUT = float(os.environ.get("COLD_START_TIMEOUT_SECONDS", "900"))
# jobs processed at once; their VLM calls share the warm GPU
WORKER_CONCURRENCY = int(os.environ.get("WORKER_CONCURRENCY", "4"))
JOB_MAX_ATTEMPTS = int(os.environ.get("JOB_MAX_ATTEMPTS", "3"))
RETENTION_DAYS = float(os.environ.get("RETENTION_DAYS", "7"))
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET") or None
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
TABLE_FORMATS = ("html", "markdown")


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
    if EMULATED:  # this service's own stub routes
        genai["server_url"] = f"http://127.0.0.1:{PORT}/emulated/v1"
        genai["client_kwargs"]["api_key"] = "emulated"
    else:
        genai["server_url"] = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/openai/v1"
        genai["client_kwargs"]["api_key"] = API_KEY
    return config


def split_options(request: dict[str, Any]) -> tuple[dict[str, Any], str, bool]:
    """Separate this service's options from the PaddleX request."""
    request = dict(request)
    return request, request.pop("tableFormat", "html"), request.pop("layoutText", False)


def validate(request: dict[str, Any]) -> str | None:
    """Error message for an invalid layout-parsing request, checked against PaddleX's own schema."""
    paddle_request, table_format, with_layout_text = split_options(request)
    if table_format not in TABLE_FORMATS:
        return f"tableFormat must be one of {TABLE_FORMATS}"
    if not isinstance(with_layout_text, bool):
        return "layoutText must be a boolean"
    try:
        InferRequest.model_validate(paddle_request)
    except ValidationError as e:
        return str(e)
    return None


def postprocess(body: dict[str, Any], table_format: str, with_layout_text: bool) -> dict[str, Any]:
    """Apply tableFormat/layoutText to a successful PaddleX response."""
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
    paddle_client = httpx.AsyncClient(transport=httpx.ASGITransport(app=paddle), base_url="http://paddle", timeout=None)
    warmup = emulated.EmulatedWarmup() if EMULATED else RunpodWarmup()

    async def parse(request: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        """Run one layout-parsing request through the official route, then apply our options."""
        paddle_request, table_format, with_layout_text = split_options(request)
        response = await paddle_client.post("/layout-parsing", json=paddle_request)
        body = response.json()
        if response.status_code == 200:
            body = postprocess(body, table_format, with_layout_text)
        warmup.touch()
        return response.status_code, body

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    store = JobStore(str(DATA_DIR / "jobs.db"))
    sender = WebhookSender(store, WEBHOOK_SECRET)
    runner = JobRunner(
        store, parse, warmup, concurrency=WORKER_CONCURRENCY, max_attempts=JOB_MAX_ATTEMPTS, on_finished=sender.wake
    )

    async def purge_old() -> None:
        while True:
            store.purge(older_than=time.time() - RETENTION_DAYS * 86400)
            await asyncio.sleep(3600)

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI):
        async with paddle.router.lifespan_context(paddle):  # loads the pipeline wrapper
            if requeued := store.requeue_interrupted():
                print(f"requeued {requeued} interrupted jobs", flush=True)
            tasks = [asyncio.create_task(t) for t in (runner.run(), sender.run(), purge_old())]
            if not WEBHOOK_SECRET:
                print("WEBHOOK_SECRET not set: webhooks are sent unsigned", flush=True)
            yield
            for task in tasks:
                task.cancel()

    app = FastAPI(title="PaddleOCR-VL 1.6", lifespan=lifespan)
    if EMULATED:
        print("OCR_BACKEND=emulated: no GPU inference, VLM answers are canned", flush=True)
        app.include_router(emulated.create_router(MODEL_NAME))
    app.include_router(create_router(store, validate=validate, on_submit=runner.wake, on_finished=sender.wake))

    @app.post("/layout-parsing", tags=["sync"])
    async def _layout_parsing(request: dict[str, Any]) -> JSONResponse:
        if problem := validate(request):
            return JSONResponse(status_code=422, content={"logId": "", "errorCode": 422, "errorMsg": problem})
        try:
            await warmup.ensure()
        except Exception as e:
            return JSONResponse(status_code=503, content={"logId": "", "errorCode": 503, "errorMsg": str(e)})
        status, body = await parse(request)
        return JSONResponse(status_code=status, content=body)

    @app.post("/warmup", tags=["sync"])
    async def _warmup() -> JSONResponse:
        try:
            await warmup.ensure()
        except Exception as e:
            return JSONResponse(status_code=503, content={"error": {"code": "backend_unavailable", "message": str(e)}})
        return JSONResponse({"status": "warm"})

    app.mount("/", paddle)  # every other official route
    return app


if __name__ == "__main__":
    uvicorn.run(build_app(), host="0.0.0.0", port=PORT)
