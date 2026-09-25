"""Emulated backend (OCR_BACKEND=emulated): exercise the whole service without GPU time.

The PaddleX pipeline, layout detection, jobs and webhooks run for real; only
the VLM is replaced by a local OpenAI-compatible stub that answers each
PaddleOCR-VL task prompt with fixed text in the model's output format. The
cold start is simulated too, so the warm-up guard behaves as it does on Runpod.
"""

import asyncio
import os
import time
import uuid
from typing import Any

from fastapi import APIRouter

COLD_START = float(os.environ.get("EMULATED_COLD_START_SECONDS", "5"))
LATENCY = float(os.environ.get("EMULATED_LATENCY_SECONDS", "0.2"))
IDLE_TIMEOUT = 300  # like the real endpoint: cold again after 5 idle minutes

# PaddleOCR-VL task prompt -> an answer shaped like the real model's output
_ANSWERS = {
    "Table Recognition:": "<fcel>Position<fcel>Menge<nl><fcel>Emuliert<fcel>1<nl>",
    "Formula Recognition:": "\\[E = mc^{2}\\]",
    "Chart Recognition:": "Kategorie | Wert\nA | 1",
    "Seal Recognition:": "Emuliertes Siegel",
}
_DEFAULT_ANSWER = "Emulierter Text (keine GPU-Inferenz)."


class EmulatedWarmup:
    """Same contract as RunpodWarmup: the first call after idling waits for a 'cold start'."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._warm_until = 0.0

    def touch(self) -> None:
        self._warm_until = time.monotonic() + IDLE_TIMEOUT

    async def ensure(self) -> None:
        async with self._lock:
            if time.monotonic() >= self._warm_until:
                print(f"emulated backend cold start ({COLD_START:.0f}s)", flush=True)
                await asyncio.sleep(COLD_START)
            self.touch()


def _prompt(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        content = message.get("content")
        if isinstance(content, str):
            return content
        for part in content or []:
            if part.get("type") == "text":
                return part["text"]
    return ""


def create_router(model_name: str) -> APIRouter:
    """OpenAI-compatible routes under /emulated/v1, used as the pipeline's VLM server."""
    router = APIRouter(prefix="/emulated/v1", include_in_schema=False)

    @router.get("/models")
    async def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": model_name, "object": "model", "owned_by": "emulated"}]}

    @router.post("/chat/completions")
    async def chat(body: dict[str, Any]) -> dict[str, Any]:
        await asyncio.sleep(LATENCY)
        prompt = _prompt(body.get("messages", [])).strip()
        text = _ANSWERS.get(prompt, _DEFAULT_ANSWER)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", model_name),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": len(text.split()), "total_tokens": len(text.split())},
        }

    return router
