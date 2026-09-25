"""Run OCR models on Runpod serverless via Flash.

Each model in `MODELS` gets its own scale-to-zero endpoint running Runpod's
vLLM worker image. Flash provisions it on first use and tracks it by name in
`.flash/resources.pkl` (relative to the working directory), so later calls
reuse the same endpoint and config changes update it in place.
"""

import base64
import mimetypes
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from runpod_flash import CudaVersion, Endpoint, GpuGroup

from .models import MODELS, ModelKey, OcrModel

# vLLM 0.28 on CUDA 13.0; supports both PaddleOCR-VL and Qwen2.5-VL natively.
WORKER_IMAGE = "runpod/worker-v1-vllm:v2.27.1"

type ImageInput = str | Path | bytes


@dataclass(frozen=True)
class OcrResult:
    text: str
    usage: dict[str, int]


@cache
def endpoint_for(key: ModelKey) -> Endpoint:
    """The Flash endpoint serving `key`. Provisioned lazily on the first job."""
    model = MODELS[key]
    return Endpoint(
        name=f"ocr-{key.replace('.', '-')}",
        image=WORKER_IMAGE,
        # 24 GB cards are plenty for ~1B models; A5000/3090/L4 first, 4090 as fallback
        gpu=[GpuGroup.AMPERE_24, GpuGroup.ADA_24],
        workers=(0, 2),
        idle_timeout=60,
        execution_timeout_ms=600_000,
        min_cuda_version=CudaVersion.V13_0,
        env={
            "MODEL_NAME": model.hf_repo,
            "MAX_MODEL_LEN": "16384",
            **model.vllm_env,
        },
    )


def image_url(image: ImageInput) -> str:
    """http(s)/data URLs pass through; paths and raw bytes become base64 data URLs."""
    if isinstance(image, str) and image.startswith(("http://", "https://", "data:")):
        return image
    if isinstance(image, bytes):
        data, mime = image, "image/png"
    else:
        path = Path(image)
        data, mime = path.read_bytes(), mimetypes.guess_type(path)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def build_request(
    model: OcrModel,
    image: ImageInput,
    *,
    task: str = "ocr",
    prompt: str | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Job input for worker-vllm's OpenAI chat-completions passthrough."""
    if prompt is None:
        if task not in model.prompts:
            raise ValueError(
                f"{model.hf_repo} has no task {task!r}; choose from {sorted(model.prompts)}"
            )
        prompt = model.prompts[task]

    messages: list[dict[str, Any]] = []
    if model.system_prompt:
        messages.append({"role": "system", "content": model.system_prompt})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url(image)}},
                {"type": "text", "text": prompt},
            ],
        }
    )
    return {
        "openai_route": "/v1/chat/completions",
        "openai_input": {
            "model": model.hf_repo,
            "messages": messages,
            "max_tokens": max_tokens or model.max_tokens,
            "temperature": 0,
        },
    }


def parse_output(output: Any) -> OcrResult:
    """Unwrap worker-vllm job output (a one-item stream aggregate) into text + usage."""
    response = output[0] if isinstance(output, list) else output
    if not isinstance(response, dict):
        raise RuntimeError(f"unexpected worker output: {output!r}")
    if "error" in response:
        raise RuntimeError(response["error"].get("message", response["error"]))
    return OcrResult(
        text=response["choices"][0]["message"]["content"],
        usage=response.get("usage", {}),
    )


async def ocr(
    image: ImageInput,
    *,
    model: ModelKey = "paddleocr-vl-1.6",
    task: str = "ocr",
    prompt: str | None = None,
    max_tokens: int | None = None,
    timeout: float = 900,
) -> OcrResult:
    """Run one image through `model`. `timeout` covers cold starts (image pull + weights)."""
    request = build_request(MODELS[model], image, task=task, prompt=prompt, max_tokens=max_tokens)
    job = await endpoint_for(model).run(request)
    await job.wait(timeout=timeout)
    if job.error:
        raise RuntimeError(f"job {job.id} failed: {job.error}")
    return parse_output(job.output)
