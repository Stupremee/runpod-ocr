"""Registry of OCR vision-language models the provider can serve.

Every model runs behind the same Runpod vLLM worker image, so adding a model is
one entry here: its Hugging Face repo, the vLLM flags it needs, and the prompt
for each task it was trained on.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True, kw_only=True)
class OcrModel:
    """An OCR VLM and everything needed to serve and prompt it."""

    hf_repo: str
    # task name -> the exact prompt the model was trained with
    prompts: Mapping[str, str]
    system_prompt: str | None = None
    max_tokens: int = 4096
    # extra worker-vllm env vars; each becomes a `vllm serve` flag
    vllm_env: Mapping[str, str] = field(default_factory=dict)


ModelKey = Literal["paddleocr-vl-1.6", "teleocr"]

MODELS: dict[ModelKey, OcrModel] = {
    # https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6
    "paddleocr-vl-1.6": OcrModel(
        hf_repo="PaddlePaddle/PaddleOCR-VL-1.6",
        prompts={
            "ocr": "OCR:",
            "table": "Table Recognition:",
            "formula": "Formula Recognition:",
            "chart": "Chart Recognition:",
            "spotting": "Spotting:",
            "seal": "Seal Recognition:",
        },
        # flags from the vLLM PaddleOCR-VL recipe
        vllm_env={
            "TRUST_REMOTE_CODE": "true",
            "MAX_NUM_BATCHED_TOKENS": "16384",
            "ENABLE_PREFIX_CACHING": "false",
            "MM_PROCESSOR_CACHE_GB": "0",
        },
    ),
    # https://huggingface.co/StarDoc-AI/TeleOCR (Qwen2.5-VL architecture)
    "teleocr": OcrModel(
        hf_repo="StarDoc-AI/TeleOCR",
        system_prompt="You are a helpful assistant.",
        prompts={
            "ocr": "Please output the text content from the image.",
            "table": "This is the image of a table. Please output the table in OTSL format.",
            "formula": "Please write out the expression of the formula in the image using LaTeX format.",
            "code": "The image contains a code snippet, please output the parsing result.",
            "layout": "Analyze the image layout.",
            "figure": "This is a scientific figure. Please extract the table implied by this figure.",
        },
    ),
}
