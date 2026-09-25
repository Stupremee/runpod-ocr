"""OCR on Runpod serverless: PaddleOCR-VL 1.6, TeleOCR, or any model in `MODELS`."""

from .cli import main
from .client import OcrResult, build_request, endpoint_for, ocr
from .models import MODELS, ModelKey, OcrModel

__all__ = [
    "MODELS",
    "ModelKey",
    "OcrModel",
    "OcrResult",
    "build_request",
    "endpoint_for",
    "main",
    "ocr",
]
