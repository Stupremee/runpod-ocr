"""The unified OCR API, deployed with `flash deploy`.

Queue request body: `{"input": <schema.OcrRequest>}`, for example
    {"input": {"document": {"url": "https://…/rechnung.pdf"},
               "model": "paddleocr-vl-1.6",
               "options": {"output": "json", "pages": [1]},
               "model_options": {"max_pixels": 1605632}}}
Output: a `schema.OcrResponse`, or `{"error": ...}` for invalid input.
"""

import os

from runpod_flash import Endpoint, GpuGroup

from .backends import BACKENDS


@Endpoint(
    name="ocr",
    # layout detection only; the VLM runs on the backend endpoints
    gpu=[GpuGroup.AMPERE_16, GpuGroup.AMPERE_24, GpuGroup.ADA_24],
    workers=(0, 3),
    idle_timeout=120,
    execution_timeout_ms=900_000,
    dependencies=[
        "paddleocr[doc-parser]==3.7.0",
        "transformers>=5,<6",
        "pypdfium2",
        "pillow",
        "pydantic>=2",
    ],
    system_dependencies=["libgl1", "libglib2.0-0"],
    env={
        "RUNPOD_API_KEY": os.environ.get("RUNPOD_API_KEY", ""),
        **{b.endpoint_id_env: os.environ.get(b.endpoint_id_env, "") for b in BACKENDS.values()},
        "PADDLE_PDX_MODEL_SOURCE": "huggingface",
        "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
    },
)
class Ocr:
    def __init__(self) -> None:
        from runpod_ocr.service import OcrService

        self._service = OcrService()

    def parse(
        self,
        document: dict,
        model: str = "paddleocr-vl-1.6",
        options: dict | None = None,
        model_options: dict | None = None,
    ) -> dict:
        from pydantic import ValidationError

        from runpod_ocr.schema import OcrRequest

        try:
            request = OcrRequest.model_validate(
                {
                    "document": document,
                    "model": model,
                    "options": options or {},
                    "model_options": model_options or {},
                }
            )
            return self._service.handle(request).model_dump(mode="json", exclude_none=True)
        except (ValidationError, ValueError) as e:
            return {"error": str(e)}
