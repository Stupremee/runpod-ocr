"""Typed Python client for the deployed OCR worker.

Other languages can call the same endpoint directly: POST
`https://api.runpod.ai/v2/<RUNPOD_OCR_ENDPOINT_ID>/run` with `{"input": <OcrRequest>}`.
"""

import base64
import os
from pathlib import Path
from typing import Any

from runpod_flash import Endpoint

from .schema import DocumentSource, ModelName, OcrOptions, OcrRequest, OcrResponse

type Document = str | Path | bytes


def document_source(document: Document) -> DocumentSource:
    """http(s) URLs are fetched by the worker; paths and bytes are sent inline (keep under ~7 MB)."""
    if isinstance(document, str) and document.startswith(("http://", "https://")):
        return DocumentSource(url=document)
    data = document if isinstance(document, bytes) else Path(document).read_bytes()
    return DocumentSource(base64=base64.b64encode(data).decode())


async def parse(
    document: Document,
    *,
    model: ModelName = "paddleocr-vl-1.6",
    options: OcrOptions | None = None,
    model_options: dict[str, Any] | None = None,
    endpoint_id: str | None = None,
    timeout: float = 1800,
) -> OcrResponse:
    """Parse a PDF or image. `timeout` covers cold starts of the worker and its backend."""
    request = OcrRequest(
        document=document_source(document),
        model=model,
        options=options or OcrOptions(),
        model_options=model_options or {},
    )
    endpoint = Endpoint(id=endpoint_id or os.environ["RUNPOD_OCR_ENDPOINT_ID"])
    job = await endpoint.run(request.model_dump(mode="json"))
    await job.wait(timeout=timeout)
    output = job.output
    if job.error or not isinstance(output, dict) or "error" in output:
        raise RuntimeError(f"OCR job {job.id} failed: {job.error or output}")
    return OcrResponse.model_validate(output)
