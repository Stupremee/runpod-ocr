"""Turn a `DocumentSource` into page images. Runs inside the worker."""

import base64
import io
import urllib.request
from typing import NamedTuple

import pypdfium2 as pdfium
from PIL import Image

from .schema import DocumentSource


class PageImage(NamedTuple):
    number: int  # 1-based page number in the source
    image: Image.Image  # RGB


def fetch(source: DocumentSource) -> bytes:
    if source.base64 is not None:
        return base64.b64decode(source.base64)
    assert source.url is not None
    request = urllib.request.Request(source.url, headers={"User-Agent": "runpod-ocr/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def render(data: bytes, *, pages: list[int] | None, dpi: int) -> tuple[int, list[PageImage]]:
    """Returns (page count of the source, selected pages rendered to RGB images)."""
    if not data.startswith(b"%PDF"):
        if pages and pages != [1]:
            raise ValueError("images have a single page")
        return 1, [PageImage(1, Image.open(io.BytesIO(data)).convert("RGB"))]

    pdf = pdfium.PdfDocument(data)
    count = len(pdf)
    wanted = pages or list(range(1, count + 1))
    if bad := [p for p in wanted if not 1 <= p <= count]:
        raise ValueError(f"pages {bad} out of range 1..{count}")
    return count, [
        PageImage(p, pdf[p - 1].render(scale=dpi / 72).to_pil().convert("RGB")) for p in wanted
    ]


def crop_png_base64(image: Image.Image, bbox: tuple[float, float, float, float]) -> str:
    buffer = io.BytesIO()
    image.crop(tuple(round(v) for v in bbox)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()
