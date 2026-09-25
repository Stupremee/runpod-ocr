"""OCR engines: one per model, all producing the unified `Block` schema.

To add a model: implement `Engine` in a new module, add its name to
`schema.ModelName`, and register a factory in `ENGINES`.
"""

from collections.abc import Callable
from typing import Any, Protocol

from ..documents import PageImage
from ..schema import Block, ModelName, OcrOptions


class Engine(Protocol):
    def parse(
        self, pages: list[PageImage], options: OcrOptions, model_options: dict[str, Any]
    ) -> list[list[Block]]:
        """Blocks per page, in reading order, before option filtering."""
        ...


def _paddleocr_vl() -> Engine:
    from .paddleocr_vl import PaddleOcrVl

    return PaddleOcrVl()


ENGINES: dict[ModelName, Callable[[], Engine]] = {
    "paddleocr-vl-1.6": _paddleocr_vl,
}
