"""PaddleOCR-VL 1.6: PP-DocLayoutV3 layout detection on this worker's GPU (via
transformers), block recognition on the vLLM backend endpoint."""

import os
import re
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from ..backends import BACKENDS
from ..documents import PageImage
from ..render import table_from_html, table_markdown
from ..schema import Block, Label, OcrOptions

_LABELS: dict[str, Label] = {
    "doc_title": "title",
    "paragraph_title": "heading",
    "display_formula": "formula",
    "inline_formula": "formula",
    "formula_number": "formula_number",
    "table": "table",
    "chart": "chart",
    "image": "image",
    "header_image": "image",
    "footer_image": "image",
    "seal": "seal",
    "figure_title": "caption",
    "chart_title": "caption",
    "table_title": "caption",
    "vision_footnote": "caption",
    "header": "header",
    "footer": "footer",
    "number": "page_number",
    "footnote": "footnote",
    "reference": "reference",
    "reference_content": "reference",
}

_FORMULA_DELIMITERS = re.compile(r"^\s*(\$\$|\$|\\\[|\\\()\s*|\s*(\$\$|\$|\\\]|\\\))\s*$")


class PaddleOcrVlOptions(BaseModel):
    """`model_options` for this engine; mirrors PaddleOCRVL.predict() knobs."""

    model_config = ConfigDict(extra="forbid")

    use_layout_detection: bool = True
    layout_nms: bool | None = None
    layout_unclip_ratio: float | None = None
    layout_merge_bboxes_mode: Literal["large", "small", "union"] | None = None
    merge_layout_blocks: bool | None = None
    temperature: float | None = None
    top_p: float | None = None
    repetition_penalty: float | None = None
    min_pixels: int | None = None
    max_pixels: int | None = None
    max_new_tokens: int | None = None


class PaddleOcrVl:
    def __init__(self) -> None:
        from paddleocr import PaddleOCRVL

        backend = BACKENDS["paddleocr-vl-1.6"]
        self._pipeline = PaddleOCRVL(
            pipeline_version="v1.6",
            engine="transformers",
            device=os.environ.get("OCR_DEVICE", "gpu:0"),  # "cpu" for local runs
            vl_rec_backend="vllm-server",
            vl_rec_server_url=backend.openai_url(os.environ[backend.endpoint_id_env]),
            vl_rec_api_key=os.environ["RUNPOD_API_KEY"],
            vl_rec_api_model_name=backend.hf_repo,
            vl_rec_max_concurrency=64,
        )

    def parse(
        self, pages: list[PageImage], options: OcrOptions, model_options: dict[str, Any]
    ) -> list[list[Block]]:
        extra = PaddleOcrVlOptions.model_validate(model_options).model_dump(exclude_none=True)
        results = self._pipeline.predict(
            # the pipeline expects OpenCV-style BGR arrays
            [np.asarray(page.image)[:, :, ::-1] for page in pages],
            use_doc_orientation_classify=options.fix_orientation,
            use_doc_unwarping=options.unwarp,
            use_chart_recognition=options.charts,
            use_seal_recognition=options.seals,
            use_ocr_for_image_block=options.image_text,
            layout_threshold=options.layout_threshold,
            **extra,
        )
        return [[_block(i, raw) for i, raw in enumerate(r["parsing_res_list"])] for r in results]


def _block(index: int, raw: Any) -> Block:
    """`raw` is a PaddleX layout block; its attributes mirror the saved JSON keys."""
    source_label: str = raw.label
    label = _LABELS.get(source_label, "text")
    content = (raw.content or "").strip()
    table = None
    if label == "formula":
        content = _FORMULA_DELIMITERS.sub("", content)
    elif content.startswith("<table"):
        table = table_from_html(content)
        content = table_markdown(table)
    x0, y0, x1, y1 = (float(v) for v in raw.bbox)
    return Block(
        id=index,
        label=label,
        source_label=source_label,
        bbox=(x0, y0, x1, y1),
        content=content,
        table=table,
    )
