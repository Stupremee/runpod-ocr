"""The unified OCR contract shared by every engine, the worker, and clients.

Engines translate their model's native output into `Block`s with a common
`Label` vocabulary; everything downstream (option filtering, Markdown/text
rendering, JSON) is engine-agnostic.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Label = Literal[
    "title",
    "heading",
    "text",
    "list",
    "table",
    "formula",
    "formula_number",
    "image",
    "chart",
    "caption",
    "code",
    "seal",
    "header",
    "footer",
    "page_number",
    "footnote",
    "reference",
]

OutputFormat = Literal["markdown", "text", "json"]

# every model the API can run; each has an engine in `runpod_ocr.engines`
ModelName = Literal["paddleocr-vl-1.6"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OcrOptions(_Strict):
    """Model-agnostic options. Engines ignore what their model cannot honour."""

    output: OutputFormat = "markdown"
    pages: list[int] | None = Field(default=None, description="1-based pages to parse; all when unset")
    dpi: int = Field(default=144, ge=72, le=300, description="PDF render resolution")
    formulas: bool = Field(default=True, description="keep formulas as LaTeX; false drops formula blocks")
    tables: bool = Field(default=True, description="keep tables; false drops table blocks")
    images: Literal["omit", "placeholder", "crop"] = Field(
        default="placeholder",
        description="image blocks: drop them, keep a reference, or also return a base64 PNG crop (json only)",
    )
    image_text: bool = Field(default=False, description="OCR text that appears inside images")
    charts: bool = Field(default=False, description="convert charts into tables")
    seals: bool = Field(default=False, description="recognize stamps/seals")
    headers_footers: bool = Field(default=False, description="include page headers, footers, page numbers")
    fix_orientation: bool = Field(default=False, description="auto-rotate pages scanned upside down or sideways")
    unwarp: bool = Field(default=False, description="flatten curved/photographed pages")
    layout_threshold: float | None = Field(default=None, ge=0, le=1)


class DocumentSource(_Strict):
    """A PDF or image, by URL or inline base64."""

    url: str | None = None
    base64: str | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> "DocumentSource":
        if (self.url is None) == (self.base64 is None):
            raise ValueError("document needs exactly one of url or base64")
        return self


class TableCell(_Strict):
    text: str
    row_span: int = 1
    col_span: int = 1


class Block(_Strict):
    """One layout region, in reading order. `bbox` is [x0, y0, x1, y1] in page pixels."""

    id: int
    label: Label
    source_label: str = Field(description="the model's own label, before normalization")
    bbox: tuple[float, float, float, float]
    content: str = Field(description="plain content: text, LaTeX for formulas, Markdown table for tables")
    table: list[list[TableCell]] | None = None
    image_png_base64: str | None = None


class Page(_Strict):
    number: int = Field(description="1-based page number in the source document")
    width: int
    height: int
    blocks: list[Block]
    markdown: str


class OcrRequest(_Strict):
    """The unified API input: `{"input": <OcrRequest>}` on the Runpod queue."""

    document: DocumentSource
    model: ModelName = "paddleocr-vl-1.6"
    options: OcrOptions = Field(default_factory=OcrOptions)
    model_options: dict[str, Any] = Field(
        default_factory=dict, description="model-specific knobs, validated by the model's engine"
    )


class OcrResponse(_Strict):
    """`markdown`/`text` are filled for those outputs; `pages` for json."""

    model: ModelName
    page_count: int = Field(description="pages in the source document")
    markdown: str | None = None
    text: str | None = None
    pages: list[Page] | None = None
