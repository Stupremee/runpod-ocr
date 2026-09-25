"""Handles one `OcrRequest` end to end, independent of which engine runs it."""

from collections.abc import Callable

from .documents import crop_png_base64, fetch, render
from .engines import ENGINES, Engine
from .render import keep_block, page_markdown, page_text
from .schema import ModelName, OcrRequest, OcrResponse, Page


class OcrService:
    """Holds one warm engine per model; build once per worker."""

    def __init__(self, engines: dict[ModelName, Callable[[], Engine]] = ENGINES) -> None:
        self._factories = engines
        self._engines: dict[ModelName, Engine] = {}

    def _engine(self, model: ModelName) -> Engine:
        if model not in self._engines:
            self._engines[model] = self._factories[model]()
        return self._engines[model]

    def handle(self, request: OcrRequest) -> OcrResponse:
        options = request.options
        page_count, images = render(fetch(request.document), pages=options.pages, dpi=options.dpi)
        blocks_per_page = self._engine(request.model).parse(images, options, request.model_options)

        pages: list[Page] = []
        for image, blocks in zip(images, blocks_per_page, strict=True):
            kept = [b for b in blocks if keep_block(b, options)]
            if options.images == "crop" and options.output == "json":
                kept = [
                    b.model_copy(update={"image_png_base64": crop_png_base64(image.image, b.bbox)})
                    if b.label in ("image", "chart", "seal")
                    else b
                    for b in kept
                ]
            pages.append(
                Page(
                    number=image.number,
                    width=image.image.width,
                    height=image.image.height,
                    blocks=kept,
                    markdown=page_markdown(kept, image.number),
                )
            )

        response = OcrResponse(model=request.model, page_count=page_count)
        match options.output:
            case "markdown":
                response.markdown = "\n\n".join(p.markdown for p in pages)
            case "text":
                response.text = "\n\n".join(page_text(p.blocks) for p in pages)
            case "json":
                response.pages = pages
        return response
