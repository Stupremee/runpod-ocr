import base64
import io

from PIL import Image

from runpod_ocr.schema import Block, DocumentSource, OcrOptions, OcrRequest
from runpod_ocr.service import OcrService


class FakeEngine:
    def parse(self, pages, options, model_options):
        return [
            [
                Block(id=0, label="title", source_label="doc_title", bbox=(0, 0, 10, 5), content="Rechnung"),
                Block(id=1, label="image", source_label="image", bbox=(0, 5, 10, 10), content=""),
                Block(id=2, label="page_number", source_label="number", bbox=(0, 9, 1, 10), content="1"),
            ]
            for _ in pages
        ]


def request(**options) -> OcrRequest:
    png = io.BytesIO()
    Image.new("RGB", (10, 10), "white").save(png, format="PNG")
    return OcrRequest(
        document=DocumentSource(base64=base64.b64encode(png.getvalue()).decode()),
        options=OcrOptions(**options),
    )


def service() -> OcrService:
    return OcrService({"paddleocr-vl-1.6": FakeEngine})


def test_markdown_output_uses_placeholders_and_drops_page_furniture():
    response = service().handle(request())
    assert response.markdown == "# Rechnung\n\n![image](page-1-block-1)"
    assert response.pages is None


def test_json_output_crops_images():
    response = service().handle(request(output="json", images="crop"))
    [page] = response.pages
    assert [b.label for b in page.blocks] == ["title", "image"]
    assert page.blocks[1].image_png_base64
    assert (page.width, page.height) == (10, 10)
