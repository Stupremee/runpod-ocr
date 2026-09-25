"""Build the HTML verification report from a /layout-parsing/batch response.

uv run --with pypdfium2 --with pillow scripts/report.py out/batch-serial.json out/report.html
"""

import base64
import html
import io
import json
import sys
from pathlib import Path

import pypdfium2 as pdfium

SOURCES = {
    "de-rechnung": "https://github.com/ZUGFeRD/corpus/blob/master/ZUGFeRDv2/correct/intarsys/EN16931/zugferd_2p0_EN16931_Einfach.pdf",
    "de-pythagoras": "https://de.wikipedia.org/wiki/Satz_des_Pythagoras",
    "de-maxwell": "https://de.wikipedia.org/wiki/Maxwell-Gleichungen",
}
# (document, 1-based page) shown in the report
PAGES = [("de-rechnung", 1), ("de-rechnung", 2), ("de-pythagoras", 3), ("de-pythagoras", 9), ("de-maxwell", 5), ("de-maxwell", 9)]
COLORS = {"table": "#f5a623", "display_formula": "#bd10e0", "inline_formula": "#bd10e0", "image": "#50e3c2",
          "chart": "#50e3c2", "paragraph_title": "#4a90e2", "doc_title": "#4a90e2", "figure_title": "#b8e986",
          "vision_footnote": "#b8e986"}
THUMB_WIDTH = 440


def thumbnail(pdf_path: Path, page: int) -> tuple[str, int, int]:
    image = pdfium.PdfDocument(pdf_path)[page - 1].render(scale=1).to_pil().convert("L")
    image.thumbnail((THUMB_WIDTH, THUMB_WIDTH * 2))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=38, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode(), image.width, image.height


def page_section(name: str, page: int, result: dict) -> str:
    pruned = result["prunedResult"]
    src, w, h = thumbnail(Path("samples") / f"{name}.pdf", page)
    scale = w / pruned["width"]
    boxes = []
    for block in pruned["parsing_res_list"]:
        x0, y0, x1, y1 = (v * scale for v in block["block_bbox"])
        color = COLORS.get(block["block_label"], "#9b9b9b")
        boxes.append(
            f'<rect x="{x0:.0f}" y="{y0:.0f}" width="{x1 - x0:.0f}" height="{y1 - y0:.0f}" fill="none" stroke="{color}" stroke-width="1.5"><title>{html.escape(block["block_label"])}</title></rect>'
        )
    labels = ", ".join(sorted({b["block_label"] for b in pruned["parsing_res_list"]}))
    return f"""
<section>
  <h2>{name}.pdf, page {page}</h2>
  <p class="meta">blocks: {len(pruned["parsing_res_list"])} · labels: {html.escape(labels)}</p>
  <div class="row">
    <svg viewBox="0 0 {w} {h}" width="{w}" role="img" aria-label="{name} page {page} with layout boxes">
      <image href="data:image/jpeg;base64,{src}" width="{w}" height="{h}"/>
      {''.join(boxes)}
    </svg>
    <pre>{html.escape(result["markdown"]["text"])}</pre>
  </div>
</section>"""


def main() -> None:
    batch, out = json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2])
    names = ["de-rechnung", "de-pythagoras", "de-maxwell"]
    by_name = dict(zip(names, batch["results"]))
    rows = "".join(
        f'<tr><td><a href="{SOURCES[n]}" target="_blank" rel="noopener noreferrer">{n}.pdf</a></td>'
        f'<td>{len(by_name[n]["result"]["layoutParsingResults"])}</td><td>{by_name[n]["errorMsg"]}</td></tr>'
        for n in names
    )
    legend = " ".join(f'<span style="color:{c}">■ {l}</span>' for l, c in COLORS.items())
    sections = "".join(page_section(n, p, by_name[n]["result"]["layoutParsingResults"][p - 1]) for n, p in PAGES)
    out.write_text(f"""<!doctype html>
<html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PaddleOCR-VL 1.6 verification</title>
<style>
body{{background:#000;color:#fff;font:14px/1.45 system-ui,sans-serif;margin:0 auto;padding:16px;max-width:1400px}}
h1{{font-size:20px;margin:0 0 8px}} h2{{font-size:16px;margin:24px 0 4px}}
a{{color:#8ab4ff}} table{{border-collapse:collapse;margin:8px 0}} td,th{{border:1px solid #333;padding:4px 8px;text-align:left}}
.meta{{color:#aaa;margin:0 0 8px;font-size:12px}}
.row{{display:flex;gap:12px;flex-wrap:wrap;align-items:flex-start}}
svg{{max-width:100%;height:auto;background:#fff;flex:0 0 auto}}
pre{{flex:1 1 420px;min-width:0;background:#111;border:1px solid #333;padding:8px;white-space:pre-wrap;word-break:break-word;font-size:12px;max-height:{THUMB_WIDTH * 1.45:.0f}px;overflow:auto;margin:0}}
</style></head><body>
<h1>PaddleOCR-VL 1.6 on Runpod: verification</h1>
<p>Official PaddleX pipeline (PP-DocLayoutV3 layout + PaddleOCR-VL-1.6 on the Runpod vLLM endpoint). One <code>/layout-parsing/batch</code> call, all three documents. Left: page with detected blocks. Right: the <code>markdown.text</code> PaddleX returned for that page, unmodified.</p>
<table><tr><th>Test PDF</th><th>Pages</th><th>Status</th></tr>{rows}</table>
<p class="meta">{legend} <span style="color:#9b9b9b">■ text/other</span></p>
{sections}
</body></html>""")
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
