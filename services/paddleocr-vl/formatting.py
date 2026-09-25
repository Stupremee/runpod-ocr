"""Optional post-processing of PaddleX layout-parsing results (stdlib only).

- `tables_to_markdown`: HTML tables -> GitHub-flavoured Markdown tables
- `layout_text`: a page's blocks placed on a character grid by their bboxes,
  like `pdftotext -layout`, built from the blocks PaddleX already returns
"""

import re
import statistics
import textwrap
from html import unescape
from html.parser import HTMLParser
from typing import Any

# an HTML table, including the centering <div> PaddleX adds when prettifying
_TABLE = re.compile(r"(?:<div[^>]*>\s*)?<table\b.*?</table>(?:\s*</div>)?", re.DOTALL | re.IGNORECASE)
_TAG = re.compile(r"<[^>]+>")
_FORMULA_DELIMITERS = re.compile(r"^\s*\${1,2}\s*|\s*\${1,2}\s*$")
# character cell height/width used to size the grid; calibrated on body text so
# wrapped lines match the page's line length
_ASPECT = 2.0

type Grid = list[list[str]]


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[tuple[str, int, int]]] = []  # (text, rowspan, colspan)
        self._cell: tuple[int, int] | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self.rows.append([])
        elif tag in ("td", "th"):
            a = dict(attrs)
            self._cell = (int(a.get("rowspan") or 1), int(a.get("colspan") or 1))
            self._text = []
        elif tag == "br":
            self._text.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._cell is not None:
            if not self.rows:
                self.rows.append([])
            # the VLM writes line breaks inside cells as a literal "\n"
            text = " ".join("".join(self._text).replace("\\n", " ").split())
            self.rows[-1].append((text, *self._cell))
            self._cell = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._text.append(data)


def table_grid(html: str) -> Grid:
    """Dense grid of cell texts; a merged cell's text sits in its top-left slot."""
    parser = _TableParser()
    parser.feed(html)
    slots: dict[tuple[int, int], str] = {}
    for r, row in enumerate(parser.rows):
        c = 0
        for text, row_span, col_span in row:
            while (r, c) in slots:
                c += 1
            for dr in range(row_span):
                for dc in range(col_span):
                    slots[(r + dr, c + dc)] = text if (dr, dc) == (0, 0) else ""
            c += col_span
    if not slots:
        return []
    height = max(r for r, _ in slots) + 1
    width = max(c for _, c in slots) + 1
    return [[slots.get((r, c), "") for c in range(width)] for r in range(height)]


def markdown_table(grid: Grid) -> str:
    def row(cells: list[str]) -> str:
        return "| " + " | ".join(cell.replace("|", "\\|") for cell in cells) + " |"

    if not grid:
        return ""
    return "\n".join([row(grid[0]), row(["---"] * len(grid[0])), *map(row, grid[1:])])


def tables_to_markdown(text: str) -> str:
    return _TABLE.sub(lambda m: "\n" + markdown_table(table_grid(m.group(0))) + "\n", text)


def _block_lines(block: dict[str, Any], width: int) -> list[str]:
    content: str = block.get("block_content") or ""
    if "<table" in content:
        grid = table_grid(content)
        widths = [max(len(row[c]) for row in grid) for c in range(len(grid[0]))] if grid else []
        return ["  ".join(cell.ljust(w) for cell, w in zip(row, widths)).rstrip() for row in grid]
    if "formula" in block.get("block_label", ""):
        return [_FORMULA_DELIMITERS.sub("", content)]  # never wrap LaTeX
    content = unescape(_TAG.sub("", content))
    lines: list[str] = []
    for line in content.splitlines():
        # never split words: a heading wider than its box just overflows
        lines += textwrap.wrap(" ".join(line.split()), width, break_long_words=False, break_on_hyphens=False) or [""]
    return lines


def layout_text(page: dict[str, Any]) -> str:
    """Render a page's `prunedResult` as monospaced text that mirrors its layout.

    Character size is estimated from the text blocks (how much text fills how
    much area), so columns, side-by-side blocks and captions land where they
    are on the page. Line breaks inside a block are re-wrapped to its width.
    """
    blocks = [b for b in page["parsing_res_list"] if (b.get("block_content") or "").strip()]
    if not blocks:
        return ""

    # treat a character cell as ASPECT times taller than wide: area = chars * cw * ASPECT*cw
    estimates = [
        ((x1 - x0) * (y1 - y0) / (_ASPECT * len(b["block_content"]))) ** 0.5
        for b in blocks
        if b["block_label"] == "text" and len(b["block_content"]) >= 40
        for x0, y0, x1, y1 in [b["block_bbox"]]
    ]
    char_w = statistics.median(estimates) if estimates else page["width"] / 100
    line_h = _ASPECT * char_w

    rows: dict[int, dict[int, str]] = {}
    for block in blocks:
        x0, y0, x1, _ = block["block_bbox"]
        col = round(x0 / char_w)
        lines = _block_lines(block, max(8, int((x1 - x0) / char_w)))
        span = max((len(line) for line in lines), default=0)
        # start at the block's row, pushed down past anything already drawn there
        top = round(y0 / line_h)
        while any(
            any(col <= c < col + span or c <= col < c + len(s) for c, s in rows.get(top + i, {}).items())
            for i in range(len(lines))
        ):
            top += 1
        for i, line in enumerate(lines):
            rows.setdefault(top + i, {})[col] = line

    out: list[str] = []
    for r in range(max(rows) + 1):
        line = ""
        for c, s in sorted(rows.get(r, {}).items()):
            line = line.ljust(c) + s if len(line) <= c else line + " " + s
        out.append(line.rstrip())
    # drop the page's left margin and runs of blank lines
    return re.sub(r"\n{3,}", "\n\n", textwrap.dedent("\n".join(out))).strip("\n")
