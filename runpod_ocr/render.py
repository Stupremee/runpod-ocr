"""Engine-agnostic post-processing: option filtering and Markdown / text rendering."""

from html.parser import HTMLParser

from .schema import Block, Label, OcrOptions, TableCell

_PAGE_FURNITURE: set[Label] = {"header", "footer", "page_number"}


def keep_block(block: Block, options: OcrOptions) -> bool:
    """Whether `options` keep this block in the output."""
    match block.label:
        case "formula" | "formula_number":
            return options.formulas
        case "table":
            return options.tables
        case "image" | "chart" | "seal":
            return options.images != "omit"
        case label if label in _PAGE_FURNITURE:
            return options.headers_footers
        case _:
            return True


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[TableCell]] = []
        self._cell: dict[str, int] | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self.rows.append([])
        elif tag in ("td", "th"):
            spans = dict(attrs)
            self._cell = {
                "row_span": int(spans.get("rowspan") or 1),
                "col_span": int(spans.get("colspan") or 1),
            }
            self._text = []
        elif tag == "br" and self._cell is not None:
            self._text.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._cell is not None:
            if not self.rows:
                self.rows.append([])
            self.rows[-1].append(TableCell(text="".join(self._text).strip(), **self._cell))
            self._cell = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._text.append(data)


def table_from_html(html: str) -> list[list[TableCell]]:
    """Parse the simple <table><tr><td> markup VLMs emit into rows of cells."""
    parser = _TableParser()
    parser.feed(html.replace("\\n", "\n"))
    return [row for row in parser.rows if row]


def _grid(rows: list[list[TableCell]]) -> list[list[str]]:
    """Expand row/col spans into a dense grid; spanned slots stay empty."""
    grid: dict[tuple[int, int], str] = {}
    width = 0
    for r, row in enumerate(rows):
        c = 0
        for cell in row:
            while (r, c) in grid:
                c += 1
            for dr in range(cell.row_span):
                for dc in range(cell.col_span):
                    grid[(r + dr, c + dc)] = cell.text if (dr, dc) == (0, 0) else ""
            c += cell.col_span
            width = max(width, c)
    height = max((r for r, _ in grid), default=-1) + 1
    return [[grid.get((r, c), "") for c in range(width)] for r in range(height)]


def table_markdown(rows: list[list[TableCell]]) -> str:
    grid = _grid(rows)
    if not grid:
        return ""

    def fmt(cells: list[str]) -> str:
        return "| " + " | ".join(t.replace("|", "\\|").replace("\n", " ") for t in cells) + " |"

    lines = [fmt(grid[0]), "| " + " | ".join("---" for _ in grid[0]) + " |"]
    lines += [fmt(row) for row in grid[1:]]
    return "\n".join(lines)


def table_text(rows: list[list[TableCell]]) -> str:
    return "\n".join("\t".join(t.replace("\n", " ") for t in row) for row in _grid(rows))


def block_markdown(block: Block, page_number: int) -> str:
    match block.label:
        case "title":
            return f"# {block.content}"
        case "heading":
            return f"## {block.content}"
        case "formula":
            return f"$$\n{block.content}\n$$"
        case "code":
            return f"```\n{block.content}\n```"
        case "image" | "chart" | "seal" if not block.content:
            return f"![{block.label}](page-{page_number}-block-{block.id})"
        case "caption":
            return f"*{block.content}*"
        case _:
            return block.content


def block_text(block: Block) -> str:
    if block.table is not None:
        return table_text(block.table)
    if block.label in ("image", "chart", "seal") and not block.content:
        return ""
    return block.content


def page_markdown(blocks: list[Block], page_number: int) -> str:
    return "\n\n".join(md for b in blocks if (md := block_markdown(b, page_number)))


def page_text(blocks: list[Block]) -> str:
    return "\n\n".join(t for b in blocks if (t := block_text(b)))
