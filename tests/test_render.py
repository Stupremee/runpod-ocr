from runpod_ocr.render import keep_block, table_from_html, table_markdown
from runpod_ocr.schema import Block, OcrOptions


def block(label, content="x") -> Block:
    return Block(id=0, label=label, source_label=label, bbox=(0, 0, 1, 1), content=content)


def test_spanned_html_table_becomes_markdown_grid():
    rows = table_from_html(
        '<table><tr><td colspan="2">Summe</td><td>A|B</td></tr>'
        '<tr><td rowspan="2">1</td><td>2</td><td>3</td></tr><tr><td>4</td><td>5</td></tr></table>'
    )
    assert rows[0][0].col_span == 2 and rows[1][0].row_span == 2
    assert table_markdown(rows).splitlines() == [
        "| Summe |  | A\\|B |",
        "| --- | --- | --- |",
        "| 1 | 2 | 3 |",
        "|  | 4 | 5 |",
    ]


def test_options_filter_blocks_by_unified_label():
    defaults = OcrOptions()
    assert keep_block(block("formula"), defaults)
    assert not keep_block(block("formula"), OcrOptions(formulas=False))
    assert not keep_block(block("page_number"), defaults)
    assert keep_block(block("footer"), OcrOptions(headers_footers=True))
    assert not keep_block(block("image"), OcrOptions(images="omit"))
