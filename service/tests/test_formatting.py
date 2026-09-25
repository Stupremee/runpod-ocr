from formatting import layout_text, tables_to_markdown

TABLE = (
    "<table border=1 style='margin: auto;'><tr><td colspan=\"2\">Summe</td><td>A|B</td></tr>"
    "<tr><td rowspan=\"2\">Pos\\n1</td><td>2</td><td>3</td></tr><tr><td>4</td><td>5</td></tr></table>"
)


def test_html_tables_become_markdown_with_spans_expanded():
    md = tables_to_markdown(f"Rechnung\n\n<div style=\"text-align: center;\">{TABLE}</div>\n\nEnde")
    assert md.splitlines() == [
        "Rechnung",
        "",
        "",
        "| Summe |  | A\\|B |",
        "| --- | --- | --- |",
        "| Pos 1 | 2 | 3 |",
        "|  | 4 | 5 |",
        "",
        "",
        "Ende",
    ]


def block(label, bbox, content):
    return {"block_label": label, "block_bbox": bbox, "block_content": content}


def test_layout_text_keeps_side_by_side_blocks_on_the_same_lines():
    body = "Wort " * 30
    page = {
        "width": 1000,
        "height": 1000,
        "parsing_res_list": [
            block("paragraph_title", [100, 100, 400, 120], "Links"),
            block("text", [100, 140, 450, 300], body),
            block("text", [550, 140, 900, 300], body),
            block("display_formula", [100, 320, 200, 340], " $$ c^{2}=a^{2}+b^{2} $$ "),
            block("image", [100, 400, 300, 600], ""),
        ],
    }
    lines = layout_text(page).splitlines()
    assert lines[0] == "Links"
    # both columns start on the same row, the right one indented past the left
    row = next(line for line in lines if line.startswith("Wort"))
    assert row.count("Wort") > row.split("  ")[0].count("Wort")
    assert "c^{2}=a^{2}+b^{2}" in lines  # formulas unwrapped, delimiters stripped
