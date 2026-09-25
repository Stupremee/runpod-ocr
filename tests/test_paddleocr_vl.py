from types import SimpleNamespace

from runpod_ocr.engines.paddleocr_vl import _block


def raw(label, content):
    return SimpleNamespace(label=label, content=content, bbox=[1, 2, 3, 4])


def test_formula_delimiters_are_stripped_to_raw_latex():
    b = _block(0, raw("display_formula", " $$ c^{2}=a^{2}+b^{2} $$ "))
    assert (b.label, b.content) == ("formula", "c^{2}=a^{2}+b^{2}")


def test_tables_are_normalized_to_markdown_and_cells():
    b = _block(3, raw("table", "<table><tr><td>Pos</td><td>Menge</td></tr><tr><td>1</td><td>20 Stk</td></tr></table>"))
    assert b.content.splitlines()[0] == "| Pos | Menge |"
    assert b.table[1][1].text == "20 Stk"


def test_unknown_labels_fall_back_to_text():
    assert _block(0, raw("vertical_text", "abc")).label == "text"
