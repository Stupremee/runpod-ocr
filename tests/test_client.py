import pytest

from runpod_ocr.client import build_request, parse_output
from runpod_ocr.models import MODELS

PNG = b"\x89PNG fake"


def test_paddle_request_uses_task_prompt_and_inlines_bytes():
    req = build_request(MODELS["paddleocr-vl-1.6"], PNG, task="table")
    body = req["openai_input"]
    assert req["openai_route"] == "/v1/chat/completions"
    assert body["model"] == "PaddlePaddle/PaddleOCR-VL-1.6"
    [user] = body["messages"]
    image, text = user["content"]
    assert image["image_url"]["url"].startswith("data:image/png;base64,")
    assert text == {"type": "text", "text": "Table Recognition:"}


def test_teleocr_request_has_system_prompt_and_passes_urls_through():
    req = build_request(MODELS["teleocr"], "https://x.test/a.jpg", prompt="custom")
    system, user = req["openai_input"]["messages"]
    assert system["role"] == "system"
    assert user["content"][0]["image_url"]["url"] == "https://x.test/a.jpg"
    assert user["content"][1]["text"] == "custom"


def test_unknown_task_lists_valid_tasks():
    with pytest.raises(ValueError, match="seal"):
        build_request(MODELS["paddleocr-vl-1.6"], PNG, task="code")


def test_parse_output_unwraps_stream_aggregate():
    out = [{"choices": [{"message": {"content": "hello"}}], "usage": {"total_tokens": 3}}]
    assert parse_output(out).text == "hello"


def test_parse_output_surfaces_worker_errors():
    with pytest.raises(RuntimeError, match="out of memory"):
        parse_output([{"error": {"message": "out of memory", "type": "startup_error"}}])
