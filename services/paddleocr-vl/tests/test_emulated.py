from fastapi import FastAPI
from fastapi.testclient import TestClient

import emulated


def test_stub_answers_each_task_prompt_in_the_models_format():
    app = FastAPI()
    app.include_router(emulated.create_router("PaddlePaddle/PaddleOCR-VL-1.6"))
    client = TestClient(app)

    def ask(prompt: str) -> str:
        messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:,"}}, {"type": "text", "text": prompt}]}]
        response = client.post("/emulated/v1/chat/completions", json={"model": "m", "messages": messages})
        return response.json()["choices"][0]["message"]["content"]

    assert ask("Table Recognition:").startswith("<fcel>")  # OTSL, like the real model
    assert ask("Formula Recognition:") == "\\[E = mc^{2}\\]"
    assert ask("OCR:") == emulated._DEFAULT_ANSWER
    assert client.get("/emulated/v1/models").json()["data"][0]["id"] == "PaddlePaddle/PaddleOCR-VL-1.6"
