# OCR on Runpod

OCR models run as **one scale-to-zero Runpod serverless endpoint per model**
(vLLM, OpenAI-compatible). Each model family gets its **own service**, which
runs that family's official pipeline and API and calls its endpoint for GPU
inference. Your backend calls the services directly. Response shapes differ
per model on purpose.

```
                       services/ (Docker, CPU)            Runpod serverless (GPU)
backend ──HTTP──▶ paddleocr-vl   PaddleX pipeline  ──▶   ocr-paddleocr-vl-1-6  (vLLM)
          └─────▶ <next model>   its own pipeline  ──▶   ocr-<next model>      (vLLM)
```

| Service | Model | Docs |
| --- | --- | --- |
| `services/paddleocr-vl` | PaddleOCR-VL 1.6 via the official PaddleX pipeline | [README](services/paddleocr-vl/README.md) |

## GPU backends

`runpod_ocr/backends.py` defines the endpoints: one worker-vllm image, one fixed
model per endpoint, tuned for batches (scale to zero, 128 concurrent requests per
worker, 5 min keep-warm, a second worker only after 30s of queueing).
Separate endpoints mean each model scales, sizes its GPU and cold-starts on its
own, and two models can run at the same time.

```bash
# .env (not committed)
RUNPOD_API_KEY=...

uv run runpod-ocr backend up <model>   # provision or update via Flash; writes <MODEL>_ENDPOINT_ID to .env
```

`backend up` is idempotent. Flash tracks endpoints by name in `.flash/`, so
config changes update the same endpoint in place.

## Adding a model

1. Add a `VllmBackend` entry to `runpod_ocr/backends.py` and run `backend up`.
2. Create `services/<model>/` with that model's official pipeline or serving app,
   pointed at the endpoint's OpenAI URL (`https://api.runpod.ai/v2/<id>/openai/v1`).
   The PaddleOCR-VL service's cold-start guard and batch route are small enough
   to copy; extract them into a shared module once a second service needs them.

## Repo

- `runpod_ocr/`: GPU backend definitions and the `runpod-ocr` CLI
- `services/`: one OCR service per model family
- `samples/`: German test PDFs
- `uv run pytest`: unit tests of all services
