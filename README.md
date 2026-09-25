# PaddleOCR-VL 1.6 service on Runpod

Your backend calls a small OCR service that runs the **official PaddleX
`PaddleOCR-VL-1.6` pipeline** and returns the standard PaddleX serving
response. Only the vision-language model runs on a GPU, on a scale-to-zero
Runpod serverless vLLM endpoint.

```
backend ──HTTP──▶ service/ (Docker, CPU)                      Runpod serverless (GPU)
                  PaddleX serving app                          worker-vllm
                  PP-DocLayoutV3 layout (PyTorch, CPU)  ──▶    PaddleOCR-VL-1.6
                  one VLM request per block, in parallel       (OpenAI-compatible)
```

## Setup

```bash
# .env (not committed)
RUNPOD_API_KEY=...

uv run runpod-ocr backend up          # provisions/updates the vLLM endpoint via Flash,
                                      # writes PADDLEOCR_VL_ENDPOINT_ID to .env
docker compose -f service/compose.yaml up -d --build   # service on :8080
```

`runpod-ocr backend up` is idempotent. Flash tracks the endpoint by name in
`.flash/`, so config changes in `runpod_ocr/backends.py` update the same
endpoint in place.

## API

The official PaddleX routes, plus a batch route and a warm-up route:

| Route | Body | Returns |
| --- | --- | --- |
| `POST /layout-parsing` | [PaddleX request](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html): `{"file": <url or base64>, "fileType": 0 (PDF) or 1 (image), ...options}` | PaddleX response: `result.layoutParsingResults[]` with `prunedResult` (blocks, bboxes, labels) and `markdown.text` per page |
| `POST /layout-parsing/batch` | `{"requests": [<layout-parsing request>, ...]}` | `{"results": [<layout-parsing response>, ...]}`, same order |
| `POST /warmup` | none | `{"status": "warm"}` once a GPU worker is serving |
| `GET /health` | none | PaddleX health check |

All PaddleX options work per request (`useLayoutDetection`, `useChartRecognition`,
`useSealRecognition`, `useOcrForImageBlock`, `markdownIgnoreLabels`, `maxPixels`,
`prettifyMarkdown`, `restructurePages`, ...). Pass `"visualize": false` unless you
want the annotated images back.

## Batching and cold starts

The GPU endpoint scales to zero, and a cold boot takes about 3 to 7 minutes
(image pull, weights, vLLM start). The service handles this as follows:

- **One cold start per batch.** Before any parsing, a shared guard sends a
  tiny probe job to Runpod and waits until it completes, meaning vLLM is
  serving. Every concurrent request waits on that same probe, so a burst never
  triggers several cold starts, and no VLM call times out while a worker boots.
- **Use `/layout-parsing/batch`** for bulk work. All files share the warm
  GPU. Each file goes through the official route, so every result is exactly
  what PaddleX returns. `BATCH_CONCURRENCY` (default 4) files run at a time,
  overlapping on the GPU (`PADDLE_PDX_SERVING_SERIAL_PIPELINE_CALLS=False`).
- **Closing the connection doesn't cancel work.** PaddleX keeps parsing a
  request after the client disconnects, so retry with care.
- **Pre-warm** with `POST /warmup` a few minutes before a known batch.
- **The endpoint stays warm for 5 minutes** after the last request
  (`idle_timeout=300`). The service assumes the GPU is warm for `WARM_TTL_SECONDS`
  (default 240) after its last traffic. Batches closer together than that
  don't cold start again.
- **One worker takes up to 128 requests at once** (`MAX_CONCURRENCY`) and
  vLLM batches them on the GPU. A second worker (another cold start) only
  starts if jobs wait over 30 seconds in the queue. Max 2 workers.

Your backend's HTTP timeout must cover a cold start plus the batch. Use 20+ minutes
for `/layout-parsing/batch`.

## Service configuration (env)

| Var | Default | |
| --- | --- | --- |
| `RUNPOD_API_KEY`, `PADDLEOCR_VL_ENDPOINT_ID` | required | Runpod credentials and backend endpoint |
| `WARM_TTL_SECONDS` | 240 | skip the warm-up probe this long after the last traffic; keep below the endpoint's 300s idle timeout |
| `COLD_START_TIMEOUT_SECONDS` | 900 | give up waiting for a worker after this |
| `BATCH_CONCURRENCY` | 4 | files per batch parsed at once; each needs about 1 to 2 GB of RAM |
| `DEVICE` | `cpu` | layout model device (`gpu:0` if the host has a GPU) |

## Costs

GPU: RTX A5000/3090/L4 class (`AMPERE_24`) at $0.69/hr, or RTX 4090 at $1.10/hr,
billed per second only while a worker is up, including the 5 minute idle tail.
The service itself needs about 2 to 8 GB of RAM depending on `BATCH_CONCURRENCY`.

## Layout

- `runpod_ocr/backends.py`: Runpod vLLM endpoint definition (Flash, image mode)
- `runpod_ocr/cli.py`: `runpod-ocr backend up`
- `service/`: the OCR service (`app.py`, the `pipeline.yaml` official config with the Runpod backend, Dockerfile, compose)
- `samples/`: German test PDFs, `scripts/report.py`: HTML verification report
