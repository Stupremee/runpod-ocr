# PaddleOCR-VL 1.6 service

Runs the **official PaddleX
`PaddleOCR-VL-1.6` pipeline** and returns the standard PaddleX serving
response. Only the vision-language model runs on a GPU, on a scale-to-zero
Runpod serverless vLLM endpoint.

```
backend ──HTTP──▶ this service (Docker, CPU)                    Runpod serverless (GPU)
                  PaddleX serving app                          worker-vllm
                  PP-DocLayoutV3 layout (PyTorch, CPU)  ──▶    PaddleOCR-VL-1.6
                  one VLM request per block, in parallel       (OpenAI-compatible)
```

## Setup

```bash
# .env (not committed)
RUNPOD_API_KEY=...

uv run runpod-ocr backend up paddleocr-vl-1.6   # provisions/updates the vLLM endpoint via Flash,
                                               # writes PADDLEOCR_VL_ENDPOINT_ID to .env
docker compose -f services/paddleocr-vl/compose.yaml up -d --build   # service on :8080
```

Run both from the repo root.

## API

The official PaddleX routes, plus a batch route and a warm-up route:

| Route | Body | Returns |
| --- | --- | --- |
| `POST /layout-parsing` | [PaddleX request](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html): `{"file": <url or base64>, "fileType": 0 (PDF) or 1 (image), ...options}`, plus `tableFormat` and `layoutText` (below) | PaddleX response: `result.layoutParsingResults[]` with `prunedResult` (blocks, bboxes, labels) and `markdown.text` per page |
| `POST /layout-parsing/batch` | `{"requests": [<layout-parsing request>, ...]}` | `{"results": [<layout-parsing response>, ...]}`, same order |
| `POST /warmup` | none | `{"status": "warm"}` once a GPU worker is serving |
| `GET /health` | none | PaddleX health check |

All PaddleX options work per request (`useLayoutDetection`, `useChartRecognition`,
`useSealRecognition`, `useOcrForImageBlock`, `markdownIgnoreLabels`, `maxPixels`,
`prettifyMarkdown`, `restructurePages`, ...). Pass `"visualize": false` unless you
want the annotated images back.

### Extra request options

These are added on top of PaddleX and work on `/layout-parsing` and per item in `/batch`:

- `tableFormat`: `"html"` (default, what PaddleX returns) or `"markdown"`. With
  `markdown`, tables in `markdown.text` and table blocks in `prunedResult` become
  GitHub Markdown tables. Markdown can't express merged cells, so a merged cell's
  text goes in its first slot and the rest stay empty. `html` is lossless.
- `layoutText`: `false` (default) or `true`. Adds `layoutText` to every page: plain
  text placed on a character grid by block position, like `pdftotext -layout`.
  Columns, side-by-side blocks and captions keep their place, and tables come out
  as aligned columns. Line breaks inside a block are re-wrapped to its width
  (block bboxes only, no per-line positions).

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

## Files

- `app.py`: the service (PaddleX app, cold-start guard, batch route, options)
- `pipeline.yaml`: official PaddleX config, pointed at the Runpod backend
- `formatting.py`: `tableFormat` and `layoutText`; tests in `tests/`
- `scripts/report.py`: HTML verification report from a batch response, run from the repo root
