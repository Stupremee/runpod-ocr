# PaddleOCR-VL 1.6 service

Runs the **official PaddleX `PaddleOCR-VL-1.6` pipeline** behind an **async job
API with webhooks**. Layout detection (PP-DocLayoutV3) runs here on CPU; the
vision-language model runs on a scale-to-zero Runpod serverless vLLM endpoint.

```
backend ──HTTP──▶ this service (Docker, CPU)                  Runpod serverless (GPU)
   ▲              job queue (SQLite) → PaddleX pipeline  ──▶  PaddleOCR-VL-1.6 (vLLM)
   └── webhooks ─ PP-DocLayoutV3 layout, one VLM request per block
```

## Setup

From the repo root:

```bash
# .env (not committed)
RUNPOD_API_KEY=...
WEBHOOK_SECRET=whsec_...   # base64 key; signs webhooks

uv run runpod-ocr backend up paddleocr-vl-1.6            # GPU endpoint; writes PADDLEOCR_VL_ENDPOINT_ID to .env
docker compose -f services/paddleocr-vl/compose.yaml up -d --build   # service on :8080
```

**Without GPU:** `OCR_BACKEND=emulated docker compose -f services/paddleocr-vl/compose.yaml up -d --build`
runs everything for real except inference. Layout detection, bboxes, jobs,
retries and webhooks all work, and the VLM is a local stub that answers every
task prompt with fixed text in the model's own formats (OTSL tables, LaTeX).
The cold start is simulated (`EMULATED_COLD_START_SECONDS`, default 5). No
Runpod credentials are needed.

## Async API

| Route | |
| --- | --- |
| `POST /jobs` | `{"request": <layout-parsing request>, "webhookUrl"?, "metadata"?}` → `202` Job |
| `POST /batches` | `{"requests": [<layout-parsing request>, ...], "webhookUrl"?, "metadata"?}` (max 500) → `202` Batch |
| `GET /jobs/{id}` | Job with `result` once succeeded (`?includeResult=false` to skip it) |
| `GET /batches/{id}` | Batch with counts and job statuses (`?includeResults=true` to inline results) |
| `DELETE /jobs/{id}` | Cancel a queued job (`409` once running or finished) |
| `DELETE /batches/{id}` | Cancel the batch's queued jobs |

Interactive docs: `GET /docs`.

- **Layout-parsing request:** the [PaddleX request](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html)
  (`{"file": <url or base64>, "fileType": 0 (PDF) or 1 (image), ...options}`) plus
  `tableFormat` and `layoutText` (below). It's validated against PaddleX's own
  schema when you submit, so a bad request gets `422` immediately instead of a failed job.
- **`result`:** PaddleX's `result` object, `layoutParsingResults[]` with `prunedResult`
  (blocks, labels, bboxes) and `markdown.text` per page.
- **Job `status`:** `queued` → `running` → `succeeded` | `failed` | `cancelled`.
  `error` is `{code, message}` with `code` one of `invalid_request` (PaddleX
  rejected the input, not retried), `processing_error` or `backend_unavailable`
  (retried up to `JOB_MAX_ATTEMPTS` with backoff; `error` shows the last failure
  while it retries).
- **Batch `status`:** `queued` → `running` → `completed` (every job finished, whatever the outcome).
- **`metadata`:** stored and echoed back on reads and webhooks, e.g. your document id.
- **`Idempotency-Key` header** on `POST`: resubmitting with the same key returns the
  existing job or batch instead of creating a new one.
- **Durability:** jobs live in SQLite on the `/data` volume. Jobs interrupted by a
  restart are requeued, and pending webhooks survive restarts. Finished jobs are
  deleted after `RETENTION_DAYS`.

### Webhooks

Set `webhookUrl` on a job or batch to get a POST when it finishes:

| Event | When | `data` |
| --- | --- | --- |
| `job.succeeded`, `job.failed`, `job.cancelled` | a job submitted via `POST /jobs` finishes | the Job, with `result` |
| `batch.completed` | every job of a batch has finished | the Batch summary (fetch results with `GET /batches/{id}?includeResults=true`) |

Body: `{"id", "type", "timestamp", "data"}`. Webhooks follow the
[Standard Webhooks](https://www.standardwebhooks.com) spec. Verify the
`webhook-signature` header with any Standard Webhooks library and `WEBHOOK_SECRET`,
and dedupe on `webhook-id`. Anything other than a 2xx is retried after
10s, 30s, 2m, 10m, 30m, 1h and 3h. Without `WEBHOOK_SECRET`, webhooks are sent unsigned.

### Extra request options

- `tableFormat`: `"html"` (default, what PaddleX returns) or `"markdown"`. With
  `markdown`, tables in `markdown.text` and table blocks in `prunedResult` become
  GitHub Markdown tables. Markdown can't express merged cells, so a merged cell's
  text goes in its first slot. `html` is lossless.
- `layoutText`: `false` (default) or `true`. Adds `layoutText` to every page: plain
  text placed on a character grid by block position, like `pdftotext -layout`.
  Columns, side-by-side blocks and captions keep their place, and tables come out
  as aligned columns. Line breaks inside a block are re-wrapped to its width
  (block bboxes only, no per-line positions).

### Synchronous route

`POST /layout-parsing` (the official PaddleX route, plus the options above) is kept
for quick tests. It blocks for the whole parse, including any cold start, so use
the job API for real work.

## Cold starts and batching

The GPU endpoint scales to zero, and a cold boot takes about 3 to 7 minutes.

- **One cold start per burst.** Before processing, a shared guard sends a tiny probe
  job to Runpod and waits until it completes (meaning vLLM is serving). Every
  worker waits on that same probe, so a burst of jobs pays for one cold start and
  no VLM call times out while a worker boots.
- **Jobs run oldest first**, `WORKER_CONCURRENCY` at a time, so a batch's documents
  run together while the GPU is warm. Their VLM calls overlap on the GPU.
- **The endpoint stays warm for 60 seconds** after the last request. Jobs submitted
  within that window don't cold start again, so send related work together (or as a batch). `POST /warmup` wakes it ahead of time.
- **One GPU worker takes up to 128 requests at once.** A second worker only starts
  after 30s of queueing. Max 2 workers.

## Configuration (env)

| Var | Default | |
| --- | --- | --- |
| `OCR_BACKEND` | `runpod` | `emulated` for testing without GPU |
| `RUNPOD_API_KEY`, `PADDLEOCR_VL_ENDPOINT_ID` | required for `runpod` | Runpod credentials and endpoint |
| `WEBHOOK_SECRET` | none | `whsec_<base64>` signing key |
| `WORKER_CONCURRENCY` | 4 | jobs processed at once; each needs about 1 to 2 GB of RAM |
| `JOB_MAX_ATTEMPTS` | 3 | attempts for retryable failures |
| `RETENTION_DAYS` | 7 | delete finished jobs after this |
| `WARM_TTL_SECONDS` | 45 | skip the warm-up probe this long after the last traffic; keep below the endpoint's 60s idle timeout |
| `COLD_START_TIMEOUT_SECONDS` | 900 | give up waiting for a GPU worker after this |
| `EMULATED_COLD_START_SECONDS`, `EMULATED_LATENCY_SECONDS` | 5, 0.2 | emulated backend timings |
| `DEVICE` | `cpu` | layout model device (`gpu:0` if the host has a GPU) |
| `DATA_DIR` | `/data` | SQLite location |

## Costs

GPU: RTX A5000/3090/L4 class at $0.69/hr, or RTX 4090 at $1.10/hr, billed per second
while a worker is up, including the 60 second idle tail. A burst costs about $0.06
fixed (cold start and idle tail) plus about $0.44 per 1,000 pages. Emulated mode costs nothing.
The service needs about 2 to 8 GB of RAM depending on `WORKER_CONCURRENCY`.

## Files

- `app.py`: wiring (PaddleX app, warm-up guard, job runner, webhooks, sync routes)
- `api.py`, `jobs.py`, `runner.py`, `webhooks.py`: async job API, SQLite queue, workers, delivery
- `emulated.py`: GPU-free backend
- `formatting.py`: `tableFormat` and `layoutText`
- `pipeline.yaml`: official PaddleX config, pointed at the Runpod backend
- `tests/`: unit tests (`uv run pytest` from the repo root)
- `scripts/report.py`: HTML verification report
- `deploy/nixos-rome.md`: NixOS deployment guide (module + auto-updates) for the nix repo

## Image

CI (`.github/workflows/paddleocr-vl-image.yml`) runs the tests and publishes
`ghcr.io/stupremee/paddleocr-vl` on every push to `main`, tagged `main` (moving)
and `sha-<commit>`.
