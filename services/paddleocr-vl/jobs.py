"""Durable job queue (SQLite) for async OCR.

A job is one layout-parsing request. A batch groups jobs submitted together;
it completes when all of its jobs have finished. Terminal transitions enqueue
webhook events (`job.succeeded|failed|cancelled`, `batch.completed`), which
`webhooks.py` delivers. All state lives in one SQLite file, so queued jobs and
pending webhooks survive restarts.
"""

import json
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any, Literal

JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]
TERMINAL: tuple[JobStatus, ...] = ("succeeded", "failed", "cancelled")

_SCHEMA = """
create table if not exists batches (
  id text primary key,
  created_at real not null,
  webhook_url text,
  metadata text,
  idempotency_key text unique
);
create table if not exists jobs (
  id text primary key,
  batch_id text references batches(id) on delete cascade,
  position integer not null default 0,
  status text not null,
  request text not null,
  webhook_url text,
  metadata text,
  idempotency_key text unique,
  attempts integer not null default 0,
  run_after real not null,
  created_at real not null,
  started_at real,
  finished_at real,
  result text,
  error text
);
create index if not exists jobs_queue on jobs (status, run_after, created_at, position);
create index if not exists jobs_batch on jobs (batch_id, position);
create table if not exists webhooks (
  id text primary key,
  event text not null,
  target_id text not null,
  url text not null,
  attempts integer not null default 0,
  next_attempt_at real,
  delivered_at real,
  last_error text,
  created_at real not null,
  unique (event, target_id)
);
create index if not exists webhooks_due on webhooks (next_attempt_at);
"""


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _iso(ts: float | None) -> str | None:
    return datetime.fromtimestamp(ts, UTC).isoformat().replace("+00:00", "Z") if ts else None


def _json(text: str | None) -> Any:
    return json.loads(text) if text else None


class JobStore:
    def __init__(self, path: str) -> None:
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("pragma journal_mode=wal")
        self._db.execute("pragma foreign_keys=on")
        self._db.executescript(_SCHEMA)
        self._lock = threading.Lock()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._db.execute("begin immediate")
            try:
                yield self._db
                self._db.execute("commit")
            except BaseException:
                self._db.execute("rollback")
                raise

    # -- submit -------------------------------------------------------------

    def create_job(
        self,
        request: dict[str, Any],
        *,
        webhook_url: str | None = None,
        metadata: Any = None,
        idempotency_key: str | None = None,
    ) -> str:
        """Returns the new job id, or the existing one for a reused idempotency key."""
        with self._tx() as db:
            if idempotency_key and (row := db.execute("select id from jobs where idempotency_key = ?", (idempotency_key,)).fetchone()):
                return row["id"]
            job_id = _new_id("job")
            now = time.time()
            db.execute(
                "insert into jobs (id, status, request, webhook_url, metadata, idempotency_key, run_after, created_at)"
                " values (?, 'queued', ?, ?, ?, ?, ?, ?)",
                (job_id, json.dumps(request), webhook_url, json.dumps(metadata), idempotency_key, now, now),
            )
            return job_id

    def create_batch(
        self,
        requests: list[dict[str, Any]],
        *,
        webhook_url: str | None = None,
        metadata: Any = None,
        idempotency_key: str | None = None,
    ) -> str:
        with self._tx() as db:
            if idempotency_key and (row := db.execute("select id from batches where idempotency_key = ?", (idempotency_key,)).fetchone()):
                return row["id"]
            batch_id = _new_id("batch")
            now = time.time()
            db.execute(
                "insert into batches (id, created_at, webhook_url, metadata, idempotency_key) values (?, ?, ?, ?, ?)",
                (batch_id, now, webhook_url, json.dumps(metadata), idempotency_key),
            )
            db.executemany(
                "insert into jobs (id, batch_id, position, status, request, run_after, created_at)"
                " values (?, ?, ?, 'queued', ?, ?, ?)",
                [(_new_id("job"), batch_id, i, json.dumps(r), now, now) for i, r in enumerate(requests)],
            )
            return batch_id

    # -- processing ---------------------------------------------------------

    def claim(self) -> tuple[str, dict[str, Any], int] | None:
        """Take the oldest runnable job: (id, request, attempt number)."""
        with self._tx() as db:
            row = db.execute(
                "update jobs set status = 'running', started_at = ?, attempts = attempts + 1"
                " where id = (select id from jobs where status = 'queued' and run_after <= ?"
                "             order by created_at, position limit 1)"
                " returning id, request, attempts",
                (time.time(), time.time()),
            ).fetchone()
        return (row["id"], json.loads(row["request"]), row["attempts"]) if row else None

    def retry_later(self, job_id: str, error: dict[str, Any], delay: float) -> None:
        with self._tx() as db:
            db.execute(
                "update jobs set status = 'queued', run_after = ?, error = ? where id = ? and status = 'running'",
                (time.time() + delay, json.dumps(error), job_id),
            )

    def finish(self, job_id: str, status: JobStatus, *, result: Any = None, error: dict[str, Any] | None = None) -> None:
        with self._tx() as db:
            db.execute(
                "update jobs set status = ?, finished_at = ?, result = ?, error = ? where id = ?",
                (status, time.time(), json.dumps(result) if result is not None else None, json.dumps(error) if error else None, job_id),
            )
            self._on_terminal(db, [job_id])

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a queued job; running and finished jobs can't be cancelled."""
        with self._tx() as db:
            changed = db.execute(
                "update jobs set status = 'cancelled', finished_at = ? where id = ? and status = 'queued'", (time.time(), job_id)
            ).rowcount
            self._on_terminal(db, [job_id] if changed else [])
            return bool(changed)

    def cancel_batch(self, batch_id: str) -> int:
        with self._tx() as db:
            ids = [
                r["id"]
                for r in db.execute(
                    "update jobs set status = 'cancelled', finished_at = ? where batch_id = ? and status = 'queued' returning id",
                    (time.time(), batch_id),
                ).fetchall()
            ]
            self._on_terminal(db, ids)
            return len(ids)

    def _on_terminal(self, db: sqlite3.Connection, job_ids: list[str]) -> None:
        """Enqueue webhooks for jobs that just finished and for batches they completed."""
        now = time.time()
        batch_ids: set[str] = set()
        for job_id in job_ids:
            job = db.execute("select status, webhook_url, batch_id from jobs where id = ?", (job_id,)).fetchone()
            if job["webhook_url"]:
                self._enqueue_webhook(db, f"job.{job['status']}", job_id, job["webhook_url"], now)
            if job["batch_id"]:
                batch_ids.add(job["batch_id"])
        for batch_id in batch_ids:
            batch = db.execute("select webhook_url from batches where id = ?", (batch_id,)).fetchone()
            open_jobs = db.execute(
                "select count(*) from jobs where batch_id = ? and status in ('queued', 'running')", (batch_id,)
            ).fetchone()[0]
            if batch["webhook_url"] and open_jobs == 0:
                self._enqueue_webhook(db, "batch.completed", batch_id, batch["webhook_url"], now)

    @staticmethod
    def _enqueue_webhook(db: sqlite3.Connection, event: str, target_id: str, url: str, now: float) -> None:
        db.execute(
            "insert or ignore into webhooks (id, event, target_id, url, next_attempt_at, created_at) values (?, ?, ?, ?, ?, ?)",
            (_new_id("msg"), event, target_id, url, now, now),
        )

    def requeue_interrupted(self) -> int:
        """Jobs left `running` by a crash or restart go back to the queue."""
        with self._tx() as db:
            return db.execute("update jobs set status = 'queued', run_after = ? where status = 'running'", (time.time(),)).rowcount

    # -- webhooks -----------------------------------------------------------

    def due_webhooks(self) -> list[sqlite3.Row]:
        with self._lock:
            return self._db.execute(
                "select * from webhooks where delivered_at is null and next_attempt_at <= ? order by next_attempt_at limit 50",
                (time.time(),),
            ).fetchall()

    def webhook_attempted(self, webhook_id: str, *, error: str | None, next_attempt_at: float | None) -> None:
        """Record an attempt: delivered (no error), retry at `next_attempt_at`, or give up (both None/error)."""
        with self._tx() as db:
            db.execute(
                "update webhooks set attempts = attempts + 1, last_error = ?, next_attempt_at = ?,"
                " delivered_at = case when ? is null then ? else null end where id = ?",
                (error, next_attempt_at, error, time.time(), webhook_id),
            )

    # -- reads --------------------------------------------------------------

    def get_job(self, job_id: str, *, include_result: bool = True) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute("select * from jobs where id = ?", (job_id,)).fetchone()
        return self._job_view(row, include_result) if row else None

    def get_batch(self, batch_id: str, *, include_results: bool = False) -> dict[str, Any] | None:
        with self._lock:
            batch = self._db.execute("select * from batches where id = ?", (batch_id,)).fetchone()
            if not batch:
                return None
            jobs = self._db.execute("select * from jobs where batch_id = ? order by position", (batch_id,)).fetchall()
        counts = {status: 0 for status in ("queued", "running", "succeeded", "failed", "cancelled")}
        for job in jobs:
            counts[job["status"]] += 1
        done = counts["queued"] + counts["running"] == 0
        started = any(job["started_at"] for job in jobs)
        return {
            "id": batch["id"],
            "status": "completed" if done else "running" if started else "queued",
            "createdAt": _iso(batch["created_at"]),
            "finishedAt": _iso(max((j["finished_at"] or 0 for j in jobs), default=0)) if done else None,
            "webhookUrl": batch["webhook_url"],
            "metadata": _json(batch["metadata"]),
            "counts": counts,
            "jobs": [self._job_view(job, include_results) for job in jobs],
        }

    @staticmethod
    def _job_view(row: sqlite3.Row, include_result: bool) -> dict[str, Any]:
        view = {
            "id": row["id"],
            "batchId": row["batch_id"],
            "status": row["status"],
            "attempts": row["attempts"],
            "createdAt": _iso(row["created_at"]),
            "startedAt": _iso(row["started_at"]),
            "finishedAt": _iso(row["finished_at"]),
            "webhookUrl": row["webhook_url"],
            "metadata": _json(row["metadata"]),
            "error": _json(row["error"]) if row["status"] != "succeeded" else None,
        }
        if include_result:
            view["result"] = _json(row["result"])
        return view

    # -- retention ----------------------------------------------------------

    def purge(self, older_than: float) -> int:
        """Delete finished jobs/batches (and their delivered webhooks) older than the cutoff."""
        with self._tx() as db:
            db.execute("delete from webhooks where delivered_at is not null and delivered_at < ?", (older_than,))
            db.execute(
                "delete from batches where created_at < ? and not exists"
                " (select 1 from jobs where batch_id = batches.id and status in ('queued', 'running'))",
                (older_than,),
            )
            return db.execute(
                "delete from jobs where batch_id is null and status in ('succeeded', 'failed', 'cancelled') and finished_at < ?",
                (older_than,),
            ).rowcount
