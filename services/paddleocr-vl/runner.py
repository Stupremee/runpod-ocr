"""Background workers that drain the job queue.

Jobs run oldest first, so the documents of a batch are processed together while
the Runpod GPU is warm. Backend or network failures are retried with backoff;
requests PaddleX rejects (4xx) fail right away.
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from jobs import JobStore

# (HTTP status, PaddleX response body) for one layout-parsing request
type Parse = Callable[[dict[str, Any]], Awaitable[tuple[int, dict[str, Any]]]]


class Warmup(Protocol):
    async def ensure(self) -> None: ...


class JobRunner:
    def __init__(
        self,
        store: JobStore,
        parse: Parse,
        warmup: Warmup,
        *,
        concurrency: int,
        max_attempts: int,
        on_finished: Callable[[], None],
        retry_base_delay: float = 30,
    ) -> None:
        self._store = store
        self._parse = parse
        self._warmup = warmup
        self._concurrency = concurrency
        self._max_attempts = max_attempts
        self._on_finished = on_finished
        self._retry_base_delay = retry_base_delay
        self._wake = asyncio.Event()

    def wake(self) -> None:
        self._wake.set()

    async def run(self) -> None:
        await asyncio.gather(*(self._worker() for _ in range(self._concurrency)))

    async def _worker(self) -> None:
        while True:
            claimed = self._store.claim()
            if claimed is None:
                self._wake.clear()
                try:  # also polls, to pick up retries whose backoff has passed
                    await asyncio.wait_for(self._wake.wait(), timeout=2)
                except TimeoutError:
                    pass
                continue
            await self.process(*claimed)

    async def process(self, job_id: str, request: dict[str, Any], attempt: int) -> None:
        try:
            await self._warmup.ensure()
            status, body = await self._parse(request)
        except Exception as e:  # Runpod unreachable, cold start timed out, ...
            return self._transient(job_id, attempt, {"code": "backend_unavailable", "message": str(e)})

        if status == 200 and body.get("errorCode") == 0:
            self._store.finish(job_id, "succeeded", result=body["result"])
        elif 400 <= status < 500:
            self._store.finish(job_id, "failed", error={"code": "invalid_request", "message": body.get("errorMsg", "")})
        else:
            return self._transient(job_id, attempt, {"code": "processing_error", "message": body.get("errorMsg", f"HTTP {status}")})
        self._on_finished()

    def _transient(self, job_id: str, attempt: int, error: dict[str, Any]) -> None:
        if attempt < self._max_attempts:
            self._store.retry_later(job_id, error, delay=self._retry_base_delay * 2 ** (attempt - 1))
        else:
            self._store.finish(job_id, "failed", error=error)
            self._on_finished()
