"""Async job API: submit layout-parsing work, poll it, or get a webhook when done.

  POST   /jobs            one document  -> 202 Job
  POST   /batches         many documents -> 202 Batch
  GET    /jobs/{id}       ?includeResult=false to skip the (large) result
  GET    /batches/{id}    ?includeResults=true to inline every job's result
  DELETE /jobs/{id}       cancel if still queued
  DELETE /batches/{id}    cancel the batch's queued jobs

`request` items are PaddleX layout-parsing requests plus this service's
`tableFormat` / `layoutText` options. `result` is PaddleX's `result` object.
Send an `Idempotency-Key` header to make submits safe to retry.
"""

from collections.abc import Callable
from typing import Any, Literal

from fastapi import APIRouter, Header, Query, status
from fastapi.responses import JSONResponse
from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field

from jobs import JobStatus, JobStore

MAX_BATCH_SIZE = 500


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


class JobCreate(_Body):
    request: dict[str, Any] = Field(description="PaddleX layout-parsing request (+ tableFormat, layoutText)")
    webhookUrl: AnyHttpUrl | None = Field(default=None, description="receives job.succeeded / job.failed / job.cancelled")
    metadata: dict[str, Any] | None = Field(default=None, description="stored and echoed back, e.g. your document id")


class BatchCreate(_Body):
    requests: list[dict[str, Any]] = Field(min_length=1, max_length=MAX_BATCH_SIZE)
    webhookUrl: AnyHttpUrl | None = Field(default=None, description="receives batch.completed once every job has finished")
    metadata: dict[str, Any] | None = None


class JobError(BaseModel):
    code: Literal["invalid_request", "processing_error", "backend_unavailable"]
    message: str


class Job(BaseModel):
    id: str
    batchId: str | None
    status: JobStatus
    attempts: int
    createdAt: str
    startedAt: str | None
    finishedAt: str | None
    webhookUrl: str | None
    metadata: dict[str, Any] | None
    error: JobError | None = Field(description="last failure; set while retrying and when failed")
    result: dict[str, Any] | None = Field(default=None, description="PaddleX result once succeeded")


class Batch(BaseModel):
    id: str
    status: Literal["queued", "running", "completed"]
    createdAt: str
    finishedAt: str | None
    webhookUrl: str | None
    metadata: dict[str, Any] | None
    counts: dict[JobStatus, int]
    jobs: list[Job]


def _error(code: int, name: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=code, content={"error": {"code": name, "message": message}})


def create_router(
    store: JobStore,
    *,
    validate: Callable[[dict[str, Any]], str | None],
    on_submit: Callable[[], None],
    on_finished: Callable[[], None],
) -> APIRouter:
    """`validate` returns an error message for an invalid request, else None."""
    router = APIRouter(tags=["jobs"])

    @router.post("/jobs", status_code=status.HTTP_202_ACCEPTED, response_model=Job, response_model_exclude_unset=True)
    def create_job(body: JobCreate, idempotency_key: str | None = Header(default=None)) -> Any:
        if problem := validate(body.request):
            return _error(422, "invalid_request", problem)
        job_id = store.create_job(
            body.request,
            webhook_url=str(body.webhookUrl) if body.webhookUrl else None,
            metadata=body.metadata,
            idempotency_key=idempotency_key,
        )
        on_submit()
        job = store.get_job(job_id, include_result=False)
        return JSONResponse(Job.model_validate(job).model_dump(exclude_unset=True), status_code=202, headers={"Location": f"/jobs/{job_id}"})

    @router.post("/batches", status_code=status.HTTP_202_ACCEPTED, response_model=Batch)
    def create_batch(body: BatchCreate, idempotency_key: str | None = Header(default=None)) -> Any:
        for i, request in enumerate(body.requests):
            if problem := validate(request):
                return _error(422, "invalid_request", f"requests[{i}]: {problem}")
        batch_id = store.create_batch(
            body.requests,
            webhook_url=str(body.webhookUrl) if body.webhookUrl else None,
            metadata=body.metadata,
            idempotency_key=idempotency_key,
        )
        on_submit()
        batch = Batch.model_validate(store.get_batch(batch_id))
        return JSONResponse(batch.model_dump(exclude_unset=True), status_code=202, headers={"Location": f"/batches/{batch_id}"})

    @router.get("/jobs/{job_id}", response_model=Job, response_model_exclude_unset=True)
    def get_job(job_id: str, includeResult: bool = Query(default=True)) -> Any:
        job = store.get_job(job_id, include_result=includeResult)
        return job if job else _error(404, "not_found", f"no job {job_id}")

    @router.get("/batches/{batch_id}", response_model=Batch, response_model_exclude_unset=True)
    def get_batch(batch_id: str, includeResults: bool = Query(default=False)) -> Any:
        batch = store.get_batch(batch_id, include_results=includeResults)
        return batch if batch else _error(404, "not_found", f"no batch {batch_id}")

    @router.delete("/jobs/{job_id}", response_model=Job, response_model_exclude_unset=True)
    def cancel_job(job_id: str) -> Any:
        if not store.cancel_job(job_id):
            job = store.get_job(job_id, include_result=False)
            if not job:
                return _error(404, "not_found", f"no job {job_id}")
            return _error(409, "not_cancellable", f"job is {job['status']}; only queued jobs can be cancelled")
        on_finished()
        return store.get_job(job_id, include_result=False)

    @router.delete("/batches/{batch_id}", response_model=Batch, response_model_exclude_unset=True)
    def cancel_batch(batch_id: str) -> Any:
        if not store.get_batch(batch_id):
            return _error(404, "not_found", f"no batch {batch_id}")
        if store.cancel_batch(batch_id):
            on_finished()
        return store.get_batch(batch_id)

    return router
