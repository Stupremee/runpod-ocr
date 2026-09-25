import asyncio
import json

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import create_router
from jobs import JobStore
from runner import JobRunner
from webhooks import WebhookSender, sign

OK = (200, {"errorCode": 0, "errorMsg": "Success", "result": {"layoutParsingResults": []}})


class NoWarmup:
    async def ensure(self) -> None:
        pass


def runner(store: JobStore, *responses: tuple[int, dict]) -> tuple[JobRunner, list[dict]]:
    """A runner whose parser answers with `responses` in order and records requests."""
    seen: list[dict] = []
    queue = list(responses)

    async def parse(request: dict) -> tuple[int, dict]:
        seen.append(request)
        return queue.pop(0)

    return JobRunner(store, parse, NoWarmup(), concurrency=1, max_attempts=3, on_finished=lambda: None, retry_base_delay=0), seen


def drain(store: JobStore, job_runner: JobRunner) -> None:
    async def go() -> None:
        while claimed := store.claim():
            await job_runner.process(*claimed)

    asyncio.run(go())


def webhook_events(store: JobStore) -> list[tuple[str, str]]:
    return [(h["event"], h["target_id"]) for h in store.due_webhooks()]


def test_batch_runs_in_order_and_fires_one_completion_webhook(tmp_path):
    store = JobStore(str(tmp_path / "jobs.db"))
    batch_id = store.create_batch([{"file": "a"}, {"file": "b"}], webhook_url="https://hook.test/b")
    job_runner, seen = runner(store, OK, OK)
    drain(store, job_runner)

    assert [r["file"] for r in seen] == ["a", "b"]
    batch = store.get_batch(batch_id)
    assert batch["status"] == "completed" and batch["counts"]["succeeded"] == 2
    assert webhook_events(store) == [("batch.completed", batch_id)]


def test_backend_errors_retry_but_rejected_requests_fail_at_once(tmp_path):
    store = JobStore(str(tmp_path / "jobs.db"))
    flaky = store.create_job({"file": "a"}, webhook_url="https://hook.test/j")
    invalid = store.create_job({"file": "b"})
    # a retried job keeps its place in the queue, so it runs again before the newer job
    job_runner, _ = runner(store, (500, {"errorMsg": "vLLM down"}), OK, (422, {"errorMsg": "bad file"}))
    drain(store, job_runner)

    assert store.get_job(flaky)["status"] == "succeeded"
    assert store.get_job(flaky)["attempts"] == 2
    failed = store.get_job(invalid)
    assert (failed["status"], failed["attempts"], failed["error"]["code"]) == ("failed", 1, "invalid_request")
    assert webhook_events(store) == [("job.succeeded", flaky)]


def test_gives_up_after_max_attempts(tmp_path):
    store = JobStore(str(tmp_path / "jobs.db"))
    job_id = store.create_job({"file": "a"})
    job_runner, _ = runner(store, *[(500, {"errorMsg": "boom"})] * 3)
    drain(store, job_runner)
    job = store.get_job(job_id)
    assert (job["status"], job["attempts"], job["error"]["message"]) == ("failed", 3, "boom")


def test_interrupted_jobs_are_requeued_and_only_queued_jobs_cancel(tmp_path):
    store = JobStore(str(tmp_path / "jobs.db"))
    running = store.create_job({"file": "a"})
    queued = store.create_job({"file": "b"}, webhook_url="https://hook.test/j")
    store.claim()  # `running` is now in progress

    assert not store.cancel_job(running)
    assert store.cancel_job(queued)
    assert webhook_events(store) == [("job.cancelled", queued)]
    assert store.requeue_interrupted() == 1
    assert store.get_job(running)["status"] == "queued"


def test_webhook_signature_matches_the_standard_webhooks_test_vector():
    # from the Standard Webhooks specification
    signature = sign(
        "whsec_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw", "msg_p5jXN8AQM9LWM0D4loKWxJek", 1614265330, '{"test": 2432232314}'
    )
    assert signature == "v1,g0hM9SsE+OTPJTGt/tmIKtSyZlE3uFJELVlNIOLJ1OE="


def test_webhooks_carry_the_job_and_retry_until_accepted(tmp_path):
    store = JobStore(str(tmp_path / "jobs.db"))
    job_id = store.create_job({"file": "a"}, webhook_url="https://hook.test/j", metadata={"doc": 7})
    drain(store, runner(store, OK)[0])
    sender = WebhookSender(store, "whsec_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw")
    received: list[httpx.Request] = []

    def receiver(request: httpx.Request) -> httpx.Response:
        received.append(request)
        return httpx.Response(503 if len(received) == 1 else 204)

    async def deliver() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(receiver)) as client:
            await sender.deliver_due(client)  # 503: scheduled for retry
            with store._lock:  # make the retry due now
                store._db.execute("update webhooks set next_attempt_at = 0")
            await sender.deliver_due(client)  # 204: delivered
            await sender.deliver_due(client)  # nothing left

    asyncio.run(deliver())
    assert len(received) == 2
    body = json.loads(received[1].content)
    assert (body["type"], body["data"]["id"], body["data"]["metadata"]) == ("job.succeeded", job_id, {"doc": 7})
    assert body["data"]["result"] == OK[1]["result"]
    headers = received[1].headers
    expected = sign("whsec_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw", headers["webhook-id"], int(headers["webhook-timestamp"]), received[1].content.decode())
    assert headers["webhook-signature"] == expected


def test_api_submit_poll_cancel(tmp_path):
    store = JobStore(str(tmp_path / "jobs.db"))
    app = FastAPI()
    app.include_router(
        create_router(
            store,
            validate=lambda r: None if "file" in r else "file is required",
            on_submit=lambda: None,
            on_finished=lambda: None,
        )
    )
    client = TestClient(app)

    created = client.post("/jobs", json={"request": {"file": "a"}, "webhookUrl": "https://hook.test/j"}, headers={"Idempotency-Key": "k1"})
    assert created.status_code == 202 and created.headers["location"] == f"/jobs/{created.json()['id']}"
    again = client.post("/jobs", json={"request": {"file": "a"}}, headers={"Idempotency-Key": "k1"})
    assert again.json()["id"] == created.json()["id"]

    assert client.post("/jobs", json={"request": {}}).json()["error"]["code"] == "invalid_request"
    assert client.post("/batches", json={"requests": [{"file": "a"}, {}]}).json()["error"]["message"].startswith("requests[1]")

    batch = client.post("/batches", json={"requests": [{"file": "a"}, {"file": "b"}]}).json()
    assert (batch["status"], len(batch["jobs"])) == ("queued", 2)
    cancelled = client.delete(f"/batches/{batch['id']}").json()
    assert (cancelled["status"], cancelled["counts"]["cancelled"]) == ("completed", 2)

    job = client.get(f"/jobs/{created.json()['id']}", params={"includeResult": False}).json()
    assert job["status"] == "queued" and "result" not in job
    assert client.delete(f"/jobs/{job['id']}").json()["status"] == "cancelled"
    assert client.delete(f"/jobs/{job['id']}").status_code == 409
    assert client.get("/jobs/job_missing").status_code == 404
