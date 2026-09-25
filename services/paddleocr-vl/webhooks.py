"""Webhook delivery, signed per the Standard Webhooks spec (standardwebhooks.com).

Each event is POSTed as `{"id", "type", "timestamp", "data"}` where `data` is
the job (with its result) or the batch summary, exactly as the GET routes
return them. Receivers verify the `webhook-signature` header with any
Standard Webhooks library and the service's WEBHOOK_SECRET (`whsec_...`),
and dedupe on `webhook-id`. Non-2xx responses and network errors are retried
on a fixed schedule; deliveries are persisted, so restarts don't lose them.
"""

import asyncio
import base64
import hashlib
import hmac
import json
import time
from typing import Any

import httpx

from jobs import JobStore

# delay before each retry; the first attempt is immediate (~5h total)
RETRY_DELAYS = (10, 30, 120, 600, 1800, 3600, 3 * 3600)


def sign(secret: str, msg_id: str, timestamp: int, body: str) -> str:
    """`webhook-signature` value for `secret` in `whsec_<base64 key>` form."""
    key = base64.b64decode(secret.removeprefix("whsec_"))
    digest = hmac.new(key, f"{msg_id}.{timestamp}.{body}".encode(), hashlib.sha256).digest()
    return f"v1,{base64.b64encode(digest).decode()}"


class WebhookSender:
    def __init__(self, store: JobStore, secret: str | None) -> None:
        self._store = store
        self._secret = secret
        self._wake = asyncio.Event()

    def wake(self) -> None:
        self._wake.set()

    def payload(self, event: str, target_id: str) -> dict[str, Any] | None:
        data = (
            self._store.get_batch(target_id)
            if event.startswith("batch.")
            else self._store.get_job(target_id, include_result=True)
        )
        return {"type": event, "data": data} if data else None

    async def run(self) -> None:
        async with httpx.AsyncClient(timeout=30) as client:
            while True:
                await self.deliver_due(client)
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=5)
                except TimeoutError:
                    pass

    async def deliver_due(self, client: httpx.AsyncClient) -> None:
        for hook in self._store.due_webhooks():
            await self._deliver(client, hook)

    async def _deliver(self, client: httpx.AsyncClient, hook: Any) -> None:
        event = self.payload(hook["event"], hook["target_id"])
        if event is None:  # target purged meanwhile
            self._store.webhook_attempted(hook["id"], error="target no longer exists", next_attempt_at=None)
            return
        timestamp = int(time.time())
        body = json.dumps({"id": hook["id"], "timestamp": timestamp, **event})
        headers = {"content-type": "application/json", "webhook-id": hook["id"], "webhook-timestamp": str(timestamp)}
        if self._secret:
            headers["webhook-signature"] = sign(self._secret, hook["id"], timestamp, body)
        try:
            response = await client.post(hook["url"], content=body, headers=headers)
            error = None if response.is_success else f"HTTP {response.status_code}"
        except httpx.HTTPError as e:
            error = f"{type(e).__name__}: {e}"
        attempt = hook["attempts"]  # attempts made before this one
        next_at = time.time() + RETRY_DELAYS[attempt] if error and attempt < len(RETRY_DELAYS) else None
        self._store.webhook_attempted(hook["id"], error=error, next_attempt_at=next_at)
        if error:
            print(f"webhook {hook['id']} ({hook['event']}) attempt {attempt + 1} failed: {error}", flush=True)
