"""runpod-ocr backend up [MODEL]: provision (or update) a vLLM backend on Runpod
and record its endpoint id in .env for the model's service."""

import argparse
import asyncio
import json
from pathlib import Path

from .backends import BACKENDS


def _set_env(path: Path, key: str, value: str) -> None:
    lines = path.read_text().splitlines() if path.exists() else []
    lines = [line for line in lines if not line.startswith(f"{key}=")] + [f"{key}={value}"]
    path.write_text("\n".join(lines) + "\n")


async def _backend_up(model: str) -> None:
    from runpod_flash.core.resources import ResourceManager

    backend = BACKENDS[model]
    endpoint = backend.endpoint()
    print(f"provisioning {backend.name} ({backend.hf_repo}); a cold boot takes a few minutes…")
    # any job provisions the endpoint (or applies config changes) and wakes a worker
    job = await endpoint.run({"openai_route": "/v1/models"})
    [(_, resource)] = ResourceManager().find_resources_by_name(backend.name)
    _set_env(Path(".env"), backend.endpoint_id_env, resource.id)
    print(f"{backend.endpoint_id_env}={resource.id} (written to .env)")
    await job.wait(timeout=1200)
    print("backend ready:", json.dumps(job.output)[:200])


def main() -> None:
    parser = argparse.ArgumentParser(prog="runpod-ocr", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    backend = commands.add_parser("backend", help="manage vLLM backends")
    backend.add_argument("action", choices=["up"])
    backend.add_argument("model", nargs="?", choices=list(BACKENDS), default="paddleocr-vl-1.6")
    args = parser.parse_args()
    asyncio.run(_backend_up(args.model))
