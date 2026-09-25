"""runpod-ocr CLI.

  runpod-ocr backend up                 provision the vLLM backend, record its id in .env
  runpod-ocr parse FILE_OR_URL [...]    OCR a PDF/image through the deployed worker
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, cast, get_args

from .schema import ModelName, OcrOptions


def _pages(spec: str) -> list[int]:
    """"1,3-5" -> [1, 3, 4, 5]"""
    pages: list[int] = []
    for part in spec.split(","):
        first, _, last = part.partition("-")
        pages += range(int(first), int(last or first) + 1)
    return pages


def _pairs(items: list[str]) -> dict[str, Any]:
    """key=value pairs; values are parsed as JSON when possible (true, 0.5, [1,2])."""
    parsed: dict[str, Any] = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep:
            raise SystemExit(f"expected key=value, got {item!r}")
        try:
            parsed[key] = json.loads(raw)
        except json.JSONDecodeError:
            parsed[key] = raw
    return parsed


def _set_env(path: Path, key: str, value: str) -> None:
    lines = path.read_text().splitlines() if path.exists() else []
    lines = [line for line in lines if not line.startswith(f"{key}=")] + [f"{key}={value}"]
    path.write_text("\n".join(lines) + "\n")


async def _backend_up(model: ModelName) -> None:
    from runpod_flash.core.resources import ResourceManager

    from .backends import BACKENDS

    backend = BACKENDS[model]
    endpoint = backend.endpoint()
    print(f"provisioning {backend.name} ({backend.hf_repo}); first boot takes a few minutes…")
    job = await endpoint.run({"openai_route": "/v1/models"})
    [(_, resource)] = ResourceManager().find_resources_by_name(backend.name)
    _set_env(Path(".env"), backend.endpoint_id_env, resource.id)
    print(f"{backend.endpoint_id_env}={resource.id} (written to .env)")
    await job.wait(timeout=1200)
    print("backend ready:", json.dumps(job.output)[:200])


def main() -> None:
    parser = argparse.ArgumentParser(prog="runpod-ocr", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    backend = commands.add_parser("backend", help="manage vLLM backends")
    backend.add_argument("action", choices=["up"])
    backend.add_argument("-m", "--model", choices=get_args(ModelName), default="paddleocr-vl-1.6")

    parse = commands.add_parser("parse", help="OCR a document")
    parse.add_argument("document", help="PDF/image path or http(s) URL")
    parse.add_argument("-m", "--model", choices=get_args(ModelName), default="paddleocr-vl-1.6")
    parse.add_argument("-f", "--format", choices=["markdown", "text", "json"], default="markdown")
    parse.add_argument("-p", "--pages", type=_pages, help="e.g. 1,3-5")
    parse.add_argument("-o", "--option", action="append", default=[], metavar="KEY=VALUE",
                       help=f"unified option, one of: {', '.join(OcrOptions.model_fields)}")
    parse.add_argument("-M", "--model-option", action="append", default=[], metavar="KEY=VALUE",
                       help="model-specific option, validated by the model's engine")
    parse.add_argument("--out", type=Path, help="write the result here instead of stdout")

    args = parser.parse_args()
    model = cast(ModelName, args.model)

    if args.command == "backend":
        asyncio.run(_backend_up(model))
        return

    from .client import parse as parse_document

    options = OcrOptions.model_validate(
        {"output": args.format, "pages": args.pages, **_pairs(args.option)}
    )
    response = asyncio.run(
        parse_document(args.document, model=model, options=options, model_options=_pairs(args.model_option))
    )
    match options.output:
        case "markdown":
            result = response.markdown or ""
        case "text":
            result = response.text or ""
        case _:
            result = response.model_dump_json(indent=2, exclude_none=True)
    if args.out:
        args.out.write_text(result)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(result)
