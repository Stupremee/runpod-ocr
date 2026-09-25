"""`runpod-ocr <image>`: OCR an image path or URL and print the text."""

import argparse
import asyncio
import sys
from typing import cast, get_args

from .client import ocr
from .models import MODELS, ModelKey


def main() -> None:
    parser = argparse.ArgumentParser(prog="runpod-ocr", description=__doc__)
    parser.add_argument("image", help="image path or http(s) URL")
    parser.add_argument("-m", "--model", choices=get_args(ModelKey), default="paddleocr-vl-1.6")
    parser.add_argument("-t", "--task", default="ocr", help="task prompt from the model registry")
    parser.add_argument("-p", "--prompt", help="raw prompt, overrides --task")
    parser.add_argument("--max-tokens", type=int)
    args = parser.parse_args()

    model = cast(ModelKey, args.model)
    if args.prompt is None and args.task not in MODELS[model].prompts:
        parser.error(f"{model} tasks: {', '.join(MODELS[model].prompts)}")

    result = asyncio.run(
        ocr(args.image, model=model, task=args.task, prompt=args.prompt, max_tokens=args.max_tokens)
    )
    print(result.text)
    print(f"[usage] {result.usage}", file=sys.stderr)
