#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_MODELS = [
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "llava-hf/LLaVA-NeXT-Video-7B-hf",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download real VLM models for KVBench experiments")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="HF model ids")
    parser.add_argument("--cache-dir", default="", help="Optional HF cache dir")
    parser.add_argument("--local-dir-root", default="models", help="Root directory for local snapshots")
    parser.add_argument("--token", default="", help="HF token (optional for public models)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.local_dir_root)
    root.mkdir(parents=True, exist_ok=True)

    for model_id in args.models:
        safe_name = model_id.replace("/", "__")
        target = root / safe_name
        target.mkdir(parents=True, exist_ok=True)
        print(f"[download] {model_id} -> {target}")
        snapshot_download(
            repo_id=model_id,
            local_dir=str(target),
            cache_dir=(args.cache_dir or None),
            token=(args.token or None),
        )
        print(f"[done] {model_id}")


if __name__ == "__main__":
    main()
