#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from kvbench.eval.offline import evaluate_offline
from kvbench.eval.trace import load_trace_npz
from kvbench.registry import build_strategy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper-style offline benchmark on one or more traces")
    parser.add_argument("--traces", nargs="+", required=True, help="NPZ trace paths")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["fullkv", "h2o", "snap", "pyramid", "vlcache", "streamingcache", "dynamic_freq_window"],
    )
    parser.add_argument("--target-tokens", type=int, default=1024)
    parser.add_argument("--important", default="16,32,64")
    parser.add_argument("--out", required=True, help="Output json path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    important_ids = {int(x) for x in args.important.split(",") if x.strip()}
    rows = []

    for trace_path in args.traces:
        states, histories = load_trace_npz(trace_path)
        for strategy_name in args.strategies:
            strategy = build_strategy(strategy_name)
            metrics = evaluate_offline(
                strategy=strategy,
                states=states,
                target_tokens=args.target_tokens,
                important_token_ids=important_ids,
                attention_histories=histories,
            )
            rows.append(
                {
                    "trace": trace_path,
                    "strategy": strategy_name,
                    "target_tokens": args.target_tokens,
                    "avg_kept_tokens": metrics.avg_kept_tokens,
                    "avg_compression_ratio": metrics.avg_compression_ratio,
                    "important_recall": metrics.important_recall,
                }
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved benchmark report: {out_path}")


if __name__ == "__main__":
    main()
