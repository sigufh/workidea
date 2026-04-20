from __future__ import annotations

import argparse
import json

from kvbench.eval.offline import evaluate_offline
from kvbench.eval.trace import load_trace_npz, synthetic_states
from kvbench.registry import build_strategy, list_strategies


def parse_kv_pairs(items: list[str]) -> dict:
    out = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --param: {item}, expected key=value")
        k, v = item.split("=", 1)
        if v.lower() in {"true", "false"}:
            out[k] = v.lower() == "true"
            continue
        try:
            out[k] = int(v)
            continue
        except ValueError:
            pass
        try:
            out[k] = float(v)
            continue
        except ValueError:
            pass
        out[k] = v
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline KV compression benchmark without local model weights")
    parser.add_argument("--strategy", required=True, choices=list_strategies())
    parser.add_argument("--target-tokens", type=int, default=512)
    parser.add_argument("--trace", type=str, default="", help="Path to NPZ trace; empty uses synthetic trace")
    parser.add_argument("--param", action="append", default=[], help="Strategy parameter in key=value format")
    parser.add_argument("--important", default="16,32,64", help="Comma-separated important token ids")
    args = parser.parse_args()

    params = parse_kv_pairs(args.param)
    strategy = build_strategy(args.strategy, **params)

    if args.trace:
        states, histories = load_trace_npz(args.trace)
    else:
        states, histories = synthetic_states()

    important_ids = {int(x) for x in args.important.split(",") if x.strip()}
    metrics = evaluate_offline(
        strategy=strategy,
        states=states,
        target_tokens=args.target_tokens,
        important_token_ids=important_ids,
        attention_histories=histories,
    )

    print(
        json.dumps(
            {
                "strategy": args.strategy,
                "target_tokens": args.target_tokens,
                "metrics": {
                    "avg_kept_tokens": metrics.avg_kept_tokens,
                    "avg_compression_ratio": metrics.avg_compression_ratio,
                    "important_recall": metrics.important_recall,
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
