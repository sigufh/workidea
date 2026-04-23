#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

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
    parser.add_argument("--target-tokens", type=int, nargs="+", default=[1024])
    parser.add_argument(
        "--important",
        default="auto",
        help="Comma-separated token ids, or 'auto' to use sink/special positions from each trace.",
    )
    parser.add_argument("--out", required=True, help="Output json path")
    return parser.parse_args()


def _quality_retention_pct(row: dict, fullkv_row: dict | None) -> float:
    """
    Academic-style primary metric: retained quality relative to FullKV (%).
    """
    if not fullkv_row:
        return 0.0
    full_mass = float(fullkv_row.get("attention_mass_recall", 0.0))
    row_mass = float(row.get("attention_mass_recall", 0.0))
    if full_mass > 1e-8:
        return 100.0 * row_mass / full_mass

    # Fallback if attention mass is unavailable.
    full_keep = float(fullkv_row.get("avg_kept_tokens", 0.0))
    row_keep = float(row.get("avg_kept_tokens", 0.0))
    if full_keep > 1e-8:
        return 100.0 * row_keep / full_keep
    return 0.0


def _resolve_auto_important_steps(states) -> set[int]:
    if not states:
        return set()
    meta = states[-1].token_meta
    special_steps = {t.timestep for t in meta if t.is_special_memory}
    if special_steps:
        return special_steps
    sink_steps = {t.timestep for t in meta if t.is_sink}
    if sink_steps:
        return sink_steps
    n = len(meta)
    if n == 0:
        return set()
    anchors = sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1})
    return set(anchors)


def _build_markdown_report(by_budget: list[dict], out_json: Path) -> None:
    lines: list[str] = []
    lines.append(f"# Academic-style benchmark ({out_json.name})")
    for budget_block in by_budget:
        lines.append("")
        lines.append(f"## Budget: {budget_block['target_tokens']}")
        lines.append("")
        lines.append("### FullKV baseline")
        lines.append("")
        base = budget_block.get("fullkv_baseline")
        if base is not None:
            lines.append("| Strategy | Retention(%) | AttnRecall | CompRatio | CompFactor |")
            lines.append("|---|---:|---:|---:|---:|")
            lines.append(
                "| {strategy} | {quality_retention_pct_mean:.2f} | {attention_mass_recall_mean:.4f} | {avg_compression_ratio_mean:.4f} | {compression_factor_mean:.2f} |".format(
                    **base
                )
            )

        lines.append("")
        lines.append("### Compressed methods ranking")
        lines.append("")
        lines.append("| Rank | Strategy | Retention(%) | AttnRecall | CompRatio | CompFactor | ImportantRecall |")
        lines.append("|---:|---|---:|---:|---:|---:|---:|")
        for row in budget_block["overall"]:
            lines.append(
                "| {rank} | {strategy} | {quality_retention_pct_mean:.2f} | {attention_mass_recall_mean:.4f} | {avg_compression_ratio_mean:.4f} | {compression_factor_mean:.2f} | {important_recall_mean:.4f} |".format(
                    **row
                )
            )

        for block in budget_block["per_trace"]:
            lines.append("")
            lines.append(f"### Trace: `{block['trace']}`")
            lines.append("")
            if block.get("fullkv_baseline"):
                base_row = block["fullkv_baseline"]
                lines.append(
                    "FullKV baseline: retention={:.2f}%, attn={:.4f}, ratio={:.4f}, factor={:.2f}".format(
                        float(base_row["quality_retention_pct"]),
                        float(base_row["attention_mass_recall"]),
                        float(base_row["avg_compression_ratio"]),
                        float(base_row["compression_factor"]),
                    )
                )
                lines.append("")
            lines.append("| Rank | Strategy | Retention(%) | AttnRecall | ImportantRecall | EffScore | CompRatio |")
            lines.append("|---:|---|---:|---:|---:|---:|---:|")
            for row in block["results"]:
                lines.append(
                    "| {rank_in_trace} | {strategy} | {quality_retention_pct:.2f} | {attention_mass_recall:.4f} | {important_recall:.4f} | {efficiency_score:.4f} | {avg_compression_ratio:.4f} |".format(
                        **row
                    )
                )

    md_path = out_json.with_suffix(".md")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    target_tokens_list = sorted(set(args.target_tokens))
    by_budget: list[dict] = []
    all_rows: list[dict] = []
    resolved_important: dict[str, dict] = {}

    for budget in target_tokens_list:
        rows: list[dict] = []
        for trace_path in args.traces:
            states, histories = load_trace_npz(trace_path)

            if args.important.strip().lower() == "auto":
                important_steps = _resolve_auto_important_steps(states)
                important_ids = None
                resolved_important[trace_path] = {
                    "mode": "auto_steps",
                    "count": len(important_steps),
                    "sample": sorted(list(important_steps))[:16],
                }
            else:
                important_ids = {int(x) for x in args.important.split(",") if x.strip()}
                important_steps = None
                resolved_important[trace_path] = {
                    "mode": "token_ids",
                    "count": len(important_ids),
                    "sample": sorted(list(important_ids))[:16],
                }

            for strategy_name in args.strategies:
                strategy = build_strategy(strategy_name)
                metrics = evaluate_offline(
                    strategy=strategy,
                    states=states,
                    target_tokens=budget,
                    important_token_ids=important_ids,
                    important_token_steps=important_steps,
                    attention_histories=histories,
                )
                row = {
                    "trace": trace_path,
                    "strategy": strategy_name,
                    "target_tokens": budget,
                    "avg_kept_tokens": metrics.avg_kept_tokens,
                    "std_kept_tokens": metrics.std_kept_tokens,
                    "avg_compression_ratio": metrics.avg_compression_ratio,
                    "std_compression_ratio": metrics.std_compression_ratio,
                    "compression_ratio_p50": metrics.compression_ratio_p50,
                    "compression_ratio_p90": metrics.compression_ratio_p90,
                    "important_recall": metrics.important_recall,
                    "sink_recall": metrics.sink_recall,
                    "special_recall": metrics.special_recall,
                    "attention_mass_recall": metrics.attention_mass_recall,
                    "efficiency_score": metrics.efficiency_score,
                }
                rows.append(row)
                all_rows.append(row.copy())

        traces = sorted({r["trace"] for r in rows})
        per_trace: list[dict] = []
        for trace in traces:
            block = [r for r in rows if r["trace"] == trace]
            fullkv_row = next((r for r in block if r["strategy"] == "fullkv"), None)
            for row in block:
                row["quality_retention_pct"] = _quality_retention_pct(row, fullkv_row)
                ratio = float(row["avg_compression_ratio"])
                row["compression_factor"] = (1.0 / ratio) if ratio > 1e-8 else 0.0

            compressed = [r for r in block if r["strategy"] != "fullkv"]
            block_sorted = sorted(
                compressed,
                key=lambda x: (x["quality_retention_pct"], -x["avg_compression_ratio"]),
                reverse=True,
            )
            for rank, row in enumerate(block_sorted, start=1):
                row["rank_in_trace"] = rank
            per_trace.append(
                {
                    "trace": trace,
                    "fullkv_baseline": fullkv_row,
                    "results": block_sorted,
                }
            )

        by_strategy: dict[str, list[dict]] = {}
        for row in rows:
            by_strategy.setdefault(row["strategy"], []).append(row)

        overall_rows_all: list[dict] = []
        for strategy, items in by_strategy.items():
            overall_rows_all.append(
                {
                    "strategy": strategy,
                    "quality_retention_pct_mean": float(np.mean([x["quality_retention_pct"] for x in items])),
                    "quality_retention_pct_std": float(np.std([x["quality_retention_pct"] for x in items])),
                    "attention_mass_recall_mean": float(np.mean([x["attention_mass_recall"] for x in items])),
                    "avg_compression_ratio_mean": float(np.mean([x["avg_compression_ratio"] for x in items])),
                    "compression_factor_mean": float(
                        np.mean([(1.0 / x["avg_compression_ratio"]) if x["avg_compression_ratio"] > 1e-8 else 0.0 for x in items])
                    ),
                    "important_recall_mean": float(np.mean([x["important_recall"] for x in items])),
                    "count_traces": len(items),
                }
            )
        fullkv_baseline = next((x for x in overall_rows_all if x["strategy"] == "fullkv"), None)
        overall_rows = [x for x in overall_rows_all if x["strategy"] != "fullkv"]
        overall_rows.sort(
            key=lambda x: (x["quality_retention_pct_mean"], -x["avg_compression_ratio_mean"]),
            reverse=True,
        )
        for rank, row in enumerate(overall_rows, start=1):
            row["rank"] = rank

        by_budget.append(
            {
                "target_tokens": budget,
                "per_trace": per_trace,
                "fullkv_baseline": fullkv_baseline,
                "overall": overall_rows,
                "rows": rows,
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "target_tokens": target_tokens_list,
            "important": args.important,
            "important_resolved": resolved_important,
            "num_traces": len(args.traces),
            "num_rows": len(all_rows),
        },
        "by_budget": by_budget,
        "per_trace": by_budget[0]["per_trace"] if by_budget else [],
        "overall": by_budget[0]["overall"] if by_budget else [],
        # keep backward compatibility for existing parsers
        "rows": all_rows,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _build_markdown_report(by_budget=by_budget, out_json=out_path)
    print(f"saved benchmark report: {out_path}")


if __name__ == "__main__":
    main()
