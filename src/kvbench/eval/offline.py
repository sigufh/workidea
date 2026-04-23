from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kvbench.strategies.base import KVStrategy
from kvbench.types import CompressionContext, KVCacheState


@dataclass(slots=True)
class OfflineMetrics:
    avg_kept_tokens: float
    std_kept_tokens: float
    avg_compression_ratio: float
    std_compression_ratio: float
    compression_ratio_p50: float
    compression_ratio_p90: float
    important_recall: float
    sink_recall: float
    special_recall: float
    attention_mass_recall: float
    efficiency_score: float


def _mean_or_zero(items: list[float]) -> float:
    if not items:
        return 0.0
    return float(np.mean(items))


def _std_or_zero(items: list[float]) -> float:
    if not items:
        return 0.0
    return float(np.std(items))


def _percentile_or_zero(items: list[float], q: float) -> float:
    if not items:
        return 0.0
    return float(np.percentile(np.asarray(items, dtype=np.float32), q))


def _aggregate_attention_per_token(state: KVCacheState) -> np.ndarray | None:
    per_layer: list[np.ndarray] = []
    for layer in state.layers:
        if layer.attention_scores is None:
            continue
        if layer.attention_scores.size == 0:
            continue
        per_layer.append(layer.attention_scores.mean(axis=0).astype(np.float32))
    if not per_layer:
        return None
    return np.stack(per_layer, axis=0).mean(axis=0)


def evaluate_offline(
    strategy: KVStrategy,
    states: list[KVCacheState],
    target_tokens: int,
    important_token_ids: set[int] | None = None,
    important_token_steps: set[int] | None = None,
    attention_histories: list[np.ndarray | None] | None = None,
) -> OfflineMetrics:
    kept_counts: list[int] = []
    ratios: list[float] = []
    recalls: list[float] = []
    sink_recalls: list[float] = []
    special_recalls: list[float] = []
    mass_recalls: list[float] = []
    efficiency_scores: list[float] = []

    for i, state in enumerate(states):
        hist = None
        if attention_histories is not None and i < len(attention_histories):
            hist = attention_histories[i]

        ctx = CompressionContext(
            target_tokens=target_tokens,
            current_step=state.current_step,
            attention_history=hist,
        )
        new_state, _ = strategy.apply(state, ctx)

        kept = new_state.token_count()
        total = max(1, state.token_count())
        kept_counts.append(kept)
        ratio = kept / total
        ratios.append(ratio)
        kept_steps = {t.timestep for t in new_state.token_meta}

        if important_token_steps:
            hit = len(kept_steps.intersection(important_token_steps))
            recalls.append(hit / max(1, len(important_token_steps)))
        elif important_token_ids:
            kept_ids = {t.token_id for t in new_state.token_meta}
            hit = len(kept_ids.intersection(important_token_ids))
            recalls.append(hit / max(1, len(important_token_ids)))

        sink_steps = {t.timestep for t in state.token_meta if t.is_sink}
        if sink_steps:
            sink_hit = len(kept_steps.intersection(sink_steps))
            sink_recalls.append(sink_hit / len(sink_steps))

        special_steps = {t.timestep for t in state.token_meta if t.is_special_memory}
        if special_steps:
            special_hit = len(kept_steps.intersection(special_steps))
            special_recalls.append(special_hit / len(special_steps))

        attn_per_token = _aggregate_attention_per_token(state)
        if attn_per_token is not None and attn_per_token.size > 0:
            keep_idx = np.asarray(sorted(kept_steps), dtype=np.int64)
            keep_idx = keep_idx[(keep_idx >= 0) & (keep_idx < attn_per_token.shape[0])]
            total_mass = float(np.sum(attn_per_token))
            if total_mass > 1e-8:
                mass = float(np.sum(attn_per_token[keep_idx]) / total_mass)
                mass_recalls.append(mass)

                gain = 1.0 - ratio
                eff = (2.0 * mass * gain) / (mass + gain + 1e-8)
                efficiency_scores.append(float(eff))

    return OfflineMetrics(
        avg_kept_tokens=_mean_or_zero(kept_counts),
        std_kept_tokens=_std_or_zero(kept_counts),
        avg_compression_ratio=_mean_or_zero(ratios),
        std_compression_ratio=_std_or_zero(ratios),
        compression_ratio_p50=_percentile_or_zero(ratios, 50),
        compression_ratio_p90=_percentile_or_zero(ratios, 90),
        important_recall=_mean_or_zero(recalls),
        sink_recall=_mean_or_zero(sink_recalls),
        special_recall=_mean_or_zero(special_recalls),
        attention_mass_recall=_mean_or_zero(mass_recalls),
        efficiency_score=_mean_or_zero(efficiency_scores),
    )
