from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kvbench.strategies.base import KVStrategy
from kvbench.types import CompressionContext, KVCacheState


@dataclass(slots=True)
class OfflineMetrics:
    avg_kept_tokens: float
    avg_compression_ratio: float
    important_recall: float


def evaluate_offline(
    strategy: KVStrategy,
    states: list[KVCacheState],
    target_tokens: int,
    important_token_ids: set[int] | None = None,
    attention_histories: list[np.ndarray | None] | None = None,
) -> OfflineMetrics:
    kept_counts: list[int] = []
    ratios: list[float] = []
    recalls: list[float] = []

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
        ratios.append(kept / total)

        if important_token_ids:
            kept_ids = {t.token_id for t in new_state.token_meta}
            hit = len(kept_ids.intersection(important_token_ids))
            recalls.append(hit / max(1, len(important_token_ids)))

    return OfflineMetrics(
        avg_kept_tokens=float(np.mean(kept_counts)) if kept_counts else 0.0,
        avg_compression_ratio=float(np.mean(ratios)) if ratios else 0.0,
        important_recall=float(np.mean(recalls)) if recalls else 0.0,
    )
