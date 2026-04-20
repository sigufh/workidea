from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kvbench.strategies.base import CompressionPlan, KVStrategy
from kvbench.strategies.utils import merge_unique, recent_window_indices, special_memory_indices, topk_indices
from kvbench.types import CompressionContext, KVCacheState


@dataclass(slots=True)
class DynamicFreqWindowStrategy(KVStrategy):
    """
    Your custom prototype:
    - Sliding window for recency
    - Frequency analysis over attention history for BaseKV vs Outlier detection
    - Special memory retention for key context
    """

    name: str = "dynamic_freq_window"
    window_size: int = 512
    basekv_ratio: float = 0.4
    outlier_z: float = 2.0

    def _frequency_scores(self, history: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # history: [tokens, T]
        fft = np.fft.rfft(history, axis=1)
        energy = np.abs(fft)

        # low-frequency dominance -> stable BaseKV candidates
        low_band = energy[:, : max(1, energy.shape[1] // 4)].mean(axis=1)
        all_band = energy.mean(axis=1) + 1e-6
        base_score = (low_band / all_band).astype(np.float32)

        # high-frequency spikes -> outlier candidates
        spike = history.max(axis=1) - history.mean(axis=1)
        z = (spike - spike.mean()) / (spike.std() + 1e-6)
        return base_score, z.astype(np.float32)

    def plan(self, state: KVCacheState, ctx: CompressionContext) -> CompressionPlan:
        per_layer: dict[int, np.ndarray] = {}
        n = state.token_count()
        special = special_memory_indices(state)
        recent = recent_window_indices(state, min(self.window_size, n))

        if ctx.attention_history is not None and ctx.attention_history.shape[0] == n:
            base_score, outlier_z = self._frequency_scores(ctx.attention_history.astype(np.float32))
        else:
            # fallback: use current attention snapshot
            if state.layers[0].attention_scores is None:
                base_score = np.zeros(n, dtype=np.float32)
            else:
                base_score = state.layers[0].attention_scores.mean(axis=0).astype(np.float32)
            outlier_z = np.zeros(n, dtype=np.float32)

        for layer_idx, layer in enumerate(state.layers):
            budget = min(ctx.target_tokens, layer.token_count())
            if budget >= layer.token_count():
                per_layer[layer_idx] = np.arange(layer.token_count(), dtype=np.int64)
                continue

            reserve = merge_unique(special, recent)
            remain = max(0, budget - reserve.size)
            base_k = int(remain * self.basekv_ratio)

            base_pick = topk_indices(base_score, base_k)
            outlier_pick = np.where(outlier_z >= self.outlier_z)[0].astype(np.int64)

            keep = merge_unique(reserve, base_pick, outlier_pick)
            if keep.size < budget:
                fill = topk_indices(base_score, budget)
                keep = merge_unique(keep, fill)
            if keep.size > budget:
                keep = np.sort(keep)[-budget:]

            per_layer[layer_idx] = keep

        return CompressionPlan(
            per_layer_indices=per_layer,
            notes={
                "strategy": self.name,
                "window_size": self.window_size,
                "basekv_ratio": self.basekv_ratio,
                "outlier_z": self.outlier_z,
            },
        )
