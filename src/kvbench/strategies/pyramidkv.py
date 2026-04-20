from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kvbench.strategies.base import CompressionPlan, KVStrategy
from kvbench.strategies.utils import merge_unique, special_memory_indices, token_scores_mean_attention, topk_indices
from kvbench.types import CompressionContext, KVCacheState


@dataclass(slots=True)
class PyramidKVStrategy(KVStrategy):
    name: str = "pyramid"
    min_ratio: float = 0.4

    def _layer_budget(self, base_budget: int, layer_idx: int, num_layers: int) -> int:
        if num_layers <= 1:
            return base_budget
        depth = layer_idx / float(num_layers - 1)
        ratio = 1.0 - depth * (1.0 - self.min_ratio)
        return max(1, int(base_budget * ratio))

    def plan(self, state: KVCacheState, ctx: CompressionContext) -> CompressionPlan:
        per_layer: dict[int, np.ndarray] = {}
        reserved = special_memory_indices(state)
        num_layers = len(state.layers)

        for layer_idx, layer in enumerate(state.layers):
            n = layer.token_count()
            budget = min(self._layer_budget(ctx.target_tokens, layer_idx, num_layers), n)
            if budget >= n:
                per_layer[layer_idx] = np.arange(n, dtype=np.int64)
                continue

            scores = token_scores_mean_attention(state, layer_idx)
            keep_main = topk_indices(scores, max(0, budget - reserved.size))
            keep = merge_unique(keep_main, reserved)

            if keep.size > budget:
                keep = keep[-budget:]
            per_layer[layer_idx] = keep

        return CompressionPlan(
            per_layer_indices=per_layer,
            notes={"strategy": self.name, "min_ratio": self.min_ratio},
        )
