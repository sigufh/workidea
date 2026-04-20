from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kvbench.strategies.base import CompressionPlan, KVStrategy
from kvbench.strategies.utils import merge_unique, special_memory_indices, topk_indices
from kvbench.types import CompressionContext, KVCacheState


@dataclass(slots=True)
class SnapKVStrategy(KVStrategy):
    name: str = "snap"
    recent_bias: float = 0.2

    def plan(self, state: KVCacheState, ctx: CompressionContext) -> CompressionPlan:
        per_layer: dict[int, np.ndarray] = {}
        reserved = special_memory_indices(state)

        for layer_idx, layer in enumerate(state.layers):
            n = layer.token_count()
            budget = min(ctx.target_tokens, n)
            if budget >= n:
                per_layer[layer_idx] = np.arange(n, dtype=np.int64)
                continue

            if layer.attention_scores is None:
                head_scores = np.zeros((layer.keys.shape[0], n), dtype=np.float32)
            else:
                head_scores = layer.attention_scores.astype(np.float32)

            pos = np.linspace(0.0, 1.0, n, dtype=np.float32)
            boosted = head_scores + self.recent_bias * pos[None, :]

            per_head_k = max(1, (budget - reserved.size) // max(1, boosted.shape[0]))
            head_selected = [topk_indices(boosted[h], per_head_k) for h in range(boosted.shape[0])]
            keep = merge_unique(*head_selected, reserved)

            if keep.size < budget:
                merged_scores = boosted.mean(axis=0)
                fill = topk_indices(merged_scores, budget)
                keep = merge_unique(keep, fill)

            if keep.size > budget:
                merged_scores = boosted.mean(axis=0)
                keep = keep[np.argsort(merged_scores[keep])[-budget:]]
                keep = np.sort(keep)

            per_layer[layer_idx] = keep

        return CompressionPlan(per_layer_indices=per_layer, notes={"strategy": self.name})
