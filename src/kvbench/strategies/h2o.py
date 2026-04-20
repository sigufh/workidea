from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kvbench.strategies.base import CompressionPlan, KVStrategy
from kvbench.strategies.utils import merge_unique, special_memory_indices, topk_indices
from kvbench.types import CompressionContext, KVCacheState


@dataclass(slots=True)
class H2OStrategy(KVStrategy):
    name: str = "h2o"
    ema_decay: float = 0.9

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
                scores = np.zeros(n, dtype=np.float32)
            else:
                attn = layer.attention_scores.mean(axis=0)
                if layer.importance_ema is not None:
                    scores = self.ema_decay * layer.importance_ema.mean(axis=0) + (1.0 - self.ema_decay) * attn
                else:
                    scores = attn

            keep_main = topk_indices(scores, max(0, budget - reserved.size))
            keep = merge_unique(keep_main, reserved)
            if keep.size > budget:
                keep = keep[-budget:]
            per_layer[layer_idx] = keep

        return CompressionPlan(per_layer_indices=per_layer, notes={"strategy": self.name})
