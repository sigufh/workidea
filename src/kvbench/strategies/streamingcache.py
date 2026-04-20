from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kvbench.strategies.base import CompressionPlan, KVStrategy
from kvbench.strategies.utils import merge_unique, recent_window_indices, special_memory_indices
from kvbench.types import CompressionContext, KVCacheState


@dataclass(slots=True)
class StreamingCacheStrategy(KVStrategy):
    name: str = "streamingcache"
    sink_size: int = 16
    window_size: int = 512
    anchor_interval: int = 64

    def plan(self, state: KVCacheState, ctx: CompressionContext) -> CompressionPlan:
        per_layer: dict[int, np.ndarray] = {}
        special = special_memory_indices(state)

        for layer_idx, layer in enumerate(state.layers):
            n = layer.token_count()
            budget = min(ctx.target_tokens, n)
            if budget >= n:
                per_layer[layer_idx] = np.arange(n, dtype=np.int64)
                continue

            sink = np.arange(min(self.sink_size, n), dtype=np.int64)
            recent = recent_window_indices(state, min(self.window_size, n))
            anchors = np.arange(0, n, max(1, self.anchor_interval), dtype=np.int64)

            keep = merge_unique(sink, recent, anchors, special)
            if keep.size > budget:
                keep = np.sort(keep)[-budget:]
            per_layer[layer_idx] = keep

        return CompressionPlan(
            per_layer_indices=per_layer,
            notes={
                "strategy": self.name,
                "sink_size": self.sink_size,
                "window_size": self.window_size,
                "anchor_interval": self.anchor_interval,
            },
        )
