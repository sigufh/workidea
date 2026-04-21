from __future__ import annotations

import numpy as np

from kvbench.strategies.base import CompressionPlan, KVStrategy
from kvbench.types import CompressionContext, KVCacheState


class FullKVStrategy(KVStrategy):
    """
    Keep all KV tokens (no compression) as an upper-bound baseline.
    """

    name: str = "fullkv"

    def plan(self, state: KVCacheState, ctx: CompressionContext) -> CompressionPlan:
        per_layer = {
            layer_idx: np.arange(layer.token_count(), dtype=np.int64)
            for layer_idx, layer in enumerate(state.layers)
        }
        return CompressionPlan(
            per_layer_indices=per_layer,
            notes={"strategy": self.name, "kept": "all"},
        )
