from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kvbench.strategies.base import CompressionPlan, KVStrategy
from kvbench.strategies.utils import merge_unique, topk_indices
from kvbench.types import CompressionContext, KVCacheState


@dataclass(slots=True)
class VLCacheStrategy(KVStrategy):
    name: str = "vlcache"
    text_ratio: float = 0.35
    keep_all_special: bool = True

    def _modality_indices(self, state: KVCacheState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        text, vision, special = [], [], []
        for i, meta in enumerate(state.token_meta):
            if self.keep_all_special and (meta.is_sink or meta.is_special_memory):
                special.append(i)
                continue
            if meta.modality == "text":
                text.append(i)
            else:
                vision.append(i)
        return (
            np.array(text, dtype=np.int64),
            np.array(vision, dtype=np.int64),
            np.array(special, dtype=np.int64),
        )

    def plan(self, state: KVCacheState, ctx: CompressionContext) -> CompressionPlan:
        per_layer: dict[int, np.ndarray] = {}
        text_idx, vision_idx, special = self._modality_indices(state)

        for layer_idx, layer in enumerate(state.layers):
            n = layer.token_count()
            budget = min(ctx.target_tokens, n)
            if budget >= n:
                per_layer[layer_idx] = np.arange(n, dtype=np.int64)
                continue

            if layer.attention_scores is None:
                scores = np.zeros(n, dtype=np.float32)
            else:
                scores = layer.attention_scores.mean(axis=0).astype(np.float32)

            remain = max(0, budget - special.size)
            text_budget = min(text_idx.size, int(remain * self.text_ratio))
            vision_budget = min(vision_idx.size, remain - text_budget)

            text_pick = text_idx[topk_indices(scores[text_idx], text_budget)] if text_idx.size > 0 and text_budget > 0 else np.array([], dtype=np.int64)
            vision_pick = vision_idx[topk_indices(scores[vision_idx], vision_budget)] if vision_idx.size > 0 and vision_budget > 0 else np.array([], dtype=np.int64)

            keep = merge_unique(special, text_pick, vision_pick)
            if keep.size < budget:
                fill = topk_indices(scores, budget)
                keep = merge_unique(keep, fill)
            if keep.size > budget:
                keep = keep[np.argsort(scores[keep])[-budget:]]
                keep = np.sort(keep)

            per_layer[layer_idx] = keep

        return CompressionPlan(
            per_layer_indices=per_layer,
            notes={"strategy": self.name, "text_ratio": self.text_ratio},
        )
