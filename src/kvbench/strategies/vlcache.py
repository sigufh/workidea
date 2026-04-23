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

    @staticmethod
    def _pick_from_candidates(scores: np.ndarray, candidates: np.ndarray, k: int) -> np.ndarray:
        if candidates.size == 0 or k <= 0:
            return np.array([], dtype=np.int64)
        k = min(k, candidates.size)
        local_scores = scores[candidates]
        local_pick = topk_indices(local_scores, k)
        return np.sort(candidates[local_pick])

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

            valid_special = special[special < n]
            valid_text = text_idx[text_idx < n]
            valid_vision = vision_idx[vision_idx < n]

            remain = max(0, budget - valid_special.size)
            text_budget = min(valid_text.size, int(remain * self.text_ratio))
            vision_budget = min(valid_vision.size, remain - text_budget)

            text_pick = self._pick_from_candidates(scores, valid_text, text_budget)
            vision_pick = self._pick_from_candidates(scores, valid_vision, vision_budget)

            keep = merge_unique(valid_special, text_pick, vision_pick)
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
