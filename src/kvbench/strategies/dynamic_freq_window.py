from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kvbench.strategies.base import CompressionPlan, KVStrategy
from kvbench.strategies.utils import merge_unique, recent_window_indices, special_memory_indices, topk_indices
from kvbench.types import CompressionContext, KVCacheState


@dataclass(slots=True)
class DynamicFreqWindowStrategy(KVStrategy):
    """
    Frequency-domain-guided Outlier-KV-aware prototype:
    - Sliding window for recency
    - Frequency analysis over attention history for BaseKV selection
    - Outlier-KV detection with adaptive quota
    - Special memory retention for key context
    """

    name: str = "dynamic_freq_window"
    window_size: int = 512
    min_window_size: int = 192
    basekv_ratio: float = 0.45
    outlier_budget_ratio: float = 0.25
    low_freq_ratio: float = 0.25
    outlier_z: float = 1.8
    recency_boost: float = 0.35
    outlier_min_keep: int = 8
    text_min_ratio: float = 0.1

    def _frequency_scores(self, history: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # history: [tokens, T]
        if history.shape[1] < 2:
            zeros = np.zeros(history.shape[0], dtype=np.float32)
            return zeros, zeros, zeros

        fft = np.fft.rfft(history, axis=1)
        energy = np.abs(fft)

        # low-frequency dominance -> stable BaseKV candidates
        low_bins = max(1, int(energy.shape[1] * self.low_freq_ratio))
        low_band = energy[:, :low_bins].mean(axis=1)
        all_band = energy.mean(axis=1) + 1e-6
        base_score = (low_band / all_band).astype(np.float32)

        # high-frequency energy + temporal spikes -> outlier candidates
        high_band = energy[:, low_bins:].mean(axis=1) if low_bins < energy.shape[1] else np.zeros_like(low_band)
        high_ratio = (high_band / all_band).astype(np.float32)
        spike = history.max(axis=1) - history.mean(axis=1)
        spike_z = ((spike - spike.mean()) / (spike.std() + 1e-6)).astype(np.float32)
        return base_score, high_ratio, spike_z

    def _adaptive_window(self, n: int, recent_attn: np.ndarray | None) -> int:
        if n <= 0:
            return 0
        if recent_attn is None or recent_attn.size == 0:
            return int(min(self.window_size, n))
        # More peaky attention -> smaller window, flatter attention -> wider window.
        sharpness = float(recent_attn.max() / (recent_attn.mean() + 1e-6))
        scale = 1.0 / (1.0 + np.log1p(max(0.0, sharpness - 1.0)))
        span = int(self.min_window_size + (self.window_size - self.min_window_size) * scale)
        return int(min(max(self.min_window_size, span), min(self.window_size, n)))

    def plan(self, state: KVCacheState, ctx: CompressionContext) -> CompressionPlan:
        per_layer: dict[int, np.ndarray] = {}
        n = state.token_count()
        special = special_memory_indices(state)

        layer0_attn = None
        if state.layers and state.layers[0].attention_scores is not None:
            layer0_attn = state.layers[0].attention_scores.mean(axis=0).astype(np.float32)
        recent = recent_window_indices(state, self._adaptive_window(n, layer0_attn))

        if ctx.attention_history is not None and ctx.attention_history.shape[0] == n:
            hist = ctx.attention_history.astype(np.float32)
            base_score, high_ratio, spike_z = self._frequency_scores(hist)
        else:
            # fallback: use current attention snapshot
            if state.layers[0].attention_scores is None:
                base_score = np.zeros(n, dtype=np.float32)
            else:
                base_score = state.layers[0].attention_scores.mean(axis=0).astype(np.float32)
            high_ratio = np.zeros(n, dtype=np.float32)
            spike_z = np.zeros(n, dtype=np.float32)

        for layer_idx, layer in enumerate(state.layers):
            budget = min(ctx.target_tokens, layer.token_count())
            if budget >= layer.token_count():
                per_layer[layer_idx] = np.arange(layer.token_count(), dtype=np.int64)
                continue

            reserve = merge_unique(special, recent)
            remain = max(0, budget - reserve.size)
            base_k = int(remain * self.basekv_ratio)

            cur_attn = (
                layer.attention_scores.mean(axis=0).astype(np.float32)
                if layer.attention_scores is not None
                else np.zeros(n, dtype=np.float32)
            )
            outlier_score = high_ratio + np.maximum(spike_z, 0.0)
            cur_z = (cur_attn - cur_attn.mean()) / (cur_attn.std() + 1e-6)
            outlier_score += np.maximum(cur_z.astype(np.float32), 0.0) * 0.5
            outlier_candidates = np.where(outlier_score >= self.outlier_z)[0].astype(np.int64)

            density = float(outlier_candidates.size) / max(1.0, float(n))
            dyn_outlier_ratio = min(0.5, self.outlier_budget_ratio * (1.0 + density * 2.0))
            outlier_quota = int(remain * dyn_outlier_ratio)
            if outlier_candidates.size > 0:
                outlier_quota = max(self.outlier_min_keep, outlier_quota)
            outlier_quota = min(outlier_quota, remain)

            mixed_base = base_score + self.recency_boost * cur_attn
            base_pick = topk_indices(mixed_base, base_k)
            if outlier_candidates.size > outlier_quota > 0:
                scores = outlier_score[outlier_candidates]
                top_outlier = np.argpartition(scores, -outlier_quota)[-outlier_quota:]
                outlier_pick = np.sort(outlier_candidates[top_outlier]).astype(np.int64)
            else:
                outlier_pick = outlier_candidates

            # Protect a minimal amount of text tokens in multimodal traces.
            text_idx = np.array(
                [i for i, t in enumerate(state.token_meta) if t.modality == "text"],
                dtype=np.int64,
            )
            text_protect_k = int(budget * self.text_min_ratio)
            text_pick = topk_indices(mixed_base[text_idx], text_protect_k) if text_idx.size > 0 else np.array([], dtype=np.int64)
            if text_pick.size > 0:
                text_pick = text_idx[text_pick]

            keep = merge_unique(reserve, base_pick, outlier_pick, text_pick)
            if keep.size < budget:
                fill = topk_indices(mixed_base, budget)
                keep = merge_unique(keep, fill)
            if keep.size > budget:
                keep_scores = mixed_base[keep]
                top = np.argpartition(keep_scores, -budget)[-budget:]
                keep = np.sort(keep[top].astype(np.int64))

            per_layer[layer_idx] = keep

        return CompressionPlan(
            per_layer_indices=per_layer,
            notes={
                "strategy": self.name,
                "window_size": self.window_size,
                "min_window_size": self.min_window_size,
                "basekv_ratio": self.basekv_ratio,
                "outlier_budget_ratio": self.outlier_budget_ratio,
                "low_freq_ratio": self.low_freq_ratio,
                "outlier_z": self.outlier_z,
                "recency_boost": self.recency_boost,
                "outlier_min_keep": self.outlier_min_keep,
                "text_min_ratio": self.text_min_ratio,
            },
        )
