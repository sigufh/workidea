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
    window_size: int = 384
    min_window_size: int = 64
    basekv_ratio: float = 0.7
    outlier_budget_ratio: float = 0.1
    low_freq_ratio: float = 0.2
    outlier_z: float = 2.2
    recency_boost: float = 0.2
    recency_min_ratio: float = 0.08
    outlier_min_keep: int = 8
    text_min_ratio: float = 0.15
    vision_min_ratio: float = 0.15
    cur_attn_weight: float = 0.55
    ema_weight: float = 0.2
    history_weight: float = 0.15
    lowfreq_weight: float = 0.1

    @staticmethod
    def _norm01(x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x.astype(np.float32)
        x = x.astype(np.float32)
        lo = float(x.min())
        hi = float(x.max())
        if hi - lo < 1e-8:
            return np.zeros_like(x, dtype=np.float32)
        return ((x - lo) / (hi - lo + 1e-8)).astype(np.float32)

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

    def _phase_ratio(self, state: KVCacheState, ctx: CompressionContext) -> float:
        # Approximate decoding phase: near 0 means prefill/early decode, near 1 means late decode.
        n = max(1, state.token_count())
        return float(min(1.0, max(0.0, ctx.current_step / float(n))))

    def _layer_ratio(self, layer_idx: int, num_layers: int) -> float:
        if num_layers <= 1:
            return 0.0
        return float(layer_idx / float(num_layers - 1))

    def _weights_for_layer_and_phase(self, layer_ratio: float, phase_ratio: float) -> tuple[float, float, float, float, float]:
        # Deeper layers and later decode rely more on current attention/recency.
        cur_w = self.cur_attn_weight + 0.2 * phase_ratio + 0.1 * layer_ratio
        ema_w = self.ema_weight + 0.05 * layer_ratio - 0.05 * phase_ratio
        hist_w = self.history_weight + 0.08 * (1.0 - layer_ratio) * (1.0 - phase_ratio)
        lowf_w = self.lowfreq_weight + 0.06 * (1.0 - layer_ratio)
        recency_w = self.recency_boost + 0.2 * phase_ratio + 0.05 * layer_ratio
        ws = np.array([cur_w, ema_w, hist_w, lowf_w, recency_w], dtype=np.float32)
        ws = np.maximum(ws, 0.0)
        s = float(ws.sum()) + 1e-8
        ws /= s
        return float(ws[0]), float(ws[1]), float(ws[2]), float(ws[3]), float(ws[4])

    @staticmethod
    def _fit_vec(vec: np.ndarray, n: int) -> np.ndarray:
        if vec.shape[0] == n:
            return vec.astype(np.float32, copy=False)
        if vec.shape[0] > n:
            return vec[:n].astype(np.float32, copy=False)
        out = np.zeros(n, dtype=np.float32)
        out[: vec.shape[0]] = vec.astype(np.float32, copy=False)
        return out

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
            hist_mean = hist.mean(axis=1).astype(np.float32)
        else:
            # fallback: use current attention snapshot
            if state.layers[0].attention_scores is None:
                base_score = np.zeros(n, dtype=np.float32)
            else:
                base_score = state.layers[0].attention_scores.mean(axis=0).astype(np.float32)
            high_ratio = np.zeros(n, dtype=np.float32)
            spike_z = np.zeros(n, dtype=np.float32)
            hist_mean = np.zeros(n, dtype=np.float32)

        base_score = self._norm01(base_score)
        high_ratio = self._norm01(high_ratio)
        spike_pos = self._norm01(np.maximum(spike_z, 0.0))
        hist_mean = self._norm01(hist_mean)

        recency_vec = np.zeros(n, dtype=np.float32)
        if recent.size > 0:
            recency_vec[recent] = np.linspace(0.2, 1.0, num=recent.size, dtype=np.float32)

        phase_ratio = self._phase_ratio(state, ctx)
        num_layers = max(1, len(state.layers))

        for layer_idx, layer in enumerate(state.layers):
            layer_n = layer.token_count()
            budget = min(ctx.target_tokens, layer_n)
            if budget >= layer_n:
                per_layer[layer_idx] = np.arange(layer_n, dtype=np.int64)
                continue

            cur_special = special[special < layer_n]
            cur_recent = recent[recent < layer_n]
            cur_base_score = self._fit_vec(base_score, layer_n)
            cur_high_ratio = self._fit_vec(high_ratio, layer_n)
            cur_spike_pos = self._fit_vec(spike_pos, layer_n)
            cur_hist_mean = self._fit_vec(hist_mean, layer_n)
            cur_recency_vec = self._fit_vec(recency_vec, layer_n)

            special_pick = cur_special
            if special_pick.size > budget:
                if layer.attention_scores is not None:
                    sp_scores = layer.attention_scores.mean(axis=0).astype(np.float32)[special_pick]
                else:
                    sp_scores = np.zeros(special_pick.size, dtype=np.float32)
                top = np.argpartition(sp_scores, -budget)[-budget:]
                special_pick = np.sort(special_pick[top]).astype(np.int64)

            remain = max(0, budget - special_pick.size)
            base_k = int(remain * self.basekv_ratio)

            cur_attn = (
                layer.attention_scores.mean(axis=0).astype(np.float32)
                if layer.attention_scores is not None
                else np.zeros(layer_n, dtype=np.float32)
            )
            cur_attn = self._fit_vec(cur_attn, layer_n)
            cur_attn_n = self._norm01(cur_attn)
            if layer.importance_ema is not None:
                ema_attn = layer.importance_ema.mean(axis=0).astype(np.float32)
                ema_attn_n = self._norm01(self._fit_vec(ema_attn, layer_n))
            else:
                ema_attn_n = cur_attn_n

            layer_ratio = self._layer_ratio(layer_idx, num_layers)
            cur_w, ema_w, hist_w, lowf_w, rec_w = self._weights_for_layer_and_phase(layer_ratio, phase_ratio)
            mixed_base = (
                cur_w * cur_attn_n
                + ema_w * ema_attn_n
                + hist_w * cur_hist_mean
                + lowf_w * cur_base_score
                + rec_w * cur_recency_vec
            ).astype(np.float32)

            outlier_score = (cur_high_ratio + cur_spike_pos).astype(np.float32)
            cur_z = (cur_attn - cur_attn.mean()) / (cur_attn.std() + 1e-6)
            outlier_score += np.maximum(cur_z.astype(np.float32), 0.0) * 0.5
            outlier_candidates = np.where(outlier_score >= self.outlier_z)[0].astype(np.int64)

            density = float(outlier_candidates.size) / max(1.0, float(layer_n))
            dyn_outlier_ratio = min(0.25, self.outlier_budget_ratio * (1.0 + density * 1.5))
            outlier_quota = int(remain * dyn_outlier_ratio)
            if outlier_candidates.size > 0:
                outlier_quota = max(self.outlier_min_keep, outlier_quota)
            outlier_quota = min(outlier_quota, remain)

            recent_quota = min(cur_recent.size, int(budget * self.recency_min_ratio))
            if recent_quota > 0:
                recent_scores = mixed_base[cur_recent]
                ridx = np.argpartition(recent_scores, -recent_quota)[-recent_quota:]
                recent_pick = np.sort(cur_recent[ridx]).astype(np.int64)
            else:
                recent_pick = np.array([], dtype=np.int64)

            base_pick = topk_indices(mixed_base, base_k)
            if outlier_candidates.size > outlier_quota > 0:
                scores = outlier_score[outlier_candidates]
                top_outlier = np.argpartition(scores, -outlier_quota)[-outlier_quota:]
                outlier_pick = np.sort(outlier_candidates[top_outlier]).astype(np.int64)
            else:
                outlier_pick = outlier_candidates

            text_idx = np.array(
                [i for i, t in enumerate(state.token_meta) if t.modality == "text"],
                dtype=np.int64,
            )
            vision_idx = np.array(
                [i for i, t in enumerate(state.token_meta) if t.modality != "text"],
                dtype=np.int64,
            )
            text_idx = text_idx[text_idx < layer_n]
            vision_idx = vision_idx[vision_idx < layer_n]
            text_protect_k = int(budget * self.text_min_ratio)
            vision_protect_k = int(budget * self.vision_min_ratio)
            text_pick = topk_indices(mixed_base[text_idx], text_protect_k) if text_idx.size > 0 else np.array([], dtype=np.int64)
            if text_pick.size > 0:
                text_pick = text_idx[text_pick]
            vision_pick = topk_indices(mixed_base[vision_idx], vision_protect_k) if vision_idx.size > 0 else np.array([], dtype=np.int64)
            if vision_pick.size > 0:
                vision_pick = vision_idx[vision_pick]

            keep = merge_unique(special_pick, recent_pick, base_pick, outlier_pick, text_pick, vision_pick)
            if keep.size < budget:
                fill = topk_indices(mixed_base, budget)
                keep = merge_unique(keep, fill)
            if keep.size > budget:
                must_keep = special_pick
                if must_keep.size >= budget:
                    mk_scores = mixed_base[must_keep]
                    top = np.argpartition(mk_scores, -budget)[-budget:]
                    keep = np.sort(must_keep[top].astype(np.int64))
                else:
                    candidate = np.setdiff1d(keep, must_keep, assume_unique=False)
                    need = budget - must_keep.size
                    if candidate.size > need:
                        cand_scores = mixed_base[candidate]
                        top = np.argpartition(cand_scores, -need)[-need:]
                        candidate = candidate[top]
                    keep = np.sort(merge_unique(must_keep, candidate).astype(np.int64))

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
                "recency_min_ratio": self.recency_min_ratio,
                "outlier_min_keep": self.outlier_min_keep,
                "text_min_ratio": self.text_min_ratio,
                "vision_min_ratio": self.vision_min_ratio,
                "cur_attn_weight": self.cur_attn_weight,
                "ema_weight": self.ema_weight,
                "history_weight": self.history_weight,
                "lowfreq_weight": self.lowfreq_weight,
                "phase_ratio": phase_ratio,
            },
        )
