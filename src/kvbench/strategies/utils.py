from __future__ import annotations

import numpy as np

from kvbench.types import KVCacheState


def token_scores_mean_attention(state: KVCacheState, layer_idx: int) -> np.ndarray:
    layer = state.layers[layer_idx]
    if layer.attention_scores is None:
        return np.zeros(layer.token_count(), dtype=np.float32)
    return layer.attention_scores.mean(axis=0).astype(np.float32)


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return np.array([], dtype=np.int64)
    k = min(k, scores.shape[0])
    part = np.argpartition(scores, -k)[-k:]
    return np.sort(part.astype(np.int64))


def special_memory_indices(state: KVCacheState) -> np.ndarray:
    idx = [i for i, t in enumerate(state.token_meta) if t.is_special_memory or t.is_sink]
    if not idx:
        return np.array([], dtype=np.int64)
    return np.array(sorted(set(idx)), dtype=np.int64)


def recent_window_indices(state: KVCacheState, window: int) -> np.ndarray:
    n = state.token_count()
    if window >= n:
        return np.arange(n, dtype=np.int64)
    start = max(0, n - window)
    return np.arange(start, n, dtype=np.int64)


def merge_unique(*arrays: np.ndarray) -> np.ndarray:
    valid = [a.astype(np.int64) for a in arrays if a is not None and a.size > 0]
    if not valid:
        return np.array([], dtype=np.int64)
    return np.unique(np.concatenate(valid))
