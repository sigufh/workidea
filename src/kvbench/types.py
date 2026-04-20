from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class TokenMeta:
    token_id: int
    timestep: int
    modality: str = "vision"
    is_sink: bool = False
    is_special_memory: bool = False
    tags: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LayerKV:
    keys: np.ndarray  # [heads, tokens, dim]
    values: np.ndarray  # [heads, tokens, dim]
    attention_scores: np.ndarray | None = None  # [heads, tokens]
    importance_ema: np.ndarray | None = None  # [heads, tokens]

    def token_count(self) -> int:
        return int(self.keys.shape[1])


@dataclass(slots=True)
class KVCacheState:
    layers: list[LayerKV]
    token_meta: list[TokenMeta]
    current_step: int = 0

    def token_count(self) -> int:
        return len(self.token_meta)

    def clone_with_indices(self, per_layer_indices: dict[int, np.ndarray]) -> "KVCacheState":
        new_layers: list[LayerKV] = []
        global_keep = None

        for layer_idx, layer in enumerate(self.layers):
            idx = per_layer_indices.get(layer_idx)
            if idx is None:
                idx = np.arange(layer.token_count(), dtype=np.int64)
            idx = np.unique(np.clip(idx.astype(np.int64), 0, layer.token_count() - 1))
            if layer_idx == 0:
                global_keep = idx

            new_layer = LayerKV(
                keys=layer.keys[:, idx, :],
                values=layer.values[:, idx, :],
                attention_scores=(layer.attention_scores[:, idx] if layer.attention_scores is not None else None),
                importance_ema=(layer.importance_ema[:, idx] if layer.importance_ema is not None else None),
            )
            new_layers.append(new_layer)

        if global_keep is None:
            global_keep = np.arange(self.token_count(), dtype=np.int64)

        new_meta = [self.token_meta[int(i)] for i in global_keep.tolist()]
        return KVCacheState(layers=new_layers, token_meta=new_meta, current_step=self.current_step)


@dataclass(slots=True)
class CompressionContext:
    target_tokens: int
    current_step: int
    attention_history: np.ndarray | None = None  # [tokens, history_len]
    extra: dict[str, Any] = field(default_factory=dict)
