from __future__ import annotations

from pathlib import Path

import numpy as np

from kvbench.types import KVCacheState, LayerKV, TokenMeta


def synthetic_states(
    steps: int = 8,
    layers: int = 4,
    heads: int = 8,
    tokens: int = 1024,
    dim: int = 64,
    seed: int = 7,
) -> tuple[list[KVCacheState], list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    states: list[KVCacheState] = []
    histories: list[np.ndarray] = []

    for s in range(steps):
        metas: list[TokenMeta] = []
        for i in range(tokens):
            modality = "text" if i % 7 == 0 else "vision"
            metas.append(
                TokenMeta(
                    token_id=i,
                    timestep=i,
                    modality=modality,
                    is_sink=(i < 8),
                    is_special_memory=(i in {16, 32, 64}),
                )
            )

        layer_list: list[LayerKV] = []
        for _ in range(layers):
            keys = rng.standard_normal((heads, tokens, dim), dtype=np.float32)
            values = rng.standard_normal((heads, tokens, dim), dtype=np.float32)
            attn = np.abs(rng.standard_normal((heads, tokens), dtype=np.float32))
            attn = attn / (attn.sum(axis=1, keepdims=True) + 1e-6)
            ema = 0.8 * attn + 0.2 * np.abs(rng.standard_normal((heads, tokens), dtype=np.float32))
            layer_list.append(LayerKV(keys=keys, values=values, attention_scores=attn, importance_ema=ema))

        history = np.abs(rng.standard_normal((tokens, 48), dtype=np.float32))
        states.append(KVCacheState(layers=layer_list, token_meta=metas, current_step=s))
        histories.append(history)

    return states, histories


def load_trace_npz(path: str | Path) -> tuple[list[KVCacheState], list[np.ndarray]]:
    """
    NPZ schema (model-free):
    - keys: [steps, layers, heads, tokens, dim]
    - values: [steps, layers, heads, tokens, dim]
    - attn: [steps, layers, heads, tokens]
    - modality: [tokens] int (0=vision,1=text)
    - sink_idx: [k]
    - special_idx: [m]
    - history: [steps, tokens, hist]
    """
    data = np.load(path)
    keys = data["keys"]
    values = data["values"]
    attn = data["attn"]
    history = data["history"] if "history" in data else None
    modality = data["modality"] if "modality" in data else np.zeros(keys.shape[3], dtype=np.int64)
    sink_idx = set(data["sink_idx"].tolist()) if "sink_idx" in data else set()
    special_idx = set(data["special_idx"].tolist()) if "special_idx" in data else set()

    steps, layers, heads, tokens, dim = keys.shape
    states: list[KVCacheState] = []
    histories: list[np.ndarray] = []

    for s in range(steps):
        metas = [
            TokenMeta(
                token_id=i,
                timestep=i,
                modality=("text" if int(modality[i]) == 1 else "vision"),
                is_sink=(i in sink_idx),
                is_special_memory=(i in special_idx),
            )
            for i in range(tokens)
        ]
        layer_list = [
            LayerKV(
                keys=keys[s, l],
                values=values[s, l],
                attention_scores=attn[s, l],
                importance_ema=None,
            )
            for l in range(layers)
        ]
        states.append(KVCacheState(layers=layer_list, token_meta=metas, current_step=s))
        histories.append(history[s] if history is not None else None)

    return states, histories
