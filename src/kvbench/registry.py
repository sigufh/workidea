from __future__ import annotations

from kvbench.strategies import (
    DynamicFreqWindowStrategy,
    H2OStrategy,
    PyramidKVStrategy,
    SnapKVStrategy,
    StreamingCacheStrategy,
    VLCacheStrategy,
)
from kvbench.strategies.base import KVStrategy


def build_strategy(name: str, **kwargs) -> KVStrategy:
    key = name.strip().lower()
    if key == "h2o":
        return H2OStrategy(**kwargs)
    if key in {"snap", "snapkv"}:
        return SnapKVStrategy(**kwargs)
    if key in {"pyramid", "pyramidkv"}:
        return PyramidKVStrategy(**kwargs)
    if key in {"vlcache", "vl-cache"}:
        return VLCacheStrategy(**kwargs)
    if key in {"streamingcache", "streaming", "streaming-cache"}:
        return StreamingCacheStrategy(**kwargs)
    if key in {"dynamic_freq_window", "dfw", "yours"}:
        return DynamicFreqWindowStrategy(**kwargs)
    raise ValueError(f"Unknown strategy: {name}")


def list_strategies() -> list[str]:
    return [
        "h2o",
        "snap",
        "pyramid",
        "vlcache",
        "streamingcache",
        "dynamic_freq_window",
    ]
