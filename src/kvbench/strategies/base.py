from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from kvbench.types import CompressionContext, KVCacheState


@dataclass(slots=True)
class CompressionPlan:
    per_layer_indices: dict[int, np.ndarray]
    notes: dict[str, Any] = field(default_factory=dict)


class KVStrategy(ABC):
    name: str = "base"

    @abstractmethod
    def plan(self, state: KVCacheState, ctx: CompressionContext) -> CompressionPlan:
        raise NotImplementedError

    def apply(self, state: KVCacheState, ctx: CompressionContext) -> tuple[KVCacheState, CompressionPlan]:
        plan = self.plan(state, ctx)
        new_state = state.clone_with_indices(plan.per_layer_indices)
        return new_state, plan
