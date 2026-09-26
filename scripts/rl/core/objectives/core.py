"""Training-objective contract and reusable implementations."""
from __future__ import annotations
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from torch import Tensor

from .losses import d3po_actor_loss


@runtime_checkable
class TrainingObjective(Protocol):
    def loss(self, batch: Any, policy: Any) -> Tensor: ...


class D3POObjective:
    """Late-stage-weighted PPO objective backed by the established loss function."""

    def __init__(self, clip_eps: float = 0.2):
        self.clip_eps = float(clip_eps)

    def loss(self, batch: Mapping[str, Tensor], policy: Any = None) -> Tensor:
        del policy
        return d3po_actor_loss(
            batch["ratio"],
            batch["advantages"],
            batch["preferences"],
            self.clip_eps,
        )
