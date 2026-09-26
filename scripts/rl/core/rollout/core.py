"""Rollout contracts and reusable advantage estimation."""
from __future__ import annotations
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .gae_functional import gae_per_objective


@runtime_checkable
class RolloutCollector(Protocol):
    def collect(self, policy: Any) -> Any: ...


@runtime_checkable
class AdvantageEstimator(Protocol):
    def compute(self, *args: Any, **kwargs: Any) -> tuple[Any, Any]: ...


class PerObjectiveGAE:
    """Reusable GAE estimator for vector-valued rewards."""

    def __init__(self, gamma: float, lam: float):
        self.gamma = float(gamma)
        self.lam = float(lam)

    def compute(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        advantages = gae_per_objective(rewards, values, dones, self.gamma, self.lam)
        returns = advantages + values[:-1]
        return advantages, returns
