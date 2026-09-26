"""Structural environment contract consumed by RL training code."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class TalonEnv(Protocol):
    """Batch-native environment interface expected by RL trainers.

    Implementations do not need to inherit this class. They only need to expose
    the declared attributes and methods.
    """

    num_envs: int
    obs_dim: int
    action_dim: int

    def reset(self) -> dict:
        """Return a batch-native transition dictionary."""
        ...

    def step(self, action: np.ndarray) -> tuple[dict, np.ndarray]:
        """Advance the environment and return (transition, done)."""
        ...
