"""Object-oriented normalization facade backed by the established implementation."""
from __future__ import annotations

import numpy as np
from .stats import RunningMeanStd


class RunningNormalizer:
    """Owns running statistics and exposes an explicit transform API."""

    def __init__(self, dim: int, *, center: bool = True, clip: float = 10.0):
        self.stats = RunningMeanStd(dim)
        self.center = bool(center)
        self.clip = float(clip)

    def update(self, x: np.ndarray) -> None:
        self.stats.update(x)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return self.stats.normalize(x, clip=self.clip, center=self.center)

    def state_dict(self) -> dict:
        return self.stats.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.stats.load_state_dict(state)
