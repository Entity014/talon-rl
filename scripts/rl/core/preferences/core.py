"""Preference contracts and reusable implementations."""
from __future__ import annotations
from typing import Protocol, runtime_checkable

import numpy as np
from talon_rl.config import PreferenceCfg, RewardVectorCfg

from .functional import floor_clip_terms, sample_preference_vector


@runtime_checkable
class PreferenceSampler(Protocol):
    def sample(self, batch_size: int, step: int = 0) -> np.ndarray: ...


@runtime_checkable
class PreferenceTransform(Protocol):
    def transform(self, w: np.ndarray) -> np.ndarray: ...


class DirichletPreferenceSampler:
    """Stateful adapter over the established Dirichlet sampling function."""

    def __init__(self, reward_cfg: RewardVectorCfg, pref_cfg: PreferenceCfg, seed: int = 0):
        self.reward_cfg = reward_cfg
        self.pref_cfg = pref_cfg
        self.rng = np.random.default_rng(seed)

    def sample(self, batch_size: int, step: int = 0) -> np.ndarray:
        return sample_preference_vector(
            self.rng, self.reward_cfg, self.pref_cfg, batch_size, step=step
        )


class FloorPreferenceTransform:
    """Simplex lower-bound projection using the tested water-filling implementation."""

    def __init__(self, term_names: tuple[str, ...], floors: dict[str, float]):
        self.term_names = tuple(term_names)
        self.floors = dict(floors)

    def transform(self, w: np.ndarray) -> np.ndarray:
        return floor_clip_terms(w, self.term_names, self.floors)
