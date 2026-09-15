"""Generalized Advantage Estimation — reusable by any on-policy algorithm,
not specific to MOPPO. See scripts/rl/core/algorithms/moppo.py for how MOPPO
calls this per reward-vector objective.
"""

from __future__ import annotations

import numpy as np


def gae_per_objective(
    rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, gamma: float, lam: float
) -> np.ndarray:
    """rewards: (T, N, K), values: (T+1, N, K), dones: (T, N) -> advantages (T, N, K).
    dones[t] masks out value bootstrapping across an episode boundary at step t."""
    T, N, K = rewards.shape
    adv = np.zeros((T, N, K), dtype=np.float32)
    gae = np.zeros((N, K), dtype=np.float32)
    for t in reversed(range(T)):
        mask = (1.0 - dones[t])[:, None]  # (N, 1), broadcasts over K
        delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        adv[t] = gae
    return adv
