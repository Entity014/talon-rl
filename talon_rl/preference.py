"""Preference vector w: sampling, rate-limiting, floor-clip.

Mirrors the Multi-Objective Module pipeline in chapter3.tex fig 3.3 — Dirichlet
sample -> rate-limiter -> floor-clip -> (OOD monitor, not implemented here since
it depends on sigma_t from the Adaptation Module, which is out of prelim scope).
"""

from __future__ import annotations

import numpy as np

from .config import PreferenceCfg, RewardVectorCfg


def sample_preference_vector(rng: np.random.Generator, reward_cfg: RewardVectorCfg, pref_cfg: PreferenceCfg) -> np.ndarray:
    """One Dirichlet(alpha) sample per episode, per chapter3.tex §3.2.3 ("w ~ Dirichlet(1.0)")."""
    alpha = np.full(reward_cfg.dim, pref_cfg.dirichlet_alpha, dtype=np.float64)
    return rng.dirichlet(alpha).astype(np.float32)


def rate_limit(w_prev: np.ndarray, w_target: np.ndarray, max_delta: float) -> np.ndarray:
    """Caps ||w_t - w_{t-1}|| so w can't jump discontinuously within an episode."""
    delta = w_target - w_prev
    norm = float(np.linalg.norm(delta))
    if norm <= max_delta or norm == 0.0:
        return w_target
    return w_prev + delta * (max_delta / norm)


def floor_clip(w: np.ndarray, term_names: tuple[str, ...], floor_eps: float, floored_term: str = "impact") -> np.ndarray:
    """w_impact >= eps always (chapter3.tex: "การตัดค่าต่ำสุด, w_impact >= epsilon") —
    never let impact-mitigation weight hit exactly zero, then renormalize to sum to 1."""
    w = w.copy()
    idx = term_names.index(floored_term)
    if w[idx] < floor_eps:
        deficit = floor_eps - w[idx]
        w[idx] = floor_eps
        # take the deficit proportionally from the other terms
        others = np.array([i for i in range(len(w)) if i != idx])
        other_sum = w[others].sum()
        if other_sum > 0:
            w[others] -= deficit * (w[others] / other_sum)
    return (w / w.sum()).astype(np.float32)
