"""Preference vector w: sampling, rate-limiting, floor-clip — batched (N, dim).
Mirrors the Multi-Objective Module pipeline in chapter3.tex fig 3.3 — Dirichlet
sample -> rate-limiter -> floor-clip -> (OOD monitor, not implemented here since
it depends on sigma_t from the Adaptation Module, which is out of prelim scope).
"""

from __future__ import annotations

import numpy as np

from talon_rl.config import PreferenceCfg, RewardVectorCfg


def sample_preference_vector(
    rng: np.random.Generator, reward_cfg: RewardVectorCfg, pref_cfg: PreferenceCfg, num_envs: int
) -> np.ndarray:
    """One Dirichlet(alpha) sample per lane, per chapter3.tex §3.2.3 ("w ~ Dirichlet(1.0)")."""
    alpha = np.full(reward_cfg.dim, pref_cfg.dirichlet_alpha, dtype=np.float64)
    return rng.dirichlet(alpha, size=num_envs).astype(np.float32)


def rate_limit(w_prev: np.ndarray, w_target: np.ndarray, max_delta: float) -> np.ndarray:
    """Caps ||w_t - w_{t-1}|| per row so w can't jump discontinuously within
    an episode. w_prev, w_target: (N, dim) -> (N, dim)."""
    delta = w_target - w_prev
    norm = np.linalg.norm(delta, axis=-1, keepdims=True)
    within_cap = (norm <= max_delta) | (norm == 0.0)
    scale = max_delta / np.maximum(norm, 1e-12)
    capped = w_prev + delta * scale
    return np.where(within_cap, w_target, capped).astype(np.float32)


def floor_clip(
    w: np.ndarray, term_names: tuple[str, ...], floor_eps: float, floored_term: str = "impact"
) -> np.ndarray:
    """w_impact >= eps always (chapter3.tex: "การตัดค่าต่ำสุด, w_impact >= epsilon") —
    never let impact-mitigation weight hit exactly zero, then renormalize each
    row to sum to 1. w: (N, dim) -> (N, dim)."""
    w = w.copy()
    idx = term_names.index(floored_term)
    below = w[:, idx] < floor_eps
    deficit = np.where(below, floor_eps - w[:, idx], 0.0)
    w[:, idx] = np.where(below, floor_eps, w[:, idx])

    others = np.array([i for i in range(w.shape[1]) if i != idx])
    other_sum = w[:, others].sum(axis=-1, keepdims=True)
    has_room = other_sum > 0
    safe_other_sum = np.where(has_room, other_sum, 1.0)
    reduction = deficit[:, None] * (w[:, others] / safe_other_sum) * has_room
    w[:, others] -= reduction

    return (w / w.sum(axis=-1, keepdims=True)).astype(np.float32)


def floor_clip_terms(
    w: np.ndarray, term_names: tuple[str, ...], floors: dict[str, float]
) -> np.ndarray:
    """Enforce several non-zero preference floors and renormalize once.

    Balance is safety-critical for locomotion, so it needs the same invariant
    treatment as impact instead of disappearing in a random Dirichlet episode.
    """
    w = w.copy()
    indices = {name: term_names.index(name) for name in floors}
    floor_values = np.array([floors[name] for name in floors], dtype=np.float32)
    if np.any(floor_values < 0) or floor_values.sum() >= 1:
        raise ValueError("preference floors must be non-negative and sum to less than 1")
    for name, index in indices.items():
        w[:, index] = np.maximum(w[:, index], floors[name])

    required = floor_values.sum(axis=0)
    total = w.sum(axis=-1, keepdims=True)
    excess = np.maximum(total - 1.0, 0.0)
    adjustable = np.ones(w.shape[1], dtype=bool)
    adjustable[list(indices.values())] = False
    adjustable_values = w[:, adjustable]
    adjustable_sum = adjustable_values.sum(axis=-1, keepdims=True)
    reduction = excess * adjustable_values / np.maximum(adjustable_sum, 1e-8)
    w[:, adjustable] = adjustable_values - reduction
    return (w / w.sum(axis=-1, keepdims=True)).astype(np.float32)
