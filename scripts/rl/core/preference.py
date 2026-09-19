"""Preference vector w: sampling, rate-limiting, floor-clip — batched (N, dim).
Mirrors the Multi-Objective Module pipeline in chapter3.tex fig 3.3 — Dirichlet
sample -> rate-limiter -> floor-clip -> (OOD monitor, not implemented here since
it depends on sigma_t from the Adaptation Module, which is out of prelim scope).
"""

from __future__ import annotations

import numpy as np

from talon_rl.config import PreferenceCfg, RewardVectorCfg


def curriculum_alpha(reward_cfg: RewardVectorCfg, pref_cfg: PreferenceCfg, step: int) -> np.ndarray:
    """Per-term Dirichlet alpha for `step` (trainer._t), linearly annealing
    `pref_cfg.curriculum_alpha_start` -> flat `dirichlet_alpha` over
    `curriculum_updates` steps -- see PreferenceCfg's own docstring for why
    this curriculum exists. Disabled (returns the flat `dirichlet_alpha`
    array unconditionally, `step` ignored) whenever `curriculum_alpha_start`
    is None or `curriculum_updates <= 0`, which is the default -- opt-in,
    matches chapter3.tex §3.2.3 ("w ~ Dirichlet(1.0)") exactly until a
    caller (train_prelim.py) explicitly configures it."""
    end = np.full(reward_cfg.dim, pref_cfg.dirichlet_alpha, dtype=np.float64)
    if pref_cfg.curriculum_alpha_start is None or pref_cfg.curriculum_updates <= 0:
        return end
    start = np.asarray(pref_cfg.curriculum_alpha_start, dtype=np.float64)
    frac = min(max(step, 0) / pref_cfg.curriculum_updates, 1.0)
    return start + (end - start) * frac


def sample_preference_vector(
    rng: np.random.Generator, reward_cfg: RewardVectorCfg, pref_cfg: PreferenceCfg, num_envs: int, step: int = 0
) -> np.ndarray:
    """One Dirichlet(alpha) sample per lane, per chapter3.tex §3.2.3 ("w ~
    Dirichlet(1.0)") -- `alpha` is per-term and may vary with `step` when
    `pref_cfg`'s curriculum is configured, see curriculum_alpha()."""
    alpha = curriculum_alpha(reward_cfg, pref_cfg, step)
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

    Found 2026-09-18 (sampling 200k draws from the actual training
    distribution): the non-floored ("adjustable") terms could go NEGATIVE.
    The reduction that pays for raising a deficient floored term is
    subtracted from the adjustable terms proportionally, but was
    unclamped -- e.g. a Dirichlet draw with impact already at ~0.90 (well
    above its own 0.05 floor, so untouched) and balance near 0 (needing
    to jump to its 0.15 floor) leaves progress+energy+smoothness combined
    with only ~0.10 of mass to give up, but paying for balance's rise
    needs ~0.14 -- every adjustable term's reduction exceeds its own
    value, driving all three negative at once. This silently violated the
    invariant CLAUDE.md documents this pipeline as existing to guarantee
    ("w always sums to 1 and never violates the impact floor"). Clamping
    each adjustable term to >= 0 before the final renormalize fixes the
    negative-weight case, which is the more severe one (a nonsensical
    value under any interpretation) -- but note it's a trade, not a full
    fix: in this same over-constrained regime, the final renormalize can
    still leave a floored term (e.g. balance) slightly BELOW its nominal
    floor, since dividing by a total > 1 shrinks every surviving
    component, floors included. Reaching that residual edge case needs
    one component to already hold most of the simplex's mass -- rare
    under Dirichlet(1) (~0.01% per draw for a >0.9 single-component
    marginal) but not impossible over a long training run. A fully
    correct fix would let over-floor terms like impact give up mass down
    to their OWN floor too, not just the adjustable terms; not implemented
    here since the current trade (never negative, floor occasionally
    undershot by a small amount in a rare case) is a strictly smaller
    problem than the bug it replaces."""
    w = w.copy()
    indices = {name: term_names.index(name) for name in floors}
    floor_values = np.array([floors[name] for name in floors], dtype=np.float32)
    if np.any(floor_values < 0) or floor_values.sum() >= 1:
        raise ValueError("preference floors must be non-negative and sum to less than 1")
    for name, index in indices.items():
        w[:, index] = np.maximum(w[:, index], floors[name])

    total = w.sum(axis=-1, keepdims=True)
    excess = np.maximum(total - 1.0, 0.0)
    adjustable = np.ones(w.shape[1], dtype=bool)
    adjustable[list(indices.values())] = False
    adjustable_values = w[:, adjustable]
    adjustable_sum = adjustable_values.sum(axis=-1, keepdims=True)
    reduction = excess * adjustable_values / np.maximum(adjustable_sum, 1e-8)
    w[:, adjustable] = np.maximum(adjustable_values - reduction, 0.0)
    return (w / w.sum(axis=-1, keepdims=True)).astype(np.float32)
