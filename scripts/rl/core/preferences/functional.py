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
    """Enforce several non-zero preference floors and renormalize -- a
    water-filling projection onto the simplex-with-lower-bounds
    {w' : sum(w')=1, w'_i >= lb_i}, lb_i = floors.get(term_names[i], 0).

    Balance is safety-critical for locomotion, so it needs the same invariant
    treatment as impact instead of disappearing in a random Dirichlet episode.

    Found 2026-09-18 (sampling 200k draws from the actual training
    distribution, then just impact+balance floored): a naive "raise deficient
    floors, pay for it by shrinking only the UNFLOORED ('adjustable') terms
    proportionally to their own current value" produced negative adjustable
    weights whenever the adjustable terms combined didn't have enough mass to
    give up -- e.g. impact already at ~0.90 (above its own floor, untouched)
    and balance near 0 (needing to jump to 0.15) leaves progress+energy+
    smoothness only ~0.10 combined to pay balance's ~0.14 rise. That
    version's fix (clamp adjustable to >= 0) stopped the negative-weight case
    but left a residual: dividing by a total > 1 at the final renormalize
    could still shrink an already-raised floor term below its own floor --
    "rare" (~0.01% per draw) with only 2 floors sharing 3 adjustable terms to
    draw from, but jumped to ~1.8% once `progress_floor_eps` (2026-09-19)
    left only 2 adjustable terms (energy, smoothness) to fund 3 floors --
    a straight-up invariant violation CLAUDE.md documents this pipeline as
    existing to guarantee ("w never violates the impact/balance/progress
    floors").

    This version fixes it properly: floored terms that hold MORE than their
    own floor (e.g. impact=0.90 above its 0.05 floor) also give up their
    slack to pay for other terms' floors, not just the always-unfloored
    ("adjustable") ones -- the "fully correct fix" the previous version's
    docstring flagged as not implemented. Iterative (bounded by `dim`
    rounds): each round, split the remaining excess proportionally across
    every term's current slack (value above ITS OWN lower bound, 0 for an
    unfloored term); any term whose share would push it below its own lower
    bound is capped there instead (consuming exactly its slack, not its
    proportional share) and excluded from the next round; terms unaffected
    by a cap this round are fully resolved. Terminates in at most `dim`
    rounds since each round with any capping permanently retires at least
    one term. `test_floor_clip_terms_never_undershoots_its_own_floor` covers
    the exact regression this fixes."""
    w = w.astype(np.float64)  # water-filling iterates a few times; avoid float32 drift
    dim = w.shape[1]
    floor_values = np.array(list(floors.values()), dtype=np.float64)
    if np.any(floor_values < 0) or floor_values.sum() >= 1:
        raise ValueError("preference floors must be non-negative and sum to less than 1")

    lb = np.zeros(dim, dtype=np.float64)
    for name, eps in floors.items():
        lb[term_names.index(name)] = eps

    result = np.maximum(w, lb[None, :])
    remaining_excess = result.sum(axis=-1, keepdims=True) - 1.0  # always >= 0, sum(w)==1 pre-floor
    active = result > lb[None, :] + 1e-12  # terms with slack, still eligible to be reduced

    for _ in range(dim):
        if not active.any():
            break
        slack = np.where(active, result - lb[None, :], 0.0)
        slack_sum = slack.sum(axis=-1, keepdims=True)
        safe_slack_sum = np.where(slack_sum > 1e-12, slack_sum, 1.0)
        share = remaining_excess * slack / safe_slack_sum
        would_be = result - share
        newly_capped = active & (would_be < lb[None, :] - 1e-9)
        row_has_cap = newly_capped.any(axis=-1, keepdims=True)

        # rows with a cap this round: only settle the capped terms at their
        # floor (consuming exactly their own slack); everything else in that
        # row is deferred to the next round with the freed-up excess.
        absorbed = np.where(newly_capped, slack, 0.0).sum(axis=-1, keepdims=True)
        result = np.where(newly_capped, lb[None, :], result)
        remaining_excess = np.where(row_has_cap, remaining_excess - absorbed, remaining_excess)
        active = active & ~newly_capped

        # rows with no cap this round: the proportional reduction is exact, resolve them now.
        clean = active & ~row_has_cap
        result = np.where(clean, would_be, result)
        remaining_excess = np.where(row_has_cap, remaining_excess, 0.0)

    result = np.maximum(result, 0.0)  # numerical safety only, should already hold from the loop
    return (result / result.sum(axis=-1, keepdims=True)).astype(np.float32)
