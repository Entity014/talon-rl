"""The 5-term reward vector from chapter3.tex table 3.3 (tab:reward-vector).
Batched (N, ...) -> (N,) throughout — this is the one place the 5 reward
formulas are written (talon_rl/tasks/locomotion/a1_env/'s manager-based env
calls compute_reward_vector directly rather than reimplementing any of this
in torch — see that module's docstring).

Each function takes batched arrays describing the current transition and
returns a per-lane (N,) float32 array; `compute_reward_vector` stacks them
into a fixed-order (N, dim) array matching RewardVectorCfg.term_names.

None of these terms know about the preference vector w — arbitration between
them happens downstream (see training/moppo.py), consistent with chapter3.tex
treating w as something applied to the reward *vector*, not baked into any
single term.
"""

from __future__ import annotations

import numpy as np

from .config import RewardVectorCfg


def progress_reward(v_actual: np.ndarray, v_command: np.ndarray, std: float) -> np.ndarray:
    """Go-anywhere navigation: exp-kernel velocity tracking. (N, 3), (N, 3) -> (N,)."""
    err = np.sum((v_actual - v_command) ** 2, axis=-1)
    return np.exp(-err / (std**2)).astype(np.float32)


def clearance_reward(obstacle_dist: np.ndarray, safe_dist: float = 0.5) -> np.ndarray:
    """Autonomous obstacle negotiation — scripted signal until the Exteroception
    Module exists (out of scope here). (N,) -> (N,)."""
    return np.clip(obstacle_dist / safe_dist, 0.0, 1.0).astype(np.float32)


def energy_reward(joint_torque: np.ndarray, joint_vel: np.ndarray) -> np.ndarray:
    """Raw power per step, negated. (N, 12), (N, 12) -> (N,)."""
    power = np.sum(np.abs(joint_torque * joint_vel), axis=-1)
    return (-power).astype(np.float32)


def impact_reward(foot_contact_force: np.ndarray, threshold: float = 50.0) -> np.ndarray:
    """Continuous impact mitigation — penalize peak landing force, not just
    falls. Threshold is an arbitrary prelim placeholder; retune against real
    A1 contact-force ranges before any hardware test. (N, 4) -> (N,)."""
    if foot_contact_force.shape[-1] == 0:
        return np.zeros(foot_contact_force.shape[0], dtype=np.float32)
    peak = np.max(np.abs(foot_contact_force), axis=-1)
    return (-np.maximum(0.0, peak - threshold) / threshold).astype(np.float32)


def smoothness_reward(action: np.ndarray, prev_action: np.ndarray, joint_acc: np.ndarray) -> np.ndarray:
    """Fixed-weight regularizer (table 3.3's 5th row) — NOT part of the
    preference vector w in the real system, but kept in the vector here so
    the smoke test exercises all 5 terms end-to-end. (N, 12) each -> (N,)."""
    action_rate = np.sum((action - prev_action) ** 2, axis=-1)
    acc = np.sum(joint_acc**2, axis=-1)
    return (-(action_rate + 0.01 * acc)).astype(np.float32)


_TERM_FUNCS = {
    "progress": lambda t, cfg: progress_reward(t["v_actual"], t["v_command"], cfg.progress_std),
    "clearance": lambda t, cfg: clearance_reward(t["obstacle_dist"]),
    "energy": lambda t, cfg: energy_reward(t["joint_torque"], t["joint_vel"]),
    "impact": lambda t, cfg: impact_reward(t["foot_contact_force"]),
    "smoothness": lambda t, cfg: smoothness_reward(t["action"], t["prev_action"], t["joint_acc"]),
}


def compute_reward_vector(transition: dict, cfg: RewardVectorCfg) -> np.ndarray:
    """Returns r as an (N, dim) array in cfg.term_names order. Inactive terms are 0."""
    n = transition["obs"].shape[0]
    values = []
    for name, active in zip(cfg.term_names, cfg.active):
        values.append(_TERM_FUNCS[name](transition, cfg) if active else np.zeros(n, dtype=np.float32))
    return np.stack(values, axis=-1).astype(np.float32)
