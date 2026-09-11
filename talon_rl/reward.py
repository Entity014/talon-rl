"""The 5-term reward vector from chapter3.tex table 3.3 (tab:reward-vector).

Each function takes a plain dict of scalars/arrays describing the current
transition (kept as a dict, not a typed struct, because the exact fields
available will change once the real Isaac Lab env replaces DummyEnv — see
envs/base_env.py). Every term returns a per-step float; `compute_reward_vector`
stacks them into a fixed-order np.ndarray matching RewardVectorCfg.term_names.

None of these terms know about the preference vector w — arbitration between
them happens downstream (see training/moppo.py), consistent with chapter3.tex
treating w as something applied to the reward *vector*, not baked into any
single term.
"""

from __future__ import annotations

import numpy as np

from .config import RewardVectorCfg


def progress_reward(v_actual: np.ndarray, v_command: np.ndarray, std: float) -> float:
    """Go-anywhere navigation: exp-kernel velocity tracking (RMA/TienKung-style)."""
    err = float(np.sum((v_actual - v_command) ** 2))
    return float(np.exp(-err / (std**2)))


def clearance_reward(obstacle_dist: float, safe_dist: float = 0.5) -> float:
    """Autonomous obstacle negotiation. Scripted signal for the prelim dummy env —
    the real pipeline gets this from the Exteroception Module (out of scope here)."""
    return float(np.clip(obstacle_dist / safe_dist, 0.0, 1.0))


def energy_reward(joint_torque: np.ndarray, joint_vel: np.ndarray) -> float:
    """Raw power per step, negated (lower power => higher reward)."""
    power = float(np.sum(np.abs(joint_torque * joint_vel)))
    return -power


def impact_reward(foot_contact_force: np.ndarray, threshold: float = 50.0) -> float:
    """Continuous impact mitigation \\cite{strauch2025crashcourse} — penalize peak
    landing force, not just falls. Threshold is an arbitrary prelim placeholder;
    retune against real A1 contact-force ranges before any hardware test."""
    peak = float(np.max(np.abs(foot_contact_force))) if foot_contact_force.size else 0.0
    return -max(0.0, peak - threshold) / threshold


def smoothness_reward(action: np.ndarray, prev_action: np.ndarray, joint_acc: np.ndarray) -> float:
    """Fixed-weight regularizer (table 3.3's 5th row) — NOT part of the preference
    vector w in the real system, but kept in the vector here for the prelim so the
    dummy-env smoke test exercises all 5 terms end-to-end. See config.py docstring."""
    action_rate = float(np.sum((action - prev_action) ** 2))
    acc = float(np.sum(joint_acc**2))
    return -(action_rate + 0.01 * acc)


_TERM_FUNCS = {
    "progress": lambda t, cfg: progress_reward(t["v_actual"], t["v_command"], cfg.progress_std),
    "clearance": lambda t, cfg: clearance_reward(t["obstacle_dist"]),
    "energy": lambda t, cfg: energy_reward(t["joint_torque"], t["joint_vel"]),
    "impact": lambda t, cfg: impact_reward(t["foot_contact_force"]),
    "smoothness": lambda t, cfg: smoothness_reward(t["action"], t["prev_action"], t["joint_acc"]),
}


def compute_reward_vector(transition: dict, cfg: RewardVectorCfg) -> np.ndarray:
    """Returns r_t as an (dim,) array in cfg.term_names order. Inactive terms are 0."""
    values = []
    for name, active in zip(cfg.term_names, cfg.active):
        values.append(_TERM_FUNCS[name](transition, cfg) if active else 0.0)
    return np.asarray(values, dtype=np.float32)
