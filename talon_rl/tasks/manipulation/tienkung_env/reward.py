# talon_rl/tasks/manipulation/tienkung_env/reward.py
"""The 5-term reward vector for TienKung's bimanual box-carry task —
same shape as talon_rl/reward.py's (this thesis's own A1 locomotion reward
vector), retargeted semantics: box-height tracking instead of velocity
tracking, arm-contact force instead of foot-contact force, 8 arm DOF
instead of 12 leg DOF. See
docs/superpowers/specs/2026-09-15-tienkung-manipulation-design.md's Reward
Vector table for the full A1-analog mapping.

Batched (N, ...) -> (N,) throughout, same convention as talon_rl/reward.py.
None of these terms know about the preference vector w — same as that
file, arbitration happens downstream in the MOPPO training loop.
"""

from __future__ import annotations

import numpy as np

from .config import RewardVectorCfg


def progress_reward(box_height: np.ndarray, target_height: np.ndarray, std: float) -> np.ndarray:
    """Exp-kernel tracking of box height toward the lift-and-hold target.
    (N,), (N,) -> (N,)."""
    err = (box_height - target_height) ** 2
    return np.exp(-err / (std**2)).astype(np.float32)


def clearance_reward(obstacle_dist: np.ndarray, safe_dist: float = 0.5) -> np.ndarray:
    """Distance to obstacles while reaching for the box — scripted signal
    until a real exteroception source exists (out of scope here), same
    formula as talon_rl.reward.clearance_reward. (N,) -> (N,)."""
    return np.clip(obstacle_dist / safe_dist, 0.0, 1.0).astype(np.float32)


def energy_reward(joint_torque: np.ndarray, joint_vel: np.ndarray) -> np.ndarray:
    """Raw arm power per step, negated. (N, 8), (N, 8) -> (N,)."""
    power = np.sum(np.abs(joint_torque * joint_vel), axis=-1)
    return (-power).astype(np.float32)


def impact_reward(arm_contact_force: np.ndarray, threshold: float = 50.0) -> np.ndarray:
    """Continuous grip-force mitigation — a privileged, sim-only training
    signal (no real force/torque sensor exists on this hardware; reward
    computation runs at training time only, so this never needs to be
    sim-to-real transferable). Threshold is an arbitrary prelim
    placeholder. (N, 2) -> (N,)."""
    if arm_contact_force.shape[-1] == 0:
        return np.zeros(arm_contact_force.shape[0], dtype=np.float32)
    peak = np.max(np.abs(arm_contact_force), axis=-1)
    return (-np.maximum(0.0, peak - threshold) / threshold).astype(np.float32)


def smoothness_reward(action: np.ndarray, prev_action: np.ndarray, joint_acc: np.ndarray) -> np.ndarray:
    """Fixed-weight regularizer, same formula as talon_rl.reward's, over 8
    arm DOF instead of 12. (N, 8) each -> (N,)."""
    action_rate = np.sum((action - prev_action) ** 2, axis=-1)
    acc = np.sum(joint_acc**2, axis=-1)
    return (-(action_rate + 0.01 * acc)).astype(np.float32)


_TERM_FUNCS = {
    "progress": lambda t, cfg: progress_reward(t["box_height"], t["target_height"], cfg.progress_std),
    "clearance": lambda t, cfg: clearance_reward(t["obstacle_dist"]),
    "energy": lambda t, cfg: energy_reward(t["joint_torque"], t["joint_vel"]),
    "impact": lambda t, cfg: impact_reward(t["arm_contact_force"]),
    "smoothness": lambda t, cfg: smoothness_reward(t["action"], t["prev_action"], t["joint_acc"]),
}


def compute_reward_vector(transition: dict, cfg: RewardVectorCfg) -> np.ndarray:
    """Returns r as an (N, dim) array in cfg.term_names order. Inactive terms are 0."""
    n = transition["obs"].shape[0]
    values = []
    for name, active in zip(cfg.term_names, cfg.active):
        values.append(_TERM_FUNCS[name](transition, cfg) if active else np.zeros(n, dtype=np.float32))
    return np.stack(values, axis=-1).astype(np.float32)
