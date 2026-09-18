"""The reward vector from chapter3.tex table 3.3 (tab:reward-vector), plus one
addition (`balance`, 2026-09-17) not yet in that table -- see its docstring
below for why. Batched (N, ...) -> (N,) throughout — this is the one place
the reward formulas are written (talon_rl/tasks/locomotion/a1_env/'s
manager-based env calls compute_reward_vector directly rather than
reimplementing any of this in torch — see that module's docstring).

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
    """Action-rate and acceleration objective. (N, 12) each -> (N,).

    It is preference-conditioned in Phase 1, as in AMOR's smoothness
    objective, so the policy can negotiate tracking versus hardware wear.
    """
    action_rate = np.sum((action - prev_action) ** 2, axis=-1)
    acc = np.sum(joint_acc**2, axis=-1)
    return (-(action_rate + 0.01 * acc)).astype(np.float32)


def balance_reward(
    roll_pitch: np.ndarray, terminal_fall: np.ndarray | None = None, fall_penalty: float = 0.0,
    alive_bonus: float = 0.0,
) -> np.ndarray:
    """Penalizes trunk tilt directly -- a dense, per-step gradient against
    falling. Not in chapter3.tex's original table 3.3; added because none of
    the other 5 terms give any signal toward staying upright before it's
    already fallen (only `base_contact` termination, after the fact) --
    diagnosed after 4 independent training runs (action scale, terrain
    curriculum, Kp/Kd randomization, network width) each failed to move
    mean_episode_length off its ~12-14/200-step floor. RMA's own reward set
    has the equivalent term (#8, Orientation: -||theta_roll,pitch||^2) for
    exactly this reason; AMOR's root-orientation term is reference-tracking
    (needs mocap) and doesn't port to this reference-free task.

    `alive_bonus` (added 2026-09-18): a flat positive reward every step the
    lane hasn't fallen, matching AMOR's constant survival bonus c_alive and
    the classic Gym alive_bonus (Hopper/Walker2d). Found necessary because
    per-step reward (Episode_Reward/progress, /smoothness) kept improving
    across a 20000-update run while mean_episode_length stayed flat at its
    ~6-8 floor the whole time -- the policy was getting better at whatever
    it experienced without any of that requiring surviving longer, since
    every other term here is a per-step rate, not tied to episode duration.
    This term is the only one that pays out purely for elapsed time alive,
    independent of how well any other objective is being tracked. (N, 2) ->
    (N,)."""
    reward = -np.sum(roll_pitch**2, axis=-1) + alive_bonus
    if terminal_fall is not None:
        reward = reward - np.asarray(terminal_fall, dtype=np.float32) * fall_penalty
    return reward.astype(np.float32)


_TERM_FUNCS = {
    "progress": lambda t, cfg: progress_reward(t["v_actual"], t["v_command"], cfg.progress_std),
    "clearance": lambda t, cfg: clearance_reward(t["obstacle_dist"]),
    "energy": lambda t, cfg: energy_reward(t["joint_torque"], t["joint_vel"]),
    "impact": lambda t, cfg: impact_reward(t["foot_contact_force"]),
    "smoothness": lambda t, cfg: smoothness_reward(t["action"], t["prev_action"], t["joint_acc"]),
    "balance": lambda t, cfg: balance_reward(
        t["roll_pitch"], t.get("terminal_fall"), cfg.fall_penalty, cfg.alive_bonus
    ),
}


def compute_reward_vector(transition: dict, cfg: RewardVectorCfg) -> np.ndarray:
    """Returns r as an (N, dim) array in cfg.term_names order. Inactive terms are 0."""
    n = transition["obs"].shape[0]
    values = []
    for name, active in zip(cfg.term_names, cfg.active):
        values.append(_TERM_FUNCS[name](transition, cfg) if active else np.zeros(n, dtype=np.float32))
    return np.stack(values, axis=-1).astype(np.float32)
