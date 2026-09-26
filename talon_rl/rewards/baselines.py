"""Baseline and thesis-stage reward-vector reconstruction helpers."""
from __future__ import annotations

import numpy as np
from collections.abc import Mapping


OBJECTIVE_ORDER = ("progress", "efficiency", "contact", "balance", "limits")
OBJECTIVE_TERMS = {
    "progress": ("track_lin_vel_xy_exp", "track_ang_vel_z_exp"),
    "efficiency": ("lin_vel_z_l2", "ang_vel_xy_l2", "dof_torques_l2", "dof_acc_l2", "action_rate_l2"),
    "contact": ("feet_air_time", "undesired_contacts"),
    "balance": ("flat_orientation_l2",),
    "limits": ("dof_pos_limits",),
}

S1_OBJECTIVE_ORDER = OBJECTIVE_ORDER
S1_OBJECTIVE_TERMS = {
    "progress": ("track_lin_vel_xy_exp", "track_ang_vel_z_exp"),
    "efficiency": ("lin_vel_z_l2", "ang_vel_xy_l2", "dof_torques_l2", "dof_acc_l2"),
    "contact": ("feet_air_time", "undesired_contacts"),
    "balance": ("flat_orientation_l2",),
    "limits": ("action_rate_l2", "dof_pos_limits"),
}

S7_OBJECTIVE_ORDER = ("progress", "balance", "efficiency")
S7_OBJECTIVE_TERMS = {
    "progress": ("track_lin_vel_xy_exp", "track_ang_vel_z_exp"),
    "balance": ("lin_vel_z_l2", "ang_vel_xy_l2", "flat_orientation_l2", "feet_air_time"),
    "efficiency": ("dof_torques_l2", "dof_acc_l2", "action_rate_l2"),
}
S7_EXCLUDED_TERMS = ("dof_pos_limits", "undesired_contacts")


# ---- b0_reward.py ----
def b0_reward(vx, roll, pitch, height, z_nominal: float, terminal_fall):
    vx=np.asarray(vx); roll=np.asarray(roll); pitch=np.asarray(pitch); height=np.asarray(height); fall=np.asarray(terminal_fall, dtype=bool)
    tracking=np.exp(-((vx-.5)/.25)**2)
    posture=-.25*(roll**2+pitch**2)
    height_term=-2.0*(height-z_nominal)**2
    return tracking+posture+height_term-10.0*fall.astype(np.float32)

# ---- m0_reward_vector.py ----
def group_stock_terms(weighted_terms: Mapping[str, np.ndarray], *, shape=None) -> np.ndarray:
    """Group already-weighted stock term outputs; missing disabled terms are zero."""
    arrays = [np.asarray(v, dtype=np.float32) for v in weighted_terms.values()]
    target = shape or (arrays[0].shape if arrays else ())
    out = []
    for names in OBJECTIVE_TERMS.values():
        acc = np.zeros(target, dtype=np.float32)
        for name in names:
            if name in weighted_terms:
                value = np.asarray(weighted_terms[name], dtype=np.float32)
                if value.shape != target:
                    raise ValueError(f"term {name} shape {value.shape} != {target}")
                if not np.isfinite(value).all():
                    raise ValueError(f"term {name} is non-finite")
                acc = acc + value
        out.append(acc)
    return np.stack(out, axis=-1)

def reconstruct_stock_scalar(reward_vector: np.ndarray, reference_weights=(1., 1., 1., 1., 1.)) -> np.ndarray:
    vector = np.asarray(reward_vector, dtype=np.float32)
    weights = np.asarray(reference_weights, dtype=np.float32)
    if vector.shape[-1] != len(OBJECTIVE_ORDER) or weights.shape != (len(OBJECTIVE_ORDER),):
        raise ValueError("reward vector/weight shape mismatch")
    result = np.sum(vector * weights, axis=-1)
    if not np.isfinite(result).all():
        raise ValueError("reconstructed stock reward is non-finite")
    return result

# ---- v1b_s1_reward_vector.py ----
def group_v1b_s1_terms(weighted_terms: Mapping[str, np.ndarray], *, shape=None) -> np.ndarray:
    """Group existing weighted stock terms without changing their sum."""
    arrays = [np.asarray(value, dtype=np.float32) for value in weighted_terms.values()]
    target = shape or (arrays[0].shape if arrays else ())
    groups = []
    for names in S1_OBJECTIVE_TERMS.values():
        accumulator = np.zeros(target, dtype=np.float32)
        for name in names:
            if name not in weighted_terms:
                continue
            value = np.asarray(weighted_terms[name], dtype=np.float32)
            if value.shape != target:
                raise ValueError(f"term {name} shape {value.shape} != {target}")
            if not np.isfinite(value).all():
                raise ValueError(f"term {name} is non-finite")
            accumulator = accumulator + value
        groups.append(accumulator)
    return np.stack(groups, axis=-1)


def reconstruct_v1b_s1_scalar(reward_vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(reward_vector, dtype=np.float32)
    if vector.shape[-1] != len(S1_OBJECTIVE_ORDER):
        raise ValueError("S1 reward vector must have five objectives")
    result = np.sum(vector, axis=-1)
    if not np.isfinite(result).all():
        raise ValueError("reconstructed S1 reward is non-finite")
    return result

# ---- v1b_s7_reward_vector.py ----
def group_v1b_s7_terms(weighted_terms: Mapping[str, np.ndarray], *, shape=None) -> np.ndarray:
    arrays = [np.asarray(value, dtype=np.float32) for value in weighted_terms.values()]
    target = shape or (arrays[0].shape if arrays else ())
    groups = []
    assigned = set()
    for names in S7_OBJECTIVE_TERMS.values():
        accumulator = np.zeros(target, dtype=np.float32)
        for name in names:
            if name not in weighted_terms:
                continue
            value = np.asarray(weighted_terms[name], dtype=np.float32)
            if value.shape != target:
                raise ValueError(f"term {name} shape {value.shape} != {target}")
            if not np.isfinite(value).all():
                raise ValueError(f"term {name} is non-finite")
            accumulator = accumulator + value
            assigned.add(name)
        groups.append(accumulator)
    unknown = set(weighted_terms) - assigned - set(S7_EXCLUDED_TERMS)
    if unknown:
        raise ValueError(f"unmapped non-safety reward terms: {sorted(unknown)}")
    return np.stack(groups, axis=-1)


def reconstruct_v1b_s7_scalar(reward_vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(reward_vector, dtype=np.float32)
    if vector.shape[-1] != len(S7_OBJECTIVE_ORDER):
        raise ValueError("S7 reward vector must have three objectives")
    result = np.sum(vector, axis=-1)
    if not np.isfinite(result).all():
        raise ValueError("reconstructed S7 scalar is non-finite")
    return result
