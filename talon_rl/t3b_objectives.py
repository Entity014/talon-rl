"""Frozen T3-B four-objective reward semantics and normalization.

All inputs are already-weighted stock IsaacLab reward terms.
All objectives use higher-is-better convention. Penalty objectives are
negative-valued and improve toward zero.
"""
from __future__ import annotations
from collections.abc import Mapping
import numpy as np

OBJECTIVE_ORDER=(
    "velocity_tracking",
    "angular_stability",
    "orientation_stability",
    "control_smoothness",
)
OBJECTIVE_TERMS={
    "velocity_tracking":("track_lin_vel_xy_exp","track_ang_vel_z_exp"),
    "angular_stability":("ang_vel_xy_l2",),
    "orientation_stability":("flat_orientation_l2",),
    "control_smoothness":("action_rate_l2",),
}
# Frozen empirical absolute-mean divisors from T3-B baseline audit:
# 3 reset seeds x 16 envs x 192 steps.
NORMALIZATION_DIVISORS=np.asarray([
    1.7194554805755615,
    0.15590913593769073,
    0.01563369482755661,
    0.08311229199171066,
],dtype=np.float32)

# Non-MORL roles retained explicitly to prevent accidental re-entry.
CONSTRAINT_TERMS={"vertical_stability":("lin_vel_z_l2",)}
REGULARIZER_TERMS={"effort":("dof_torques_l2",)}
AUXILIARY_TERMS={"joint_smoothness":("dof_acc_l2",),"gait_contact":("feet_air_time",)}
EXCLUDED_ZERO_WEIGHT_TERMS=("dof_pos_limits",)

def raw_objective_vector(weighted_terms:Mapping[str,np.ndarray],*,shape=None)->np.ndarray:
    arrays=[np.asarray(v,dtype=np.float32) for v in weighted_terms.values()]
    target=shape or (arrays[0].shape if arrays else ())
    out=[]
    for names in OBJECTIVE_TERMS.values():
        acc=np.zeros(target,dtype=np.float32)
        for name in names:
            if name not in weighted_terms:
                raise KeyError(f"missing required reward term: {name}")
            value=np.asarray(weighted_terms[name],dtype=np.float32)
            if value.shape!=target:
                raise ValueError(f"term {name} shape {value.shape} != {target}")
            if not np.isfinite(value).all():
                raise ValueError(f"term {name} is non-finite")
            acc+=value
        out.append(acc)
    return np.stack(out,axis=-1)

def normalize_objectives(raw:np.ndarray)->np.ndarray:
    x=np.asarray(raw,dtype=np.float32)
    if x.shape[-1]!=len(OBJECTIVE_ORDER):
        raise ValueError("last dimension must be four frozen T3-B objectives")
    if not np.isfinite(x).all():
        raise ValueError("raw objective vector is non-finite")
    return x/NORMALIZATION_DIVISORS

def normalized_objective_vector(weighted_terms:Mapping[str,np.ndarray],*,shape=None)->np.ndarray:
    return normalize_objectives(raw_objective_vector(weighted_terms,shape=shape))

def scalarize(normalized:np.ndarray,w:np.ndarray)->np.ndarray:
    x=np.asarray(normalized,dtype=np.float32); ww=np.asarray(w,dtype=np.float32)
    if x.shape[-1]!=4 or ww.shape[-1]!=4:
        raise ValueError("normalized objectives and preference must end in dimension 4")
    if not np.isfinite(ww).all() or np.any(ww<0):
        raise ValueError("preference must be finite and non-negative")
    if not np.allclose(ww.sum(axis=-1),1.0,atol=1e-6):
        raise ValueError("preference must lie on the four-objective simplex")
    return np.sum(x*ww,axis=-1)
