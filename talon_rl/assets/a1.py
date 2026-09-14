# talon_rl/assets/a1.py
"""Talon's Unitree A1 config — UNITREE_A1_CFG with the actuator gains
overridden to RMA's (Kumar et al. 2021) Kp=55/Kd=0.8, confirmed against the
real installed UNITREE_A1_CFG's actuator group key ("base_legs", not the
first guess of "legs") on 2026-09-13/14. See
docs/superpowers/specs/2026-09-13-isaac-lab-env-setup-design.md's Risks
section for why RMA's value was chosen over legged_gym's.
"""

from __future__ import annotations

from isaaclab_assets import UNITREE_A1_CFG

_A1_ACTUATOR_GROUP = "base_legs"
_A1_KP = 55.0
_A1_KD = 0.8

TALON_A1_CFG = UNITREE_A1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
TALON_A1_CFG.actuators[_A1_ACTUATOR_GROUP].stiffness = _A1_KP
TALON_A1_CFG.actuators[_A1_ACTUATOR_GROUP].damping = _A1_KD
