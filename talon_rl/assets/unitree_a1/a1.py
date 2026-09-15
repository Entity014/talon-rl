# talon_rl/assets/unitree_a1/a1.py
"""Talon's Unitree A1 config — UNITREE_A1_CFG with the actuator gains
overridden to RMA's (Kumar et al. 2021) Kp=55/Kd=0.8, confirmed against the
real installed UNITREE_A1_CFG's actuator group key ("base_legs", not the
first guess of "legs") on 2026-09-13/14. See
docs/superpowers/specs/2026-09-13-isaac-lab-env-setup-design.md's Risks
section for why RMA's value was chosen over legged_gym's.

`usd_path` is overridden to a locally vendored copy
(data/Robots/unitree_a1/a1.usd, ~44MB, fetched from Nucleus 2026-09-15) instead
of UNITREE_A1_CFG's stock Nucleus-hosted path — the robot itself no longer
needs live Nucleus/CDN resolution at construction time. Note this does NOT
remove IsaacLabTalonEnv's Nucleus dependency entirely: the ground plane
(`sim_utils.GroundPlaneCfg()` in a1_env_cfg.py) still resolves its default
grid texture from Nucleus, so the get_assets_root_path() + carb
asset-root-cloud-setting workaround (2026-09-14) stays necessary regardless.
Mirrors jaykorea/Isaac-RL-Two-wheel-Legged-Bot's assets/<robot>/ convention —
vendoring USD/mesh/texture files under a shared TALON_ASSETS_DATA_DIR rather
than depending on a remote asset server, though our case is a stock asset
vendored for reliability, not a custom robot with no other source.
"""

from __future__ import annotations

from isaaclab_assets import UNITREE_A1_CFG

from . import TALON_ASSETS_DATA_DIR

_A1_ACTUATOR_GROUP = "base_legs"
_A1_KP = 55.0
_A1_KD = 0.8

TALON_A1_CFG = UNITREE_A1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
TALON_A1_CFG.spawn.usd_path = f"{TALON_ASSETS_DATA_DIR}/Robots/unitree_a1/a1.usd"
TALON_A1_CFG.actuators[_A1_ACTUATOR_GROUP].stiffness = _A1_KP
TALON_A1_CFG.actuators[_A1_ACTUATOR_GROUP].damping = _A1_KD
