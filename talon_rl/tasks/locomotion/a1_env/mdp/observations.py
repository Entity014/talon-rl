# talon_rl/tasks/locomotion/a1_env/mdp/observations.py
"""Observation terms for IsaacLabTalonEnv. Reuses Isaac Lab's own
isaaclab.envs.mdp.joint_pos, .joint_vel, .last_action for those fields (see
mdp/__init__.py's re-export) — roll_pitch, foot_contact_binary, and
v_command have no Isaac Lab builtin matching ObservationSpaceCfg's exact
shape/meaning, so they're written here.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def roll_pitch(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """(roll, pitch) in radians from the root quaternion (w, x, y, z),
    standard aerospace convention — same formula as the single-env
    IsaacLabTalonEnv's _quat_to_roll_pitch helper (2026-09-13), ported to
    batched torch. (N, 2)."""
    asset: Articulation = env.scene[asset_cfg.name]
    quat = asset.data.root_quat_w
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = 2.0 * (w * y - z * x)
    pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))
    return torch.stack([roll, pitch], dim=-1)


def foot_contact_binary(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor"), threshold: float = 1.0
) -> torch.Tensor:
    """(N, 4) binary contact per foot, matching ObservationSpaceCfg.foot_contact_dim."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force_mag = torch.norm(sensor.data.net_forces_w, dim=-1)
    return (force_mag > threshold).float()


def v_command(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Scripted constant forward-velocity command — no real CommandManager
    in this prelim (matches the single-env IsaacLabTalonEnv's
    self.v_command, 2026-09-13). (N, 3)."""
    return env.v_command_buf
