# talon_rl/tasks/locomotion/a1_env/mdp/rewards.py
"""Stock Isaac Lab velocity-tracking kernels, reading this task's
v_command_buf instead of a CommandManager (the task has none).

The bodies are copied from isaaclab.envs.mdp.rewards so the V3 T/A/O/S
objective semantics (talon_rl/rewards/objectives.py) carry over to the
canonical V4 env unchanged; only the command source differs. The A/O/S
terms need no port and use the stock functions directly.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def track_lin_vel_xy_exp(env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_vel_error = torch.sum(torch.square(env.v_command_buf[:, :2] - asset.data.root_lin_vel_b[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_exp(env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.v_command_buf[:, 2] - asset.data.root_ang_vel_b[:, 2])
    return torch.exp(-ang_vel_error / std**2)
