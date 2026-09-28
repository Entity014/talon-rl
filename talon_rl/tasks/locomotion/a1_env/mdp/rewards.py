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
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

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


class action_jerk_l2(ManagerTermBase):
    """Squared second difference of the policy action, ||a_t - 2a_{t-1} + a_{t-2}||^2.

    V4-C2S-R1 smoothness candidate S1: a ramp in the action costs nothing,
    a jerk does. The action manager keeps only a_t and a_{t-1}, so this term
    keeps a_{t-2} itself and zeroes it on reset, when the action manager
    zeroes its own action history.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._a2 = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)

    def reset(self, env_ids=None) -> None:
        self._a2[slice(None) if env_ids is None else env_ids] = 0.0

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        a, a1 = env.action_manager.action, env.action_manager.prev_action
        out = torch.sum(torch.square(a - 2.0 * a1 + self._a2), dim=1)
        self._a2 = a1.clone()
        return out


class body_height_osc_l2(ManagerTermBase):
    """Squared vertical oscillation of the base about its own moving mean,
    (z - EMA(z))^2, EMA time constant HEIGHT_EMA_TAU (0.5 s).

    F8 V3 candidate for Vertical Stability. Same definition as the
    measurement library (talon_rl.rewards.measurements): a steady height
    offset (posture) costs nothing, bouncing does. The mean restarts at the
    current height after an env reset.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._mean = torch.zeros(env.num_envs, device=env.device)
        self._fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None) -> None:
        self._fresh[slice(None) if env_ids is None else env_ids] = True

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        from talon_rl.rewards.measurements import ema_update
        z = env.scene["robot"].data.root_pos_w[:, 2]
        self._mean = torch.where(self._fresh, z, self._mean)
        self._fresh[:] = False
        self._mean = ema_update(self._mean, z, env.step_dt)
        return torch.square(z - self._mean)
