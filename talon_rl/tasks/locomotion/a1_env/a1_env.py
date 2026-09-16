# talon_rl/tasks/locomotion/a1_env/a1_env.py
"""IsaacLabTalonEnv(ManagerBasedRLEnv) — the class registered as
Isaac-Talon-A1-v0. Scene/observations/actions/terminations/events all come
from IsaacLabTalonEnvCfg's manager configs (a1_env_cfg.py) — this class
only adds: (1) the v_command/obstacle_ahead scripted buffers
mdp/observations.py and mdp/terminations.py read, and (2) step()/reset()
overrides that return this repo's BaseTalonEnv-shaped
(transition_dict, done_array) instead of gym.Env's raw tuple, since nothing
outside this repo needs generic gym.Env compliance from this class
(train_prelim.py drives it directly via BaseTalonEnv) — see
docs/superpowers/specs/2026-09-14-vectorized-isaac-lab-env-design.md.
"""

from __future__ import annotations

import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnv

from talon_rl.envs.base_env import BaseTalonEnv

from .a1_env_cfg import IsaacLabTalonEnvCfg


class IsaacLabTalonEnv(ManagerBasedRLEnv, BaseTalonEnv):
    cfg: IsaacLabTalonEnvCfg

    def __init__(self, cfg: IsaacLabTalonEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)
        # NOTE deviation from the brief's literal code (verified against the
        # installed isaaclab 0.48.0 source, not guessed): ManagerBasedEnv.num_envs
        # is a read-only @property (`return self.scene.num_envs`, no setter) --
        # `self.num_envs = self.cfg.scene.num_envs` here raised
        # AttributeError: can't set attribute. The inherited property already
        # returns the same value, so the assignment is simply dropped.
        # v_command_buf/obstacle_ahead_buf construction moved to the
        # load_managers() override below -- see its docstring for why.
        self.obs_dim = self.cfg.obs_dim
        self.action_dim = self.cfg.action_dim

    def load_managers(self) -> None:
        """Overridden (deviation from the brief's literal code, verified against
        isaaclab.envs.manager_based_env.ManagerBasedEnv.__init__/.load_managers,
        not guessed): ManagerBasedEnv.__init__ builds self.scene, THEN calls
        self.load_managers() -- which synchronously constructs ObservationManager,
        invoking every term function once (incl. mdp.v_command /
        mdp.obstacle_reached) to infer shapes -- all before control ever returns
        to IsaacLabTalonEnv.__init__'s own body. So v_command_buf/obstacle_ahead_buf
        must exist by the time load_managers() runs, not after super().__init__()
        returns. self.scene already exists at this point (scene creation happens
        before ManagerBasedEnv.__init__ calls self.load_managers()), so
        self.device/self.num_envs (both properties reading self.scene) are valid
        here.
        """
        self.v_command_buf = torch.tensor([0.5, 0.0, 0.0], device=self.device).expand(self.num_envs, 3).contiguous()
        self.obstacle_ahead_buf = torch.full((self.num_envs,), 5.0, device=self.device)
        super().load_managers()

    def reset(self, **kwargs) -> dict:
        obs_dict, _extras = super().reset(**kwargs)
        return self._transition(obs_dict)

    def step(self, action: np.ndarray) -> tuple[dict, np.ndarray]:
        action_t = torch.from_numpy(action).to(self.device)
        obs_dict, _reward_buf, terminated, truncated, _extras = super().step(action_t)
        transition = self._transition(obs_dict)
        done = (terminated | truncated).cpu().numpy()
        return transition, done

    def _transition(self, obs_dict: dict) -> dict:
        robot = self.scene["robot"]
        prev_action = self.action_manager.prev_action.cpu().numpy()
        action = self.action_manager.action.cpu().numpy()
        transition = {
            "obs": obs_dict["policy"].cpu().numpy().astype(np.float32),
            "extrinsics": obs_dict["privileged"].cpu().numpy().astype(np.float32),
            # v_command is (v_x, v_y, omega_z) (config.py's command_dim comment) —
            # root_lin_vel_b alone is (v_x, v_y, v_z), so its 3rd column was being
            # compared against a yaw-rate target instead of the robot's actual yaw
            # rate. Swap in root_ang_vel_b's z-component so progress_reward's
            # exp-kernel tracks what v_command actually specifies.
            "v_actual": torch.cat(
                [robot.data.root_lin_vel_b[:, :2], robot.data.root_ang_vel_b[:, 2:3]], dim=-1
            ).cpu().numpy(),
            "v_command": self.v_command_buf.cpu().numpy(),
            "obstacle_dist": np.maximum(
                0.0, (self.obstacle_ahead_buf - robot.data.root_pos_w[:, 0]).cpu().numpy()
            ),
            "joint_torque": robot.data.applied_torque.cpu().numpy(),
            "joint_vel": robot.data.joint_vel.cpu().numpy(),
            "joint_acc": (action - prev_action) / self.step_dt,
            "foot_contact_force": torch.norm(
                self.scene.sensors["contact_sensor"].data.net_forces_w, dim=-1
            ).cpu().numpy(),
            "action": action,
            "prev_action": prev_action,
        }
        return transition
