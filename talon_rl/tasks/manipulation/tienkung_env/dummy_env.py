# talon_rl/tasks/manipulation/tienkung_env/dummy_env.py
"""Physics-free smoke-test env for TienKung's bimanual box-carry —
same batch-native, gym.vector.SyncVectorEnv-backed structure as
scripts/moppo/dummy_env.py's DummyEnv/DummyTalonEnv (this thesis's own A1
dummy env), own toy physics: action[0] drives box-lift progress,
action[1] drives grip firmness (higher = firmer hold = more arm-contact
force / impact risk, lower = gentler but risks the box slipping) —
retargets that file's action[0]=accel/action[1]=softness trade-off
pattern from locomotion to bimanual carrying.

No termination condition beyond horizon this round (box-dropped
termination logic is deferred to the real Isaac Lab env, a later
increment) — every episode truncates at `horizon` steps, never
terminates early.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from talon_rl.envs.base_env import BaseTalonEnv

from .config import ActionSpaceCfg, ObservationSpaceCfg


class DummyEnv(gym.Env):
    """Single lane. action[0] drives box-lift height toward target_height;
    action[1] drives grip firmness (arm_contact_force magnitude)."""

    def __init__(
        self,
        obs_cfg: ObservationSpaceCfg,
        action_cfg: ActionSpaceCfg,
        horizon: int = 200,
        dt: float = 0.02,
        target_height: float = 1.0,
    ):
        super().__init__()
        self.obs_cfg = obs_cfg
        self.action_cfg = action_cfg
        self.horizon = horizon
        self.dt = dt
        self.target_height = target_height
        self.obs_dim = obs_cfg.total_dim - obs_cfg.preference_dim
        self.action_dim = action_cfg.dim
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.action_dim,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.height = 0.0
        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self._obstacle_dist = float(self.np_random.uniform(3.0, 6.0))
        obs = self._compute_transition(np.zeros(self.action_dim, dtype=np.float32))
        return obs, {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        lift_rate = max(0.0, float(action[0])) * 0.5
        self.height = float(np.clip(self.height + lift_rate * self.dt, 0.0, 2.0))
        self.t += 1

        obs = self._compute_transition(action)
        self.prev_action = action  # advance AFTER _compute_transition reads the old value

        terminated = False  # no drop condition this round — see module docstring
        truncated = bool(self.t >= self.horizon)
        return obs, 0.0, terminated, truncated, {}

    def _compute_transition(self, action: np.ndarray) -> np.ndarray:
        """Computes and stores this step's transition fields (read
        externally by DummyTalonEnv after step()/reset() return) using
        self.prev_action's CURRENT (pre-advance) value — must run before
        step() advances self.prev_action to `action`, matching the A1
        DummyEnv's own ordering rule."""
        firmness = float(np.clip(action[1], 0.0, 1.0)) if action.size > 1 else 0.0

        self.box_height = np.float32(self.height)
        self.target_height_out = np.float32(self.target_height)
        self.obstacle_dist = self._obstacle_dist
        self.joint_vel = np.full(self.action_dim, self.height, dtype=np.float32)
        self.joint_torque = action * (1.0 + firmness)
        self.joint_acc = (action - self.prev_action) / self.dt
        self.arm_contact_force = np.full(2, firmness * 60.0, dtype=np.float32)
        self.action = action
        self.prev_action_out = self.prev_action

        obs = np.concatenate([
            np.zeros(self.obs_cfg.joint_pos_dim, dtype=np.float32),
            np.full(self.obs_cfg.joint_vel_dim, self.height, dtype=np.float32),
            np.array([0.0, 0.0, self.height], dtype=np.float32)[: self.obs_cfg.box_relative_pos_dim],
            np.ones(self.obs_cfg.arm_contact_dim, dtype=np.float32),
            self.prev_action[: self.obs_cfg.prev_action_dim],
        ])
        return obs.astype(np.float32)


class DummyTalonEnv(BaseTalonEnv):
    def __init__(
        self,
        obs_cfg: ObservationSpaceCfg,
        action_cfg: ActionSpaceCfg,
        num_envs: int = 1,
        horizon: int = 200,
        dt: float = 0.02,
        seed: int = 0,
        target_height: float = 1.0,
    ):
        self.num_envs = num_envs
        self.obs_dim = obs_cfg.total_dim - obs_cfg.preference_dim
        self.action_dim = action_cfg.dim
        self._seed = seed
        self._vec_env = gym.vector.SyncVectorEnv(
            [
                (lambda: DummyEnv(obs_cfg, action_cfg, horizon=horizon, dt=dt, target_height=target_height))
                for _ in range(num_envs)
            ],
            autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
        )

    def reset(self) -> dict:
        obs, _infos = self._vec_env.reset(seed=[self._seed + i for i in range(self.num_envs)])
        return self._collect_transition(obs)

    def step(self, action: np.ndarray) -> tuple[dict, np.ndarray]:
        obs, _rewards, terminated, truncated, _infos = self._vec_env.step(action)
        done = np.logical_or(terminated, truncated)
        return self._collect_transition(obs), done

    def _collect_transition(self, obs: np.ndarray) -> dict:
        transition = {"obs": obs.astype(np.float32)}
        for key, attr in (
            ("box_height", "box_height"),
            ("target_height", "target_height_out"),
            ("obstacle_dist", "obstacle_dist"),
            ("joint_torque", "joint_torque"),
            ("joint_vel", "joint_vel"),
            ("joint_acc", "joint_acc"),
            ("arm_contact_force", "arm_contact_force"),
            ("action", "action"),
        ):
            transition[key] = np.stack([getattr(env.unwrapped, attr) for env in self._vec_env.envs])
        transition["prev_action"] = np.stack([env.unwrapped.prev_action_out for env in self._vec_env.envs])
        return transition

    def close(self) -> None:
        self._vec_env.close()
