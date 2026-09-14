"""Physics-free smoke-test env — batch-native, gym.vector.SyncVectorEnv-backed.

Still NOT a locomotion simulator (see the original 2026-09-11 docstring for
the "why" of the toy physics) — the only change here is vectorization.
DummyEnv(gym.Env) implements one lane's kinematics using gym's own
reset()/step() contract, and DummyTalonEnv wraps `num_envs` of them in
gym.vector.SyncVectorEnv — this reuses gym's own tested per-lane auto-reset
logic instead of hand-rolling it a second time (IsaacLabTalonEnv's
auto-reset already comes from Isaac Lab's manager machinery for free, so
both real envs now lean on a tested implementation rather than this repo's
own). gymnasium's automatic info-dict vectorization doesn't cleanly carry
array-valued fields (only scalar diagnostics), so DummyTalonEnv reads the
extra reward-computation fields directly off each sub-env's post-step
instance state (`self._vec_env.envs`) instead of through `info`.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ..config import ActionSpaceCfg, ObservationSpaceCfg
from .base_env import BaseTalonEnv


class DummyEnv(gym.Env):
    """Single lane. action[0] drives forward accel, action[1] softens
    landing (reduces impact, costs extra energy) — see 2026-09-11's
    DummyTalonEnv docstring for the full rationale, unchanged here."""

    def __init__(self, obs_cfg: ObservationSpaceCfg, action_cfg: ActionSpaceCfg, horizon: int = 200, dt: float = 0.02):
        super().__init__()
        self.obs_cfg = obs_cfg
        self.action_cfg = action_cfg
        self.horizon = horizon
        self.dt = dt
        self.obs_dim = obs_cfg.total_dim - obs_cfg.preference_dim
        self.action_dim = action_cfg.dim
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.action_dim,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.pos = 0.0
        self.vel = 0.0
        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self.v_command = np.array([self.np_random.uniform(0.3, 1.0), 0.0, 0.0], dtype=np.float32)
        self.obstacle_ahead = float(self.np_random.uniform(3.0, 6.0))
        obs = self._compute_transition(np.zeros(self.action_dim, dtype=np.float32))
        return obs, {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        accel = float(action[0]) * 2.0
        self.vel = float(np.clip(self.vel + accel * self.dt, 0.0, 2.0))
        self.pos += self.vel * self.dt
        self.t += 1

        obs = self._compute_transition(action)
        self.prev_action = action  # advance AFTER _compute_transition reads the old value

        terminated = bool(self.pos >= self.obstacle_ahead)
        truncated = bool(self.t >= self.horizon)
        return obs, 0.0, terminated, truncated, {}

    def _compute_transition(self, action: np.ndarray) -> np.ndarray:
        """Computes and stores this step's transition fields (read externally
        by DummyTalonEnv after step()/reset() return) using self.prev_action's
        CURRENT (pre-advance) value — must run before step() advances
        self.prev_action to `action`, or prev_action_out would equal this
        step's action instead of the previous one."""
        softness = float(np.clip(action[1], 0.0, 1.0)) if action.size > 1 else 0.0

        self.v_actual = np.array([self.vel, 0.0, 0.0], dtype=np.float32)
        self.v_command_out = self.v_command
        self.obstacle_dist = max(0.0, self.obstacle_ahead - self.pos)
        self.joint_vel = np.full(self.action_dim, self.vel, dtype=np.float32)
        self.joint_torque = action * (1.0 + softness)
        self.joint_acc = (action - self.prev_action) / self.dt
        self.foot_contact_force = np.full(4, self.vel * 40.0 * (1.0 - 0.5 * softness), dtype=np.float32)
        self.action = action
        self.prev_action_out = self.prev_action

        obs = np.concatenate([
            np.zeros(self.obs_cfg.joint_pos_dim, dtype=np.float32),
            np.full(self.obs_cfg.joint_vel_dim, self.vel, dtype=np.float32),
            np.zeros(self.obs_cfg.roll_pitch_dim, dtype=np.float32),
            np.ones(self.obs_cfg.foot_contact_dim, dtype=np.float32),
            self.prev_action[: self.obs_cfg.prev_action_dim],
            self.v_command,
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
    ):
        self.num_envs = num_envs
        self.obs_dim = obs_cfg.total_dim - obs_cfg.preference_dim
        self.action_dim = action_cfg.dim
        self._seed = seed
        self._vec_env = gym.vector.SyncVectorEnv(
            [(lambda: DummyEnv(obs_cfg, action_cfg, horizon=horizon, dt=dt)) for _ in range(num_envs)],
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
        for key in ("v_actual", "obstacle_dist", "joint_torque", "joint_vel",
                    "joint_acc", "foot_contact_force", "action"):
            transition[key] = np.stack([getattr(env.unwrapped, key) for env in self._vec_env.envs])
        transition["v_command"] = np.stack([env.unwrapped.v_command_out for env in self._vec_env.envs])
        transition["prev_action"] = np.stack([env.unwrapped.prev_action_out for env in self._vec_env.envs])
        return transition

    def close(self) -> None:
        self._vec_env.close()
