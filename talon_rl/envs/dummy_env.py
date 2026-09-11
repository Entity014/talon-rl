"""Physics-free smoke-test env.

This is NOT a locomotion simulator — it's a 1D toy standing in for "a robot that
moves forward, can brake softly to reduce impact at a torque/energy cost, and
approaches a scripted obstacle." Its only job is to let training/moppo.py run
end-to-end (obs -> policy(w) -> action -> reward vector -> vector critic -> PPO
update) with numbers that respond sensibly to the preference vector, before
anyone spends GPU time wiring up the real Isaac Lab environment.

Concretely: action[0] drives forward acceleration, action[1] is a "soften
landing" effort (reduces the impact term, costs extra energy) — the rest of
the 12 action dims are along for the ride so the obs/action *shapes* match
config.ActionSpaceCfg exactly, and get penalized by the smoothness term like
real actuator noise would be. Do not read any locomotion conclusions from
this — it exists to prove the RL loop is wired correctly, nothing else.
"""

from __future__ import annotations

import numpy as np

from ..config import ActionSpaceCfg, ObservationSpaceCfg
from .base_env import BaseTalonEnv


class DummyTalonEnv(BaseTalonEnv):
    def __init__(self, obs_cfg: ObservationSpaceCfg, action_cfg: ActionSpaceCfg, horizon: int = 200, dt: float = 0.02, seed: int = 0):
        self.num_envs = 1
        self.obs_cfg = obs_cfg
        self.action_cfg = action_cfg
        # obs_dim excludes the preference vector — moppo.py appends w itself (see base_env.py)
        self.obs_dim = obs_cfg.total_dim - obs_cfg.preference_dim
        self.action_dim = action_cfg.dim
        self.horizon = horizon
        self.dt = dt
        self._rng = np.random.default_rng(seed)
        self._reset_state()

    def _reset_state(self) -> None:
        self.t = 0
        self.pos = 0.0
        self.vel = 0.0
        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self.v_command = np.array([self._rng.uniform(0.3, 1.0), 0.0, 0.0], dtype=np.float32)
        self.obstacle_ahead = self._rng.uniform(3.0, 6.0)

    def reset(self) -> dict:
        self._reset_state()
        return self._build_transition(action=np.zeros(self.action_dim, dtype=np.float32))

    def step(self, action: np.ndarray) -> tuple[dict, bool]:
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        accel = float(action[0]) * 2.0  # forward accel driven by action[0]
        self.vel = float(np.clip(self.vel + accel * self.dt, 0.0, 2.0))
        self.pos += self.vel * self.dt
        self.t += 1

        transition = self._build_transition(action)
        self.prev_action = action

        done = self.t >= self.horizon or self.pos >= self.obstacle_ahead
        return transition, done

    def _build_transition(self, action: np.ndarray) -> dict:
        softness = float(np.clip(action[1], 0.0, 1.0)) if action.size > 1 else 0.0

        obs = np.concatenate(
            [
                np.zeros(self.obs_cfg.joint_pos_dim, dtype=np.float32),
                np.full(self.obs_cfg.joint_vel_dim, self.vel, dtype=np.float32),
                np.zeros(self.obs_cfg.roll_pitch_dim, dtype=np.float32),
                np.ones(self.obs_cfg.foot_contact_dim, dtype=np.float32),
                self.prev_action[: self.obs_cfg.prev_action_dim],
                self.v_command,
            ]
        )

        joint_vel = np.full(self.action_dim, self.vel, dtype=np.float32)
        joint_torque = action * (1.0 + softness)  # softening costs extra torque/energy
        joint_acc = (action - self.prev_action) / self.dt
        foot_contact_force = np.full(4, self.vel * 40.0 * (1.0 - 0.5 * softness), dtype=np.float32)

        return {
            "obs": obs,
            "v_actual": np.array([self.vel, 0.0, 0.0], dtype=np.float32),
            "v_command": self.v_command,
            "obstacle_dist": max(0.0, self.obstacle_ahead - self.pos),
            "joint_torque": joint_torque,
            "joint_vel": joint_vel,
            "joint_acc": joint_acc,
            "foot_contact_force": foot_contact_force,
            "action": action,
            "prev_action": self.prev_action,
        }
