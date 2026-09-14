"""Env interface contract — batch-native (num_envs, ...) shapes throughout.

Both DummyTalonEnv and IsaacLabTalonEnv (constructed via
gym.make("Isaac-Talon-A1-v0")) satisfy this directly, no wrapper.
reset()/step() return every transition value with a leading
(num_envs, ...) axis; step()'s `done` array marks which lanes just
auto-reset internally — that lane's "obs" row (and every other field) is
already the fresh post-reset value, not the terminal one, matching gym
VectorEnv / Isaac Lab ManagerBasedRLEnv auto-reset semantics.

A transition dict must carry every key talon_rl.reward._TERM_FUNCS expects
(v_actual, v_command, obstacle_dist, joint_torque, joint_vel,
foot_contact_force, action, prev_action, joint_acc), plus "obs" — see
reward.py for the exact shapes each key needs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseTalonEnv(ABC):
    num_envs: int
    obs_dim: int  # WITHOUT the preference vector — moppo.py appends w itself
    action_dim: int

    @abstractmethod
    def reset(self) -> dict:
        """Returns the first transition dict; every value has a leading (num_envs, ...) axis."""

    @abstractmethod
    def step(self, action: np.ndarray) -> tuple[dict, np.ndarray]:
        """action: (num_envs, action_dim). Returns (transition, done); done: (num_envs,) bool.
        Any lane with done[i] == True has already been auto-reset internally."""
