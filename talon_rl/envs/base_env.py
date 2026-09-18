"""Env interface contract — batch-native (num_envs, ...) shapes throughout.

Both DummyTalonEnv and IsaacLabTalonEnv (constructed via
gym.make("Isaac-Talon-A1-v0")) satisfy this directly, no wrapper.
reset()/step() return every transition value with a leading
(num_envs, ...) axis; step()'s `done` array marks which lanes just
auto-reset internally. Its "obs" row is therefore the fresh post-reset
observation, ready for the next action. When a transition ends an episode,
the optional `reward_transition` entry holds the same reward inputs from the
terminal frame; it is used only to assign the action's reward correctly.

A transition dict must carry every key the task's own reward.py's
_TERM_FUNCS expects (e.g. talon_rl.reward for the A1,
talon_rl.tasks.manipulation.tienkung_env.reward for TienKung). For the A1's
own reward.py that's (v_actual, v_command, joint_torque, joint_vel,
foot_contact_force, action, prev_action, joint_acc, roll_pitch), plus "obs" — see
reward.py for the exact shapes each key needs.

`reward_transition` has all normal transition fields and the same batch
shape. For non-terminal lanes it equals the normal transition; for a
terminal lane, only reward inputs are replaced with their pre-reset values.
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
