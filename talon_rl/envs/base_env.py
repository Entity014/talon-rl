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

Known limitation — terminal-step reward/action mispairing under auto-reset:
because done[i]==True means lane i's fields are ALREADY the fresh
post-reset values (not the terminal frame that actually caused
termination), the reward vector computed from that step's transition dict
is computed from the NEXT episode's first frame, not from the action that
was actually taken to cause the termination — e.g. "action"/"prev_action"
come back as zeros and "obstacle_dist" is the new episode's fresh
randomized draw, instead of whatever the terminal frame actually was.
Concretely: reset()'s fields, not the terminal frame's, get used for that
step's reward. This happens consistently in both DummyTalonEnv and
IsaacLabTalonEnv (so `--env dummy` -> `--env isaac_lab` stays a true
drop-in swap), and is a known, deliberately-out-of-scope-for-this-prelim
limitation — roughly 1-in-`horizon` steps gets slightly wrong credit
assignment. Not fixed here; see README's Known gaps.
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
