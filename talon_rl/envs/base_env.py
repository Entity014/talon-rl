"""Env interface contract.

Whoever writes the real Isaac Lab / Isaac Gym environment (the next milestone,
once there's a GPU box to run it on) implements this exact interface so
training/moppo.py doesn't change at all when DummyEnv is swapped out.

reset() and step() must return a `transition` dict with (at least) the keys
that talon_rl.reward._TERM_FUNCS expects: v_actual, v_command, obstacle_dist,
joint_torque, joint_vel, foot_contact_force, action, prev_action, joint_acc —
plus "obs" (the flat np.ndarray matching config.ObservationSpaceCfg.total_dim,
*excluding* the preference vector w, which training/moppo.py appends itself).
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
        """Returns the first transition dict of a fresh episode."""

    @abstractmethod
    def step(self, action: np.ndarray) -> tuple[dict, bool]:
        """Returns (transition, done)."""
