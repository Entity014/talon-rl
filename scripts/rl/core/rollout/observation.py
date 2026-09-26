"""num_policy_stacks / num_critic_stacks pattern, after Flamingo
(jaykorea/Isaac-RL-Two-wheel-Legged-Bot) — batched (N, stacks, obs_dim).

Keeps separate temporal-history buffers for the actor and the critic, since
the two don't need the same context length: RMA's Adaptation Module needs
~0.5s of proprioceptive history \\cite{kumar2021} to regress $\\hat z_t$, but
a vector critic evaluating $V(s,c,w)$ doesn't have that requirement and can
run on fewer (or just the current) frame. Splitting this out as its own
wrapper — rather than baking stacking into the env — means `TalonEnv`
implementations stay single-timestep and simple; only `training/moppo.py`
needs to know stacking exists.

`push(obs, done_mask)` zero-fills any lane whose `done_mask[i]` is True
before pushing its new frame — a fresh episode shouldn't see the previous
episode's stale history (mirrors `reset()`'s zero-warm-up, but per-lane).
"""

from __future__ import annotations

import numpy as np


class ObservationStack:
    def __init__(self, num_envs: int, obs_dim: int, num_policy_stacks: int = 1, num_critic_stacks: int = 1):
        if num_policy_stacks < 1 or num_critic_stacks < 1:
            raise ValueError("stack counts must be >= 1 (1 = no stacking, current frame only)")
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.num_policy_stacks = num_policy_stacks
        self.num_critic_stacks = num_critic_stacks
        self._policy_buf = np.zeros((num_envs, num_policy_stacks, obs_dim), dtype=np.float32)
        self._critic_buf = np.zeros((num_envs, num_critic_stacks, obs_dim), dtype=np.float32)

    @property
    def policy_obs_dim(self) -> int:
        return self.obs_dim * self.num_policy_stacks

    @property
    def critic_obs_dim(self) -> int:
        return self.obs_dim * self.num_critic_stacks

    def reset(self, obs: np.ndarray) -> None:
        """obs: (num_envs, obs_dim) — resets every lane's history to zero,
        then pushes the first real observation into all of them."""
        self._policy_buf[:] = 0.0
        self._critic_buf[:] = 0.0
        self.push(obs, done_mask=np.ones(self.num_envs, dtype=bool))

    def push(self, obs: np.ndarray, done_mask: np.ndarray) -> None:
        """obs: (num_envs, obs_dim), done_mask: (num_envs,) bool."""
        self._policy_buf[done_mask] = 0.0
        self._critic_buf[done_mask] = 0.0
        self._policy_buf = np.roll(self._policy_buf, shift=-1, axis=1)
        self._critic_buf = np.roll(self._critic_buf, shift=-1, axis=1)
        self._policy_buf[:, -1, :] = obs
        self._critic_buf[:, -1, :] = obs

    @property
    def policy_obs(self) -> np.ndarray:
        return self._policy_buf.reshape(self.num_envs, -1)

    @property
    def critic_obs(self) -> np.ndarray:
        return self._critic_buf.reshape(self.num_envs, -1)
