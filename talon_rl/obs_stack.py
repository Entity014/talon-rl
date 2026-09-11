"""num_policy_stacks / num_critic_stacks pattern, after Flamingo
(jaykorea/Isaac-RL-Two-wheel-Legged-Bot).

Keeps separate temporal-history buffers for the actor and the critic, since
the two don't need the same context length: RMA's Adaptation Module needs
~0.5s of proprioceptive history \\cite{kumar2021} to regress $\\hat z_t$, but
a vector critic evaluating $V(s,c,w)$ doesn't have that requirement and can
run on fewer (or just the current) frame. Splitting this out as its own
wrapper — rather than baking stacking into the env — means `BaseTalonEnv`
implementations (including the eventual Isaac Lab one) stay single-timestep
and simple; only `training/moppo.py` needs to know stacking exists.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class ObservationStack:
    def __init__(self, obs_dim: int, num_policy_stacks: int = 1, num_critic_stacks: int = 1):
        if num_policy_stacks < 1 or num_critic_stacks < 1:
            raise ValueError("stack counts must be >= 1 (1 = no stacking, current frame only)")
        self.obs_dim = obs_dim
        self.num_policy_stacks = num_policy_stacks
        self.num_critic_stacks = num_critic_stacks
        self._policy_buf: deque[np.ndarray] = deque(maxlen=num_policy_stacks)
        self._critic_buf: deque[np.ndarray] = deque(maxlen=num_critic_stacks)

    @property
    def policy_obs_dim(self) -> int:
        return self.obs_dim * self.num_policy_stacks

    @property
    def critic_obs_dim(self) -> int:
        return self.obs_dim * self.num_critic_stacks

    def reset(self, obs: np.ndarray) -> None:
        """Fills both buffers with zeros, then pushes the first real observation
        (so a fresh episode doesn't need `num_stacks - 1` warm-up steps of noise)."""
        zero = np.zeros_like(obs)
        self._policy_buf.clear()
        self._critic_buf.clear()
        for _ in range(self.num_policy_stacks - 1):
            self._policy_buf.append(zero)
        for _ in range(self.num_critic_stacks - 1):
            self._critic_buf.append(zero)
        self.push(obs)

    def push(self, obs: np.ndarray) -> None:
        self._policy_buf.append(obs)
        self._critic_buf.append(obs)

    @property
    def policy_obs(self) -> np.ndarray:
        return np.concatenate(list(self._policy_buf)).astype(np.float32)

    @property
    def critic_obs(self) -> np.ndarray:
        return np.concatenate(list(self._critic_buf)).astype(np.float32)
