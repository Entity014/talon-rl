"""Actor-critic network shape — reusable across on-policy algorithms, not
specific to MOPPO's preference-conditioned loss. See
scripts/rl/core/algorithms/moppo.py for how MOPPO builds and updates it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    def __init__(self, actor_obs_dim: int, critic_obs_dim: int, action_dim: int, reward_dim: int, hidden_dim: int):
        super().__init__()
        self.actor_body = nn.Sequential(
            nn.Linear(actor_obs_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
        )
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        self.critic_body = nn.Sequential(
            nn.Linear(critic_obs_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
        )
        self.critic_head = nn.Linear(hidden_dim, reward_dim)

    def act(self, actor_obs_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.actor_mean(self.actor_body(actor_obs_w))
        dist = Normal(mean, self.log_std.exp())
        action = dist.sample()
        logp = dist.log_prob(action).sum(-1)
        return action, logp

    def act_inference(self, actor_obs_w: torch.Tensor) -> torch.Tensor:
        """Deterministic action (the Normal's mean, no sampling) — for
        play/deployment, not training. Same input always gives the same
        output, unlike act(). Also what gets exported for sim2sim/hardware:
        see scripts/rl/core/wrapper/exporter.py."""
        return self.actor_mean(self.actor_body(actor_obs_w))

    def logp(self, actor_obs_w: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        mean = self.actor_mean(self.actor_body(actor_obs_w))
        dist = Normal(mean, self.log_std.exp())
        return dist.log_prob(action).sum(-1)

    def value(self, critic_obs_w: torch.Tensor) -> torch.Tensor:
        return self.critic_head(self.critic_body(critic_obs_w))
