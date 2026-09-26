"""V2-R1 single-variable FiLM-authority redesign.

Contract: identical V2 parameterization/state schema, latent, anchors, critic,
and insertion point. The only mechanism change is FiLM authority alpha:
0.1 -> 0.5. Identity initialization remains exact because gamma/beta heads
are zero-initialized by the V2 base class.
"""
from __future__ import annotations

import torch
from torch import Tensor

from talon_rl.models.behavior.latent import V2BehaviorActorCritic

FILM_AUTHORITY_R1 = 0.5


class V2R1BehaviorActorCritic(V2BehaviorActorCritic):
    """V2 with fixed higher-authority FiLM envelope and no new parameters."""

    FILM_AUTHORITY = FILM_AUTHORITY_R1

    def actor_features(self, obs: Tensor, w: Tensor) -> Tensor:
        self._validate_w(obs, w)
        z = self.behavior_z(w)
        h = torch.nn.functional.elu(self.actor_pre(obs))
        alpha = self.FILM_AUTHORITY
        h = (1.0 + alpha * torch.tanh(self.film_gamma(z))) * h + alpha * torch.tanh(self.film_beta(z))
        return self.actor_rest(h)
