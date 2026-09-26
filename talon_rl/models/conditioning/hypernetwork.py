"""V2-H: V2-B plus a preference-conditioned low-rank policy-parameter manifold.

The hypernetwork does not route among objective-labelled experts.  Instead,
preference w generates coefficients over latent low-rank weight bases for the
actor action head:

    W(w) = W0 + sum_k c_k(w) U_k V_k

The coefficient network output is zero-initialized.  Therefore V2-H is exactly
function-preserving with respect to a frozen V2-B checkpoint at H0.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from talon_rl.models.conditioning.single_site_film import V2BSingleSiteFiLMActorCritic
from talon_rl.models.foundations.preference_embedding import NUM_OBJECTIVES


class V2HPreferenceHyperActorCritic(V2BSingleSiteFiLMActorCritic):
    NUM_BASES = 4
    BASIS_RANK = 4
    COEFF_HIDDEN = 16

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: list[int] | None = None,
    ):
        hidden_dims = hidden_dims or [128, 128, 128]
        super().__init__(obs_dim, action_dim, hidden_dims)

        feature_dim = self.actor_mean.in_features
        self.HYPER_FEATURE_DIM = feature_dim
        self.HYPER_ACTION_DIM = action_dim

        # Preference -> latent policy-manifold coordinates.
        self.preference_hyper = nn.Sequential(
            nn.Linear(NUM_OBJECTIVES, self.COEFF_HIDDEN),
            nn.ELU(),
            nn.Linear(self.COEFF_HIDDEN, self.NUM_BASES),
        )

        # Latent low-rank weight bases.  These carry no semantic labels.
        self.hyper_u = nn.Parameter(
            torch.empty(self.NUM_BASES, action_dim, self.BASIS_RANK)
        )
        self.hyper_v = nn.Parameter(
            torch.empty(self.NUM_BASES, self.BASIS_RANK, feature_dim)
        )

        nn.init.orthogonal_(self.hyper_u.reshape(self.NUM_BASES * action_dim, self.BASIS_RANK))
        nn.init.orthogonal_(self.hyper_v.reshape(self.NUM_BASES * self.BASIS_RANK, feature_dim))

        # Keep basis magnitude moderate; the zero coefficient output gives exact
        # function preservation while allowing immediate coefficient gradients.
        with torch.no_grad():
            self.hyper_u.mul_(0.10)
            self.hyper_v.mul_(0.10)
            final = self.preference_hyper[-1]
            final.weight.zero_()
            final.bias.zero_()

    def hyper_coefficients(self, w: Tensor) -> Tensor:
        return self.preference_hyper(w)

    def generated_weight_delta(self, w: Tensor) -> Tensor:
        """Return [B, action_dim, feature_dim] low-rank weight delta."""
        coeff = self.hyper_coefficients(w)
        basis = torch.einsum("kar,krf->kaf", self.hyper_u, self.hyper_v)
        return torch.einsum("bk,kaf->baf", coeff, basis)

    def _conditional_actor_mean(self, obs: Tensor, w: Tensor) -> Tensor:
        features = self._actor_features_v2a(obs, w)
        base = self.actor_mean(features)
        delta_w = self.generated_weight_delta(w)
        delta = torch.einsum("baf,bf->ba", delta_w, features)
        return base + delta

    def _pre_tanh_dist_with_preference(self, obs: Tensor, w: Tensor):
        mean = self._conditional_actor_mean(obs, w)
        std = self.log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def act_inference_with_preference(self, obs: Tensor, w: Tensor):
        mean = self._conditional_actor_mean(obs, w)
        return torch.tanh(mean) * self.ACTION_CLIP

    def act_with_preference_latent(self, obs: Tensor, w: Tensor):
        dist = self._pre_tanh_dist_with_preference(obs, w)
        u = dist.sample()
        action = torch.tanh(u) * self.ACTION_CLIP
        logp = (dist.log_prob(u) - self._log_det_jacobian(u)).sum(-1)
        return action, logp, u

    def logp_from_pre_tanh_with_preference(self, obs: Tensor, w: Tensor, u: Tensor):
        dist = self._pre_tanh_dist_with_preference(obs, w)
        return (dist.log_prob(u) - self._log_det_jacobian(u)).sum(-1)


def initialize_from_v2b(
    model: V2HPreferenceHyperActorCritic,
    v2b_state: dict,
) -> None:
    """Copy V2-B exactly and zero only the generated manifold coordinates."""
    target = model.state_dict()
    source = v2b_state.get("model", v2b_state.get("model_state_dict", v2b_state))

    for key in list(target):
        if key in source and target[key].shape == source[key].shape:
            target[key].copy_(source[key])

    target["preference_hyper.2.weight"].zero_()
    target["preference_hyper.2.bias"].zero_()

    model.load_state_dict(target)
