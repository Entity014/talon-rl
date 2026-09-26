"""Function-preserving FiLM-conditioned actor for Post-V1 D3."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from rl.core.modules.actor_critic import ActorCritic
from talon_rl.models.foundations.three_objective import NUM_OBJECTIVES


class D3FiLMActorCritic(ActorCritic):
    """V1-C critic with a single FiLM modulation in the actor."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list[int] | None = None):
        hidden_dims = hidden_dims or [128, 128, 128]
        super().__init__(obs_dim + NUM_OBJECTIVES, obs_dim + NUM_OBJECTIVES, action_dim, NUM_OBJECTIVES, hidden_dims)
        self.physical_obs_dim = obs_dim
        # Retain the V1-C first-layer mapping exactly; FiLM is the new
        # feature-wise path layered on top of that function-preserving route.
        self.actor_pre = nn.Linear(obs_dim + NUM_OBJECTIVES, hidden_dims[0])
        self.film_gamma = nn.Linear(NUM_OBJECTIVES, hidden_dims[0])
        self.film_beta = nn.Linear(NUM_OBJECTIVES, hidden_dims[0])
        self.actor_rest = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ELU(),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ELU(),
        )
        self.actor_body = nn.Identity()

    @staticmethod
    def _with_w(obs: Tensor, w: Tensor) -> Tensor:
        if obs.ndim != 2 or w.ndim != 2 or obs.shape[0] != w.shape[0] or w.shape[1] != NUM_OBJECTIVES:
            raise ValueError("obs and w must be [batch,dim] and [batch,3]")
        if not torch.isfinite(w).all() or (w < 0).any() or not torch.allclose(w.sum(-1), torch.ones(w.shape[0], device=w.device), atol=1e-6):
            raise ValueError("w must be finite and on the simplex")
        return torch.cat((obs, w), dim=-1)

    def actor_features(self, obs: Tensor, w: Tensor) -> Tensor:
        h = self.actor_pre(self._with_w(obs, w))
        h = torch.nn.functional.elu(h)
        return self.actor_rest(self.film_gamma(w) * h + self.film_beta(w))

    def actor_mean_with_preference(self, obs: Tensor, w: Tensor) -> Tensor:
        return self.actor_mean(self.actor_features(obs, w))

    def _pre_tanh_dist_with_preference(self, obs: Tensor, w: Tensor):
        from torch.distributions import Normal
        mean = self.actor_mean_with_preference(obs, w)
        std = (self.log_std if self.exploration_mode == "learned" else self.scheduled_log_std).exp()
        return Normal(mean, std)

    def act_with_preference(self, obs: Tensor, w: Tensor):
        dist = self._pre_tanh_dist_with_preference(obs, w)
        u = dist.sample()
        return self._squash(dist, u)

    def act_inference_with_preference(self, obs: Tensor, w: Tensor):
        dist = self._pre_tanh_dist_with_preference(obs, w)
        return torch.tanh(dist.mean) * self.ACTION_CLIP

    def logp_with_preference(self, obs: Tensor, w: Tensor, action: Tensor):
        return self.logp(self._with_w(obs, w), action) if False else self._logp_for_action(self._pre_tanh_dist_with_preference(obs, w), action)

    def _logp_for_action(self, dist, action: Tensor):
        u = torch.atanh((action / self.ACTION_CLIP).clamp(-1 + self._ATANH_EPS, 1 - self._ATANH_EPS))
        _, logp = self._squash(dist, u)
        return logp

    def value_with_preference(self, obs: Tensor, w: Tensor):
        return self.value(self._with_w(obs, w))


def initialize_from_v1c(model: D3FiLMActorCritic, source_model_state: dict, *, w_ref=(1 / 3, 1 / 3, 1 / 3)) -> None:
    """Copy V1-C state and absorb its first-layer preference effect at w_ref."""
    src = source_model_state
    with torch.no_grad():
        w = torch.as_tensor(w_ref, device=model.actor_pre.weight.device, dtype=model.actor_pre.weight.dtype)
        model.actor_pre.weight.copy_(src["actor_body.0.weight"])
        model.actor_pre.bias.copy_(src["actor_body.0.bias"])
        model.actor_rest[0].load_state_dict({"weight": src["actor_body.2.weight"], "bias": src["actor_body.2.bias"]}) if False else None
        model.actor_rest[0].weight.copy_(src["actor_body.2.weight"])
        model.actor_rest[0].bias.copy_(src["actor_body.2.bias"])
        model.actor_rest[2].weight.copy_(src["actor_body.4.weight"])
        model.actor_rest[2].bias.copy_(src["actor_body.4.bias"])
        model.actor_mean.weight.copy_(src["actor_mean.weight"])
        model.actor_mean.bias.copy_(src["actor_mean.bias"])
        model.critic_body.load_state_dict({k.removeprefix("critic_body."): v for k, v in src.items() if k.startswith("critic_body.")})
        model.critic_head.weight.copy_(src["critic_head.weight"])
        model.critic_head.bias.copy_(src["critic_head.bias"])
        model.log_std.copy_(src["log_std"])
        model.scheduled_log_std.copy_(src["scheduled_log_std"])
        nn.init.zeros_(model.film_gamma.weight); nn.init.ones_(model.film_gamma.bias)
        nn.init.zeros_(model.film_beta.weight); nn.init.zeros_(model.film_beta.bias)
