"""V2 explicit continuous behavior-latent actor/critic primitives.

The module is deliberately separate from V1.  It contains no routing,
curriculum, replay, or action-distillation mechanism.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.distributions import Normal

from rl.core.modules.actor_critic import ActorCritic
from talon_rl.models.foundations.three_objective import NUM_OBJECTIVES


LATENT_DIM = 2


class V2BehaviorActorCritic(ActorCritic):
    """Single actor conditioned through an explicit continuous behavior z."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        anchors: Tensor | None = None,
        hidden_dims: list[int] | None = None,
        *,
        film_alpha: float = 0.1,
    ):
        hidden_dims = hidden_dims or [128, 128, 128]
        # Critic input is obs+z; actor input is obs only and is modulated by z.
        super().__init__(obs_dim, obs_dim + LATENT_DIM, action_dim, NUM_OBJECTIVES, hidden_dims)
        self.physical_obs_dim = obs_dim
        self.film_alpha = float(film_alpha)
        if not torch.isfinite(torch.tensor(self.film_alpha)) or self.film_alpha < 0:
            raise ValueError("film_alpha must be finite and non-negative")
        self.behavior_encoder = nn.Sequential(nn.Linear(NUM_OBJECTIVES, 32), nn.ELU(), nn.Linear(32, LATENT_DIM))
        self.actor_pre = nn.Linear(obs_dim, hidden_dims[0])
        self.film_gamma = nn.Linear(LATENT_DIM, hidden_dims[0])
        self.film_beta = nn.Linear(LATENT_DIM, hidden_dims[0])
        self.actor_rest = nn.Sequential(nn.Linear(hidden_dims[0], hidden_dims[1]), nn.ELU(), nn.Linear(hidden_dims[1], hidden_dims[2]), nn.ELU())
        # Keep base names available to shared tooling; V2 uses explicit paths.
        self.actor_body = nn.Identity()
        self.register_buffer("behavior_anchors", torch.zeros(NUM_OBJECTIVES, LATENT_DIM) if anchors is None else anchors.detach().clone(), persistent=True)
        self._init_identity_conditioning()

    def _init_identity_conditioning(self) -> None:
        # Keep the hidden encoder path alive.  Zeroing this layer as well as
        # the output layer makes the manifold gradient reach only the output
        # bias, producing a constant z(w) and permanently killing input
        # dependence.  The final layer remains zero so z=0 at initialization.
        nn.init.xavier_uniform_(self.behavior_encoder[0].weight); nn.init.zeros_(self.behavior_encoder[0].bias)
        nn.init.zeros_(self.behavior_encoder[2].weight); nn.init.zeros_(self.behavior_encoder[2].bias)
        nn.init.zeros_(self.film_gamma.weight); nn.init.zeros_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight); nn.init.zeros_(self.film_beta.bias)

    @staticmethod
    def _validate_w(obs: Tensor, w: Tensor) -> None:
        if obs.ndim != 2 or w.ndim != 2 or obs.shape[0] != w.shape[0] or w.shape[1] != NUM_OBJECTIVES:
            raise ValueError("obs and w must be [batch,dim] and [batch,3]")
        if not torch.isfinite(w).all() or (w < 0).any() or not torch.allclose(w.sum(-1), torch.ones(w.shape[0], device=w.device), atol=1e-6):
            raise ValueError("w must be finite and on the simplex")

    def behavior_z(self, w: Tensor) -> Tensor:
        if w.ndim != 2 or w.shape[1] != NUM_OBJECTIVES or not torch.isfinite(w).all():
            raise ValueError("w must be finite [batch,3]")
        return self.behavior_encoder(w)

    def z_reference(self, w: Tensor) -> Tensor:
        if w.ndim != 2 or w.shape[1] != NUM_OBJECTIVES or not torch.isfinite(w).all():
            raise ValueError("w must be finite [batch,3]")
        return w @ self.behavior_anchors

    def manifold_loss(self, w: Tensor) -> Tensor:
        z = self.behavior_z(w)
        return (z - self.z_reference(w).detach()).pow(2).mean()

    def actor_features(self, obs: Tensor, w: Tensor) -> Tensor:
        self._validate_w(obs, w)
        z = self.behavior_z(w)
        h = torch.nn.functional.elu(self.actor_pre(obs))
        h = (1.0 + self.film_alpha * torch.tanh(self.film_gamma(z))) * h + self.film_alpha * torch.tanh(self.film_beta(z))
        return self.actor_rest(h)

    def _dist(self, obs: Tensor, w: Tensor) -> Normal:
        return Normal(self.actor_mean(self.actor_features(obs, w)), (self.log_std if self.exploration_mode == "learned" else self.scheduled_log_std).exp())

    def act_with_preference(self, obs: Tensor, w: Tensor):
        dist = self._dist(obs, w); u = dist.sample(); return self._squash(dist, u)

    def act_inference_with_preference(self, obs: Tensor, w: Tensor) -> Tensor:
        return torch.tanh(self._dist(obs, w).mean) * self.ACTION_CLIP

    def logp_with_preference(self, obs: Tensor, w: Tensor, action: Tensor) -> Tensor:
        dist = self._dist(obs, w); u = torch.atanh((action / self.ACTION_CLIP).clamp(-1 + self._ATANH_EPS, 1 - self._ATANH_EPS)); _, logp = self._squash(dist, u); return logp

    def value_with_preference(self, obs: Tensor, w: Tensor) -> Tensor:
        self._validate_w(obs, w)
        return self.critic_head(self.critic_body(torch.cat((obs, self.behavior_z(w)), dim=-1)))


def initialize_from_v1c(model: V2BehaviorActorCritic, source_state: dict, *, w_ref=(1 / 3, 1 / 3, 1 / 3)) -> None:
    """Function-preserving V1-C -> V2 initialization at w_ref."""
    w = torch.as_tensor(w_ref, dtype=model.actor_pre.weight.dtype, device=model.actor_pre.weight.device)
    with torch.no_grad():
        # V1 first layer: physical observation plus w. Absorb w_ref into bias.
        model.actor_pre.weight.copy_(source_state["actor_body.0.weight"][:, :model.physical_obs_dim])
        actor_w = source_state["actor_body.0.weight"].to(device=w.device, dtype=w.dtype)
        actor_b = source_state["actor_body.0.bias"].to(device=w.device, dtype=w.dtype)
        model.actor_pre.bias.copy_(actor_b + actor_w[:, model.physical_obs_dim:] @ w)
        model.actor_rest[0].weight.copy_(source_state["actor_body.2.weight"]); model.actor_rest[0].bias.copy_(source_state["actor_body.2.bias"])
        model.actor_rest[2].weight.copy_(source_state["actor_body.4.weight"]); model.actor_rest[2].bias.copy_(source_state["actor_body.4.bias"])
        model.actor_mean.weight.copy_(source_state["actor_mean.weight"]); model.actor_mean.bias.copy_(source_state["actor_mean.bias"])
        # V1 critic: initialize at w_ref, while z columns remain trainable from zero.
        model.critic_body[0].weight[:, :model.physical_obs_dim].copy_(source_state["critic_body.0.weight"][:, :model.physical_obs_dim])
        model.critic_body[0].weight[:, model.physical_obs_dim:].zero_()
        critic_w = source_state["critic_body.0.weight"].to(device=w.device, dtype=w.dtype)
        critic_b = source_state["critic_body.0.bias"].to(device=w.device, dtype=w.dtype)
        model.critic_body[0].bias.copy_(critic_b + critic_w[:, model.physical_obs_dim:] @ w)
        model.critic_body[2].weight.copy_(source_state["critic_body.2.weight"]); model.critic_body[2].bias.copy_(source_state["critic_body.2.bias"])
        model.critic_body[4].weight.copy_(source_state["critic_body.4.weight"]); model.critic_body[4].bias.copy_(source_state["critic_body.4.bias"])
        model.critic_head.weight.copy_(source_state["critic_head.weight"]); model.critic_head.bias.copy_(source_state["critic_head.bias"])
        model.log_std.copy_(source_state["log_std"]); model.scheduled_log_std.copy_(source_state["scheduled_log_std"])
