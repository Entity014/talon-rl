"""V1-A function-preserving residual objective-adapter actor.

V1-A deliberately contains no learned preference gate, vector critic,
objective-specific advantage, curriculum, rehearsal, or anchor logic.  The
preference vector is a simplex weight applied directly to five residual
adapters after the shared actor backbone.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from .actor_critic import ActorCritic


OBJECTIVES = ("progress", "efficiency", "contact", "balance", "limits")
NUM_OBJECTIVES = len(OBJECTIVES)


def adapter_parameter_count(hidden_dim: int, bottleneck_dim: int, num_adapters: int = NUM_OBJECTIVES) -> int:
    """Count Linear(D) + Linear(U) parameters for all residual adapters."""
    if hidden_dim < 1 or bottleneck_dim < 1 or num_adapters < 1:
        raise ValueError("hidden_dim, bottleneck_dim, and num_adapters must be positive")
    one = hidden_dim * bottleneck_dim + bottleneck_dim + bottleneck_dim * hidden_dim + hidden_dim
    return num_adapters * one


def choose_adapter_bottleneck(
    baseline_parameter_count: int,
    hidden_dim: int,
    overhead_fraction: float = 0.30,
    num_adapters: int = NUM_OBJECTIVES,
) -> int:
    """Choose the largest bottleneck whose adapter overhead fits the budget."""
    if baseline_parameter_count < 1 or hidden_dim < 1:
        raise ValueError("baseline_parameter_count and hidden_dim must be positive")
    if not 0.0 < overhead_fraction:
        raise ValueError("overhead_fraction must be positive")
    budget = int(baseline_parameter_count * overhead_fraction)
    bottleneck = 0
    for candidate in range(1, hidden_dim + 1):
        if adapter_parameter_count(hidden_dim, candidate, num_adapters) <= budget:
            bottleneck = candidate
        else:
            break
    if bottleneck == 0:
        raise ValueError("parameter budget is too small for one adapter bottleneck unit")
    return bottleneck


class ResidualObjectiveAdapter(nn.Module):
    """One objective adapter: ``U sigma(D h)``.

    ``up`` is zero-initialized by design.  Consequently the complete V1-A
    actor is exactly the baseline actor at initialization, while ``down`` is
    normally initialized and becomes trainable after ``up`` receives its
    first non-zero update.
    """

    def __init__(self, hidden_dim: int, bottleneck_dim: int):
        super().__init__()
        self.down = nn.Linear(hidden_dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, hidden_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, h: Tensor) -> Tensor:
        return self.up(torch.sigmoid(self.down(h)))


class V1AActorCritic(ActorCritic):
    """Baseline ActorCritic with direct simplex-weighted residual adapters."""

    def __init__(
        self,
        actor_obs_dim: int,
        critic_obs_dim: int,
        action_dim: int,
        reward_dim: int,
        hidden_dims: list[int],
        bottleneck_dim: int,
        objectives: Sequence[str] = OBJECTIVES,
        reconstruction_dim: int = 0,
    ):
        if tuple(objectives) != OBJECTIVES:
            raise ValueError(f"V1-A objective order is frozen as {OBJECTIVES}")
        if bottleneck_dim < 1:
            raise ValueError("bottleneck_dim must be positive")
        super().__init__(actor_obs_dim, critic_obs_dim, action_dim, reward_dim, hidden_dims, reconstruction_dim)
        self.objectives = OBJECTIVES
        self.bottleneck_dim = bottleneck_dim
        self.adapters = nn.ModuleList(
            ResidualObjectiveAdapter(hidden_dims[-1], bottleneck_dim) for _ in self.objectives
        )

    @staticmethod
    def validate_preference(w: Tensor, batch_size: int | None = None) -> Tensor:
        if w.ndim == 1:
            if w.shape[0] != NUM_OBJECTIVES:
                raise ValueError(f"w must have {NUM_OBJECTIVES} objective weights")
            w = w.unsqueeze(0)
        if w.ndim != 2 or w.shape[1] != NUM_OBJECTIVES:
            raise ValueError(f"w must have shape [batch, {NUM_OBJECTIVES}]")
        if batch_size is not None and w.shape[0] not in (1, batch_size):
            raise ValueError("w batch dimension must be 1 or match the observation batch")
        if not torch.isfinite(w).all() or (w < 0).any():
            raise ValueError("w must be finite and non-negative")
        if not torch.allclose(w.sum(dim=-1), torch.ones(w.shape[0], device=w.device, dtype=w.dtype), atol=1e-6, rtol=0):
            raise ValueError("w must lie on the probability simplex")
        return w

    def fused_latent(self, actor_obs: Tensor, w: Tensor) -> Tensor:
        h = self.actor_body(actor_obs)
        weights = self.validate_preference(w, h.shape[0]).to(dtype=h.dtype, device=h.device)
        if weights.shape[0] == 1 and h.shape[0] != 1:
            weights = weights.expand(h.shape[0], -1)
        residual = torch.stack([adapter(h) for adapter in self.adapters], dim=1)
        return h + (weights.unsqueeze(-1) * residual).sum(dim=1)

    def _pre_tanh_dist_with_preference(self, actor_obs: Tensor, w: Tensor):
        from torch.distributions import Normal

        mean = self.actor_mean(self.fused_latent(actor_obs, w))
        std = (self.log_std if self.exploration_mode == "learned" else self.scheduled_log_std).exp()
        return Normal(mean, std)

    def act_inference_with_preference(self, actor_obs: Tensor, w: Tensor) -> Tensor:
        return torch.tanh(self.actor_mean(self.fused_latent(actor_obs, w))) * self.ACTION_CLIP

    def raw_mean_with_preference(self, actor_obs: Tensor, w: Tensor) -> Tensor:
        return self.actor_mean(self.fused_latent(actor_obs, w))

    def logp_with_preference(self, actor_obs: Tensor, w: Tensor, action: Tensor) -> Tensor:
        dist = self._pre_tanh_dist_with_preference(actor_obs, w)
        normalized = (action / self.ACTION_CLIP).clamp(-1.0 + self._ATANH_EPS, 1.0 - self._ATANH_EPS)
        u = torch.atanh(normalized)
        return (dist.log_prob(u) - self._log_det_jacobian(u)).sum(-1)

    def entropy_with_preference(self, actor_obs: Tensor, w: Tensor) -> Tensor:
        return self._pre_tanh_dist_with_preference(actor_obs, w).entropy().sum(-1)

    @classmethod
    def from_baseline(
        cls,
        baseline: ActorCritic,
        bottleneck_dim: int,
        objectives: Sequence[str] = OBJECTIVES,
    ) -> "V1AActorCritic":
        """Create V1-A and copy every compatible baseline parameter exactly."""
        model = cls(
            baseline.actor_body[0].in_features,
            baseline.critic_body[0].in_features,
            baseline.actor_mean.out_features,
            baseline.critic_head.out_features,
            [layer.out_features for layer in baseline.actor_body if isinstance(layer, nn.Linear)],
            bottleneck_dim,
            objectives,
            baseline.reconstruction_head.out_features if baseline.reconstruction_head is not None else 0,
        )
        own = model.state_dict()
        for key, value in baseline.state_dict().items():
            if key in own and own[key].shape == value.shape:
                own[key].copy_(value)
        model.load_state_dict(own)
        return model
