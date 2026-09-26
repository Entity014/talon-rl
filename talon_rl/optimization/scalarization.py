"""Pure P0 actor-loss constructions for matched late-vs-early PPO tests."""
from __future__ import annotations

import torch
from torch import Tensor


def _validate(ratio: Tensor, advantages: Tensor, weights: Tensor) -> None:
    if ratio.ndim != 1 or advantages.ndim != 2 or advantages.shape[1] != 3:
        raise ValueError("ratio must be [batch] and advantages must be [batch,3]")
    if weights.shape != advantages.shape or weights.ndim != 2:
        raise ValueError("weights must have shape [batch,3]")
    if ratio.shape[0] != advantages.shape[0]:
        raise ValueError("ratio/advantages batch mismatch")
    if not torch.isfinite(ratio).all() or not torch.isfinite(advantages).all() or not torch.isfinite(weights).all():
        raise ValueError("P0 loss inputs must be finite")


def early_scalarized_ppo_loss(
    ratio: Tensor, advantages: Tensor, weights: Tensor, clip_eps: float = 0.2
) -> Tensor:
    """Scalarize advantages first, then apply one PPO clipped surrogate."""
    _validate(ratio, advantages, weights)
    scalar_advantage = (weights.detach() * advantages).sum(dim=-1)
    clipped_ratio = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps)
    surrogate = torch.minimum(ratio * scalar_advantage, clipped_ratio * scalar_advantage)
    return -surrogate.mean()


def late_weighted_ppo_loss(
    ratio: Tensor, advantages: Tensor, weights: Tensor, clip_eps: float = 0.2
) -> Tensor:
    """Apply PPO clipping per objective, then weight and sum the surrogates."""
    _validate(ratio, advantages, weights)
    clipped_ratio = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps)
    per_objective = torch.minimum(
        ratio[:, None] * advantages,
        clipped_ratio[:, None] * advantages,
    )
    return -(weights.detach() * per_objective).sum(dim=-1).mean()


def p0_loss_diagnostics(
    ratio: Tensor, advantages: Tensor, weights: Tensor, clip_eps: float = 0.2
) -> dict[str, Tensor]:
    """Return auditable P0 intermediates without changing either loss."""
    _validate(ratio, advantages, weights)
    clipped_ratio = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps)
    scalar_advantage = (weights.detach() * advantages).sum(dim=-1)
    per_objective = torch.minimum(
        ratio[:, None] * advantages,
        clipped_ratio[:, None] * advantages,
    )
    return {
        "scalar_advantage": scalar_advantage,
        "per_objective_surrogate": per_objective,
        "weighted_actor_loss": -(weights.detach() * per_objective).sum(dim=-1).mean(),
        "early_actor_loss": -torch.minimum(
            ratio * scalar_advantage,
            clipped_ratio * scalar_advantage,
        ).mean(),
        "ratio": ratio,
        "clipped_ratio": clipped_ratio,
        "clip_fraction": (ratio != clipped_ratio).to(ratio.dtype).mean(),
    }
