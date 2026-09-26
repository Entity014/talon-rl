"""Explicit V1-B objective-gradient routing primitives.

This module is intentionally trainer-independent.  It computes routed
gradients without changing the PPO ratio or loss equations.  Adapter ``i``
keeps only the gradient of objective loss ``L_i``; shared parameters receive
the weighted aggregate across all objective losses.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


def normalize_objective_advantages(advantages: Tensor, eps: float = 1e-8) -> Tensor:
    """Normalize each objective column independently over the batch."""
    if advantages.ndim != 2 or advantages.shape[1] < 1:
        raise ValueError("advantages must have shape [batch, objectives]")
    mean = advantages.mean(dim=0, keepdim=True)
    std = advantages.std(dim=0, unbiased=False, keepdim=True)
    return (advantages - mean) / std.clamp_min(eps)


def route_objective_gradients(
    losses: Sequence[Tensor],
    adapter_groups: Sequence[Sequence[nn.Parameter]],
    shared_parameters: Sequence[nn.Parameter],
    weights: Tensor,
) -> dict[nn.Parameter, Tensor]:
    """Return gradients implementing the frozen V1-B routing contract.

    ``losses[i]`` may depend on every adapter through the fused policy.  The
    returned gradient explicitly discards off-diagonal adapter paths while
    preserving the weighted aggregate for shared parameters.
    """
    n = len(losses)
    if len(adapter_groups) != n or weights.numel() != n:
        raise ValueError("losses, adapter_groups, and weights must have equal length")
    weights = weights.reshape(-1)
    if (weights < 0).any() or not torch.allclose(weights.sum(), torch.ones_like(weights.sum()), atol=1e-6):
        raise ValueError("weights must lie on the simplex")
    all_adapter = [parameter for group in adapter_groups for parameter in group]
    routed: dict[nn.Parameter, Tensor] = {}
    for i, loss in enumerate(losses):
        params = list(adapter_groups[i]) + list(shared_parameters)
        # Keep the graph available for routing observability (the 5x5
        # gradient matrix is computed from the same real minibatch).
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        for parameter, gradient in zip(params, grads):
            if gradient is None:
                continue
            contribution = weights[i] * gradient
            # Parameter.__eq__ is elementwise; use identity so a scalar
            # parameter with the same value as an adapter is not misclassified.
            if any(parameter is candidate for candidate in adapter_groups[i]):
                routed[parameter] = contribution
            else:
                routed[parameter] = routed.get(parameter, torch.zeros_like(gradient)) + contribution
    for parameter in all_adapter:
        routed.setdefault(parameter, torch.zeros_like(parameter))
    return routed


class SharedObjectiveCritic(nn.Module):
    """Shared critic encoder with one value head per objective."""

    def __init__(self, input_dim: int, hidden_dim: int, objectives: int = 5):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, hidden_dim), nn.ELU())
        self.heads = nn.ModuleList(nn.Linear(hidden_dim, 1) for _ in range(objectives))

    def forward(self, observations: Tensor) -> Tensor:
        latent = self.encoder(observations)
        return torch.cat([head(latent) for head in self.heads], dim=-1)


def objective_critic_loss(values: Tensor, returns: Tensor) -> Tensor:
    """Frozen V1-B critic scale: mean objective-wise MSE."""
    if values.shape != returns.shape or values.ndim != 2:
        raise ValueError("values and returns must have identical [batch, objectives] shapes")
    return torch.mean((values - returns) ** 2)
