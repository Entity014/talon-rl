from __future__ import annotations

import torch
from torch import nn

from rl.core.objectives.routing import (
    SharedObjectiveCritic,
    normalize_objective_advantages,
    objective_critic_loss,
    route_objective_gradients,
)


def test_advantages_are_normalized_per_objective():
    x = torch.tensor([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])
    y = normalize_objective_advantages(x)
    assert torch.allclose(y.mean(0), torch.zeros(2), atol=1e-6)
    assert torch.allclose(y.std(0, unbiased=False), torch.ones(2), atol=1e-6)


def test_off_diagonal_adapter_gradients_are_zero_and_shared_is_aggregate():
    shared = nn.Parameter(torch.tensor(2.0))
    adapters = [nn.Parameter(torch.tensor(float(i + 1))) for i in range(3)]
    losses = [(shared + adapters[i]) ** 2 for i in range(3)]
    routed = route_objective_gradients(losses, [[p] for p in adapters], [shared], torch.tensor([0.2, 0.3, 0.5]))
    for i, parameter in enumerate(adapters):
        expected = torch.tensor(0.2 if i == 0 else 0.3 if i == 1 else 0.5) * 2 * (2.0 + float(i + 1))
        assert torch.allclose(routed[parameter], expected)
    expected_shared = sum(w * 2 * (2.0 + float(i + 1)) for i, w in enumerate((0.2, 0.3, 0.5)))
    assert torch.allclose(routed[shared], torch.tensor(expected_shared))


def test_shared_objective_critic_has_separate_heads_and_aggregate_encoder_gradient():
    critic = SharedObjectiveCritic(4, 8, objectives=3)
    values = critic(torch.randn(6, 4))
    returns = torch.randn(6, 3)
    loss = objective_critic_loss(values, returns)
    loss.backward()
    assert values.shape == (6, 3)
    assert all(head.weight.grad is not None for head in critic.heads)
    assert critic.encoder[0].weight.grad is not None
