import pytest
import torch

from rl.core.modules.actor_critic import ActorCritic
from rl.core.modules.v1a_actor_critic import (
    V1AActorCritic,
    adapter_parameter_count,
    choose_adapter_bottleneck,
)


def test_v1a_initialization_is_function_preserving_for_action_and_logp():
    torch.manual_seed(11)
    base = ActorCritic(9, 9, 4, 1, [16, 12])
    model = V1AActorCritic.from_baseline(base, bottleneck_dim=2)
    obs = torch.randn(32, 9)
    w = torch.full((32, 5), 0.2)
    action = base.act_inference(obs)
    assert torch.allclose(action, model.act_inference_with_preference(obs, w), atol=1e-6, rtol=1e-6)
    assert torch.allclose(base.logp(obs, action), model.logp_with_preference(obs, w, action), atol=1e-6, rtol=1e-6)


def test_v1a_requires_simplex_weights_and_directly_uses_them():
    model = V1AActorCritic(4, 4, 2, 1, [8, 8], bottleneck_dim=1)
    obs = torch.randn(3, 4)
    with pytest.raises(ValueError, match="simplex"):
        model.act_inference_with_preference(obs, torch.ones(3, 5))
    with pytest.raises(ValueError, match="non-negative"):
        model.act_inference_with_preference(obs, torch.tensor([[1.2, -.2, 0., 0., 0.]]).expand(3, -1))


def test_v1a_gradient_semantics_are_expected_for_zero_up_projection():
    torch.manual_seed(3)
    model = V1AActorCritic(5, 5, 2, 1, [8, 8], bottleneck_dim=3)
    obs = torch.randn(6, 5)
    w = torch.full((6, 5), 0.2)
    model.act_inference_with_preference(obs, w).sum().backward()
    assert all(adapter.up.weight.grad.abs().sum() > 0 for adapter in model.adapters)
    assert all(adapter.down.weight.grad.abs().sum() == 0 for adapter in model.adapters)
    with torch.no_grad():
        for adapter in model.adapters:
            adapter.up.weight.add_(0.01)
    model.zero_grad()
    model.act_inference_with_preference(obs, w).sum().backward()
    assert all(adapter.down.weight.grad.abs().sum() > 0 for adapter in model.adapters)


def test_bottleneck_is_solved_from_the_frozen_overhead_budget():
    baseline_params = 1000
    b = choose_adapter_bottleneck(baseline_params, hidden_dim=16, overhead_fraction=0.30)
    assert b >= 1
    assert adapter_parameter_count(16, b) <= 300
    assert adapter_parameter_count(16, b + 1) > 300
