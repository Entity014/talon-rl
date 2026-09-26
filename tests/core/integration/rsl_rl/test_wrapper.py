import torch
from torch import nn
from torch.distributions import Normal

from rl.core.integration.rsl_rl.v1a_wrapper import RslRlV1AWrapper


class FakeRslActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(6, 8), nn.ELU(), nn.Linear(8, 8), nn.ELU(), nn.Linear(8, 3))
        self.critic = nn.Sequential(nn.Linear(6, 8), nn.ELU(), nn.Linear(8, 1))
        self.std = nn.Parameter(torch.full((3,), 0.25))
        self.distribution = None

    @property
    def action_std(self):
        return self.distribution.stddev


def test_wrapper_preserves_rsl_state_keys_and_function_at_zero_residual():
    torch.manual_seed(19)
    base = FakeRslActorCritic()
    wrapper = RslRlV1AWrapper(base, bottleneck_dim=2)
    obs = torch.randn(10, 6)
    w = torch.full((10, 5), 0.2)
    assert set(base.state_dict()) <= set(wrapper.state_dict())
    assert all(
        key.startswith("v1a_adapters.") or key == "w_ref"
        for key in wrapper.state_dict()
        if key not in base.state_dict()
    )
    assert torch.allclose(base.actor(obs), wrapper.act_inference(obs, w), atol=1e-6, rtol=1e-6)
    assert torch.equal(base.std, wrapper.state_dict()["std"])
    assert torch.allclose(base.critic(obs), wrapper.evaluate(obs), atol=1e-6, rtol=1e-6)


def test_wrapper_preserves_distribution_parameters_log_prob_and_entropy():
    torch.manual_seed(23)
    base = FakeRslActorCritic()
    wrapper = RslRlV1AWrapper(base, bottleneck_dim=2)
    obs = torch.randn(7, 6)
    w = torch.full((7, 5), 0.2)
    baseline_mean = base.actor(obs)
    wrapper.update_distribution(obs, w)
    baseline_dist = Normal(baseline_mean, base.std.expand_as(baseline_mean))
    actions = torch.randn(7, 3)
    assert torch.allclose(wrapper._base.distribution.loc, baseline_dist.loc, atol=1e-6, rtol=1e-6)
    assert torch.allclose(wrapper._base.distribution.scale, baseline_dist.scale, atol=1e-6, rtol=1e-6)
    assert torch.allclose(wrapper.get_actions_log_prob(actions), baseline_dist.log_prob(actions).sum(-1), atol=1e-6, rtol=1e-6)
    assert torch.allclose(wrapper.entropy, baseline_dist.entropy().sum(-1), atol=1e-6, rtol=1e-6)


def test_wrapper_first_gradient_reaches_up_not_down():
    torch.manual_seed(29)
    wrapper = RslRlV1AWrapper(FakeRslActorCritic(), bottleneck_dim=2)
    obs = torch.randn(4, 6)
    w = torch.full((4, 5), 0.2)
    wrapper.act_inference(obs, w).sum().backward()
    assert all(adapter.up.weight.grad.abs().sum() > 0 for adapter in wrapper.v1a_adapters)
    assert all(adapter.down.weight.grad.abs().sum() == 0 for adapter in wrapper.v1a_adapters)


def test_wrapper_round_trips_baseline_plus_extension_state_dict():
    torch.manual_seed(31)
    source = RslRlV1AWrapper(FakeRslActorCritic(), bottleneck_dim=2)
    with torch.no_grad():
        for adapter in source.v1a_adapters:
            adapter.up.weight.add_(0.01)
    target = RslRlV1AWrapper(FakeRslActorCritic(), bottleneck_dim=2)
    result = target.load_state_dict(source.state_dict())
    assert not result.missing_keys and not result.unexpected_keys
    for key, value in source.state_dict().items():
        assert torch.equal(value, target.state_dict()[key])


def test_wrapper_supports_s7_three_objective_configuration():
    torch.manual_seed(37)
    objectives = ("progress", "balance", "efficiency")
    wrapper = RslRlV1AWrapper(
        FakeRslActorCritic(),
        bottleneck_dim=2,
        objectives=objectives,
        w_ref=(1 / 3,) * 3,
    )
    obs = torch.randn(6, 6)
    action = wrapper.act_inference(obs, torch.full((6, 3), 1 / 3))
    assert action.shape == (6, 3)
    assert len(wrapper.v1a_adapters) == 3
    assert torch.allclose(action, wrapper._base.actor(obs), atol=1e-6, rtol=1e-6)
