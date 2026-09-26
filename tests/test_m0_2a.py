import torch

from rl.core.algorithms.vector_ppo import (
    M02aActorCritic,
    late_weighted_actor_loss,
    vector_gae,
    vector_value_loss,
    adaptive_kl_lr,
)


def test_uniform_late_loss_is_algebraically_equivalent_when_advantages_sum():
    ratio = torch.tensor([1.05, 0.97, 1.01], requires_grad=True)
    adv = torch.tensor([[1.0, -0.2, 0.3, 2.0, -0.1],
                        [-0.4, 0.1, 0.2, -0.3, 0.7],
                        [0.5, 0.2, -0.1, 0.4, -0.2]], requires_grad=True)
    w = torch.full_like(adv, 0.2)
    scalar_adv = adv.sum(-1)
    actual = late_weighted_actor_loss(ratio, adv, w)
    expected = -(ratio * scalar_adv).mean()
    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-7)
    ga, = torch.autograd.grad(actual, ratio, retain_graph=True)
    ge, = torch.autograd.grad(expected, ratio)
    assert torch.allclose(ga, ge, atol=1e-6, rtol=1e-6)


def test_preference_actor_and_vector_critic_shapes():
    model = M02aActorCritic(obs_dim=11, action_dim=4, hidden_dims=[16, 16])
    obs = torch.randn(7, 11)
    w = torch.full((7, 5), 0.2)
    action, logp = model.act_with_preference(obs, w)
    value = model.value_with_preference(obs, w)
    assert action.shape == (7, 4)
    assert logp.shape == (7,)
    assert value.shape == (7, 5)
    assert torch.isfinite(action).all() and torch.isfinite(logp).all()


def test_vector_gae_and_value_loss_are_finite():
    reward = torch.randn(4, 3, 5)
    value = torch.randn(4, 3, 5)
    next_value = torch.randn(3, 5)
    done = torch.zeros(4, 3, dtype=torch.bool)
    adv, returns = vector_gae(reward, value, next_value, done)
    assert adv.shape == returns.shape == reward.shape
    assert torch.isfinite(adv).all() and torch.isfinite(returns).all()
    assert torch.isfinite(vector_value_loss(value, returns))


def test_adaptive_kl_lr_events_and_bounds():
    p = torch.nn.Parameter(torch.ones(()))
    opt = torch.optim.Adam([p], lr=1e-3)
    assert adaptive_kl_lr(opt, torch.tensor(0.001)) == "up"
    assert opt.param_groups[0]["lr"] == 1.5e-3
    assert adaptive_kl_lr(opt, torch.tensor(0.01)) == "hold"
    assert adaptive_kl_lr(opt, torch.tensor(0.1)) == "down"
    assert 1e-5 <= opt.param_groups[0]["lr"] <= 1e-2
