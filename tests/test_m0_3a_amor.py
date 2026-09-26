import torch

from rl.core.algorithms.amor import (
    AmorActorCritic, sample_episode_preferences, scalarize_advantages,
    normalize_scalar_advantages, standard_clipped_actor_loss, vector_gae,
)


def test_preferences_are_simplex_and_finite():
    w = sample_episode_preferences(32)
    assert w.shape == (32, 5)
    assert torch.isfinite(w).all() and torch.all(w > 0)
    assert torch.allclose(w.sum(-1), torch.ones(32), atol=1e-6)


def test_early_scalarization_and_standard_clip():
    adv = torch.tensor([[1., -1., .5, 0., 2.], [.2, .1, -.3, .4, -.2]])
    w = torch.full_like(adv, .2)
    scalar = scalarize_advantages(adv, w)
    ratio = torch.tensor([1.1, .8])
    expected = -(ratio * scalar).mean()
    assert torch.allclose(standard_clipped_actor_loss(ratio, scalar), expected)
    norm = normalize_scalar_advantages(scalar)
    assert abs(float(norm.mean())) < 1e-6


def test_model_and_vector_gae_shapes():
    m = AmorActorCritic(11, 4, [16, 16])
    obs, w = torch.randn(7, 11), torch.full((7, 5), .2)
    action, logp = m.act_with_preference(obs, w)
    assert action.shape == (7, 4) and logp.shape == (7,)
    assert m.value_with_preference(obs, w).shape == (7, 5)
    r = torch.randn(4, 3, 5); v = torch.randn(4, 3, 5)
    a, ret = vector_gae(r, v, torch.randn(3, 5), torch.zeros(4, 3, dtype=torch.bool))
    assert a.shape == ret.shape == r.shape and torch.isfinite(ret).all()
