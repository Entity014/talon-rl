"""Tests for AMOR algorithm variants."""

# -----------------------------------------------------------------------------
# Former: test_m0_3a_amor.py
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# Former: test_m0_3b_amor.py
# -----------------------------------------------------------------------------

import torch
from rl.core.algorithms.amor import RunningObservationNorm, ScaleAlignedAmorActorCritic

def test_running_norm_roundtrip_and_freeze():
    n = RunningObservationNorm(3); x = torch.randn(20,3); n.update(x); before=n.mean.clone(); n.freeze(); n.update(torch.randn(20,3)); assert torch.equal(before,n.mean)
    state=n.state_dict(); extra=n.state_dict_extra(); m=RunningObservationNorm(3); m.load_state_dict(state); m.load_state_dict_extra(extra); assert torch.allclose(m.mean,n.mean) and m.frozen

def test_scale_model_shapes():
    m=ScaleAlignedAmorActorCritic(11,4,[8,8,8,8]); o=torch.randn(5,11); w=torch.full((5,5),.2); a,lp=m.act_with_preference(o,w); assert a.shape==(5,4) and m.value_with_preference(o,w).shape==(5,5) and torch.isfinite(lp).all()
