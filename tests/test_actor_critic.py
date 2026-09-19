"""Regression tests for ActorCritic's tanh-squashed action bound -- found
2026-09-17: with no bound at all, a standing-only rollout showed the raw
action mean growing 0.5 -> 8.3 in magnitude over 5 steps, surviving 4.7x
worse than zero action on the identical env. Two hard-clamp variants were
tried and each broke training a different way (see
scripts/rl/core/modules/actor_critic.py's ACTION_CLIP comment for the full
diagnosis) before landing on tanh-squashing."""

import torch

from rl.core.modules.actor_critic import ActorCritic


def _make_model():
    return ActorCritic(actor_obs_dim=8, critic_obs_dim=8, action_dim=4, reward_dim=3, hidden_dims=[16, 16])


def test_act_inference_is_bounded_even_for_extreme_actor_mean():
    model = _make_model()
    with torch.no_grad():
        model.actor_mean.weight.fill_(0.0)
        model.actor_mean.bias.fill_(50.0)  # forces the raw mean far outside any sane range
    action = model.act_inference(torch.zeros(2, 8))
    assert torch.all(action.abs() <= ActorCritic.ACTION_CLIP)


def test_act_is_bounded_for_every_sample_even_with_large_std():
    model = _make_model()
    with torch.no_grad():
        model.actor_mean.weight.fill_(0.0)
        model.actor_mean.bias.fill_(0.0)
        model.log_std.fill_(5.0)  # huge std -> pre-tanh samples routinely far outside [-1, 1]
    action, logp = model.act(torch.zeros(100, 8))
    assert torch.all(action.abs() <= ActorCritic.ACTION_CLIP)
    assert torch.all(torch.isfinite(logp))


def test_logp_round_trips_through_the_tanh_inversion():
    """logp() must recover (up to float precision) the log_prob that act()
    computed for the same action -- this is what PPO's importance ratio
    depends on. Exercises the atanh-based inversion, not just the forward
    squash used by act() itself."""
    model = _make_model()
    obs = torch.randn(50, 8)
    action, logp_from_act = model.act(obs)
    logp_recomputed = model.logp(obs, action)
    assert torch.allclose(logp_from_act, logp_recomputed, atol=1e-4)


def test_logp_stays_finite_for_an_action_at_the_clip_boundary():
    """An action exactly at +/-ACTION_CLIP (e.g. loaded from a checkpoint
    trained before this fix, or produced by float rounding) must not send
    atanh to +/-inf and poison the loss with NaN."""
    model = _make_model()
    obs = torch.zeros(2, 8)
    boundary_action = torch.full((2, 4), ActorCritic.ACTION_CLIP)
    logp = model.logp(obs, boundary_action)
    assert torch.all(torch.isfinite(logp))


def test_entropy_increases_with_log_std():
    """PPO's entropy bonus (moppo.py's update()) was missing entirely before
    2026-09-17 -- every training run this session improved for ~20-30
    iterations then got stuck or declined for the rest of a 2000-update run,
    consistent with premature exploration collapse. entropy() must actually
    track log_std (a wider Gaussian = higher entropy) for that bonus to mean
    anything as a training signal."""
    model = _make_model()
    obs = torch.randn(10, 8)
    with torch.no_grad():
        model.log_std.fill_(-2.0)
    narrow = model.entropy(obs)
    with torch.no_grad():
        model.log_std.fill_(2.0)
    wide = model.entropy(obs)
    assert torch.all(torch.isfinite(narrow)) and torch.all(torch.isfinite(wide))
    assert torch.all(wide > narrow)


def test_pre_tanh_dist_does_not_bound_log_std_itself():
    """ActorCritic deliberately does NOT clamp/squash log_std anywhere in
    its forward pass (see LOG_STD_MIN/LOG_STD_MAX's shared docstring for
    the two rejected in-graph approaches -- a hard clamp gave a dead
    gradient one direction 2026-09-18, and a differentiable double-softplus
    ceiling was tried and rejected the same day, both a middle-range
    distortion and, at higher beta, the identical dead-gradient failure via
    float32 underflow). Bounding is MOPPOTrainer.update()'s job, via a
    post-optimizer-step clamp on the raw parameter -- see
    test_moppo_smoke.py's log_std bound tests. This just locks in that
    `std` tracks log_std directly (plain exp, no transform) so nobody
    reintroduces an in-graph bound here without noticing this test breaks."""
    model = _make_model()
    with torch.no_grad():
        model.log_std.fill_(-10.0)  # a value LOG_STD_MIN=-1.6 would normally floor
    dist = model._pre_tanh_dist(torch.zeros(2, 8))
    assert torch.allclose(dist.stddev, torch.tensor(-10.0).exp().expand_as(dist.stddev))


def test_log_std_gradient_is_always_nonzero():
    """A plain exp() has a real, nonzero gradient at every input -- unlike
    the hard-clamp floor this replaced (2026-09-18, see LOG_STD_MIN's
    docstring), there's no value log_std can hold that zeroes this out."""
    model = _make_model()
    with torch.no_grad():
        model.log_std.fill_(-10.0)
    entropy = model.entropy(torch.zeros(2, 8)).sum()
    entropy.backward()
    assert torch.all(model.log_std.grad != 0)


def test_raw_mean_matches_act_inference_before_the_tanh_squash():
    """raw_mean() feeds MOPPOTrainer's mean-magnitude regularizer
    (MOPPOConfig.mean_reg_coef) -- must be the exact same pre-tanh value
    act_inference() squashes, or the penalty would be regularizing a
    different quantity than the one actually saturating."""
    model = _make_model()
    obs = torch.randn(5, 8)
    raw = model.raw_mean(obs)
    action = model.act_inference(obs)
    assert torch.allclose(torch.tanh(raw) * ActorCritic.ACTION_CLIP, action)
