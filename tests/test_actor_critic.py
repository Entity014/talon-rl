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


def test_log_std_has_a_floor_below_which_it_cannot_shrink():
    """Found 2026-09-17 (phase1_longrun, 20000 updates): entropy_coef alone
    (moppo.py) slowed but didn't stop log_std shrinking to ~-2.66 (std~=0.07)
    over a long run, after which the policy couldn't explore back to a much
    better behavior it had briefly found (mean_episode_len 70.45 -> 6-8
    floor, never recovered). Even if the parameter itself is driven far
    below the floor, the std actually used for sampling/entropy must stay
    close to exp(LOG_STD_MIN) -- softplus only asymptotes to the floor, so
    this checks "close" (1e-3), not bitwise equal."""
    model = _make_model()
    with torch.no_grad():
        model.log_std.fill_(-10.0)  # far below LOG_STD_MIN
    dist = model._pre_tanh_dist(torch.zeros(2, 8))
    expected_std = torch.tensor(ActorCritic.LOG_STD_MIN).exp()
    assert torch.allclose(dist.stddev, expected_std.expand_as(dist.stddev), atol=1e-3)


def test_log_std_gradient_is_nonzero_even_below_the_floor():
    """Found 2026-09-18 (phase1_longrun5): a hard `torch.clamp` floor gives
    exactly zero gradient outside its range, so once log_std overshot
    LOG_STD_MIN (Adam momentum alone carried one dim to -1.89), BOTH the
    policy-loss gradient and the entropy-bonus gradient into log_std became
    permanently 0 -- entropy went dead flat at the floor's exact value for
    7000+ iterations straight, unrecoverable no matter entropy_coef. A
    softplus floor must keep a nonzero (if small) gradient past the
    boundary so the entropy bonus can still pull log_std back up."""
    model = _make_model()
    with torch.no_grad():
        model.log_std.fill_(-10.0)  # far below LOG_STD_MIN
    entropy = model.entropy(torch.zeros(2, 8)).sum()
    entropy.backward()
    assert torch.all(model.log_std.grad != 0)
