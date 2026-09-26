"""V4-C0b gates for the objective-set PPO core (scripts/rl/core/algorithms/objective_set_ppo.py).

Each test names the failure it exists to prevent: a MORL loss that does not
reduce to plain PPO on one objective, weights that leak across objectives,
padding that is not inert, value loss moving the actor, or an optimization
shell that silently departs from rsl_rl (M0)."""
import itertools

import numpy as np
import pytest
import torch

from rl.core.algorithms.objective_set_ppo import (
    PPOConfig, actor_surrogate, adapt_lr, gae, gaussian_kl, normalize_advantages,
    sample_objective_sets, update, value_loss,
)
from rl.core.rollout.gae_functional import gae_per_objective
from talon_rl.models.authority.teacher_v4 import TeacherV4

CLIP = 0.2


def _clip_surrogate(logp, old, a):
    """Single-objective rsl_rl surrogate loss: mean of max(-A r, -A clip(r))."""
    r = torch.exp(logp - old)
    return torch.max(-a * r, -a * r.clamp(1 - CLIP, 1 + CLIP)).mean()


def _batch(B=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, generator=g) * .3, torch.randn(B, generator=g) * .3, torch.randn(B, 4, generator=g)


def test_one_hot_set_reduces_to_single_objective_ppo():
    logp, old, adv = _batch()
    for k in range(4):
        w = torch.zeros(64, 4); w[:, k] = 1
        loss, _ = actor_surrogate(logp, old, adv, w, w > 0, CLIP)
        assert torch.allclose(loss, _clip_surrogate(logp, old, adv[:, k]), atol=1e-6)


def test_two_objective_set_is_cardinality_times_weighted_sum():
    logp, old, adv = _batch()
    w = torch.zeros(64, 4); w[:, 0], w[:, 2] = .7, .3
    loss, _ = actor_surrogate(logp, old, adv, w, w > 0, CLIP)
    ref = 2 * (.7 * _clip_surrogate(logp, old, adv[:, 0]) + .3 * _clip_surrogate(logp, old, adv[:, 2]))
    assert torch.allclose(loss, ref, atol=1e-6)


def test_objective_permutation_leaves_losses_unchanged():
    logp, old, adv = _batch()
    w = torch.tensor([.4, .1, .3, .2]).expand(64, -1)
    mask = w > 0
    ret, v, ov = torch.randn(64, 4), torch.randn(64, 4), torch.randn(64, 4)
    ref_a = actor_surrogate(logp, old, adv, w, mask, CLIP)[0]
    ref_v = value_loss(v, ov, ret, mask, CLIP)
    for p in itertools.permutations(range(4)):
        p = list(p)
        assert torch.allclose(actor_surrogate(logp, old, adv[:, p], w[:, p], mask[:, p], CLIP)[0], ref_a, atol=1e-6)
        assert torch.allclose(value_loss(v[:, p], ov[:, p], ret[:, p], mask[:, p], CLIP), ref_v, atol=1e-6)


def test_inactive_objectives_contribute_nothing():
    logp, old, adv = _batch()
    w = torch.zeros(64, 4); w[:, 1], w[:, 3] = .5, .5
    mask = w > 0
    ret, v, ov = torch.randn(64, 4), torch.randn(64, 4), torch.randn(64, 4)
    a0 = actor_surrogate(logp, old, adv, w, mask, CLIP)[0]
    v0 = value_loss(v, ov, ret, mask, CLIP)
    junk = adv.clone(); junk[:, [0, 2]] = 1e3
    vjunk = v.clone(); vjunk[:, [0, 2]] = -1e3
    assert torch.equal(actor_surrogate(logp, old, junk, w, mask, CLIP)[0], a0)
    assert torch.equal(value_loss(vjunk, ov, ret, mask, CLIP), v0)
    na = normalize_advantages(adv, w, mask)
    assert torch.equal(na[:, [0, 2]], torch.zeros(64, 2))


def test_value_loss_matches_rsl_rl_clipped_loss_on_one_objective():
    v, ov, ret = torch.randn(64, 4), torch.randn(64, 4), torch.randn(64, 4)
    mask = torch.zeros(64, 4, dtype=torch.bool); mask[:, 1] = True
    c = ov[:, 1] + (v[:, 1] - ov[:, 1]).clamp(-CLIP, CLIP)
    ref = torch.max((v[:, 1] - ret[:, 1]).pow(2), (c - ret[:, 1]).pow(2)).mean()
    assert torch.allclose(value_loss(v, ov, ret, mask, CLIP), ref, atol=1e-6)


def test_gae_matches_numpy_reference_including_last_value_bootstrap():
    g = torch.Generator().manual_seed(1)
    T, N, K = 24, 8, 4
    r, v, lv = torch.randn(T, N, K, generator=g), torch.randn(T, N, K, generator=g), torch.randn(N, K, generator=g)
    d = torch.rand(T, N, generator=g) < .1
    ret, adv = gae(r, v, lv, d, .99, .95)
    ref = gae_per_objective(r.numpy(), torch.cat([v, lv[None]]).numpy(), d.numpy().astype(np.float32), .99, .95)
    assert np.allclose(adv.numpy(), ref, atol=1e-5)
    assert torch.allclose(ret, adv + v)


@pytest.mark.parametrize("kl,expected", [(0.03, 1e-3 / 1.5), (0.004, 1.5e-3), (0.01, 1e-3), (0.0, 1e-3), (1.0, 1e-3 / 1.5)])
def test_adaptive_lr_is_rsl_rl_rule(kl, expected):
    assert adapt_lr(1e-3, kl, 0.01) == pytest.approx(expected)


def test_adaptive_lr_respects_rsl_rl_bounds():
    assert adapt_lr(1.2e-5, 1.0, 0.01) == 1e-5
    assert adapt_lr(9e-3, 1e-4, 0.01) == 1e-2


def test_gaussian_kl_matches_torch_kl():
    g = torch.Generator().manual_seed(2)
    mu0, mu1 = torch.randn(32, 12, generator=g), torch.randn(32, 12, generator=g) * .1
    s0, s1 = torch.rand(32, 12, generator=g) + .5, torch.rand(32, 12, generator=g) + .5
    ref = torch.distributions.kl_divergence(torch.distributions.Normal(mu0, s0), torch.distributions.Normal(mu0 + mu1, s1)).sum(-1)
    assert torch.allclose(gaussian_kl(mu0, s0, mu0 + mu1, s1), ref, atol=1e-3)


def test_objective_set_sampler_contract():
    w, mask = sample_objective_sets(2000, (2, 4), torch.Generator().manual_seed(3))
    assert torch.allclose(w.sum(-1), torch.ones(2000), atol=1e-6)
    assert set(mask.sum(-1).tolist()) == {2, 4}
    assert torch.equal(w > 0, mask)
    w2, _ = sample_objective_sets(2000, (2, 4), torch.Generator().manual_seed(3))
    assert torch.equal(w, w2)
    one, m1 = sample_objective_sets(50, (1,), torch.Generator().manual_seed(4))
    assert torch.equal(one.max(-1).values, torch.ones(50)) and set(m1.sum(-1).tolist()) == {1}


def _rollout_batch(model, B=256, seed=5):
    g = torch.Generator().manual_seed(seed)
    obs, env = torch.randn(B, 48, generator=g), torch.randn(B, 12, generator=g)
    w, mask = sample_objective_sets(B, (1, 2, 3, 4), g)
    ids = torch.arange(4).expand(B, -1).contiguous()
    with torch.no_grad():
        dist = model._dist(obs, env, ids, w)
        u = dist.sample()
        old_logp = (dist.log_prob(u) - model._log_det_jacobian(u)).sum(-1)
        ov = model.query_values(obs, env, ids, w, ids)
    ret = ov + torch.randn(B, 4, generator=g)
    return dict(obs=obs, env=env, ids=ids, w=w, mask=mask, u=u, old_logp=old_logp, old_mu=dist.loc, old_sigma=dist.scale,
                old_values=ov, returns=ret, adv=normalize_advantages(ret - ov, w, mask))


def test_ratio_is_exactly_one_at_unchanged_policy():
    torch.manual_seed(0); m = TeacherV4()
    b = _rollout_batch(m)
    dist = m._dist(b["obs"], b["env"], b["ids"], b["w"])
    logp = (dist.log_prob(b["u"]) - m._log_det_jacobian(b["u"])).sum(-1)
    _, ratio = actor_surrogate(logp, b["old_logp"], b["adv"], b["w"], b["mask"], CLIP)
    assert torch.equal(ratio, torch.ones_like(ratio))


def test_actor_and_critic_losses_only_reach_their_own_parameters():
    torch.manual_seed(0); m = TeacherV4()
    b = _rollout_batch(m)
    dist = m._dist(b["obs"], b["env"], b["ids"], b["w"])
    logp = (dist.log_prob(b["u"]) - m._log_det_jacobian(b["u"])).sum(-1)
    (actor_surrogate(logp, b["old_logp"], b["adv"], b["w"], b["mask"], CLIP)[0] - .01 * dist.entropy().sum(-1).mean()).backward()
    assert all(p.grad is None for p in m.critic_parameters())
    m.zero_grad(set_to_none=True)
    value_loss(m.query_values(b["obs"], b["env"], b["ids"], b["w"], b["ids"]), b["old_values"], b["returns"], b["mask"], CLIP).backward()
    assert all(p.grad is None for p in m.actor_parameters())


def test_update_runs_the_m0_schedule_and_moves_both_heads():
    torch.manual_seed(0); m = TeacherV4()
    b = _rollout_batch(m, B=256)
    cfg = PPOConfig()
    aopt = torch.optim.Adam(m.actor_parameters(), lr=cfg.lr)
    copt = torch.optim.Adam(m.critic_parameters(), lr=cfg.lr)
    a0 = [p.detach().clone() for p in m.actor_parameters()]
    c0 = [p.detach().clone() for p in m.critic_parameters()]
    lr, stats = update(m, aopt, copt, b, cfg, cfg.lr, torch.Generator().manual_seed(0))
    assert all(np.isfinite(v) for v in stats.values())
    assert any(not torch.equal(p, q) for p, q in zip(m.actor_parameters(), a0))
    assert any(not torch.equal(p, q) for p, q in zip(m.critic_parameters(), c0))
    assert 1e-5 <= lr <= 1e-2
    assert all(pg["lr"] == lr for o in (aopt, copt) for pg in o.param_groups)
