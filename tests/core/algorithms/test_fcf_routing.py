"""FC-F preflight: routing leaves FULL unchanged and blocks only new task gradients."""
import copy
from dataclasses import replace

import torch

from rl.core.algorithms.objective_set_ppo import PPOConfig, update
from talon_rl.models.authority.teacher_v4 import TeacherV4


def batch(model, n=24):
    gen = torch.Generator().manual_seed(101)
    obs = torch.randn(n, 48, generator=gen)
    env = torch.randn(n, 12, generator=gen)
    w = torch.rand(n, 2, generator=gen)
    w = w / w.sum(-1, keepdim=True)
    ids = torch.tensor([1, 2]).expand(n, -1).contiguous()
    qids = torch.arange(3).expand(n, -1).contiguous()
    with torch.no_grad():
        dist = model._dist(obs, env, ids, w)
        u = dist.sample()
        old_lp = (dist.log_prob(u) - model._log_det_jacobian(u)).sum(-1)
        old_v = model.query_values(obs, env, ids, w, qids)
    lw = torch.cat((torch.full((n, 1), .44), w), -1)
    lm = torch.ones(n, 3, dtype=torch.bool)
    return dict(obs=obs, env=env, ids=ids, w=w, mask=w > 0, u=u,
                old_logp=old_lp, old_mu=dist.loc.detach(), old_sigma=dist.scale.detach(),
                old_values=old_v, returns=old_v + torch.randn(n, 3, generator=gen) * .1,
                adv=torch.randn(n, 3, generator=gen), loss_w=lw, loss_mask=lm, query_ids=qids)


def optimizers(model):
    return torch.optim.Adam(model.actor_parameters(), lr=1e-3), torch.optim.Adam(model.critic_parameters(), lr=1e-3)


def close_state(a, b):
    for name in a:
        va, vb = a[name], b[name]
        if isinstance(va, torch.Tensor):
            assert torch.allclose(va, vb, atol=1e-6, rtol=1e-5), name
        elif isinstance(va, dict):
            close_state(va, vb)
        else:
            assert va == vb


def test_full_route_matches_existing_update_after_adam_state_is_loaded():
    torch.manual_seed(12)
    model = TeacherV4(num_objectives=3)
    cfg = replace(PPOConfig(), epochs=1, minibatches=1)
    aopt, copt = optimizers(model)
    b = batch(model)
    update(model, aopt, copt, b, cfg, cfg.lr, torch.Generator().manual_seed(1))
    routed = copy.deepcopy(model)
    raopt, rcopt = optimizers(routed)
    raopt.load_state_dict(copy.deepcopy(aopt.state_dict()))
    rcopt.load_state_dict(copy.deepcopy(copt.state_dict()))
    b = batch(model)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    lr0, st0 = update(model, aopt, copt, b, cfg, cfg.lr, torch.Generator().manual_seed(2))
    allowed = {id(p) for ps in routed.actor_parameter_groups().values() for p in ps}
    lr1, st1 = update(routed, raopt, rcopt, b, cfg, cfg.lr, torch.Generator().manual_seed(2), allowed)
    assert lr0 == lr1
    assert st0 == {k: st1[k] for k in st0}
    for (n, p), (_, q) in zip(model.named_parameters(), routed.named_parameters()):
        assert torch.allclose(p - before[n], q - before[n], atol=1e-6, rtol=1e-5), n
    close_state(aopt.state_dict(), raopt.state_dict())
    close_state(copt.state_dict(), rcopt.state_dict())
    assert 0 < st1["actor_clip_coef"] <= 1
    assert st1["actor_clip_min"] <= st1["actor_clip_coef"]


def test_task_only_batch_moves_only_allowed_groups():
    torch.manual_seed(13)
    base = TeacherV4(num_objectives=3)
    b = batch(base)
    b["adv"][:, 1:] = 0
    cfg = replace(PPOConfig(), epochs=1, minibatches=1, entropy_coef=0.0, value_coef=0.0, max_grad_norm=1000.0)
    routes = {
        "NO-PREF-PATH": {"G1", "G2", "G3", "G7"},
        "NO-BASES": {"G1", "G2", "G5", "G6", "G7"},
        "SHARED-FEATURES-ONLY": {"G1", "G2", "G7"},
        "PREF-ONLY": {"G4", "G5", "G6", "G7"},
    }
    for names in routes.values():
        model = copy.deepcopy(base)
        groups = model.actor_parameter_groups()
        before = {id(p): p.detach().clone() for ps in groups.values() for p in ps}
        allowed = {id(p) for g in names for p in groups[g]}
        aopt, copt = optimizers(model)
        update(model, aopt, copt, b, cfg, cfg.lr, torch.Generator().manual_seed(3), allowed)
        assert any(not torch.equal(p, before[id(p)]) for g in names for p in groups[g])
        assert all(torch.equal(p, before[id(p)]) for g, ps in groups.items() if g not in names for p in ps)


def test_preference_only_update_is_route_invariant():
    torch.manual_seed(14)
    base = TeacherV4(num_objectives=3)
    b = batch(base)
    b["adv"][:, 0] = 0
    cfg = replace(PPOConfig(), epochs=1, minibatches=1)
    full = copy.deepcopy(base)
    aopt, copt = optimizers(full)
    full_ids = {id(p) for ps in full.actor_parameter_groups().values() for p in ps}
    update(full, aopt, copt, b, cfg, cfg.lr, torch.Generator().manual_seed(4), full_ids)
    routed = copy.deepcopy(base)
    raopt, rcopt = optimizers(routed)
    groups = routed.actor_parameter_groups()
    only_g7 = {id(p) for p in groups["G7"]}
    update(routed, raopt, rcopt, b, cfg, cfg.lr, torch.Generator().manual_seed(4), only_g7)
    for (name, p), (_, q) in zip(full.named_parameters(), routed.named_parameters()):
        assert torch.allclose(p, q, atol=1e-6, rtol=1e-5), name
