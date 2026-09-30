"""FC-G optimizer and composition preflight on a shared PPO minibatch."""
import copy
from dataclasses import replace

import torch

from rl.core.algorithms.objective_set_ppo import PPOConfig, compose_actor_gradients, update
from talon_rl.models.authority.teacher_v4 import TeacherV4
from tests.core.algorithms.test_fcf_routing import batch, close_state, optimizers


def test_composition_formula_and_exposure():
    t = [torch.tensor([2.0, 0.0])]
    n = [torch.tensor([0.0, 0.5])]
    g0, m0 = compose_actor_gradients(t, n, 1.0, "GLOBAL")
    g1, m1 = compose_actor_gradients(t, n, 1.0, "SPLIT")
    g2, m2 = compose_actor_gradients(t, n, 1.0, "SPLIT-NORM-MATCHED")
    assert torch.allclose(g1[0], torch.tensor([1.0, 0.5]), atol=2e-6)
    assert torch.allclose(torch.linalg.vector_norm(g0[0]), torch.linalg.vector_norm(g2[0]), atol=1e-6)
    assert m0["fcg_coef_diff_frac"] == 1
    assert m0["fcg_global_suppresses_non_task_frac"] == 1
    assert m1["fcg_split_norm"] > 1
    assert m2["fcg_cos_global_normmatched"] < 1
    z, mz = compose_actor_gradients([torch.zeros(2)], [torch.zeros(2)], 1.0, "SPLIT-NORM-MATCHED")
    assert torch.equal(z[0], torch.zeros(2)) and mz["fcg_cos_global_normmatched"] == 1


def test_global_matches_legacy_with_adam_history_and_critic():
    torch.manual_seed(21)
    base = TeacherV4(num_objectives=3)
    cfg = replace(PPOConfig(), epochs=1, minibatches=1)
    a0, c0 = optimizers(base)
    first = batch(base)
    update(base, a0, c0, first, cfg, cfg.lr, torch.Generator().manual_seed(1))
    routed = copy.deepcopy(base)
    a1, c1 = optimizers(routed)
    a1.load_state_dict(copy.deepcopy(a0.state_dict()))
    c1.load_state_dict(copy.deepcopy(c0.state_dict()))
    shared = batch(base)
    lr0, st0 = update(base, a0, c0, shared, cfg, cfg.lr, torch.Generator().manual_seed(2))
    lr1, st1 = update(routed, a1, c1, shared, cfg, cfg.lr, torch.Generator().manual_seed(2),
                      gradient_composition="GLOBAL")
    assert lr0 == lr1
    assert st0 == {k: st1[k] for k in st0}
    close_state(base.state_dict(), routed.state_dict())
    close_state(a0.state_dict(), a1.state_dict())
    close_state(c0.state_dict(), c1.state_dict())
    assert st1["fcg_task_norm"] > 0 and st1["fcg_non_task_norm"] > 0


def test_split_changes_actor_but_not_critic_on_shared_batch():
    torch.manual_seed(22)
    base = TeacherV4(num_objectives=3)
    split = copy.deepcopy(base)
    a0, c0 = optimizers(base)
    a1, c1 = optimizers(split)
    b = batch(base)
    cfg = replace(PPOConfig(), epochs=1, minibatches=1, max_grad_norm=0.01)
    _, s0 = update(base, a0, c0, b, cfg, cfg.lr, torch.Generator().manual_seed(3), gradient_composition="GLOBAL")
    _, s1 = update(split, a1, c1, b, cfg, cfg.lr, torch.Generator().manual_seed(3), gradient_composition="SPLIT")
    assert s0["kl"] == s1["kl"]
    assert any(not torch.equal(p, q) for p, q in zip(base.actor_parameters(), split.actor_parameters()))
    for p, q in zip(base.critic_parameters(), split.critic_parameters()):
        assert torch.equal(p, q)
    close_state(c0.state_dict(), c1.state_dict())
