"""D3PO's Late-Stage Weighting (LSW) + diversity regularizer — Ambadkar et al.
2026, arXiv:2602.07764. See scripts/rl/core/algorithms/moppo.py for how
MOPPOTrainer wires this in place of AMOR's early-scalarization.
"""

import numpy as np
import pytest
import torch

from rl.core.objectives.losses import d3po_actor_loss, diversity_regularizer_loss, normalize_per_objective


def test_normalize_per_objective_gives_zero_mean_unit_std_per_column():
    adv = np.array([[1.0, 100.0], [3.0, 300.0], [5.0, 500.0]], dtype=np.float32)
    out = normalize_per_objective(adv)
    assert np.allclose(out.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(out.std(axis=0), 1.0, atol=1e-4)


def test_d3po_actor_loss_matches_clip_then_weight_formula_when_ratio_clipped():
    # ratio outside [1-eps, 1+eps] so clipping actually engages differently
    # per objective (this is the exact behavior that differs from AMOR's
    # early scalarization — see next test).
    ratio = torch.tensor([1.5])
    adv = torch.tensor([[2.0, -2.0]])  # objective 0 positive, objective 1 negative
    w = torch.tensor([[0.5, 0.5]])
    clip_eps = 0.2

    loss = d3po_actor_loss(ratio, adv, w, clip_eps)

    # hand-computed: clip(1.5, 0.8, 1.2) = 1.2
    # obj0: min(1.5*2.0, 1.2*2.0) = min(3.0, 2.4) = 2.4
    # obj1: min(1.5*-2.0, 1.2*-2.0) = min(-3.0, -2.4) = -3.0
    # weighted = 0.5*2.4 + 0.5*(-3.0) = -0.3 ; loss = -mean(weighted) = 0.3
    assert torch.isclose(loss, torch.tensor(0.3), atol=1e-5)


def test_d3po_actor_loss_differs_from_early_scalarization_when_clipped():
    # Regression test for the exact bug this replaces: AMOR clips the
    # w-scalarized advantage BEFORE clipping (losing signal when objectives
    # conflict); D3PO clips each objective independently first. The two
    # must diverge once ratio pushes clipping into effect on
    # opposite-signed per-objective advantages.
    ratio = torch.tensor([1.5])
    adv = torch.tensor([[2.0, -2.0]])
    w = torch.tensor([[0.5, 0.5]])
    clip_eps = 0.2

    d3po_loss = d3po_actor_loss(ratio, adv, w, clip_eps)

    scalar_adv = (adv * w).sum(-1)  # AMOR: scalarize before clip
    clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    amor_loss = -torch.min(ratio * scalar_adv, clipped_ratio * scalar_adv).mean()

    assert not torch.isclose(d3po_loss, amor_loss, atol=1e-5)


def test_diversity_loss_zero_when_kl_matches_scaled_l1_distance():
    # If the analytic KL between the two preference-conditioned action
    # distributions exactly equals alpha * ||w - w'||_1, the squared
    # residual (the regularizer's whole point) must be exactly zero.
    std = torch.tensor([1.0, 1.0])
    alpha = 2.0
    w = torch.tensor([[0.6, 0.4]])
    w_prime = torch.tensor([[0.5, 0.5]])
    l1 = (w - w_prime).abs().sum(-1)  # 0.2
    target_kl = alpha * l1  # 0.4
    # KL(N(m1,std)||N(m2,std)) = sum (m1-m2)^2 / (2*std^2); solve for a
    # mean gap that gives exactly target_kl with std=1: (delta)^2/2 = 0.4
    delta = torch.sqrt(target_kl * 2)
    mean_w = torch.zeros(1, 2)
    mean_w_prime = torch.zeros(1, 2)
    mean_w_prime[0, 0] = delta.item()

    loss = diversity_regularizer_loss(mean_w, mean_w_prime, w, w_prime, std, alpha)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)


def test_diversity_loss_positive_when_kl_and_l1_mismatch():
    std = torch.tensor([1.0])
    mean_w = torch.zeros(1, 1)
    mean_w_prime = torch.zeros(1, 1)  # identical means -> KL = 0
    w = torch.tensor([[0.9]])
    w_prime = torch.tensor([[0.1]])  # far apart -> alpha*L1 > 0
    loss = diversity_regularizer_loss(mean_w, mean_w_prime, w, w_prime, std, alpha=5.0)
    assert loss.item() > 0.0


def test_diversity_loss_stays_bounded_when_std_is_collapsed_near_the_exploration_floor():
    """Found 2026-09-18: re-enabling diversity_lambda (0 -> 0.05) while the
    policy's actual std sat near ActorCritic.LOG_STD_MIN (~0.2) produced a
    policy loss in the billions within ~10 updates -- 2*std**2 in this
    formula's denominator is tiny at that std, so even a modest few-unit
    gap between mean_w and mean_w' blows kl into the hundreds before it's
    squared again. min_std (default 0.5) floors the denominator so this
    term stays numerically sane regardless of how collapsed exploration
    currently is -- this reproduces that incident's std (0.2) and a
    realistic max mean gap (2*ACTION_CLIP=6 per dim) and checks the loss
    stays in a PPO-sane range, not the millions."""
    std = torch.full((12,), 0.2)  # matches ActorCritic std at LOG_STD_MIN
    mean_w = torch.full((1, 12), -3.0)
    mean_w_prime = torch.full((1, 12), 3.0)  # worst case: every dim at opposite ACTION_CLIP extremes
    w = torch.tensor([[0.9, 0.025, 0.025, 0.025, 0.025]])
    w_prime = torch.tensor([[0.025, 0.025, 0.025, 0.025, 0.9]])

    loss = diversity_regularizer_loss(mean_w, mean_w_prime, w, w_prime, std, alpha=1.0)
    assert torch.isfinite(loss)
    assert loss.item() <= 100.0, f"loss={loss.item()} -- exceeds max_loss, the hard safety net"


def test_diversity_loss_max_loss_clamp_engages_even_with_min_std_already_applied():
    """max_loss is a second, independent safety net on top of min_std (see
    that param's docstring) -- proves it actually clamps, not just that
    min_std alone happens to keep things bounded, by using a mean gap wide
    enough to exceed max_loss=100 even at min_std's floor of 0.5."""
    std = torch.full((12,), 0.2)
    mean_w = torch.full((1, 12), -3.0)
    mean_w_prime = torch.full((1, 12), 3.0)
    w = torch.tensor([[0.9, 0.025, 0.025, 0.025, 0.025]])
    w_prime = torch.tensor([[0.025, 0.025, 0.025, 0.025, 0.9]])

    loss = diversity_regularizer_loss(mean_w, mean_w_prime, w, w_prime, std, alpha=1.0, max_loss=100.0)
    assert loss.item() == pytest.approx(100.0)
