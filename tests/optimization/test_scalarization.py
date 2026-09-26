import torch

from talon_rl.optimization.scalarization import (
    early_scalarized_ppo_loss,
    late_weighted_ppo_loss,
    p0_loss_diagnostics,
)


def test_p0_losses_are_equivalent_when_clipping_is_inactive():
    ratio = torch.tensor([0.95, 1.05, 1.0])
    advantages = torch.tensor([[1.0, -0.2, 0.4], [-0.5, 0.7, 0.1], [0.2, 0.3, -0.9]])
    weights = torch.tensor([[0.2, 0.3, 0.5], [0.6, 0.1, 0.3], [1 / 3, 1 / 3, 1 / 3]])
    early = early_scalarized_ppo_loss(ratio, advantages, weights)
    late = late_weighted_ppo_loss(ratio, advantages, weights)
    assert torch.allclose(early, late, atol=1e-7, rtol=0.0)


def test_p0_losses_diverge_when_clipping_is_active_with_mixed_signs():
    ratio = torch.tensor([1.3])
    advantages = torch.tensor([[1.0, -1.0, 0.0]])
    weights = torch.tensor([[0.5, 0.5, 0.0]])
    early = early_scalarized_ppo_loss(ratio, advantages, weights)
    late = late_weighted_ppo_loss(ratio, advantages, weights)
    assert torch.allclose(early, torch.tensor(0.0), atol=1e-7)
    assert not torch.allclose(early, late, atol=1e-7)


def test_zero_weight_objective_has_no_late_loss_contribution():
    ratio = torch.tensor([1.3, 0.7])
    advantages = torch.tensor([[1.0, 2.0, 100.0], [2.0, -1.0, -100.0]])
    weights = torch.tensor([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]])
    reduced = advantages.clone()
    reduced[:, 2] = 0.0
    assert torch.allclose(
        late_weighted_ppo_loss(ratio, advantages, weights),
        late_weighted_ppo_loss(ratio, reduced, weights),
        atol=1e-7,
    )


def test_loss_gradients_are_finite_and_diagnostics_are_consistent():
    ratio = torch.tensor([0.8, 1.25], requires_grad=True)
    advantages = torch.tensor([[0.4, -0.2, 0.1], [-0.3, 0.7, 0.2]])
    weights = torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.8, 0.1]])
    diagnostics = p0_loss_diagnostics(ratio, advantages, weights)
    loss = diagnostics["weighted_actor_loss"]
    loss.backward()
    assert torch.isfinite(ratio.grad).all()
    assert torch.allclose(loss, late_weighted_ppo_loss(ratio.detach(), advantages, weights))
    assert 0.0 <= float(diagnostics["clip_fraction"]) <= 1.0
