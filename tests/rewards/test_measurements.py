"""Raw measurement library (talon_rl/rewards/measurements.py): the pure pieces
of the stateful logger. Each test names the mistake it prevents."""
import torch

from talon_rl.rewards.measurements import MEASUREMENT_LIBRARY, NAMES, foot_slip, impact_sq, joint_power_abs, vertical_acc_sq


def test_vertical_acceleration_is_a_finite_difference_over_the_policy_step():
    # not v_z^2 (that is lin_vel_z_l2) and not divided by the physics dt
    assert torch.allclose(vertical_acc_sq(torch.tensor([0.3]), torch.tensor([0.1]), 0.02), torch.tensor([100.0]))


def test_slip_counts_only_feet_in_contact():
    vel = torch.tensor([[[3.0, 4.0], [1.0, 0.0]]])
    force = torch.tensor([[[0.0, 0.0, 10.0], [0.0, 0.0, 0.5]]])  # second foot below the 1 N contact threshold
    assert torch.allclose(foot_slip(vel, force), torch.tensor([5.0]))


def test_impact_sums_squared_force_change_over_feet():
    f0 = torch.zeros(1, 2, 3); f1 = torch.tensor([[[0.0, 0.0, 3.0], [4.0, 0.0, 0.0]]])
    assert torch.allclose(impact_sq(f1, f0), torch.tensor([25.0]))


def test_power_is_absolute_so_negative_work_is_not_a_reward():
    assert torch.allclose(joint_power_abs(torch.tensor([[2.0, -3.0]]), torch.tensor([[1.0, 1.0]])), torch.tensor([5.0]))


def test_every_library_entry_has_a_role_and_logger_column_order_is_the_library_order():
    assert NAMES == tuple(MEASUREMENT_LIBRARY) and all(len(v) == 3 for v in MEASUREMENT_LIBRARY.values())


def test_height_oscillation_ignores_a_steady_offset():
    # a constant height (any posture) converges to zero oscillation; only motion about the mean counts
    from talon_rl.rewards.measurements import ema_update
    m = torch.tensor([0.42])
    for _ in range(500):
        m = ema_update(m, torch.tensor([0.25]), 0.02)
    assert torch.allclose((torch.tensor([0.25]) - m) ** 2, torch.tensor([0.0]), atol=1e-12)
