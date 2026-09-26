import numpy as np

from talon_rl.rewards.baselines import reconstruct_stock_scalar
from talon_rl.rewards.baselines import group_v1b_s1_terms, reconstruct_v1b_s1_scalar


def test_s1_regrouping_preserves_stock_scalar():
    terms = {
        "track_lin_vel_xy_exp": np.array([1.0, 2.0]),
        "track_ang_vel_z_exp": np.array([0.5, 0.25]),
        "lin_vel_z_l2": np.array([-0.2, -0.1]),
        "ang_vel_xy_l2": np.array([-0.03, -0.04]),
        "dof_torques_l2": np.array([-0.001, -0.002]),
        "dof_acc_l2": np.array([-0.0001, -0.0002]),
        "action_rate_l2": np.array([-0.01, -0.02]),
        "feet_air_time": np.array([0.1, 0.2]),
        "flat_orientation_l2": np.array([-0.3, -0.4]),
        "dof_pos_limits": np.zeros(2),
    }
    grouped = group_v1b_s1_terms(terms)
    assert np.array_equal(reconstruct_v1b_s1_scalar(grouped), reconstruct_stock_scalar(grouped))
    assert np.allclose(grouped[:, 4], terms["action_rate_l2"])


def test_s1_limits_is_informative_when_action_rate_is_active():
    terms = {"action_rate_l2": np.array([-0.1, -0.2]), "dof_pos_limits": np.zeros(2)}
    grouped = group_v1b_s1_terms(terms)
    assert np.count_nonzero(grouped[:, 4]) == 2
    assert np.std(grouped[:, 4]) > 0.0
