import numpy as np

from talon_rl.rewards.baselines import (
    S7_OBJECTIVE_ORDER,
    group_v1b_s7_terms,
    reconstruct_v1b_s7_scalar,
)


def test_s7_pass3_mapping_reconstructs_stock_scalar_and_excludes_zero_limit_term():
    terms = {
        "track_lin_vel_xy_exp": np.array([1.0, 2.0]),
        "track_ang_vel_z_exp": np.array([0.5, 0.25]),
        "lin_vel_z_l2": np.array([-0.2, -0.3]),
        "ang_vel_xy_l2": np.array([-0.1, -0.2]),
        "flat_orientation_l2": np.array([-0.4, -0.5]),
        "feet_air_time": np.array([0.05, 0.0]),
        "dof_torques_l2": np.array([-0.01, -0.02]),
        "dof_acc_l2": np.array([-0.03, -0.04]),
        "action_rate_l2": np.array([-0.05, -0.06]),
        "dof_pos_limits": np.zeros(2),
    }
    grouped = group_v1b_s7_terms(terms)
    assert S7_OBJECTIVE_ORDER == ("progress", "balance", "efficiency")
    np.testing.assert_allclose(reconstruct_v1b_s7_scalar(grouped), sum(terms.values()))
    assert grouped.shape == (2, 3)
