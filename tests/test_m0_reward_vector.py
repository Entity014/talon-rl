import numpy as np
from talon_rl.rewards.baselines import group_stock_terms, reconstruct_stock_scalar

def test_stock_terms_reconstruct_exactly_with_disabled_contact_term():
    terms = {
        "track_lin_vel_xy_exp": np.array([1., 2.]),
        "track_ang_vel_z_exp": np.array([.5, .25]),
        "lin_vel_z_l2": np.array([-1., -2.]),
        "ang_vel_xy_l2": np.array([-.1, -.2]),
        "dof_torques_l2": np.array([-.01, -.02]),
        "dof_acc_l2": np.array([-.001, -.002]),
        "action_rate_l2": np.array([-.03, -.04]),
        "feet_air_time": np.array([.1, .2]),
        "flat_orientation_l2": np.array([-.4, -.5]),
        "dof_pos_limits": np.zeros(2),
    }
    vector = group_stock_terms(terms)
    np.testing.assert_allclose(reconstruct_stock_scalar(vector), sum(terms.values()), atol=1e-6)

def test_disabled_terms_are_zero_and_shape_preserved():
    terms = {"track_lin_vel_xy_exp": np.ones((2, 3))}
    vector = group_stock_terms(terms, shape=(2, 3))
    assert vector.shape == (2, 3, 5)
    np.testing.assert_allclose(vector[..., 0], 1.)
    np.testing.assert_allclose(vector[..., 2:], 0.)
