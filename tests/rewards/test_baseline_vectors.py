"""Tests for baseline and bridge reward-vector semantics."""

# -----------------------------------------------------------------------------
# Former: test_b0_reward.py
# -----------------------------------------------------------------------------

import numpy as np
from talon_rl.rewards.baselines import b0_reward

def test_tracking_shape_and_symmetry():
    z=.32
    assert np.isclose(b0_reward(.5,0,0,z,z,False),1)
    assert np.isclose(b0_reward(.25,0,0,z,z,False),b0_reward(.75,0,0,z,z,False))
    assert np.isclose(b0_reward(0,0,0,z,z,False),np.exp(-4))
    assert b0_reward(-.5,0,0,z,z,False) < .001

def test_penalties_and_single_terminal_charge():
    z=.32; base=b0_reward(.5,0,0,z,z,False)
    assert b0_reward(.5,.1,0,z,z,False) < base
    assert b0_reward(.5,0,0,z+.1,z,False) < base
    assert np.isclose(b0_reward(.5,0,0,z,z,True)-base,-10)

# -----------------------------------------------------------------------------
# Former: test_m0_reward_vector.py
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# Former: test_v1b_s1_reward_vector.py
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# Former: test_v1b_s7_reward_vector.py
# -----------------------------------------------------------------------------

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
