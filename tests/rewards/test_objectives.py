import numpy as np
from talon_rl.rewards.objectives import (
    OBJECTIVE_ORDER,NORMALIZATION_DIVISORS,raw_objective_vector,
    normalize_objectives,normalized_objective_vector,scalarize,
    REGULARIZER_TERMS,CONSTRAINT_TERMS,
)

def sample():
    return {
      "track_lin_vel_xy_exp":np.array([1.0,1.5],np.float32),
      "track_ang_vel_z_exp":np.array([0.5,0.25],np.float32),
      "ang_vel_xy_l2":np.array([-0.1,-0.2],np.float32),
      "flat_orientation_l2":np.array([-0.01,-0.02],np.float32),
      "action_rate_l2":np.array([-0.04,-0.08],np.float32),
      "dof_torques_l2":np.array([-0.07,-0.08],np.float32),
      "lin_vel_z_l2":np.array([-0.05,-0.04],np.float32),
    }

def test_frozen_order_and_roles():
    assert OBJECTIVE_ORDER==("velocity_tracking","angular_stability","orientation_stability","control_smoothness")
    assert "effort" not in OBJECTIVE_ORDER and "effort" in REGULARIZER_TERMS
    assert "vertical_stability" not in OBJECTIVE_ORDER and "vertical_stability" in CONSTRAINT_TERMS
    assert np.all(NORMALIZATION_DIVISORS>0)

def test_raw_grouping_exact():
    r=raw_objective_vector(sample())
    expected=np.array([[1.5,-.1,-.01,-.04],[1.75,-.2,-.02,-.08]],np.float32)
    assert np.allclose(r,expected,atol=0,rtol=0)

def test_normalization_and_uniform_contribution_deterministic():
    r=raw_objective_vector(sample());n=normalize_objectives(r)
    assert np.allclose(n,r/NORMALIZATION_DIVISORS)
    assert np.allclose(normalized_objective_vector(sample()),n)
    w=np.full((2,4),.25,np.float32)
    assert np.allclose(scalarize(n,w),np.sum(n*w,axis=-1))

def test_higher_is_better_penalty_convention():
    # Less-negative penalty reward must be better after positive normalization.
    r=np.array([[1.,-.2,-.02,-.1],[1.,-.1,-.01,-.05]],np.float32)
    n=normalize_objectives(r)
    assert np.all(n[1,1:]>n[0,1:])


def test_shared_substrate_is_preference_invariant_and_never_an_objective():
    """F3: R_shared must add the same amount to w^T R for every preference on
    the simplex, must not touch objective terms, and 'none' must be exactly
    zero so B0 is the unchanged V4-C3 trainer."""
    from talon_rl.rewards.objectives import OBJECTIVE_TERMS, SHARED_ARMS, SHARED_SCALE, shared_vector
    names = ["track_lin_vel_xy_exp", "track_ang_vel_z_exp", "lin_vel_z_l2", "ang_vel_xy_l2", "dof_torques_l2",
             "dof_acc_l2", "action_rate_l2", "feet_air_time", "flat_orientation_l2", "dof_pos_limits"]
    objective_terms = {t for ts in OBJECTIVE_TERMS.values() for t in ts}
    assert not np.any(shared_vector(names, "none"))
    for arm, terms in SHARED_ARMS.items():
        v = shared_vector(names, arm)
        assert not objective_terms & set(terms)
        assert {names[i] for i in np.flatnonzero(v)} == set(terms)
        assert np.allclose(v[v != 0], SHARED_SCALE)
    raw = np.random.default_rng(0).normal(size=len(names))
    v = shared_vector(names, "all")
    for w in ([1, 0, 0], [1 / 3] * 3, [.15, .7, .15], [0, .3, .7]):
        # each objective row gets +v; weights sum to one, so the gain is v.raw for every w
        assert np.isclose(sum(wi * (raw @ v) for wi in w), raw @ v)
