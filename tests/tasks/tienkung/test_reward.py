import numpy as np

from talon_rl.tasks.manipulation.tienkung_env.config import RewardVectorCfg
from talon_rl.tasks.manipulation.tienkung_env.reward import (
    clearance_reward,
    compute_reward_vector,
    energy_reward,
    impact_reward,
    progress_reward,
    smoothness_reward,
)


def test_progress_reward_is_max_at_zero_error():
    height = np.array([1.0, 1.0, 1.0])
    target = np.array([1.0, 1.0, 1.0])
    r_same = progress_reward(height, target, std=0.3)
    r_diff = progress_reward(height, target * 2.0, std=0.3)
    assert r_same.shape == (3,)
    assert np.allclose(r_same, 1.0)
    assert np.all(r_diff < 1.0)


def test_clearance_reward_clips_to_unit_interval():
    dist = np.array([10.0, 0.0])
    r = clearance_reward(dist, safe_dist=0.5)
    assert r.shape == (2,)
    assert r[0] == 1.0
    assert r[1] == 0.0


def test_energy_reward_is_nonpositive():
    torque = np.array([[1.0, -2.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0] * 8])
    vel = np.ones((2, 8))
    r = energy_reward(torque, vel)
    assert r.shape == (2,)
    assert r[0] <= 0.0
    assert r[1] == 0.0


def test_impact_reward_penalizes_only_above_threshold():
    forces = np.array([[10.0, 10.0], [200.0, 0.0]])
    r = impact_reward(forces, threshold=50.0)
    assert r.shape == (2,)
    assert r[0] == 0.0
    assert r[1] < 0.0


def test_smoothness_reward_penalizes_action_change():
    a = np.zeros((2, 8))
    b = np.ones((2, 8))
    acc = np.zeros((2, 8))
    r_same = smoothness_reward(a, a, acc)
    r_diff = smoothness_reward(b, a, acc)
    assert r_same.shape == (2,)
    assert np.allclose(r_same, 0.0)
    assert np.all(r_diff < 0.0)


def test_compute_reward_vector_respects_active_mask_and_order():
    cfg = RewardVectorCfg(active=(True, False, True, False, True))
    n = 4
    transition = {
        "obs": np.zeros((n, 1)),
        "box_height": np.ones(n), "target_height": np.ones(n),
        "obstacle_dist": np.ones(n),
        "joint_torque": np.ones((n, 8)), "joint_vel": np.ones((n, 8)),
        "arm_contact_force": np.zeros((n, 2)),
        "action": np.zeros((n, 8)), "prev_action": np.zeros((n, 8)), "joint_acc": np.zeros((n, 8)),
    }
    r = compute_reward_vector(transition, cfg)
    assert r.shape == (n, 5)
    assert np.allclose(r[:, 1], 0.0)  # clearance masked off
    assert np.allclose(r[:, 3], 0.0)  # impact masked off
    assert np.all(r[:, 0] != 0.0)     # progress active
