import numpy as np

from talon_rl.config import RewardVectorCfg
from talon_rl.reward import (
    clearance_reward,
    compute_reward_vector,
    energy_reward,
    impact_reward,
    progress_reward,
    smoothness_reward,
)


def test_progress_reward_is_max_at_zero_error():
    v = np.array([0.5, 0.0, 0.0])
    assert progress_reward(v, v, std=0.5) == 1.0
    assert progress_reward(v, v * 2, std=0.5) < 1.0


def test_clearance_reward_clips_to_unit_interval():
    assert clearance_reward(obstacle_dist=10.0, safe_dist=0.5) == 1.0
    assert clearance_reward(obstacle_dist=0.0, safe_dist=0.5) == 0.0


def test_energy_reward_is_nonpositive():
    torque = np.array([1.0, -2.0, 0.5])
    vel = np.array([1.0, 1.0, 1.0])
    assert energy_reward(torque, vel) <= 0.0
    assert energy_reward(np.zeros(3), vel) == 0.0


def test_impact_reward_penalizes_only_above_threshold():
    below = np.array([10.0, 10.0, 10.0, 10.0])
    above = np.array([200.0, 0.0, 0.0, 0.0])
    assert impact_reward(below, threshold=50.0) == 0.0
    assert impact_reward(above, threshold=50.0) < 0.0


def test_smoothness_reward_penalizes_action_change():
    a = np.zeros(12)
    b = np.ones(12)
    acc = np.zeros(12)
    assert smoothness_reward(a, a, acc) == 0.0
    assert smoothness_reward(b, a, acc) < 0.0


def test_compute_reward_vector_respects_active_mask_and_order():
    cfg = RewardVectorCfg(active=(True, False, True, False, True))
    transition = {
        "v_actual": np.zeros(3), "v_command": np.zeros(3),
        "obstacle_dist": 1.0,
        "joint_torque": np.ones(12), "joint_vel": np.ones(12),
        "foot_contact_force": np.zeros(4),
        "action": np.zeros(12), "prev_action": np.zeros(12), "joint_acc": np.zeros(12),
    }
    r = compute_reward_vector(transition, cfg)
    assert r.shape == (5,)
    assert r[1] == 0.0  # clearance masked off
    assert r[3] == 0.0  # impact masked off
    assert r[0] != 0.0  # progress active (max reward since v_actual==v_command==0 -> exp(0)=1)
