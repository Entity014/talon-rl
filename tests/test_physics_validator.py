import os
import tempfile

import numpy as np

from rl.core.physics_validator import PhysicsValidator


def test_survival_counts_falls_and_caps_at_steps_for_survivors():
    """A lane that never falls should count as surviving the FULL rollout,
    not whatever step the loop happened to stop at by coincidence."""
    v = PhysicsValidator(num_envs=2, action_clip=3.0)
    action = np.zeros((2, 12))
    for step in range(5):
        done = np.array([step == 2, False])  # lane 0 falls at step 2, lane 1 never falls
        v.record({}, action, done)
    s = v.summary()
    assert s["steps"] == 5
    assert v.fall_step[0] == 2
    assert v.fall_step[1] == -1
    assert s["mean_survival"] == (2 + 5) / 2  # lane 0 capped at its fall step, lane 1 capped at total steps
    assert s["pct_full_survival"] == 50.0


def test_action_saturation_only_counts_alive_lanes():
    """A lane's action after it has already fallen (post-reset, arbitrary
    new-episode action) shouldn't count toward saturation stats for the
    episode that just ended."""
    v = PhysicsValidator(num_envs=2, action_clip=3.0)
    saturated_action = np.full((2, 12), 2.9)  # >= 0.95*3.0
    unsaturated_action = np.full((2, 12), 0.1)

    v.record({}, saturated_action, done=np.array([True, False]))  # lane 0 falls THIS step, still counts as alive
    v.record({}, unsaturated_action, done=np.array([False, False]))  # lane 0 now dead, shouldn't count

    s = v.summary()
    # Step 1 both alive (sat=1.0), step 2 only lane 1 alive (sat=0.0) -> mean of [1.0, 0.0]
    assert s["mean_action_saturation_pct"] == 50.0


def test_tracking_ratio_and_error_computed_from_v_actual_and_v_command():
    v = PhysicsValidator(num_envs=1, action_clip=3.0)
    action = np.zeros((1, 12))
    transition = {
        "v_actual": np.array([[0.25, 0.0, 0.0]]),
        "v_command": np.array([[0.5, 0.0, 0.0]]),
    }
    v.record(transition, action, done=np.array([False]))
    s = v.summary()
    assert s["tracking_ratio_pct"] == 50.0
    assert s["mean_tracking_error"] == 0.25


def test_undesired_contact_tracked_across_all_lanes_not_just_alive():
    """Unlike torque/roll_pitch/saturation (which reset per-episode
    relevance), undesired contact should be tracked for every lane every
    step regardless of alive status -- a fallen lane's post-reset dragging
    is still a real behavior worth counting."""
    v = PhysicsValidator(num_envs=2, action_clip=3.0)
    action = np.zeros((2, 12))
    v.record({"undesired_contact_count": np.array([1.0, 0.0])}, action, done=np.array([False, False]))
    v.record({"undesired_contact_count": np.array([0.0, 2.0])}, action, done=np.array([False, False]))
    s = v.summary()
    assert s["mean_undesired_contact_count"] == np.mean([1.0, 0.0, 0.0, 2.0])
    assert s["frac_steps_with_undesired_contact"] == 0.5  # 2 of 4 (lane,step) entries nonzero


def test_optional_fields_absent_do_not_crash_or_appear_in_summary():
    """DummyTalonEnv/an env missing some transition keys shouldn't crash
    the validator -- absent metrics are just omitted from the summary."""
    v = PhysicsValidator(num_envs=1, action_clip=3.0)
    action = np.zeros((1, 12))
    v.record({}, action, done=np.array([False]))
    s = v.summary()
    assert "max_torque_nm" not in s
    assert "tracking_ratio_pct" not in s
    assert "mean_undesired_contact_count" not in s
    v.print_summary()  # must not raise
    with tempfile.TemporaryDirectory() as tmp_dir:
        v.save_plots(tmp_dir)  # must not raise even with every optional series empty


def test_save_plots_writes_one_png_per_collected_series():
    v = PhysicsValidator(num_envs=1, action_clip=3.0)
    action = np.full((1, 12), 2.9)
    transition = {
        "joint_torque": np.full((1, 12), 10.0),
        "roll_pitch": np.zeros((1, 2)),
        "v_actual": np.array([[0.25, 0.0, 0.0]]),
        "v_command": np.array([[0.5, 0.0, 0.0]]),
        "undesired_contact_count": np.array([1.0]),
    }
    for _ in range(3):
        v.record(transition, action, done=np.array([False]))

    with tempfile.TemporaryDirectory() as tmp_dir:
        v.save_plots(tmp_dir)
        written = set(os.listdir(tmp_dir))
        assert "validate_action_saturation.png" in written
        # v_actual/v_command share ONE combined plot (easier to compare
        # tracking against target than two separately-scaled PNGs), not
        # separate validate_v_actual_x.png / validate_v_command_x.png files.
        assert "validate_velocity_tracking.png" in written
        assert "validate_v_actual_x.png" not in written
        assert "validate_v_command_x.png" not in written
        assert "validate_tracking_error.png" in written
        assert "validate_undesired_contact_count.png" in written
