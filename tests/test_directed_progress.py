import numpy as np

from talon_rl.rewards.directed_progress import DirectedProgressState, WINDOW_SECONDS, update


def _state_at_origin(n=1, command_x=0.5):
    return DirectedProgressState.initial(
        root_xy=np.zeros((n, 2)), heading=np.zeros(n), command_x=np.full(n, command_x),
    )


def test_stationary_lane_scores_zero():
    state = _state_at_origin(command_x=0.5)
    dp, _ = update(state, root_xy=np.zeros((1, 2)), heading=np.zeros(1), command_x=np.full(1, 0.5), reset_mask=np.zeros(1, dtype=bool))
    assert np.allclose(dp, 0.0)


def test_exact_forward_tracking_saturates_to_one():
    """A lane moving exactly at the commanded speed for the full window
    (0.5s @ 0.5 m/s = 0.25m) must read ~1.0 -- the ceiling of the [0,1] range."""
    command_x = 0.5
    state = _state_at_origin(command_x=command_x)
    expected_distance = command_x * WINDOW_SECONDS  # 0.25 m
    root_xy = np.array([[expected_distance, 0.0]])
    dp, _ = update(state, root_xy=root_xy, heading=np.zeros(1), command_x=np.full(1, command_x), reset_mask=np.zeros(1, dtype=bool))
    assert np.allclose(dp, 1.0, atol=1e-6)


def test_half_speed_progress_scores_half():
    command_x = 0.5
    state = _state_at_origin(command_x=command_x)
    half_distance = 0.5 * command_x * WINDOW_SECONDS
    root_xy = np.array([[half_distance, 0.0]])
    dp, _ = update(state, root_xy=root_xy, heading=np.zeros(1), command_x=np.full(1, command_x), reset_mask=np.zeros(1, dtype=bool))
    assert np.allclose(dp, 0.5, atol=1e-4)


def test_wrong_direction_scores_zero():
    command_x = 0.5
    state = _state_at_origin(command_x=command_x)
    root_xy = np.array([[-0.1, 0.0]])  # moved backward under a forward command
    dp, _ = update(state, root_xy=root_xy, heading=np.zeros(1), command_x=np.full(1, command_x), reset_mask=np.zeros(1, dtype=bool))
    assert np.allclose(dp, 0.0)


def test_reverse_command_exact_tracking_saturates_to_one():
    """Sign convention: a reverse command (c_x<0) must reward NEGATIVE
    displacement, not positive -- a sign error here would reward driving
    forward under a "go backward" command."""
    command_x = -0.5
    state = _state_at_origin(command_x=command_x)
    expected_distance = command_x * WINDOW_SECONDS  # negative: -0.25 m
    root_xy = np.array([[expected_distance, 0.0]])
    dp, _ = update(state, root_xy=root_xy, heading=np.zeros(1), command_x=np.full(1, command_x), reset_mask=np.zeros(1, dtype=bool))
    assert np.allclose(dp, 1.0, atol=1e-6)

    # moving forward under the same reverse command must score 0, not the mirrored value
    wrong_way = np.array([[abs(expected_distance), 0.0]])
    dp_wrong, _ = update(state, root_xy=wrong_way, heading=np.zeros(1), command_x=np.full(1, command_x), reset_mask=np.zeros(1, dtype=bool))
    assert np.allclose(dp_wrong, 0.0)


def test_zero_command_always_scores_zero_regardless_of_motion():
    state = _state_at_origin(command_x=0.0)
    root_xy = np.array([[1.0, 1.0]])  # drifted a lot, but there's no command to be "directed" toward
    dp, _ = update(state, root_xy=root_xy, heading=np.zeros(1), command_x=np.zeros(1), reset_mask=np.zeros(1, dtype=bool))
    assert np.allclose(dp, 0.0)


def test_command_change_resets_history_without_spiking():
    """A lane that had already covered the old command's full window
    (dp~1.0) must NOT carry that displacement over when the command
    changes -- the new window starts at the CURRENT position, so the very
    step the command changes must read exactly 0, not some stale nonzero
    value from before the change."""
    command_x = 0.5
    state = _state_at_origin(command_x=command_x)
    saturated_xy = np.array([[command_x * WINDOW_SECONDS, 0.0]])
    dp_before, state_after = update(state, root_xy=saturated_xy, heading=np.zeros(1), command_x=np.full(1, command_x), reset_mask=np.zeros(1, dtype=bool))
    assert np.allclose(dp_before, 1.0, atol=1e-6)

    new_command_x = 0.75
    dp_on_change, state_reset = update(
        state_after, root_xy=saturated_xy, heading=np.zeros(1), command_x=np.full(1, new_command_x), reset_mask=np.zeros(1, dtype=bool),
    )
    assert np.allclose(dp_on_change, 0.0)  # no spike/carryover on the change step
    assert np.allclose(state_reset.origin_xy, saturated_xy)  # window re-anchored HERE, not the old origin
    assert np.allclose(state_reset.origin_command_x, new_command_x)


def test_episode_reset_resets_history_without_spiking():
    command_x = 0.5
    state = _state_at_origin(command_x=command_x)
    saturated_xy = np.array([[command_x * WINDOW_SECONDS, 0.0]])
    dp_before, state_after = update(state, root_xy=saturated_xy, heading=np.zeros(1), command_x=np.full(1, command_x), reset_mask=np.zeros(1, dtype=bool))
    assert np.allclose(dp_before, 1.0, atol=1e-6)

    # lane reset (terminated + respawned) at a totally different position/heading, same command
    respawn_xy = np.array([[3.0, -2.0]])
    respawn_heading = np.array([1.2])
    dp_reset, state_reset = update(
        state_after, root_xy=respawn_xy, heading=respawn_heading, command_x=np.full(1, command_x), reset_mask=np.ones(1, dtype=bool),
    )
    assert np.allclose(dp_reset, 0.0)
    assert np.allclose(state_reset.origin_xy, respawn_xy)
    assert np.allclose(state_reset.origin_heading, respawn_heading)


def test_projection_uses_heading_at_window_start_not_current_heading():
    """A lane that's since turned must still be scored against the
    direction it faced when the window opened -- using CURRENT heading
    instead would let a lane game the projection by turning toward
    whichever direction it drifted, regardless of the command."""
    command_x = 0.5
    # window opened facing +x (heading=0)
    state = DirectedProgressState.initial(root_xy=np.zeros((1, 2)), heading=np.zeros(1), command_x=np.full(1, command_x))
    # lane actually displaced along +x (correct for heading=0), but report a
    # very different CURRENT heading (e.g. it spun around) -- must not affect the projection
    root_xy = np.array([[command_x * WINDOW_SECONDS, 0.0]])
    current_heading_spun = np.array([np.pi])  # facing backward now, irrelevant to the projection
    dp, _ = update(state, root_xy=root_xy, heading=current_heading_spun, command_x=np.full(1, command_x), reset_mask=np.zeros(1, dtype=bool))
    assert np.allclose(dp, 1.0, atol=1e-6)  # scored against origin heading (0), not current (pi)


def test_output_is_clipped_to_unit_interval_on_overshoot():
    command_x = 0.5
    state = _state_at_origin(command_x=command_x)
    overshoot_xy = np.array([[10 * command_x * WINDOW_SECONDS, 0.0]])
    dp, _ = update(state, root_xy=overshoot_xy, heading=np.zeros(1), command_x=np.full(1, command_x), reset_mask=np.zeros(1, dtype=bool))
    assert np.allclose(dp, 1.0)


def test_batched_lanes_are_independent():
    n = 3
    state = DirectedProgressState.initial(root_xy=np.zeros((n, 2)), heading=np.zeros(n), command_x=np.array([0.5, -0.5, 0.0]))
    root_xy = np.array([
        [0.5 * WINDOW_SECONDS, 0.0],   # lane 0: exact forward tracking -> 1.0
        [0.0, 0.0],                    # lane 1: stationary under reverse command -> 0.0
        [5.0, 5.0],                    # lane 2: zero command, drifted -> 0.0
    ])
    dp, _ = update(state, root_xy=root_xy, heading=np.zeros(n), command_x=np.array([0.5, -0.5, 0.0]), reset_mask=np.zeros(n, dtype=bool))
    assert np.allclose(dp, [1.0, 0.0, 0.0], atol=1e-6)
