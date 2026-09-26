import numpy as np

from talon_rl.curricula.command_schedule import G1CommandSchedule


def test_zero_hold_is_exactly_200_and_switch_is_pre_observation():
    s = G1CommandSchedule(1, np.random.default_rng(4), zero_probability=1.0)
    for step in range(199):
        e = s.before_observation()
        assert e.command[0].tolist() == [0.0, 0.0, 0.0]
        assert not e.progress_reset[0]
    e = s.before_observation()
    assert e.progress_reset[0]
    assert e.command[0, 0] != 0.0
    assert e.hold_step[0] == 200


def test_nonzero_lane_does_not_resample_and_done_lane_gets_fresh_draw():
    s = G1CommandSchedule(2, np.random.default_rng(8), zero_probability=0.0)
    first = s.before_observation().command.copy()
    for _ in range(20):
        assert np.array_equal(s.before_observation().command, first)
    s.reset_lanes(np.array([True, False]))
    after = s.before_observation().command
    assert not np.array_equal(after[0], first[0])
    assert np.array_equal(after[1], first[1])


def test_seed_reproducibility_and_lane_independence():
    a = G1CommandSchedule(8, np.random.default_rng(99))
    b = G1CommandSchedule(8, np.random.default_rng(99))
    for _ in range(240):
        assert np.array_equal(a.before_observation().command, b.before_observation().command)
    a.reset_lanes(np.array([True, False, False, False, False, False, False, False]))
    assert np.array_equal(a.before_observation().command[1:], b.before_observation().command[1:])
