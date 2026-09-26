"""mdp.terrain_levels_vel's promotion-dwell requirement (added 2026-09-18)
lives in talon_rl.curricula.streak as pure tensor math specifically so it
can be tested without importing the a1_env package, which requires a live
Isaac Sim `carb` context just to import (see
talon_rl/tasks/locomotion/a1_env/__init__.py) -- test_a1_env.py is the
GPU/Isaac-Sim-gated integration test, this is the logic-only unit test.
"""

import torch

from talon_rl.curricula.streak import apply_promote_streak

REQUIRED = 3


def test_move_up_requires_required_consecutive_qualifying_episodes():
    """A single qualifying episode must NOT promote -- promoting off one
    lucky rollout (the pre-fix behavior) risked promoting a policy before
    its success was a consolidated skill, a candidate cause of the
    spike-then-collapse pattern seen in phase1_longrun."""
    streak = torch.zeros(1, dtype=torch.long)
    for i in range(REQUIRED - 1):
        streak, move_up = apply_promote_streak(streak, torch.tensor([True]), REQUIRED)
        assert not move_up[0], f"promoted too early on success #{i + 1}"
    streak, move_up = apply_promote_streak(streak, torch.tensor([True]), REQUIRED)
    assert move_up[0], "should promote on the Nth consecutive qualifying episode"
    assert streak[0] == 0, "streak must reset after promotion so the next level starts a fresh dwell"


def test_a_non_qualifying_episode_resets_the_streak():
    streak = torch.zeros(1, dtype=torch.long)
    streak, _ = apply_promote_streak(streak, torch.tensor([True]), REQUIRED)
    assert streak[0] == 1
    streak, move_up = apply_promote_streak(streak, torch.tensor([False]), REQUIRED)
    assert streak[0] == 0, "a fall or short episode must reset the streak, not just fail to increment it"
    assert not move_up[0]


def test_streaks_are_independent_per_env():
    streak = torch.zeros(2, dtype=torch.long)
    qualifies = torch.tensor([True, False])
    for _ in range(REQUIRED - 1):
        streak, move_up = apply_promote_streak(streak, qualifies, REQUIRED)
        assert not move_up.any()
    streak, move_up = apply_promote_streak(streak, qualifies, REQUIRED)
    assert move_up.tolist() == [True, False]
