"""End-to-end smoke test: does the whole MOPPO loop run on a vectorized
DummyTalonEnv without error, and do the reported numbers stay finite? This
is NOT a convergence test — the dummy env has no locomotion physics, so
"the policy got better" isn't a meaningful claim here. It only proves
obs -> policy(w) -> action -> reward vector -> vector critic -> PPO update
is wired correctly end to end, now with N parallel lanes and persistent
rollout collection across update() calls.
"""

import os
import tempfile

import numpy as np
import torch

from talon_rl.config import ActionSpaceCfg, ObservationSpaceCfg, ObservationStackCfg, PreferenceCfg, RewardVectorCfg

from rl.core.algorithms.moppo import MOPPOConfig, MOPPOTrainer
from rl.core.dummy_env import DummyTalonEnv


def test_moppo_runs_a_few_updates_without_nans():
    obs_cfg = ObservationSpaceCfg()
    action_cfg = ActionSpaceCfg()
    reward_cfg = RewardVectorCfg()
    pref_cfg = PreferenceCfg()

    env = DummyTalonEnv(obs_cfg, action_cfg, num_envs=8, horizon=40, seed=0)
    trainer = MOPPOTrainer(
        env, obs_cfg, reward_cfg, pref_cfg,
        moppo_cfg=MOPPOConfig(num_steps=10, epochs_per_update=2),
        seed=0,
    )

    for _ in range(3):
        stats = trainer.update()
        assert np.isfinite(stats["policy_loss"])
        assert np.isfinite(stats["value_loss"])
        assert np.all(np.isfinite(stats["mean_reward_vec"]))
        assert stats["mean_reward_vec"].shape == (reward_cfg.dim,)


def test_moppo_runs_with_mismatched_actor_critic_stacks():
    """Exercises the Flamingo-style num_policy_stacks != num_critic_stacks path —
    actor sees 4 frames of history, critic sees only the current frame."""
    obs_cfg = ObservationSpaceCfg()
    action_cfg = ActionSpaceCfg()
    reward_cfg = RewardVectorCfg()
    pref_cfg = PreferenceCfg()
    stack_cfg = ObservationStackCfg(num_policy_stacks=4, num_critic_stacks=1)

    env = DummyTalonEnv(obs_cfg, action_cfg, num_envs=4, horizon=40, seed=0)
    trainer = MOPPOTrainer(
        env, obs_cfg, reward_cfg, pref_cfg,
        moppo_cfg=MOPPOConfig(num_steps=10, epochs_per_update=2),
        stack_cfg=stack_cfg,
        seed=0,
    )

    stats = trainer.update()
    assert np.isfinite(stats["policy_loss"])
    assert np.isfinite(stats["value_loss"])


def test_rollout_is_persistent_across_update_calls():
    """The rollout must continue where the previous update() left off, not
    reset every call — regression test for the episode-based -> persistent
    rollout behavioral change (see design doc's moppo.py section)."""
    obs_cfg = ObservationSpaceCfg()
    action_cfg = ActionSpaceCfg()
    reward_cfg = RewardVectorCfg()
    pref_cfg = PreferenceCfg()

    env = DummyTalonEnv(obs_cfg, action_cfg, num_envs=2, horizon=1000, seed=0)
    trainer = MOPPOTrainer(
        env, obs_cfg, reward_cfg, pref_cfg,
        moppo_cfg=MOPPOConfig(num_steps=5, epochs_per_update=1),
        seed=0,
    )
    t_before = trainer._t  # internal step counter this task's implementation must expose
    trainer.update()
    t_after = trainer._t
    assert t_after == t_before + 5  # advanced by exactly num_steps, not reset to 0


def test_save_load_round_trips_model_and_optimizer_state():
    """A checkpoint saved from one trainer must reproduce identical model
    weights (and optimizer state) in a FRESH trainer instance — proves the
    state actually transferred through the file, not just "didn't crash"."""
    obs_cfg = ObservationSpaceCfg()
    action_cfg = ActionSpaceCfg()
    reward_cfg = RewardVectorCfg()
    pref_cfg = PreferenceCfg()
    moppo_cfg = MOPPOConfig(num_steps=5, epochs_per_update=1)

    env = DummyTalonEnv(obs_cfg, action_cfg, num_envs=4, horizon=40, seed=0)
    trainer = MOPPOTrainer(env, obs_cfg, reward_cfg, pref_cfg, moppo_cfg=moppo_cfg, seed=0)
    trainer.update()  # advance weights away from their initial random values

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = os.path.join(tmp_dir, "checkpoint.pt")
        trainer.save(ckpt_path)

        # A fresh trainer — different seed, so its initial (pre-load) weights
        # are NOT the same as trainer's, ruling out a test that would pass
        # even if load() were a no-op.
        env2 = DummyTalonEnv(obs_cfg, action_cfg, num_envs=4, horizon=40, seed=1)
        fresh_trainer = MOPPOTrainer(env2, obs_cfg, reward_cfg, pref_cfg, moppo_cfg=moppo_cfg, seed=1)
        for p1, p2 in zip(trainer.model.parameters(), fresh_trainer.model.parameters()):
            assert not torch.equal(p1, p2)  # sanity: genuinely different before load

        fresh_trainer.load(ckpt_path)

        for p1, p2 in zip(trainer.model.parameters(), fresh_trainer.model.parameters()):
            assert torch.equal(p1, p2)

        # Optimizer state round-trips too (same param groups' state keys).
        assert trainer.optim.state_dict()["param_groups"] == fresh_trainer.optim.state_dict()["param_groups"]


def test_save_creates_parent_directories():
    """train_prelim.py's --save_path may point at a directory that doesn't
    exist yet (e.g. a fresh logs/<run>/ dir) — save() must create it."""
    obs_cfg = ObservationSpaceCfg()
    action_cfg = ActionSpaceCfg()
    reward_cfg = RewardVectorCfg()
    pref_cfg = PreferenceCfg()

    env = DummyTalonEnv(obs_cfg, action_cfg, num_envs=2, horizon=40, seed=0)
    trainer = MOPPOTrainer(
        env, obs_cfg, reward_cfg, pref_cfg, moppo_cfg=MOPPOConfig(num_steps=5, epochs_per_update=1), seed=0
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = os.path.join(tmp_dir, "nested", "run_dir", "checkpoint.pt")
        trainer.save(ckpt_path)
        assert os.path.isfile(ckpt_path)
