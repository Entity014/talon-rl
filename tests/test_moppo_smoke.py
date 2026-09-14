"""End-to-end smoke test: does the whole MOPPO loop run on a vectorized
DummyTalonEnv without error, and do the reported numbers stay finite? This
is NOT a convergence test — the dummy env has no locomotion physics, so
"the policy got better" isn't a meaningful claim here. It only proves
obs -> policy(w) -> action -> reward vector -> vector critic -> PPO update
is wired correctly end to end, now with N parallel lanes and persistent
rollout collection across update() calls.
"""

import numpy as np

from talon_rl.config import ActionSpaceCfg, ObservationSpaceCfg, ObservationStackCfg, PreferenceCfg, RewardVectorCfg

from moppo.dummy_env import DummyTalonEnv
from moppo.moppo import MOPPOConfig, MOPPOTrainer


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
