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


def test_sample_diversity_w_respects_floor_clip_invariant():
    # D3PO's diversity regularizer needs a second preference vector w' — it
    # must go through the SAME floor_clip pipeline as the real w, or it
    # silently violates the w_impact >= eps invariant the rest of the
    # system depends on (flagged explicitly in the D3PO literature review).
    obs_cfg = ObservationSpaceCfg()
    action_cfg = ActionSpaceCfg()
    reward_cfg = RewardVectorCfg()
    pref_cfg = PreferenceCfg()

    env = DummyTalonEnv(obs_cfg, action_cfg, num_envs=4, horizon=40, seed=0)
    trainer = MOPPOTrainer(env, obs_cfg, reward_cfg, pref_cfg, seed=0)

    w_prime = trainer._sample_diversity_w(100)
    impact_idx = reward_cfg.term_names.index("impact")
    assert np.all(w_prime[:, impact_idx] >= reward_cfg.impact_floor_eps - 1e-6)
    assert np.allclose(w_prime.sum(axis=-1), 1.0, atol=1e-5)


def test_d3po_update_produces_finite_losses():
    """MOPPOTrainer.update() now runs D3PO's LSW + diversity regularizer
    (not AMOR early-scalarization) — must still produce finite, sane loss
    values end to end on the dummy env."""
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


def test_reward_normalization_bounds_reward_magnitude():
    """CLAUDE.md/docs/mdp.md document that smoothness sits around -300 while
    progress sits around 0-1 in raw scale — running per-objective
    normalization must bring every stored reward within the normalizer's
    clip range regardless of that raw-scale gap."""
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
        assert np.all(np.abs(stats["mean_reward_vec"]) <= 10.0)


def test_reward_norm_state_round_trips_through_checkpoint():
    """A resumed run must keep the same reward scale as the run it resumes
    from — reloading fresh (unfit) normalizer stats would shock the reward
    magnitude the value function was trained against."""
    obs_cfg = ObservationSpaceCfg()
    action_cfg = ActionSpaceCfg()
    reward_cfg = RewardVectorCfg()
    pref_cfg = PreferenceCfg()
    moppo_cfg = MOPPOConfig(num_steps=5, epochs_per_update=1)

    env = DummyTalonEnv(obs_cfg, action_cfg, num_envs=4, horizon=40, seed=0)
    trainer = MOPPOTrainer(env, obs_cfg, reward_cfg, pref_cfg, moppo_cfg=moppo_cfg, seed=0)
    trainer.update()
    trainer.update()  # accumulate non-trivial running stats

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = os.path.join(tmp_dir, "checkpoint.pt")
        trainer.save(ckpt_path)

        env2 = DummyTalonEnv(obs_cfg, action_cfg, num_envs=4, horizon=40, seed=1)
        fresh_trainer = MOPPOTrainer(env2, obs_cfg, reward_cfg, pref_cfg, moppo_cfg=moppo_cfg, seed=1)
        assert not np.allclose(trainer.reward_norm.mean, fresh_trainer.reward_norm.mean)

        fresh_trainer.load(ckpt_path)
        assert np.allclose(trainer.reward_norm.mean, fresh_trainer.reward_norm.mean)
        assert np.allclose(trainer.reward_norm.var, fresh_trainer.reward_norm.var)
        assert trainer.reward_norm.count == fresh_trainer.reward_norm.count


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


from talon_rl.config import ExtrinsicsCfg


class _ExtrinsicsDummyEnv(DummyTalonEnv):
    """DummyTalonEnv plus a fake extrinsics vector in the transition dict —
    exercises MOPPOTrainer's encoder wiring without needing Isaac Sim."""

    def __init__(self, *args, extrinsics_dim: int, **kwargs):
        super().__init__(*args, **kwargs)
        self._extrinsics_dim = extrinsics_dim

    def _with_extrinsics(self, transition):
        transition["extrinsics"] = np.random.randn(self.num_envs, self._extrinsics_dim).astype(np.float32)
        return transition

    def reset(self):
        return self._with_extrinsics(super().reset())

    def step(self, action):
        transition, done = super().step(action)
        return self._with_extrinsics(transition), done


def test_encoder_concatenates_z_t_into_actor_obs():
    obs_cfg = ObservationSpaceCfg()
    action_cfg = ActionSpaceCfg()
    reward_cfg = RewardVectorCfg()
    pref_cfg = PreferenceCfg()
    extrinsics_cfg = ExtrinsicsCfg()

    env = _ExtrinsicsDummyEnv(obs_cfg, action_cfg, num_envs=4, horizon=40, seed=0, extrinsics_dim=extrinsics_cfg.dim)
    trainer = MOPPOTrainer(
        env, obs_cfg, reward_cfg, pref_cfg,
        moppo_cfg=MOPPOConfig(num_steps=5, epochs_per_update=1),
        extrinsics_cfg=extrinsics_cfg,
        seed=0,
    )
    expected_actor_dim = trainer.stack.policy_obs_dim + reward_cfg.dim + extrinsics_cfg.adaptation_latent_dim
    assert trainer.model.actor_body[0].in_features == expected_actor_dim


def test_encoder_params_are_in_the_optimizer():
    obs_cfg = ObservationSpaceCfg()
    action_cfg = ActionSpaceCfg()
    reward_cfg = RewardVectorCfg()
    pref_cfg = PreferenceCfg()
    extrinsics_cfg = ExtrinsicsCfg()

    env = _ExtrinsicsDummyEnv(obs_cfg, action_cfg, num_envs=4, horizon=40, seed=0, extrinsics_dim=extrinsics_cfg.dim)
    trainer = MOPPOTrainer(
        env, obs_cfg, reward_cfg, pref_cfg,
        moppo_cfg=MOPPOConfig(num_steps=5, epochs_per_update=1),
        extrinsics_cfg=extrinsics_cfg,
        seed=0,
    )
    optim_param_ids = {id(p) for group in trainer.optim.param_groups for p in group["params"]}
    for p in trainer.encoder.parameters():
        assert id(p) in optim_param_ids


def test_update_runs_end_to_end_with_encoder():
    obs_cfg = ObservationSpaceCfg()
    action_cfg = ActionSpaceCfg()
    reward_cfg = RewardVectorCfg()
    pref_cfg = PreferenceCfg()
    extrinsics_cfg = ExtrinsicsCfg()

    env = _ExtrinsicsDummyEnv(obs_cfg, action_cfg, num_envs=4, horizon=40, seed=0, extrinsics_dim=extrinsics_cfg.dim)
    trainer = MOPPOTrainer(
        env, obs_cfg, reward_cfg, pref_cfg,
        moppo_cfg=MOPPOConfig(num_steps=5, epochs_per_update=1),
        extrinsics_cfg=extrinsics_cfg,
        seed=0,
    )
    stats = trainer.update()
    assert np.isfinite(stats["policy_loss"])
    assert np.isfinite(stats["value_loss"])


def test_encoder_checkpoint_round_trips():
    obs_cfg = ObservationSpaceCfg()
    action_cfg = ActionSpaceCfg()
    reward_cfg = RewardVectorCfg()
    pref_cfg = PreferenceCfg()
    extrinsics_cfg = ExtrinsicsCfg()
    moppo_cfg = MOPPOConfig(num_steps=5, epochs_per_update=1)

    env = _ExtrinsicsDummyEnv(obs_cfg, action_cfg, num_envs=4, horizon=40, seed=0, extrinsics_dim=extrinsics_cfg.dim)
    trainer = MOPPOTrainer(
        env, obs_cfg, reward_cfg, pref_cfg, moppo_cfg=moppo_cfg, extrinsics_cfg=extrinsics_cfg, seed=0
    )
    trainer.update()

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = os.path.join(tmp_dir, "checkpoint.pt")
        trainer.save(ckpt_path)

        env2 = _ExtrinsicsDummyEnv(obs_cfg, action_cfg, num_envs=4, horizon=40, seed=1, extrinsics_dim=extrinsics_cfg.dim)
        fresh_trainer = MOPPOTrainer(
            env2, obs_cfg, reward_cfg, pref_cfg, moppo_cfg=moppo_cfg, extrinsics_cfg=extrinsics_cfg, seed=1
        )
        for p1, p2 in zip(trainer.encoder.parameters(), fresh_trainer.encoder.parameters()):
            assert not torch.equal(p1, p2)

        fresh_trainer.load(ckpt_path)
        for p1, p2 in zip(trainer.encoder.parameters(), fresh_trainer.encoder.parameters()):
            assert torch.equal(p1, p2)


def test_dummy_env_without_extrinsics_still_works():
    # --env dummy must keep working unmodified when extrinsics_cfg is None.
    obs_cfg = ObservationSpaceCfg()
    action_cfg = ActionSpaceCfg()
    reward_cfg = RewardVectorCfg()
    pref_cfg = PreferenceCfg()

    env = DummyTalonEnv(obs_cfg, action_cfg, num_envs=4, horizon=40, seed=0)
    trainer = MOPPOTrainer(
        env, obs_cfg, reward_cfg, pref_cfg, moppo_cfg=MOPPOConfig(num_steps=5, epochs_per_update=1), seed=0
    )
    assert not hasattr(trainer, "encoder") or trainer.encoder is None
    stats = trainer.update()
    assert np.isfinite(stats["policy_loss"])


def test_update_backprops_into_encoder_parameters():
    # Non-obvious requirement: self.encoder being in self.optim's param
    # groups is not sufficient — update() must actually route a live,
    # gradient-carrying z_t through the loss (not just replay the numpy-
    # frozen z_t baked into the rollout buffer at collection time), or
    # optim.step() is a structural no-op for the encoder despite membership.
    obs_cfg = ObservationSpaceCfg()
    action_cfg = ActionSpaceCfg()
    reward_cfg = RewardVectorCfg()
    pref_cfg = PreferenceCfg()
    extrinsics_cfg = ExtrinsicsCfg()

    env = _ExtrinsicsDummyEnv(obs_cfg, action_cfg, num_envs=4, horizon=40, seed=0, extrinsics_dim=extrinsics_cfg.dim)
    trainer = MOPPOTrainer(
        env, obs_cfg, reward_cfg, pref_cfg,
        moppo_cfg=MOPPOConfig(num_steps=5, epochs_per_update=1),
        extrinsics_cfg=extrinsics_cfg,
        seed=0,
    )
    before = [p.clone() for p in trainer.encoder.parameters()]
    trainer.update()
    after = list(trainer.encoder.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))
