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
import pytest
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


def test_preference_is_constant_until_an_episode_resets():
    """Phase-1 MOPPO follows AMOR: w is sampled once at episode start, not
    continuously resampled inside an episode."""
    obs_cfg = ObservationSpaceCfg()
    action_cfg = ActionSpaceCfg()
    reward_cfg = RewardVectorCfg()
    pref_cfg = PreferenceCfg()
    env = DummyTalonEnv(obs_cfg, action_cfg, num_envs=3, horizon=1000, seed=0)
    trainer = MOPPOTrainer(
        env, obs_cfg, reward_cfg, pref_cfg,
        moppo_cfg=MOPPOConfig(num_steps=5, epochs_per_update=1), seed=0,
    )

    w_initial = trainer.w.copy()
    trainer.update()
    assert np.allclose(trainer.w, w_initial)


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


class _ShiftedExtrinsicsDummyEnv(DummyTalonEnv):
    """Like _ExtrinsicsDummyEnv but with an artificially large fixed offset
    and scale on one channel — mimics the real Isaac Lab env's ~1000x
    per-channel dynamic range (e.g. actuator stiffness ~44-66 vs. CoM
    offset ~±0.05), which np.random.randn's unit-scale output does not
    exercise (see Finding 2: the dummy-env test fixture was blind to this
    class of bug for exactly that reason)."""

    def __init__(self, *args, extrinsics_dim: int, **kwargs):
        super().__init__(*args, **kwargs)
        self._extrinsics_dim = extrinsics_dim

    def _with_extrinsics(self, transition):
        raw = np.random.randn(self.num_envs, self._extrinsics_dim).astype(np.float32)
        raw[:, 0] = raw[:, 0] * 5.0 + 55.0  # channel 0 at a real stiffness-like scale
        transition["extrinsics"] = raw
        return transition

    def reset(self):
        return self._with_extrinsics(super().reset())

    def step(self, action):
        transition, done = super().step(action)
        return self._with_extrinsics(transition), done


def test_extrinsics_normalization_running_stats_move_from_init():
    """Finding 2 regression: raw extrinsics fed straight into
    EnvFactorEncoder span a ~1000x per-channel range, so the large-magnitude
    channel dominates the small ones' gradient contribution for a long
    time. Proves trainer.extrinsics_norm's running mean/var actually move
    away from RunningMeanStd(dim)'s init defaults (mean=0, var=1) once fed
    extrinsics far from zero-mean/unit-variance, and that the shifted
    channel's mean lands near its true offset (55) rather than near 0 —
    i.e. the stats are per-channel-correct, not just "moved" — mirroring
    test_reward_norm_state_round_trips_through_checkpoint's structure for
    reward_norm."""
    obs_cfg = ObservationSpaceCfg()
    action_cfg = ActionSpaceCfg()
    reward_cfg = RewardVectorCfg()
    pref_cfg = PreferenceCfg()
    extrinsics_cfg = ExtrinsicsCfg()

    env = _ShiftedExtrinsicsDummyEnv(
        obs_cfg, action_cfg, num_envs=8, horizon=40, seed=0, extrinsics_dim=extrinsics_cfg.dim
    )
    trainer = MOPPOTrainer(
        env, obs_cfg, reward_cfg, pref_cfg,
        moppo_cfg=MOPPOConfig(num_steps=10, epochs_per_update=2),
        extrinsics_cfg=extrinsics_cfg,
        seed=0,
    )
    assert trainer.extrinsics_norm is not None
    assert np.allclose(trainer.extrinsics_norm.mean, 0.0)  # RunningMeanStd's pre-fit default
    assert np.allclose(trainer.extrinsics_norm.var, 1.0)

    for _ in range(3):
        trainer.update()

    assert not np.allclose(trainer.extrinsics_norm.mean, 0.0)
    assert not np.allclose(trainer.extrinsics_norm.var, 1.0)
    assert trainer.extrinsics_norm.mean[0] > 20.0  # channel 0's true offset is 55


def test_extrinsics_norm_state_round_trips_through_checkpoint():
    """Same rationale as test_reward_norm_state_round_trips_through_checkpoint:
    a resumed run must keep the same extrinsics scale as the run it resumes
    from, or the encoder sees a different input distribution than the one
    its weights were trained against."""
    obs_cfg = ObservationSpaceCfg()
    action_cfg = ActionSpaceCfg()
    reward_cfg = RewardVectorCfg()
    pref_cfg = PreferenceCfg()
    extrinsics_cfg = ExtrinsicsCfg()
    moppo_cfg = MOPPOConfig(num_steps=5, epochs_per_update=1)

    env = _ShiftedExtrinsicsDummyEnv(
        obs_cfg, action_cfg, num_envs=4, horizon=40, seed=0, extrinsics_dim=extrinsics_cfg.dim
    )
    trainer = MOPPOTrainer(
        env, obs_cfg, reward_cfg, pref_cfg, moppo_cfg=moppo_cfg, extrinsics_cfg=extrinsics_cfg, seed=0
    )
    trainer.update()
    trainer.update()  # accumulate non-trivial running stats

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = os.path.join(tmp_dir, "checkpoint.pt")
        trainer.save(ckpt_path)

        env2 = _ShiftedExtrinsicsDummyEnv(
            obs_cfg, action_cfg, num_envs=4, horizon=40, seed=1, extrinsics_dim=extrinsics_cfg.dim
        )
        fresh_trainer = MOPPOTrainer(
            env2, obs_cfg, reward_cfg, pref_cfg, moppo_cfg=moppo_cfg, extrinsics_cfg=extrinsics_cfg, seed=1
        )
        assert not np.allclose(trainer.extrinsics_norm.mean, fresh_trainer.extrinsics_norm.mean)

        fresh_trainer.load(ckpt_path)
        assert np.allclose(trainer.extrinsics_norm.mean, fresh_trainer.extrinsics_norm.mean)
        assert np.allclose(trainer.extrinsics_norm.var, fresh_trainer.extrinsics_norm.var)
        assert trainer.extrinsics_norm.count == fresh_trainer.extrinsics_norm.count


def test_extrinsics_norm_is_none_on_dummy_path():
    # --env dummy (extrinsics_cfg=None) must not construct an extrinsics
    # normalizer either — mirrors the existing self.encoder is None guard.
    obs_cfg = ObservationSpaceCfg()
    action_cfg = ActionSpaceCfg()
    reward_cfg = RewardVectorCfg()
    pref_cfg = PreferenceCfg()

    env = DummyTalonEnv(obs_cfg, action_cfg, num_envs=4, horizon=40, seed=0)
    trainer = MOPPOTrainer(
        env, obs_cfg, reward_cfg, pref_cfg, moppo_cfg=MOPPOConfig(num_steps=5, epochs_per_update=1), seed=0
    )
    assert trainer.extrinsics_norm is None


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


class _TruncatingDummyEnv(DummyTalonEnv):
    """DummyTalonEnv plus a controllable `terminal_fall` that's False even
    when `done` is True on lane 0's second step — simulates IsaacLabTalonEnv
    reporting a time-out (episode cut off, more reward was still possible)
    rather than a real fall (base_contact, no more reward possible)."""

    def step(self, action):
        transition, done = super().step(action)
        transition["terminal_fall"] = np.zeros(self.num_envs, dtype=bool)
        return transition, done


class _ActionRecordingDummyEnv(DummyTalonEnv):
    """DummyTalonEnv plus recording of the exact action array step() received
    (DummyEnv.step() clips its own copy to [-1, 1] internally, which would
    hide whether the action passed in was already bounded)."""

    def step(self, action):
        self.last_action_received = action.copy()
        return super().step(action)


def test_collect_rollout_action_is_bounded_and_matches_what_env_received():
    """ActorCritic.act() tanh-squashes internally now (see its ACTION_CLIP
    comment: two hard-clamp variants tried first each broke training a
    different way), so the action applied to the env and the action stored
    for training (act_list/logp_list) must be the exact same bounded value
    -- no separate raw-vs-applied split needed or wanted anymore."""
    obs_cfg = ObservationSpaceCfg()
    action_cfg = ActionSpaceCfg()
    reward_cfg = RewardVectorCfg()
    pref_cfg = PreferenceCfg()

    env = _ActionRecordingDummyEnv(obs_cfg, action_cfg, num_envs=4, horizon=40, seed=0)
    trainer = MOPPOTrainer(
        env, obs_cfg, reward_cfg, pref_cfg,
        moppo_cfg=MOPPOConfig(num_steps=3, epochs_per_update=1),
        seed=0,
    )
    with torch.no_grad():
        trainer.model.actor_mean.weight.fill_(0.0)
        trainer.model.actor_mean.bias.fill_(50.0)  # forces the pre-tanh mean far past ACTION_CLIP

    r = trainer._collect_rollout()
    assert np.all(np.abs(r["actions"]) <= trainer.model.ACTION_CLIP)
    np.testing.assert_array_equal(env.last_action_received, r["actions"][-1])


def test_gae_dones_uses_terminal_fall_not_raw_done():
    """`done` conflates real termination with time-out truncation
    (a1_env.py's `terminated | truncated`) — GAE must only zero the value
    bootstrap on the former, or a full-horizon time-out gets treated
    identically to a fall and there's no training signal left favoring
    "survive longer" once episodes approach the horizon (found 2026-09-17
    diagnosing mean_episode_length regressing across training runs)."""
    obs_cfg = ObservationSpaceCfg()
    action_cfg = ActionSpaceCfg()
    reward_cfg = RewardVectorCfg()
    pref_cfg = PreferenceCfg()

    env = _TruncatingDummyEnv(obs_cfg, action_cfg, num_envs=4, horizon=3, seed=0)
    trainer = MOPPOTrainer(
        env, obs_cfg, reward_cfg, pref_cfg,
        moppo_cfg=MOPPOConfig(num_steps=5, epochs_per_update=1),
        seed=0,
    )
    r = trainer._collect_rollout()
    # horizon=3 means every lane times out (done=True) at t=2 (0-indexed) —
    # _TruncatingDummyEnv reports terminal_fall=False throughout, so the
    # collected "dones" (what GAE bootstraps against) must stay all-False
    # despite done being True at that step.
    assert not r["dones"].any()


def test_penalty_curriculum_ramps_up_each_update_and_survives_checkpoint():
    """RMA-style penalty curriculum (MOPPOConfig.penalty_curriculum_init) —
    k starts small and ramps toward 1 via k = k ** growth each update() call,
    tried 2026-09-17 after phase1_longrun (20000 updates) showed
    mean_episode_len flat at its ~6-8 floor for the first ~12,000 iterations
    regardless of raw iteration count. Must actually advance (not stay
    frozen at init), and must round-trip through save/load like the reward
    normalizer's stats do -- a resume that resets it back to k_0 would undo
    whatever ramp progress a long run had already made."""
    obs_cfg = ObservationSpaceCfg()
    action_cfg = ActionSpaceCfg()
    reward_cfg = RewardVectorCfg()
    pref_cfg = PreferenceCfg()

    env = DummyTalonEnv(obs_cfg, action_cfg, num_envs=4, horizon=40, seed=0)
    cfg = MOPPOConfig(num_steps=5, epochs_per_update=1, penalty_curriculum_init=0.03, penalty_curriculum_growth=0.997)
    trainer = MOPPOTrainer(env, obs_cfg, reward_cfg, pref_cfg, moppo_cfg=cfg, seed=0)

    k_before = trainer._penalty_k
    assert k_before == cfg.penalty_curriculum_init
    stats = trainer.update()
    assert stats["penalty_curriculum_k"] == k_before  # this call used the pre-advance value
    assert trainer._penalty_k == k_before ** cfg.penalty_curriculum_growth
    assert trainer._penalty_k > k_before  # k in (0, 1) raised to a power < 1 increases it

    for _ in range(10):
        trainer.update()
    k_after_many = trainer._penalty_k
    assert k_after_many > k_before
    assert 0.0 < k_after_many < 1.0

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "ckpt.pt")
        trainer.save(path)
        trainer2 = MOPPOTrainer(env, obs_cfg, reward_cfg, pref_cfg, moppo_cfg=cfg, seed=1)
        trainer2.load(path)
        assert trainer2._penalty_k == k_after_many


def test_termination_reason_counts_fall_back_to_terminal_for_envs_without_the_keys():
    """DummyTalonEnv has no term_time_out/term_obstacle_reached/
    term_base_contact keys (only IsaacLabTalonEnv provides the real
    breakdown, added 2026-09-18 -- see a1_env.py) -- _collect_rollout must
    not crash on a `.get()` miss, and its fallback (treat any unattributed
    `done` as base_contact, since that's the historical default before this
    breakdown existed) must actually fire, not silently drop the count."""
    obs_cfg = ObservationSpaceCfg()
    action_cfg = ActionSpaceCfg()
    reward_cfg = RewardVectorCfg()
    pref_cfg = PreferenceCfg()

    env = DummyTalonEnv(obs_cfg, action_cfg, num_envs=4, horizon=3, seed=0)
    trainer = MOPPOTrainer(
        env, obs_cfg, reward_cfg, pref_cfg,
        moppo_cfg=MOPPOConfig(num_steps=5, epochs_per_update=1),
        seed=0,
    )
    r = trainer._collect_rollout()
    assert r["done_count"] > 0
    assert r["term_reason_counts"]["base_contact"] == r["done_count"]
    assert r["term_reason_counts"]["time_out"] == 0
    assert r["term_reason_counts"]["obstacle_reached"] == 0


def test_update_stats_include_termination_fractions_that_sum_to_at_most_one():
    obs_cfg = ObservationSpaceCfg()
    action_cfg = ActionSpaceCfg()
    reward_cfg = RewardVectorCfg()
    pref_cfg = PreferenceCfg()

    env = DummyTalonEnv(obs_cfg, action_cfg, num_envs=4, horizon=3, seed=0)
    trainer = MOPPOTrainer(
        env, obs_cfg, reward_cfg, pref_cfg,
        moppo_cfg=MOPPOConfig(num_steps=5, epochs_per_update=1),
        seed=0,
    )
    stats = trainer.update()
    total_frac = (
        stats["term_frac_base_contact"] + stats["term_frac_time_out"] + stats["term_frac_obstacle_reached"]
    )
    assert 0.0 <= total_frac <= 1.0 + 1e-6


def test_log_std_is_clamped_to_LOG_STD_MIN_after_update():
    """Found 2026-09-18: an in-graph clamp/softplus floor (tried twice,
    once as a hard clamp, once as a differentiable double-softplus) either
    dead-gradients or distorts the whole usable range (see
    ActorCritic.LOG_STD_MIN's docstring). Bounding moved to a post-step
    clamp on the raw parameter instead -- this checks the floor half:
    however far below LOG_STD_MIN a gradient step pushes log_std, the
    value must be restored to at least LOG_STD_MIN before the next forward
    pass reads it."""
    obs_cfg = ObservationSpaceCfg()
    action_cfg = ActionSpaceCfg()
    reward_cfg = RewardVectorCfg()
    pref_cfg = PreferenceCfg()

    env = DummyTalonEnv(obs_cfg, action_cfg, num_envs=4, horizon=40, seed=0)
    trainer = MOPPOTrainer(
        env, obs_cfg, reward_cfg, pref_cfg,
        moppo_cfg=MOPPOConfig(num_steps=5, epochs_per_update=1), seed=0,
    )
    with torch.no_grad():
        trainer.model.log_std.fill_(-5.0)  # already below LOG_STD_MIN
    trainer.update()
    assert torch.all(trainer.model.log_std >= trainer.model.LOG_STD_MIN)


def test_log_std_is_clamped_to_LOG_STD_MAX_after_update():
    """Ceiling half of the same fix -- found 2026-09-18 (phase1_longrun5,
    resumed8): the performance-gated log_std mechanism blocks log_std from
    DECREASING on any update() whose mean_episode_len underperforms its own
    baseline, with nothing stopping an INCREASE once that gate is
    perpetually triggered (mean_episode_len was declining every update for
    ~750 updates straight) -- entropy climbed past +4.3 and kept rising,
    the mirror-image runaway of the original collapse-to-the-floor bug."""
    obs_cfg = ObservationSpaceCfg()
    action_cfg = ActionSpaceCfg()
    reward_cfg = RewardVectorCfg()
    pref_cfg = PreferenceCfg()

    env = DummyTalonEnv(obs_cfg, action_cfg, num_envs=4, horizon=40, seed=0)
    trainer = MOPPOTrainer(
        env, obs_cfg, reward_cfg, pref_cfg,
        moppo_cfg=MOPPOConfig(num_steps=5, epochs_per_update=1), seed=0,
    )
    with torch.no_grad():
        trainer.model.log_std.fill_(5.0)  # already above LOG_STD_MAX
    trainer.update()
    assert torch.all(trainer.model.log_std <= trainer.model.LOG_STD_MAX)


def test_mean_reg_penalizes_and_shrinks_a_large_raw_actor_mean():
    """Found 2026-09-18: actor_body's hidden-layer activations and
    actor_mean's raw output kept growing through training (activation max
    5->10->15+, raw mean max up to 24) even with weight_decay=1e-4 on
    every parameter -- a generic weight penalty wasn't targeted at the
    actual symptom. mean_reg_coef adds an SAC-style direct penalty on
    mean.pow(2). Starting actor_mean's bias artificially large and running
    a few update() calls with a healthy mean_reg_coef must pull the raw
    mean's magnitude down, not leave it where weight_decay alone couldn't."""
    obs_cfg = ObservationSpaceCfg()
    action_cfg = ActionSpaceCfg()
    reward_cfg = RewardVectorCfg()
    pref_cfg = PreferenceCfg()

    env = DummyTalonEnv(obs_cfg, action_cfg, num_envs=4, horizon=40, seed=0)
    trainer = MOPPOTrainer(
        env, obs_cfg, reward_cfg, pref_cfg,
        moppo_cfg=MOPPOConfig(num_steps=5, epochs_per_update=1, mean_reg_coef=1e-1, weight_decay=0.0, lr=1e-2),
        seed=0,
    )
    with torch.no_grad():
        trainer.model.actor_mean.bias.fill_(10.0)  # artificially large raw mean

    obs = torch.zeros(1, trainer.stack.policy_obs_dim + trainer.reward_cfg.dim)
    raw_mean_before = trainer.model.raw_mean(obs).abs().mean().item()
    for _ in range(20):
        trainer.update()
    raw_mean_after = trainer.model.raw_mean(obs).abs().mean().item()
    assert raw_mean_after < raw_mean_before


def test_log_std_max_anneals_toward_final_each_update_call():
    """Replacement for the removed performance-gated log_std mechanism
    (2026-09-18): instead of blocking log_std from shrinking, the CEILING
    itself now decays on a fixed schedule -- log_std stays free to use any
    value below whatever the current ceiling is."""
    obs_cfg = ObservationSpaceCfg()
    action_cfg = ActionSpaceCfg()
    reward_cfg = RewardVectorCfg()
    pref_cfg = PreferenceCfg()

    env = DummyTalonEnv(obs_cfg, action_cfg, num_envs=4, horizon=40, seed=0)
    cfg = MOPPOConfig(
        num_steps=5, epochs_per_update=1,
        log_std_max_anneal_final=-0.5, log_std_max_anneal_decay=0.9,
    )
    trainer = MOPPOTrainer(env, obs_cfg, reward_cfg, pref_cfg, moppo_cfg=cfg, seed=0)

    ceiling_before = trainer._log_std_max
    assert ceiling_before == trainer.model.LOG_STD_MAX  # starts un-annealed
    stats = trainer.update()
    assert stats["log_std_max"] == ceiling_before  # reports the value USED this call, pre-advance
    expected_after = cfg.log_std_max_anneal_final + (ceiling_before - cfg.log_std_max_anneal_final) * cfg.log_std_max_anneal_decay
    assert trainer._log_std_max == pytest.approx(expected_after)
    assert trainer._log_std_max < ceiling_before  # moved toward final (which is lower)

    for _ in range(50):
        trainer.update()
    assert trainer._log_std_max < expected_after  # kept decaying
    assert trainer._log_std_max > cfg.log_std_max_anneal_final  # asymptotic, never overshoots


def test_log_std_never_exceeds_the_annealed_ceiling_after_update():
    """The post-step clamp must use the CURRENT (possibly-annealed)
    ceiling, not the un-annealed ActorCritic.LOG_STD_MAX constant --
    forces log_std above the constant-but-already-annealed-below ceiling
    and checks it gets clamped to the lower, annealed value."""
    obs_cfg = ObservationSpaceCfg()
    action_cfg = ActionSpaceCfg()
    reward_cfg = RewardVectorCfg()
    pref_cfg = PreferenceCfg()

    env = DummyTalonEnv(obs_cfg, action_cfg, num_envs=4, horizon=40, seed=0)
    cfg = MOPPOConfig(num_steps=5, epochs_per_update=1, log_std_max_anneal_final=-0.5, log_std_max_anneal_decay=0.9)
    trainer = MOPPOTrainer(env, obs_cfg, reward_cfg, pref_cfg, moppo_cfg=cfg, seed=0)
    trainer._log_std_max = -0.5  # simulate having already annealed down
    with torch.no_grad():
        trainer.model.log_std.fill_(0.0)  # above the annealed ceiling, at the un-annealed one

    trainer.update()
    assert torch.all(trainer.model.log_std <= -0.5 + 1e-6)


def test_log_std_max_survives_checkpoint_round_trip():
    obs_cfg = ObservationSpaceCfg()
    action_cfg = ActionSpaceCfg()
    reward_cfg = RewardVectorCfg()
    pref_cfg = PreferenceCfg()

    env = DummyTalonEnv(obs_cfg, action_cfg, num_envs=4, horizon=40, seed=0)
    trainer = MOPPOTrainer(
        env, obs_cfg, reward_cfg, pref_cfg,
        moppo_cfg=MOPPOConfig(num_steps=5, epochs_per_update=1, log_std_max_anneal_decay=0.5), seed=0,
    )
    trainer.update()
    ceiling_after_update = trainer._log_std_max
    assert ceiling_after_update != trainer.model.LOG_STD_MAX

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "ckpt.pt")
        trainer.save(path)
        trainer2 = MOPPOTrainer(env, obs_cfg, reward_cfg, pref_cfg, moppo_cfg=MOPPOConfig(), seed=1)
        trainer2.load(path)
        assert trainer2._log_std_max == ceiling_after_update


def test_push_obs_normalizes_before_pushing_onto_the_stack():
    """Found 2026-09-18: reward_norm/extrinsics_norm normalized their inputs
    but the actual policy observation never went through any normalizer at
    all -- "What Matters in On-Policy RL" (Andrychowicz et al. 2021) found
    this the single strongest lever in their whole study. push_obs must be
    the one place every caller normalizes through, so self.stack never
    holds a raw (unnormalized) frame."""
    obs_cfg = ObservationSpaceCfg()
    action_cfg = ActionSpaceCfg()
    reward_cfg = RewardVectorCfg()
    pref_cfg = PreferenceCfg()

    env = DummyTalonEnv(obs_cfg, action_cfg, num_envs=4, horizon=40, seed=0)
    trainer = MOPPOTrainer(env, obs_cfg, reward_cfg, pref_cfg, moppo_cfg=MOPPOConfig(), seed=0)

    raw_obs = np.full((4, env.obs_dim), 1000.0, dtype=np.float32)  # far outside any real obs's scale
    trainer.push_obs(raw_obs, done_mask=np.zeros(4, dtype=bool))
    pushed = trainer.stack.policy_obs
    assert not np.allclose(pushed, raw_obs.reshape(pushed.shape))
    assert np.all(np.abs(pushed) <= 10.0 + 1e-4)  # RunningMeanStd's default clip


def test_obs_norm_state_round_trips_through_checkpoint():
    """Same resume-shouldn't-shock-the-input-scale reasoning as
    reward_norm/extrinsics_norm's own round-trip tests -- a resumed run
    must keep the SAME obs_norm running stats, not reset them."""
    obs_cfg = ObservationSpaceCfg()
    action_cfg = ActionSpaceCfg()
    reward_cfg = RewardVectorCfg()
    pref_cfg = PreferenceCfg()
    moppo_cfg = MOPPOConfig(num_steps=5, epochs_per_update=1)

    env = DummyTalonEnv(obs_cfg, action_cfg, num_envs=4, horizon=40, seed=0)
    trainer = MOPPOTrainer(env, obs_cfg, reward_cfg, pref_cfg, moppo_cfg=moppo_cfg, seed=0)
    trainer.update()
    trainer.update()

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = os.path.join(tmp_dir, "checkpoint.pt")
        trainer.save(ckpt_path)

        env2 = DummyTalonEnv(obs_cfg, action_cfg, num_envs=4, horizon=40, seed=1)
        fresh_trainer = MOPPOTrainer(env2, obs_cfg, reward_cfg, pref_cfg, moppo_cfg=moppo_cfg, seed=1)
        assert not np.allclose(trainer.obs_norm.mean, fresh_trainer.obs_norm.mean)

        fresh_trainer.load(ckpt_path)
        assert np.allclose(trainer.obs_norm.mean, fresh_trainer.obs_norm.mean)
        assert np.allclose(trainer.obs_norm.var, fresh_trainer.obs_norm.var)
        assert trainer.obs_norm.count == fresh_trainer.obs_norm.count
