#!/usr/bin/env python3
"""Return/advantage decomposition under a FIXED preference w, per objective --
raw reward, GAE return, raw advantage, normalized advantage, and the
w-weighted normalized-advantage magnitude that actually feeds
d3po_actor_loss's weighted sum (losses.py).

Built 2026-09-19 to answer a specific question raised mid-session: does a
reward term with large raw magnitude (`alive_bonus`, inside `balance`)
dominate the policy gradient despite D3PO's per-objective advantage
normalization, or does normalize_per_objective actually neutralize that
before w is applied? Answer found empirically: influence % tracks w almost
exactly regardless of raw reward/advantage scale differences across terms
(see talon-thesis/03_Daily_Notes/2026-09-19.md for the full readout) --
kept as permanent tooling since this question (is w-weighting real at the
gradient level, or just at the reward level) will come up again for any
future reward-vector change.

Reuses MOPPOTrainer._collect_rollout() verbatim (same reward pipeline,
penalty_curriculum_k scaling, reward_norm) by monkey-patching
sample_preference_vector for this process only, so the collected rollout
never resamples away from the forced w -- everything else (obs stacking,
reward normalization, GAE) is IDENTICAL to what training itself would do
under this w. _collect_rollout is read-only w.r.t. the model (no
optimizer.step() anywhere in it) but DOES mutate reward_norm's running
stats -- run this once per (checkpoint, w) combination in a fresh process,
don't reuse a trainer instance across multiple forced-w passes.

    PYTHONPATH="$(pwd):$(pwd)/scripts" python scripts/rl/diagnostics.py advantage-decomposition \
        runs/<run>/checkpoints/checkpoint.pt 0.7 0.1 0.1 0.1 --steps 24
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from talon_rl.config import (
    ActionSpaceCfg, ExtrinsicsCfg, ObservationSpaceCfg, ObservationStackCfg, PreferenceCfg, RewardVectorCfg,
)

from rl.core.algorithms import MOPPOConfig, MOPPOTrainer
import rl.core.algorithms.moppo as moppo_mod
from rl.core.losses import normalize_per_objective
from rl.core.storage.rollout_storage import gae_per_objective


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("w", type=float, nargs=4, help="Force this w (RewardVectorCfg.term_names order), need not sum to 1")
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=24, help="Rollout window length; default matches MOPPOConfig.num_steps")
    parser.add_argument("--command", type=float, nargs=3, default=(0.5, 0.0, 0.0), metavar=("VX", "VY", "WZ"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_encoder", action="store_true")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint

    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher({"headless": True, "enable_cameras": False})
    simulation_app = app_launcher.app  # noqa: F841 -- kept alive for the process lifetime

    import gymnasium as gym
    import talon_rl.tasks.locomotion.a1_env  # noqa: F401 -- registers Isaac-Talon-A1-v0
    from talon_rl.tasks.locomotion.a1_env.a1_env_cfg import IsaacLabTalonEnvCfg

    obs_cfg = ObservationSpaceCfg()
    action_cfg = ActionSpaceCfg()
    reward_cfg = RewardVectorCfg()
    pref_cfg = PreferenceCfg()
    stack_cfg = ObservationStackCfg()
    extrinsics_cfg = None if args.no_encoder else ExtrinsicsCfg()

    cfg = IsaacLabTalonEnvCfg()
    cfg.scene.num_envs = args.num_envs
    env = gym.make("Isaac-Talon-A1-v0", cfg=cfg, render_mode=None).unwrapped

    moppo_cfg = MOPPOConfig(num_steps=args.steps)
    trainer = MOPPOTrainer(
        env, obs_cfg, reward_cfg, pref_cfg, moppo_cfg=moppo_cfg, stack_cfg=stack_cfg,
        extrinsics_cfg=extrinsics_cfg, seed=args.seed,
    )
    trainer.load(checkpoint_path)
    print(f"loaded checkpoint (t={trainer._t}, penalty_k={trainer._penalty_k:.4f})")

    w_forced = np.tile(np.array(args.w, dtype=np.float32) / sum(args.w), (args.num_envs, 1)).astype(np.float32)
    trainer.w = w_forced.copy()

    # Force v_command the same way play.py's --command does, so this rollout
    # tests under a real sustained command, not whatever the env samples.
    env.v_command_buf[:] = torch.tensor(args.command, device=env.device)

    # Pin w for the whole rollout: _collect_rollout resamples via
    # sample_preference_vector only on lanes whose PREVIOUS step was `done`
    # -- patch it to always hand back the forced w so a mid-rollout reset
    # can't drift the lane onto a different preference.
    orig_sample = moppo_mod.sample_preference_vector

    def _fixed_sample(rng, reward_cfg_, pref_cfg_, n, step=0):
        return w_forced[:n].copy()

    moppo_mod.sample_preference_vector = _fixed_sample
    try:
        r = trainer._collect_rollout()
    finally:
        moppo_mod.sample_preference_vector = orig_sample

    values_with_final = np.concatenate([r["values"], r["final_value"][None]], axis=0)
    adv_raw = gae_per_objective(r["rewards"], values_with_final, r["dones"], moppo_cfg.gamma, moppo_cfg.gae_lambda)
    returns = adv_raw + r["values"]

    T, N = r["dones"].shape
    adv_norm = normalize_per_objective(adv_raw.reshape(T * N, -1)).reshape(T, N, -1)

    names = reward_cfg.term_names
    w_normed = np.array(args.w, dtype=np.float32) / sum(args.w)
    print(f"\nw = {dict(zip(names, args.w))}  (steps={T}, envs={N})")
    print(f"\n{'term':<12}{'raw reward':>14}{'raw |adv|':>14}{'return':>14}{'norm |adv|':>14}{'w_i*norm|adv|':>14}")
    norm_adv_mags = []
    for k, name in enumerate(names):
        raw_reward = r["rewards"][:, :, k].mean()
        raw_adv_mag = np.abs(adv_raw[:, :, k]).mean()
        ret = returns[:, :, k].mean()
        norm_adv_mag = np.abs(adv_norm[:, :, k]).mean()
        norm_adv_mags.append(norm_adv_mag)
        # w is the SAME constant vector for every sample in this rollout
        # (forced, not sampled) -- mean(w_i * norm_adv_i) over samples
        # would average to ~0 since norm_adv is zero-mean by construction
        # (normalize_per_objective mean-subtracts each column). The
        # informative quantity under a FIXED w is the scalar magnitude
        # w_i * mean(|norm_adv_i|): how much this objective's typical
        # advantage swing actually counts toward d3po_actor_loss's
        # weighted sum, given this w.
        w_contribution = w_normed[k] * norm_adv_mag
        print(f"{name:<12}{raw_reward:>14.4f}{raw_adv_mag:>14.4f}{ret:>14.4f}{norm_adv_mag:>14.4f}{w_contribution:>14.4f}")

    total_contribution = sum(w_normed[k] * norm_adv_mags[k] for k in range(len(names)))
    print(f"\nsum(w_i * norm|adv_i|): {total_contribution:.4f}  (each term's share of this = its % influence on the weighted-sum loss)")
    for k, name in enumerate(names):
        share = w_normed[k] * norm_adv_mags[k] / total_contribution * 100
        print(f"  {name}: {share:.1f}%")
    print(f"mean episode length this window: {trainer._lane_step_count.mean():.2f}")


if __name__ == "__main__":
    main()
