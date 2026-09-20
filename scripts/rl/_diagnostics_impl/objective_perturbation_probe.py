#!/usr/bin/env python3
"""Perturbation probe for the local-attractor question objective_segment_
audit.py opened: seeds 0/1's stable low-v_x states have near-zero/negative
signed progress advantage while forward states have strongly positive
advantage, with progress dominating the weighted-sum gradient share
(~77-80%) in BOTH groups and near-identical action magnitude between
them. That ruled out reward-weighting (Hypothesis A) and leaned toward
policy/exploration (Hypothesis B) over action-to-motion decoupling
(Hypothesis C) -- this script tests that more directly by asking the
actor and critic what THEY think should happen if the command pressure
on a frozen stable state were increased, distinguishing:

  (a) actor doesn't change its action meaningfully even when the command
      pressure is turned up -- the policy itself has converged away from
      producing forward-locomoting actions in this state, independent of
      what the critic says. Policy-level/exploration attractor.
  (b) actor DOES change its action under stronger command pressure, but
      the critic's value estimate for the state barely moves -- the
      value landscape gives little incentive to actually follow through,
      even though the policy COULD act differently. Critic/local-value
      issue.
  (c) both actor and critic respond (action changes, value increases)
      but a short forward rollout from that same physical state still
      doesn't produce sustained v_x -- points back to the action-to-
      motion/contact pathway, a DIFFERENT question from 2C/2D/2E's
      controller-tracks-commanded-JOINT-TARGET diagnosis (this would be
      about whether the chosen ACTION, faithfully executed, produces
      forward motion at all).

Method: collects one long rollout (reusing _collect_rollout verbatim,
same env.step-wrapping trick as objective_segment_audit.py, for the
same warmup/fall-exclusion masking), identifies "stable" (|v_x|<
--stable_vx_thresh) samples, and for a random subsample of them:
  1. Takes that step's ACTUAL actor_obs/critic_obs row (the real state
     the trained policy was in -- not a synthetic one).
  2. Builds a perturbed copy with the v_command_x slice boosted by
     +--perturb_delta (locating that slice from ObservationSpaceCfg's
     own declared dims, not a hardcoded index).
  3. Runs both through act_inference (deterministic -- mean action, no
     sampling noise, so a measured difference is the policy's actual
     response, not RNG) and value(), and reports: |Δaction| (L2, whole
     12-dim action), cosine similarity between baseline and perturbed
     action (0=orthogonal/unrelated, 1=identical direction), and Δvalue
     for EVERY objective (not just progress -- balance/efficiency/impact
     could shift too and that's informative context, per this arc's
     standing rule against collapsing a mechanism away).

Does NOT run a physical short rollout from the frozen state (case (c)
above) -- that requires teleporting the robot's full joint/root state
back into the sim (write_joint_state_to_sim/write_root_state_to_sim,
same pattern gravity_finite_diff_check.py already used) and is left as
a deliberate follow-up ONLY if (a)/(b) come back inconclusive; per the
audit's own design, this script should not be extended to attempt it
without that decision being made explicitly first.

    PYTHONPATH="$(pwd):$(pwd)/scripts" python scripts/rl/diagnostics.py objective-perturbation-probe \
        runs/phase1_hipact_dt01_seed1_2026-09-20/checkpoints/checkpoint.pt 0.7 0.1 0.1 0.1 \
        --steps 200 --sim_dt 0.01 --perturb_delta 0.3
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("w", type=float, nargs=4, help="Force this w (RewardVectorCfg.term_names order), need not sum to 1")
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--stable_vx_thresh", type=float, default=0.15)
    parser.add_argument("--fall_exclude_window", type=int, default=5)
    parser.add_argument("--perturb_delta", type=float, nargs="+", default=[0.3, 0.6], help="One or more v_command_x boosts to test, e.g. 0.3 0.6")
    parser.add_argument("--n_probe", type=int, default=1000, help="Max number of (t, lane) stable samples to probe (random subsample if more are available)")
    parser.add_argument("--command", type=float, nargs=3, default=(0.5, 0.0, 0.0), metavar=("VX", "VY", "WZ"))
    parser.add_argument("--sim_dt", type=float, default=0.02)
    parser.add_argument("--decimation", type=int, default=1)
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
    action_cfg = ActionSpaceCfg()  # noqa: F841
    reward_cfg = RewardVectorCfg()
    pref_cfg = PreferenceCfg()
    stack_cfg = ObservationStackCfg()
    extrinsics_cfg = None if args.no_encoder else ExtrinsicsCfg()

    cfg = IsaacLabTalonEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.sim.dt = args.sim_dt
    cfg.decimation = args.decimation
    cfg.sim.render_interval = cfg.decimation
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
    env.v_command_buf[:] = torch.tensor(args.command, device=env.device)

    orig_sample = moppo_mod.sample_preference_vector

    def _fixed_sample(rng, reward_cfg_, pref_cfg_, n, step=0):
        return w_forced[:n].copy()

    moppo_mod.sample_preference_vector = _fixed_sample

    T, N = args.steps, args.num_envs
    v_x_rec = np.zeros((T, N), dtype=np.float32)
    _step_idx = [0]
    orig_step = env.step

    def _wrapped_step(action):
        transition, done = orig_step(action)
        v_x_rec[_step_idx[0]] = transition["v_actual"][:, 0]
        _step_idx[0] += 1
        return transition, done

    env.step = _wrapped_step
    try:
        r = trainer._collect_rollout()
    finally:
        env.step = orig_step
        moppo_mod.sample_preference_vector = orig_sample

    terminal_mask = r["dones"]
    fall_exclude = terminal_mask.copy()
    for shift in range(1, args.fall_exclude_window + 1):
        fall_exclude[:-shift] |= terminal_mask[shift:]
        fall_exclude[shift:] |= terminal_mask[:-shift]
    valid = np.ones((T, N), dtype=bool)
    valid[:args.warmup] = False
    valid[fall_exclude] = False
    stable_mask = valid & (np.abs(v_x_rec) < args.stable_vx_thresh)

    t_idx, n_idx = np.nonzero(stable_mask)
    n_available = len(t_idx)
    print(f"\n{n_available} stable samples available (warmup={args.warmup}, fall_exclude_window={args.fall_exclude_window})")
    if n_available == 0:
        print("no stable samples found -- nothing to probe")
        return
    rng = np.random.default_rng(args.seed)
    n_probe = min(args.n_probe, n_available)
    sel = rng.choice(n_available, size=n_probe, replace=False)
    t_sel, n_sel = t_idx[sel], n_idx[sel]

    actor_obs = r["actor_obs"][t_sel, n_sel]    # (n_probe, actor_dim)
    critic_obs = r["critic_obs"][t_sel, n_sel]  # (n_probe, critic_dim)

    # Locate v_command's slice inside the raw (unstacked) obs block --
    # obs_stack with num_policy_stacks/num_critic_stacks=1 (this checkpoint's
    # config) means policy_obs/critic_obs IS the raw per-step obs, in
    # ObservationSpaceCfg's own declared field order.
    cmd_start = (
        obs_cfg.joint_pos_dim + obs_cfg.joint_vel_dim + obs_cfg.roll_pitch_dim
        + obs_cfg.foot_contact_dim + obs_cfg.prev_action_dim
    )
    cmd_x_idx = cmd_start  # v_command's first component is v_x

    device = trainer.device
    actor_obs_t = torch.from_numpy(actor_obs).to(device)
    critic_obs_t = torch.from_numpy(critic_obs).to(device)

    with torch.no_grad():
        baseline_action = trainer.model.act_inference(actor_obs_t)
        baseline_value = trainer._value(critic_obs_t)

    names = reward_cfg.term_names
    baseline_action_np = baseline_action.cpu().numpy()
    baseline_value_np = baseline_value.cpu().numpy()

    print(f"\nProbing {n_probe} stable states (|v_x|<{args.stable_vx_thresh}), command_x slice at obs index {cmd_x_idx}")
    print(f"baseline: mean |action|={np.linalg.norm(baseline_action_np, axis=-1).mean():.4f}  "
          f"mean value per objective: {dict(zip(names, baseline_value_np.mean(axis=0).round(4)))}")

    for delta in args.perturb_delta:
        perturbed_actor_obs = actor_obs.copy()
        perturbed_actor_obs[:, cmd_x_idx] += delta
        perturbed_critic_obs = critic_obs.copy()
        perturbed_critic_obs[:, cmd_x_idx] += delta

        with torch.no_grad():
            perturbed_action = trainer.model.act_inference(torch.from_numpy(perturbed_actor_obs).to(device))
            perturbed_value = trainer._value(torch.from_numpy(perturbed_critic_obs).to(device))

        perturbed_action_np = perturbed_action.cpu().numpy()
        perturbed_value_np = perturbed_value.cpu().numpy()

        delta_action = perturbed_action_np - baseline_action_np
        action_diff_mag = np.linalg.norm(delta_action, axis=-1)
        cos_sim = np.sum(baseline_action_np * perturbed_action_np, axis=-1) / (
            np.linalg.norm(baseline_action_np, axis=-1) * np.linalg.norm(perturbed_action_np, axis=-1) + 1e-8
        )

        print(f"\n=== perturb v_command_x += {delta} ===")
        print(f"  |Δaction| (L2, 12-dim): mean={action_diff_mag.mean():.4f}  std={action_diff_mag.std():.4f}  "
              f"median={np.median(action_diff_mag):.4f}")
        print(f"  cos_sim(baseline, perturbed action): mean={cos_sim.mean():.4f}  "
              f"(1.0=identical direction, 0=orthogonal)")
        delta_value = perturbed_value_np - baseline_value_np
        print(f"  {'term':<12}{'Δvalue (mean)':>16}{'Δvalue (median)':>18}")
        for k, name in enumerate(names):
            print(f"  {name:<12}{delta_value[:, k].mean():>16.4f}{np.median(delta_value[:, k]):>18.4f}")


if __name__ == "__main__":
    main()
