#!/usr/bin/env python3
"""Pre-fall vs normal-window comparison: for every fall in a rollout, looks
at [t_fall-window, t_fall) and compares |pitch|, |pitch_rate| (real
body-frame angular velocity from a1_env.py's transition dict, not a
finite-difference approximation), height, balance_reward, and
action_magnitude against the same quantities elsewhere in the trajectory
(the complement of all pre-fall windows) -- calibrates what tilt-angle/
tilt-rate/height range balance_tilt_coef/balance_tilt_rate_coef/
height_coef should actually target, rather than guessing candidate values
with no grounding in what pre-fall states look like. Also reports a
lead-time snapshot (value at fixed offsets before each fall) to check
whether pitch_rate is a genuine early-warning signal or only spikes in
the last step or two.

height is tracked unconditionally (not just pre-fall/normal split) so
this same script run against a checkpoint already known to crouch
(2026-09-20: A/B/D all settle around height 0.21-0.25 vs target 0.42
regardless of balance_tilt_coef) gives that checkpoint's own height
percentiles -- comparing a genuinely-walking baseline's normal-window
height against a crouching checkpoint's height tells us whether
target_height=0.42 is even a physically-reachable region for this gait,
before picking a height_coef strong enough to push toward it.

Reuses the same rollout mechanics as balance_progress_trace.py (deterministic
act_inference under a forced w/command) but aggregates over ALL lanes'
fall events, not just one plotted lane, for a less noisy read.

    PYTHONPATH="$(pwd):$(pwd)/scripts" python scripts/rl/prefall_window_analysis.py \
        runs/<run>/checkpoints/checkpoint.pt 0.7 0.1 0.1 0.1 --steps 200 --window 10
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from talon_rl.config import (
    ActionSpaceCfg, ExtrinsicsCfg, ObservationSpaceCfg, ObservationStackCfg, PreferenceCfg, RewardVectorCfg,
)
from talon_rl.reward import compute_reward_vector

from rl.core.algorithms import MOPPOConfig, MOPPOTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("w", type=float, nargs=4)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--window", type=int, default=10, help="How many steps before each fall counts as 'pre-fall'")
    parser.add_argument("--command", type=float, nargs=3, default=(0.5, 0.0, 0.0), metavar=("VX", "VY", "WZ"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target_height", type=float, default=0.42)
    args = parser.parse_args()

    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher({"headless": True, "enable_cameras": False})
    simulation_app = app_launcher.app  # noqa: F841

    import gymnasium as gym
    import talon_rl.tasks.locomotion.a1_env  # noqa: F401
    from talon_rl.tasks.locomotion.a1_env.a1_env_cfg import IsaacLabTalonEnvCfg

    obs_cfg = ObservationSpaceCfg()
    action_cfg = ActionSpaceCfg()
    reward_cfg = RewardVectorCfg()
    pref_cfg = PreferenceCfg()
    stack_cfg = ObservationStackCfg()
    extrinsics_cfg = ExtrinsicsCfg()

    cfg = IsaacLabTalonEnvCfg()
    cfg.scene.num_envs = args.num_envs
    env = gym.make("Isaac-Talon-A1-v0", cfg=cfg, render_mode=None).unwrapped

    trainer = MOPPOTrainer(
        env, obs_cfg, reward_cfg, pref_cfg, moppo_cfg=MOPPOConfig(), stack_cfg=stack_cfg,
        extrinsics_cfg=extrinsics_cfg, seed=args.seed,
    )
    trainer.load(args.checkpoint)
    print(f"loaded checkpoint (t={trainer._t})")

    trainer.w = np.tile(np.array(args.w, dtype=np.float32) / sum(args.w), (args.num_envs, 1)).astype(np.float32)
    balance_idx = reward_cfg.term_names.index("balance")

    pitch = np.zeros((args.steps, args.num_envs), dtype=np.float32)
    roll = np.zeros((args.steps, args.num_envs), dtype=np.float32)
    pitch_rate = np.zeros((args.steps, args.num_envs), dtype=np.float32)
    height = np.zeros((args.steps, args.num_envs), dtype=np.float32)
    r_balance = np.zeros((args.steps, args.num_envs), dtype=np.float32)
    action_mag = np.zeros((args.steps, args.num_envs), dtype=np.float32)
    fell = np.zeros((args.steps, args.num_envs), dtype=bool)

    trainer.model.eval()
    for t in range(args.steps):
        env.v_command_buf[:] = torch.tensor(args.command, device=env.device)
        action = trainer.act_inference()
        transition, done = env.step(action)
        trainer._last_extrinsics = transition.get("extrinsics") if trainer.encoder else None
        trainer.push_obs(transition["obs"], done_mask=done)

        reward_vec = compute_reward_vector(transition, reward_cfg)
        roll[t] = transition["roll_pitch"][:, 0]
        pitch[t] = transition["roll_pitch"][:, 1]
        # Real body-frame angular velocity (root_ang_vel_b), not a
        # finite-difference approximation of pitch across steps -- see
        # a1_env.py's transition dict / balance_reward's tilt_rate_coef.
        pitch_rate[t] = transition["roll_pitch_rate"][:, 1]
        height[t] = transition["height"]
        r_balance[t] = reward_vec[:, balance_idx]
        action_mag[t] = np.abs(action).mean(axis=-1)
        fell[t] = transition.get("terminal_fall", done)

    pre_fall_mask = np.zeros_like(fell)
    for t in range(args.steps):
        if fell[t].any():
            lo = max(0, t - args.window)
            pre_fall_mask[lo:t, fell[t]] = True
    normal_mask = ~pre_fall_mask
    # Exclude the fall step itself (roll_pitch/pitch_rate there reflect the
    # already-toppled pose, not the lead-up) from BOTH groups.
    normal_mask &= ~fell

    def stats(mask: np.ndarray, arr: np.ndarray) -> str:
        vals = arr[mask]
        if vals.size == 0:
            return "n/a"
        p50, p90, p95 = np.percentile(vals, [50, 90, 95])
        return f"mean={np.mean(vals):.4f} p50={p50:.4f} p90={p90:.4f} p95={p95:.4f} (n={vals.size})"

    n_falls = int(fell.sum())
    print(f"\ntotal fall events across {args.num_envs} lanes x {args.steps} steps: {n_falls}")
    print(f"pre-fall window: last {args.window} steps before each fall\n")
    for name, arr in [
        ("|pitch|", np.abs(pitch)), ("|roll|", np.abs(roll)),
        ("|pitch_rate|", np.abs(pitch_rate)), ("height", height),
        ("balance_reward", r_balance), ("action_magnitude", action_mag),
    ]:
        print(f"{name}:")
        print(f"  pre-fall: {stats(pre_fall_mask, arr)}")
        print(f"  normal:   {stats(normal_mask, arr)}")
        if name == "height":
            print(f"  all:      {stats(~fell, arr)}")

    # Lead-time check: is pitch_rate an EARLY warning signal (rising steadily
    # before the fall) or does it only spike in the last step or two (i.e.
    # a symptom of falling, not a precursor)? Snapshots |pitch| and
    # |pitch_rate| at fixed offsets before each fall, not windowed averages.
    print("\nlead-time snapshot (value AT exactly N steps before each fall, not windowed):")
    print(f"{'steps before fall':<20}{'|pitch|':>14}{'|pitch_rate|':>16}")
    for offset in (50, 30, 20, 10, 5, 1):
        p_vals, pr_vals = [], []
        for t in range(args.steps):
            if not fell[t].any():
                continue
            src_t = t - offset
            if src_t < 0:
                continue
            lanes = fell[t]
            p_vals.append(np.abs(pitch[src_t, lanes]))
            pr_vals.append(np.abs(pitch_rate[src_t, lanes]))
        if not p_vals:
            print(f"{-offset:<20}{'n/a':>14}{'n/a':>16}")
            continue
        p_all, pr_all = np.concatenate(p_vals), np.concatenate(pr_vals)
        print(f"{-offset:<20}{np.median(p_all):>14.4f}{np.median(pr_all):>16.4f}")
    p_all_normal = np.abs(pitch)[normal_mask]
    pr_all_normal = np.abs(pitch_rate)[normal_mask]
    print(f"{'(normal, ref)':<20}{np.median(p_all_normal):>14.4f}{np.median(pr_all_normal):>16.4f}")

    # k_theta suggestion: solve k * p90(|pitch|)^2 = target_penalty for a
    # couple of target penalty magnitudes, using the PRE-FALL p90 (not max)
    # as the "this level of tilt should matter" anchor -- p90 rather than
    # p50 since we want the coefficient to register before most of the
    # pre-fall population, not only the median case.
    pitch_p90_prefall = np.percentile(np.abs(pitch)[pre_fall_mask], 90) if pre_fall_mask.any() else None
    if pitch_p90_prefall:
        print(f"\nk_theta suggestions (solving k * pre-fall_p90(|pitch|)^2 = target penalty, pre-fall p90(|pitch|)={pitch_p90_prefall:.4f}):")
        for target in (0.1, 0.2, 0.5):
            k = target / (pitch_p90_prefall ** 2)
            print(f"  target penalty={target}: k_theta ~= {k:.2f}")
    pitch_rate_p90_prefall = np.percentile(np.abs(pitch_rate)[pre_fall_mask], 90) if pre_fall_mask.any() else None
    if pitch_rate_p90_prefall:
        print(f"\nk_theta_dot suggestions (pre-fall p90(|pitch_rate|)={pitch_rate_p90_prefall:.4f} rad/s):")
        for target in (0.1, 0.2, 0.5):
            k = target / (pitch_rate_p90_prefall ** 2)
            print(f"  target penalty={target}: k_theta_dot ~= {k:.4f}")

    # height_coef suggestion: same p90-solve method, but anchored on the
    # non-fall window's height ERROR (not pre-fall -- crouching isn't a
    # fall-precursor, it's a steady-state loophole, so the whole
    # surviving trajectory is the relevant population here).
    height_err_all = np.abs(height - args.target_height)[~fell]
    if height_err_all.size:
        height_err_p90 = np.percentile(height_err_all, 90)
        print(f"\nheight_coef suggestions (solving k * p90(|height-target|)^2 = target penalty, "
              f"p90(|height-{args.target_height}|)={height_err_p90:.4f}):")
        for target in (0.1, 0.2, 0.5):
            k = target / (height_err_p90 ** 2)
            print(f"  target penalty={target}: height_coef ~= {k:.2f}")


if __name__ == "__main__":
    main()
