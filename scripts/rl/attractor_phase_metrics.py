#!/usr/bin/env python3
"""Phase-segmented post-transient analysis -- follow-up to
initial_drop_metrics.py's finding that steps 0-15 are dominated by a
checkpoint-independent, seed-dependent actuator-saturation transient (peak
torque pins at the A1's 33.5 Nm limit in 100% of lanes, both A and h025,
all 3 paired seeds). That answered "who drops more in the first 15 steps?"
(nobody differs consistently). The real question is what happens AFTER
that transient: does each checkpoint settle into a walking gait, a
stationary crouch, a repeated collapse-reset cycle, or recover into
locomotion?

Splits a full rollout (same paired cfg.seed mechanism as
initial_drop_metrics.py -- see that module's docstring for why bare
torch.manual_seed() after env creation doesn't reproduce trajectories)
into three phases and aggregates each separately:

  Phase A (0-15):    already characterized by initial_drop_metrics.py,
                      not recomputed here.
  Phase B (15-50):   immediate post-transient -- mean/std height, mean v_z,
                      mean v_x, pitch/pitch_rate, torque saturation
                      fraction, action magnitude, fall occurrence, and the
                      individual balance_reward sub-terms (not just the
                      combined total, same "don't collapse the mechanism
                      away" reasoning as balance_decomposition.py).
  Phase C (50-200):  longer-horizon attractor -- same metrics, to see
                      whether Phase B's state persists, drifts, or gives
                      way to repeated fall/reset cycling.

Per-step masking follows prefall_window_analysis.py's convention (exclude
only the exact terminal_fall step from each phase's stats, not everything
after it cumulatively) -- appropriate here since a lane can fall and reset
multiple times within a 35- or 150-step window, unlike PhysicsValidator's
cumulative-since-rollout-start masking (which would zero out all
post-first-fall data, discarding most of Phase C for any lane that ever
fell).

Also computes time-to-first-stable-band per lane: the first step (>=15)
starting a `--stable_window`-length run where |v_z| stays below
`--vz_thresh`, height stays within `--height_std_thresh` of its own
windowed mean, torque saturation fraction stays below `--sat_thresh`, and
no fall occurs in the window. Thresholds are starting points to separate
transient from settled behavior, not independently calibrated.

    PYTHONPATH="$(pwd):$(pwd)/scripts" python scripts/rl/attractor_phase_metrics.py \
        runs/<run>/checkpoints/checkpoint.pt 0.7 0.1 0.1 0.1 --seed 0 --steps 200 \
        --progress_std 0.5 --balance_tilt_coef 1.0 --target_height 0.25
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

A1_TORQUE_LIMIT_NM = 33.5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("w", type=float, nargs=4)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--command", type=float, nargs=3, default=(0.5, 0.0, 0.0), metavar=("VX", "VY", "WZ"))
    parser.add_argument("--seed", type=int, default=0, help="Sets cfg.seed BEFORE env creation (not just trainer.rng)")
    parser.add_argument("--phase_b", type=int, nargs=2, default=(15, 50), metavar=("LO", "HI"))
    parser.add_argument("--phase_c", type=int, nargs=2, default=(50, 200), metavar=("LO", "HI"))
    parser.add_argument("--stable_window", type=int, default=10)
    parser.add_argument("--vz_thresh", type=float, default=0.15)
    parser.add_argument("--height_std_thresh", type=float, default=0.02)
    parser.add_argument("--sat_thresh", type=float, default=0.05)
    parser.add_argument("--progress_std", type=float, default=None, help="Must match what the checkpoint was trained with")
    parser.add_argument("--balance_tilt_coef", type=float, default=None, help="Must match what the checkpoint was trained with")
    parser.add_argument("--balance_tilt_rate_coef", type=float, default=None, help="Must match what the checkpoint was trained with")
    parser.add_argument("--balance_height_coef", type=float, default=None, help="Must match what the checkpoint was trained with")
    parser.add_argument("--target_height", type=float, default=None, help="Must match what the checkpoint was trained with")
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
    reward_cfg_overrides = {}
    if args.progress_std is not None:
        reward_cfg_overrides["progress_std"] = args.progress_std
    if args.balance_tilt_coef is not None:
        reward_cfg_overrides["balance_tilt_coef"] = args.balance_tilt_coef
    if args.balance_tilt_rate_coef is not None:
        reward_cfg_overrides["balance_tilt_rate_coef"] = args.balance_tilt_rate_coef
    if args.balance_height_coef is not None:
        reward_cfg_overrides["balance_height_coef"] = args.balance_height_coef
    if args.target_height is not None:
        reward_cfg_overrides["target_height"] = args.target_height
    reward_cfg = RewardVectorCfg(**reward_cfg_overrides)
    pref_cfg = PreferenceCfg()
    stack_cfg = ObservationStackCfg()
    extrinsics_cfg = ExtrinsicsCfg()

    cfg = IsaacLabTalonEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed  # see module docstring -- must be set before gym.make()
    env = gym.make("Isaac-Talon-A1-v0", cfg=cfg, render_mode=None).unwrapped

    trainer = MOPPOTrainer(
        env, obs_cfg, reward_cfg, pref_cfg, moppo_cfg=MOPPOConfig(), stack_cfg=stack_cfg,
        extrinsics_cfg=extrinsics_cfg, seed=args.seed,
    )
    trainer.load(args.checkpoint)
    print(f"loaded checkpoint (t={trainer._t}), env seed={args.seed}")

    trainer.w = np.tile(np.array(args.w, dtype=np.float32) / sum(args.w), (args.num_envs, 1)).astype(np.float32)

    T, N = args.steps, args.num_envs
    height = np.zeros((T, N), dtype=np.float32)
    v_z = np.zeros((T, N), dtype=np.float32)
    v_x = np.zeros((T, N), dtype=np.float32)
    pitch = np.zeros((T, N), dtype=np.float32)
    pitch_rate = np.zeros((T, N), dtype=np.float32)
    action_mag = np.zeros((T, N), dtype=np.float32)
    torque_sat = np.zeros((T, N), dtype=bool)
    terminal_fall = np.zeros((T, N), dtype=bool)
    tilt_penalty = np.zeros((T, N), dtype=np.float32)
    v_z_penalty = np.zeros((T, N), dtype=np.float32)
    height_penalty = np.zeros((T, N), dtype=np.float32)

    trainer.model.eval()
    for t in range(T):
        env.v_command_buf[:] = torch.tensor(args.command, device=env.device)
        action = trainer.act_inference()
        transition, done = env.step(action)
        trainer._last_extrinsics = transition.get("extrinsics") if trainer.encoder else None
        trainer.push_obs(transition["obs"], done_mask=done)

        height[t] = transition["height"]
        v_z[t] = transition["v_z"]
        v_x[t] = transition["v_actual"][:, 0]
        pitch[t] = transition["roll_pitch"][:, 1]
        pitch_rate[t] = transition["roll_pitch_rate"][:, 1]
        action_mag[t] = np.abs(action).mean(axis=-1)
        torque = transition["joint_torque"]
        torque_sat[t] = (np.abs(torque) >= 0.95 * A1_TORQUE_LIMIT_NM).any(axis=-1)
        terminal_fall[t] = transition.get("terminal_fall", done).astype(bool)

        roll_pitch = transition["roll_pitch"]
        tilt_penalty[t] = -reward_cfg.balance_tilt_coef * np.sum(roll_pitch ** 2, axis=-1)
        v_z_penalty[t] = -(transition["v_z"] ** 2)
        height_penalty[t] = -reward_cfg.balance_height_coef * (transition["height"] - reward_cfg.target_height) ** 2

    def phase_stats(lo: int, hi: int, label: str) -> None:
        mask = ~terminal_fall[lo:hi]  # exclude only the instantaneous fall/reset step, not everything after
        n_valid = mask.sum()
        print(f"\n=== {label} [{lo},{hi}) ===")
        if n_valid == 0:
            print("  no alive steps in this window")
            return

        def m(arr: np.ndarray) -> float:
            return float(arr[lo:hi][mask].mean())

        def s(arr: np.ndarray) -> float:
            return float(arr[lo:hi][mask].std())

        print(f"  mean height (std)        {m(height):.4f} ({s(height):.4f})")
        print(f"  mean v_z                 {m(v_z):.4f}")
        print(f"  mean v_x (cmd {args.command[0]})       {m(v_x):.4f}")
        print(f"  mean |pitch|              {float(np.abs(pitch[lo:hi])[mask].mean()):.4f}")
        print(f"  mean |pitch_rate|         {float(np.abs(pitch_rate[lo:hi])[mask].mean()):.4f}")
        print(f"  torque saturation frac    {m(torque_sat.astype(np.float32)):.4f}")
        print(f"  mean action magnitude     {m(action_mag):.4f}")
        falls_in_window = terminal_fall[lo:hi]
        print(f"  falls per lane (mean)     {falls_in_window.sum(axis=0).mean():.4f}")
        print(f"  frac lanes with >=1 fall  {(falls_in_window.any(axis=0)).mean():.4f}")
        print(f"  balance sub-terms: tilt_penalty={m(tilt_penalty):.4f}  "
              f"v_z_penalty={m(v_z_penalty):.4f}  height_penalty={m(height_penalty):.4f}  "
              f"alive_bonus={reward_cfg.alive_bonus:.4f}")

    phase_stats(*args.phase_b, "Phase B (post-transient)")
    phase_stats(*args.phase_c, "Phase C (long-horizon attractor)")

    # time-to-first-stable-band: first step >=15 starting a
    # stable_window-length run meeting all three settled-behavior criteria.
    print(f"\n=== time-to-first-stable-band (window={args.stable_window}, "
          f"vz<{args.vz_thresh}, height_std<{args.height_std_thresh}, sat<{args.sat_thresh}) ===")
    first_stable = np.full(N, -1, dtype=np.int32)
    w = args.stable_window
    for start in range(15, T - w + 1):
        undecided = first_stable < 0
        if not undecided.any():
            break
        win_vz = np.abs(v_z[start:start + w])
        win_height = height[start:start + w]
        win_sat = torque_sat[start:start + w].astype(np.float32)
        win_fall = terminal_fall[start:start + w]
        ok = (
            (win_vz < args.vz_thresh).all(axis=0)
            & (win_height.std(axis=0) < args.height_std_thresh)
            & (win_sat.mean(axis=0) < args.sat_thresh)
            & (~win_fall.any(axis=0))
        )
        newly_stable = undecided & ok
        first_stable[newly_stable] = start

    stabilized = first_stable >= 0
    print(f"  fraction of lanes that ever stabilize: {stabilized.mean():.4f}")
    if stabilized.any():
        print(f"  time-to-stable among those: mean={first_stable[stabilized].mean():.1f}  "
              f"median={np.median(first_stable[stabilized]):.1f}")


if __name__ == "__main__":
    main()
