#!/usr/bin/env python3
"""Experiment 2A.3 (revised) -- Hip/Gait Asymmetry Diagnosis. The K=5
progress-leak-fix retrain (moppo.py's progress_leak_window) changed the
reward ranking and improved action saturation/pitch peaks in most seeds,
but did NOT fix falls/lane or the persistent hip L/R asymmetry
gait_joint_trace.py found (hip_L actively oscillating, hip_R nearly
frozen, same shape before and after the leak fix) -- the leak wasn't the
bottleneck. This asks the sharper question directly: does the POLICY
intend asymmetric legs (q_target itself differs L/R from early in each
cycle), or do both legs get a similar target and the SYSTEM responds
differently (q_actual/torque diverge while q_target stays close)?

For hip, thigh, and calf (L = {FL, RL}, R = {FR, RR}, averaged within
each side -- same grouping as gait_joint_trace.py), computes per step,
aggregated across ALL lanes (not one traced lane):

    dq_target = |q_target_L - q_target_R|
    dq        = |q_actual_L - q_actual_R|
    dtau_desired = |tau_desired_L - tau_desired_R|
    dtau_applied = |tau_applied_L - tau_applied_R|

then reports, matching fall_cycle_analysis.py's own methodology (pre-fall
window vs normal-elsewhere percentiles, plus a fine-grained lead-time
snapshot near each fall) for each of these four series per joint type.
Also reports each series' value in the FIRST 10 steps after a reset vs
the LAST 10 steps before the next fall (same lane's own inter-fall
segment) -- directly tests whether asymmetry is present from the start
of a cycle (learned/structural) or only grows late (a balance-loss
consequence), the "target ต่างเฉพาะหลังเริ่มเสีย balance" row in the
diagnosis table this responds to.

Reading the four series together answers the six-row diagnosis table:
  - dq_target large even early in a cycle -> policy intends asymmetric
    legs (learned/structural gait choice).
  - dq_target small but dq large -> control/dynamics/contact divergence,
    not a policy choice.
  - dq_target only grows in the last few steps before a fall -> asymmetry
    is a CONSEQUENCE of balance loss, not its cause.
  - dtau large while dq stays small -> actuator/contact loading
    difference (same position, different force needed to hold it).
  - Consistent shape across all 3 seeds -> structural gait pattern.
  - Magnitude/timing varies a lot by seed -> more stochastic/policy-
    behavior-dependent than a fixed structural cause.

    PYTHONPATH="$(pwd):$(pwd)/scripts" python scripts/analysis/run_diagnostics.py hip-asymmetry-analysis \
        runs/<run>/checkpoints/checkpoint.pt 0.7 0.1 0.1 0.1 --seed 0 --steps 200 --window 10 \
        --progress_std 0.5 --balance_tilt_coef 1.0 --target_height 0.42
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("w", type=float, nargs=4)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--window", type=int, default=10, help="Pre-fall window / early-vs-late segment length")
    parser.add_argument("--command", type=float, nargs=3, default=(0.5, 0.0, 0.0), metavar=("VX", "VY", "WZ"))
    parser.add_argument("--seed", type=int, default=0, help="Sets cfg.seed BEFORE env creation (not just trainer.rng)")
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
    cfg.seed = args.seed  # see feedback_talon_rl_env_seeding memory
    env = gym.make("Isaac-Talon-A1-v0", cfg=cfg, render_mode=None).unwrapped

    trainer = MOPPOTrainer(
        env, obs_cfg, reward_cfg, pref_cfg, moppo_cfg=MOPPOConfig(), stack_cfg=stack_cfg,
        extrinsics_cfg=extrinsics_cfg, seed=args.seed,
    )
    trainer.load(args.checkpoint)
    print(f"loaded checkpoint (t={trainer._t}), env seed={args.seed}")

    trainer.w = np.tile(np.array(args.w, dtype=np.float32) / sum(args.w), (args.num_envs, 1)).astype(np.float32)

    robot = env.scene["robot"]
    action_term = env.action_manager._terms["joint_pos"]
    joint_names = list(action_term._joint_names)
    action_scale = float(action_term.cfg.scale)
    default_joint_pos = action_term._offset[0].cpu().numpy()
    kp = float(cfg.scene.robot.actuators["base_legs"].stiffness)
    kd = float(cfg.scene.robot.actuators["base_legs"].damping)

    def side_idx(joint_type: str, side: str) -> list[int]:
        legs = ("FL", "RL") if side == "L" else ("FR", "RR")
        return [i for i, n in enumerate(joint_names) if joint_type in n and any(n.startswith(leg) for leg in legs)]

    joint_types = ("hip", "thigh", "calf")
    idx = {(jt, side): side_idx(jt, side) for jt in joint_types for side in ("L", "R")}

    T, N = args.steps, args.num_envs
    series = {
        f"{jt}_{metric}": np.zeros((T, N), dtype=np.float32)
        for jt in joint_types for metric in ("dq_target", "dq", "dtau_desired", "dtau_applied")
    }
    fell = np.zeros((T, N), dtype=bool)

    trainer.model.eval()
    for t in range(T):
        env.v_command_buf[:] = torch.tensor(args.command, device=env.device)
        action = trainer.act_inference()

        joint_pos_pre = robot.data.joint_pos.cpu().numpy()  # (N, 12)
        joint_vel_pre = robot.data.joint_vel.cpu().numpy()

        transition, done = env.step(action)
        trainer._last_extrinsics = transition.get("extrinsics") if trainer.encoder else None
        trainer.push_obs(transition["obs"], done_mask=done)

        fell[t] = transition.get("terminal_fall", done).astype(bool)

        target = action * action_scale + default_joint_pos[None, :]  # (N, 12)
        tau_desired_full = kp * (target - joint_pos_pre) - kd * joint_vel_pre  # (N, 12)
        tau_applied_full = transition["joint_torque"]  # (N, 12)

        for jt in joint_types:
            q_target_L = target[:, idx[(jt, "L")]].mean(axis=-1)
            q_target_R = target[:, idx[(jt, "R")]].mean(axis=-1)
            q_L = joint_pos_pre[:, idx[(jt, "L")]].mean(axis=-1)
            q_R = joint_pos_pre[:, idx[(jt, "R")]].mean(axis=-1)
            tau_d_L = tau_desired_full[:, idx[(jt, "L")]].mean(axis=-1)
            tau_d_R = tau_desired_full[:, idx[(jt, "R")]].mean(axis=-1)
            tau_a_L = tau_applied_full[:, idx[(jt, "L")]].mean(axis=-1)
            tau_a_R = tau_applied_full[:, idx[(jt, "R")]].mean(axis=-1)

            series[f"{jt}_dq_target"][t] = np.abs(q_target_L - q_target_R)
            series[f"{jt}_dq"][t] = np.abs(q_L - q_R)
            series[f"{jt}_dtau_desired"][t] = np.abs(tau_d_L - tau_d_R)
            series[f"{jt}_dtau_applied"][t] = np.abs(tau_a_L - tau_a_R)

    w = args.window
    pre_fall_mask = np.zeros_like(fell)
    for t in range(T):
        if fell[t].any():
            lo = max(0, t - w)
            pre_fall_mask[lo:t, fell[t]] = True
    normal_mask = ~pre_fall_mask & ~fell

    def pctstats(mask: np.ndarray, arr: np.ndarray) -> str:
        vals = arr[mask]
        if vals.size == 0:
            return "n/a"
        p50, p90 = np.percentile(vals, [50, 90])
        return f"mean={np.mean(vals):.4f} p50={p50:.4f} p90={p90:.4f} (n={vals.size})"

    n_falls = int(fell.sum())
    print(f"\ntotal fall events across {N} lanes x {T} steps: {n_falls}")
    print(f"pre-fall window: last {w} steps before each fall\n")

    for jt in joint_types:
        print(f"=== {jt} ===")
        for metric in ("dq_target", "dq", "dtau_desired", "dtau_applied"):
            arr = series[f"{jt}_{metric}"]
            print(f"  {metric}:")
            print(f"    pre-fall: {pctstats(pre_fall_mask, arr)}")
            print(f"    normal:   {pctstats(normal_mask, arr)}")
        print()

    # Early-vs-late within each inter-fall cycle: first `window` steps
    # after a reset vs last `window` steps before the next fall, SAME
    # lane's own segment -- tests whether asymmetry is present from the
    # start of a cycle or only grows late.
    print(f"early-vs-late per inter-fall cycle (first {w} steps after reset vs last {w} steps before next fall):")
    fall_steps_per_lane = [np.where(fell[:, n])[0] for n in range(N)]
    for jt in joint_types:
        for metric in ("dq_target", "dq", "dtau_desired", "dtau_applied"):
            arr = series[f"{jt}_{metric}"]
            early_vals, late_vals = [], []
            for n in range(N):
                fs = fall_steps_per_lane[n]
                segment_starts = np.concatenate(([0], fs[:-1] + 1)) if len(fs) else np.array([], dtype=int)
                for seg_start, seg_end in zip(segment_starts, fs):
                    if seg_end - seg_start < 2 * w:
                        continue  # too short to split cleanly into early/late halves
                    early_vals.append(arr[seg_start:seg_start + w, n])
                    late_vals.append(arr[seg_end - w:seg_end, n])
            if early_vals:
                early_mean = float(np.concatenate(early_vals).mean())
                late_mean = float(np.concatenate(late_vals).mean())
                print(f"  {jt}_{metric:<14} early={early_mean:.4f}  late={late_mean:.4f}  "
                      f"late/early={late_mean / early_mean if early_mean else float('nan'):.2f}x  "
                      f"(n_segments={len(early_vals)})")
            else:
                print(f"  {jt}_{metric:<14} no segments long enough (>= {2 * w} steps) to split")


if __name__ == "__main__":
    main()
