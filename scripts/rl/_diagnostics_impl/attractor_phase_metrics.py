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

2026-09-20 (Experiment 2A, locomotion baseline): also reports v_x
tracking MAE/RMSE against the forced command (not just mean v_x --
comparable mean can hide oscillation around the target) and action
saturation fraction (|action|>=95% of ACTION_CLIP, same convention as
PhysicsValidator) alongside the existing torque saturation fraction --
action saturation is a policy-output-scale question, torque saturation
is an actuator-authority question, and conflating them was exactly what
this arc's decision tree (spawn/transient -> locomotion establishment ->
long-horizon stability, each branch pointing at a different root cause:
actuator/domain-randomization, balance control/reward, action
authority/dynamics, or locomotion learning) is trying to avoid. Also
reports max (not just mean) |pitch|/|pitch_rate| per phase, since a
runaway is a peak/escalation question a windowed mean can dilute away.

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

    PYTHONPATH="$(pwd):$(pwd)/scripts" python scripts/rl/diagnostics.py attractor-phase-metrics \
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
from talon_rl.reward import compute_reward_vector

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
    parser.add_argument(
        "--functional_vz_thresh", type=float, default=0.3,
        help="Functional-locomotion-fraction: |v_z| below this counts as controlled vertical motion",
    )
    parser.add_argument(
        "--functional_height_band", type=float, nargs=2, default=(0.15, 0.35), metavar=("LO", "HI"),
        help="Functional-locomotion-fraction: height must sit in this band (empirically the "
             "non-collapsed operating range seen this session, not target_height -- no checkpoint "
             "tested has actually held target_height=0.42)",
    )
    parser.add_argument(
        "--functional_vx_thresh", type=float, default=0.2,
        help="Functional-locomotion-fraction: |v_x - command| below this counts as tracking",
    )
    parser.add_argument("--progress_std", type=float, default=None, help="Must match what the checkpoint was trained with")
    parser.add_argument("--balance_tilt_coef", type=float, default=None, help="Must match what the checkpoint was trained with")
    parser.add_argument("--balance_tilt_rate_coef", type=float, default=None, help="Must match what the checkpoint was trained with")
    parser.add_argument("--balance_height_coef", type=float, default=None, help="Must match what the checkpoint was trained with")
    parser.add_argument("--target_height", type=float, default=None, help="Must match what the checkpoint was trained with")
    parser.add_argument("--balance_hip_activation_coef", type=float, default=None, help="Must match what the checkpoint was trained with")
    parser.add_argument("--balance_hip_sym_coef", type=float, default=None, help="Must match what the checkpoint was trained with")
    parser.add_argument("--sim_dt", type=float, default=0.02, help="Physics timestep, s -- 2E locked dt=0.01 as the numerically-validated diagnostic timestep (2026-09-20); this lets a checkpoint TRAINED at dt=0.02 be EVALUATED at a different dt to test training/eval timestep sensitivity, independent of retraining")
    parser.add_argument("--decimation", type=int, default=1)
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
    if args.balance_hip_activation_coef is not None:
        reward_cfg_overrides["balance_hip_activation_coef"] = args.balance_hip_activation_coef
    if args.balance_hip_sym_coef is not None:
        reward_cfg_overrides["balance_hip_sym_coef"] = args.balance_hip_sym_coef
    reward_cfg = RewardVectorCfg(**reward_cfg_overrides)
    pref_cfg = PreferenceCfg()
    stack_cfg = ObservationStackCfg()
    extrinsics_cfg = ExtrinsicsCfg()

    cfg = IsaacLabTalonEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed  # see module docstring -- must be set before gym.make()
    cfg.sim.dt = args.sim_dt
    cfg.decimation = args.decimation
    cfg.sim.render_interval = cfg.decimation
    env = gym.make("Isaac-Talon-A1-v0", cfg=cfg, render_mode=None).unwrapped

    trainer = MOPPOTrainer(
        env, obs_cfg, reward_cfg, pref_cfg, moppo_cfg=MOPPOConfig(), stack_cfg=stack_cfg,
        extrinsics_cfg=extrinsics_cfg, seed=args.seed,
    )
    trainer.load(args.checkpoint)
    action_clip = trainer.model.ACTION_CLIP
    print(f"loaded checkpoint (t={trainer._t}), env seed={args.seed}, ACTION_CLIP={action_clip}")

    trainer.w = np.tile(np.array(args.w, dtype=np.float32) / sum(args.w), (args.num_envs, 1)).astype(np.float32)

    CONTACT_THRESHOLD_N = 1.0
    foot_body_ids = env._foot_body_ids
    foot_names = [env.scene["robot"].body_names[i] for i in foot_body_ids]
    left_feet = [i for i, n in enumerate(foot_names) if n.startswith("FL") or n.startswith("RL")]
    right_feet = [i for i, n in enumerate(foot_names) if n.startswith("FR") or n.startswith("RR")]

    T, N = args.steps, args.num_envs
    height = np.zeros((T, N), dtype=np.float32)
    v_z = np.zeros((T, N), dtype=np.float32)
    v_x = np.zeros((T, N), dtype=np.float32)
    pitch = np.zeros((T, N), dtype=np.float32)
    pitch_rate = np.zeros((T, N), dtype=np.float32)
    action_mag = np.zeros((T, N), dtype=np.float32)
    action_sat = np.zeros((T, N), dtype=bool)
    torque_sat = np.zeros((T, N), dtype=bool)
    terminal_fall = np.zeros((T, N), dtype=bool)
    tilt_penalty = np.zeros((T, N), dtype=np.float32)
    v_z_penalty = np.zeros((T, N), dtype=np.float32)
    height_penalty = np.zeros((T, N), dtype=np.float32)
    r_progress = np.zeros((T, N), dtype=np.float32)
    r_balance = np.zeros((T, N), dtype=np.float32)
    n_feet_contact = np.zeros((T, N), dtype=np.float32)
    contact_L = np.zeros((T, N), dtype=np.float32)
    contact_R = np.zeros((T, N), dtype=np.float32)
    hip_qdot_L = np.zeros((T, N), dtype=np.float32)
    hip_qdot_R = np.zeros((T, N), dtype=np.float32)

    progress_idx = reward_cfg.term_names.index("progress")
    balance_idx = reward_cfg.term_names.index("balance")

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
        action_sat[t] = (np.abs(action) >= 0.95 * action_clip).any(axis=-1)
        torque = transition["joint_torque"]
        torque_sat[t] = (np.abs(torque) >= 0.95 * A1_TORQUE_LIMIT_NM).any(axis=-1)
        terminal_fall[t] = transition.get("terminal_fall", done).astype(bool)

        roll_pitch = transition["roll_pitch"]
        tilt_penalty[t] = -reward_cfg.balance_tilt_coef * np.sum(roll_pitch ** 2, axis=-1)
        v_z_penalty[t] = -(transition["v_z"] ** 2)
        height_penalty[t] = -reward_cfg.balance_height_coef * (transition["height"] - reward_cfg.target_height) ** 2

        reward_vec = compute_reward_vector(transition, reward_cfg)
        r_progress[t] = reward_vec[:, progress_idx]
        r_balance[t] = reward_vec[:, balance_idx]

        contact = transition["foot_contact_force"] > CONTACT_THRESHOLD_N  # (N, 4)
        n_feet_contact[t] = contact.sum(axis=-1)
        contact_L[t] = contact[:, left_feet].mean(axis=-1)
        contact_R[t] = contact[:, right_feet].mean(axis=-1)
        hip_qdot_L[t] = transition["hip_qdot_L"]
        hip_qdot_R[t] = transition["hip_qdot_R"]

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

        v_x_err = v_x[lo:hi] - args.command[0]
        v_x_mae = float(np.abs(v_x_err)[mask].mean())
        v_x_rmse = float(np.sqrt((v_x_err[mask] ** 2).mean()))
        abs_pitch = np.abs(pitch[lo:hi])
        abs_pitch_rate = np.abs(pitch_rate[lo:hi])

        print(f"  mean height (std)        {m(height):.4f} ({s(height):.4f})")
        print(f"  mean v_z                 {m(v_z):.4f}")
        print(f"  mean v_x (cmd {args.command[0]})       {m(v_x):.4f}")
        print(f"  v_x tracking MAE / RMSE   {v_x_mae:.4f} / {v_x_rmse:.4f}")
        print(f"  mean/max |pitch|          {float(abs_pitch[mask].mean()):.4f} / {float(abs_pitch[mask].max()):.4f}")
        print(f"  mean/max |pitch_rate|     {float(abs_pitch_rate[mask].mean()):.4f} / {float(abs_pitch_rate[mask].max()):.4f}")
        print(f"  torque saturation frac    {m(torque_sat.astype(np.float32)):.4f}")
        print(f"  action saturation frac    {m(action_sat.astype(np.float32)):.4f}")
        print(f"  mean action magnitude     {m(action_mag):.4f}")
        falls_in_window = terminal_fall[lo:hi]
        print(f"  falls per lane (mean)     {falls_in_window.sum(axis=0).mean():.4f}")
        print(f"  frac lanes with >=1 fall  {(falls_in_window.any(axis=0)).mean():.4f}")
        print(f"  balance sub-terms: tilt_penalty={m(tilt_penalty):.4f}  "
              f"v_z_penalty={m(v_z_penalty):.4f}  height_penalty={m(height_penalty):.4f}  "
              f"alive_bonus={reward_cfg.alive_bonus:.4f}")
        print(f"  mean progress_reward      {m(r_progress):.4f}")
        print(f"  mean balance_reward       {m(r_balance):.4f}")
        print(f"  mean n_feet_contact       {m(n_feet_contact):.4f}")
        print(f"  mean contact_L / contact_R {m(contact_L):.4f} / {m(contact_R):.4f}")
        print(f"  mean |hip_qdot_L| / |hip_qdot_R| "
              f"{float(np.abs(hip_qdot_L[lo:hi])[mask].mean()):.4f} / "
              f"{float(np.abs(hip_qdot_R[lo:hi])[mask].mean()):.4f}")
        # Contact-conditioned hip activity: mean |hip_qdot| while that
        # side's own feet ARE in contact (stance) vs are NOT (swing) --
        # answers whether hip motion concentrates in swing phase (normal
        # gait) or is uniform regardless of contact (more consistent with
        # jitter/reward-satisfying motion than a coordinated step cycle).
        for side, qdot, contact in (("L", hip_qdot_L, contact_L), ("R", hip_qdot_R, contact_R)):
            c = contact[lo:hi][mask]
            q = np.abs(qdot[lo:hi][mask])
            stance = c >= 0.5
            swing = ~stance
            stance_val = float(q[stance].mean()) if stance.any() else float("nan")
            swing_val = float(q[swing].mean()) if swing.any() else float("nan")
            print(f"  hip_{side} |qdot| stance/swing {stance_val:.4f} / {swing_val:.4f}")

        # Functional-locomotion fraction: fraction of alive steps meeting
        # ALL THREE simultaneously (controlled v_z, height in the
        # empirically-observed non-collapsed band, v_x tracking within
        # tolerance). A blunter "falls/lane" count can't distinguish a
        # policy with real (if intermittent) functional segments from one
        # that never achieves anything -- this measures the segments
        # directly, per Experiment 2A.5's own finding that improvement
        # showed up as periods of good behavior, not a uniformly-better
        # rollout.
        height_lo, height_hi = args.functional_height_band
        functional = (
            (np.abs(v_z[lo:hi]) < args.functional_vz_thresh)
            & (height[lo:hi] >= height_lo) & (height[lo:hi] <= height_hi)
            & (np.abs(v_x[lo:hi] - args.command[0]) < args.functional_vx_thresh)
        )
        print(f"  functional locomotion frac {float(functional[mask].mean()):.4f} "
              f"(|vz|<{args.functional_vz_thresh}, h in [{height_lo},{height_hi}], "
              f"|vx-cmd|<{args.functional_vx_thresh})")

    # Whole-rollout survival (first-fall time, capped at T) -- distinct from
    # per-phase "falls per lane" (which counts every fall/reset cycle within
    # a window); this is the classic "how long before it first goes down"
    # number, same convention as PhysicsValidator.mean_survival.
    first_fall = np.where(terminal_fall.any(axis=0), terminal_fall.argmax(axis=0), T)
    print(f"\n=== survival (first fall, capped at {T}) ===")
    print(f"  mean={first_fall.mean():.1f}  median={np.median(first_fall):.1f}  "
          f"pct_never_fell={(first_fall == T).mean() * 100:.1f}%")

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
