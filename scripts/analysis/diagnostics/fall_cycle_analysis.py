#!/usr/bin/env python3
"""Fall-cycle-aligned diagnosis -- Experiment 2A.1, follow-up to
attractor_phase_metrics.py's baseline finding that checkpoint A never
establishes stable locomotion under progress-heavy w + forced 0.5 m/s
command: falls recur roughly every 13-15 steps throughout the ENTIRE
rollout (median first-fall time ~12-13, steady-state Phase C fall rate
matches the same period), not just during an initial transient that then
clears. That makes the Phase B (15-50) / Phase C (50-200) split a
reporting convenience, not a mechanistically meaningful boundary for this
checkpoint -- the real unit of analysis is the ~13-15-step fall cycle
itself, repeated many times per lane.

Aggregate numbers (mean over a whole phase) already showed torque
saturation, action saturation, pitch-rate spikes, and vertical
instability are all elevated together -- but aggregates can't say which
one starts deviating from normal FIRST. That ordering is what actually
constrains which intervention (reward shaping vs action scale vs gait/
contact dynamics) is worth trying next -- see this script's own decision
tree in the module docstring's caller (talon-thesis 03_Daily_Notes).

For every terminal_fall event across ALL lanes and ALL steps (not just
one phase), extracts the last `--window` steps before it (t=-window..-1,
fall at t=0) for: pitch, roll, pitch_rate, height, v_z, v_x, action
magnitude, action saturation, torque magnitude, torque saturation,
balance_reward, progress_reward. Reports:

  1. pre-fall vs normal-elsewhere stats (same window-aggregate style as
     prefall_window_analysis.py, extended to the full variable set).
  2. A fine-grained lead-time snapshot (median value at exact offsets,
     not windowed) near the fall.
  3. A "first onset offset" per variable: scanning from -window toward
     -1, the first offset where the pre-fall population's median value
     (or, for the two saturation booleans, fraction of lanes saturating)
     deviates from the normal-elsewhere reference by more than
     `--z_thresh` normal-population standard deviations (or, for the
     saturation fractions, more than `--sat_ratio_thresh` times the
     normal fraction). This is a first statistical cut at "who moves
     first", not a rigorous change-point detector -- read the raw
     offset-by-offset numbers alongside it, not the onset ranking alone.

    PYTHONPATH="$(pwd):$(pwd)/scripts" python scripts/analysis/run_diagnostics.py fall-cycle-analysis \
        runs/<run>/checkpoints/checkpoint.pt 0.7 0.1 0.1 0.1 --seed 0 --steps 200 --window 15 \
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
from talon_rl.rewards.locomotion import compute_reward_vector

from rl.core.algorithms import MOPPOConfig, MOPPOTrainer

A1_TORQUE_LIMIT_NM = 33.5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("w", type=float, nargs=4)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--window", type=int, default=15, help="Fall-cycle length to analyze before each fall")
    parser.add_argument("--command", type=float, nargs=3, default=(0.5, 0.0, 0.0), metavar=("VX", "VY", "WZ"))
    parser.add_argument("--seed", type=int, default=0, help="Sets cfg.seed BEFORE env creation (not just trainer.rng)")
    parser.add_argument("--z_thresh", type=float, default=2.0, help="Onset detection: |z-score| vs normal population")
    parser.add_argument("--sat_ratio_thresh", type=float, default=2.0, help="Onset detection for sat fractions: ratio vs normal")
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
    cfg.seed = args.seed  # see fall_cycle_analysis module docstring / feedback_talon_rl_env_seeding memory
    env = gym.make("Isaac-Talon-A1-v0", cfg=cfg, render_mode=None).unwrapped

    trainer = MOPPOTrainer(
        env, obs_cfg, reward_cfg, pref_cfg, moppo_cfg=MOPPOConfig(), stack_cfg=stack_cfg,
        extrinsics_cfg=extrinsics_cfg, seed=args.seed,
    )
    trainer.load(args.checkpoint)
    action_clip = trainer.model.ACTION_CLIP
    print(f"loaded checkpoint (t={trainer._t}), env seed={args.seed}, ACTION_CLIP={action_clip}")

    trainer.w = np.tile(np.array(args.w, dtype=np.float32) / sum(args.w), (args.num_envs, 1)).astype(np.float32)
    progress_idx = reward_cfg.term_names.index("progress")
    balance_idx = reward_cfg.term_names.index("balance")

    T, N = args.steps, args.num_envs
    pitch = np.zeros((T, N), dtype=np.float32)
    roll = np.zeros((T, N), dtype=np.float32)
    pitch_rate = np.zeros((T, N), dtype=np.float32)
    height = np.zeros((T, N), dtype=np.float32)
    v_z = np.zeros((T, N), dtype=np.float32)
    v_x = np.zeros((T, N), dtype=np.float32)
    action_mag = np.zeros((T, N), dtype=np.float32)
    action_sat = np.zeros((T, N), dtype=np.float32)  # 0/1, kept float for windowed-mean convenience
    torque_mag = np.zeros((T, N), dtype=np.float32)
    torque_sat = np.zeros((T, N), dtype=np.float32)
    r_balance = np.zeros((T, N), dtype=np.float32)
    r_progress = np.zeros((T, N), dtype=np.float32)
    fell = np.zeros((T, N), dtype=bool)

    trainer.model.eval()
    for t in range(T):
        env.v_command_buf[:] = torch.tensor(args.command, device=env.device)
        action = trainer.act_inference()
        transition, done = env.step(action)
        trainer._last_extrinsics = transition.get("extrinsics") if trainer.encoder else None
        trainer.push_obs(transition["obs"], done_mask=done)

        pitch[t] = transition["roll_pitch"][:, 1]
        roll[t] = transition["roll_pitch"][:, 0]
        pitch_rate[t] = transition["roll_pitch_rate"][:, 1]
        height[t] = transition["height"]
        v_z[t] = transition["v_z"]
        v_x[t] = transition["v_actual"][:, 0]
        action_mag[t] = np.abs(action).mean(axis=-1)
        action_sat[t] = (np.abs(action) >= 0.95 * action_clip).any(axis=-1).astype(np.float32)
        torque = transition["joint_torque"]
        torque_mag[t] = np.abs(torque).mean(axis=-1)
        torque_sat[t] = (np.abs(torque) >= 0.95 * A1_TORQUE_LIMIT_NM).any(axis=-1).astype(np.float32)
        fell[t] = transition.get("terminal_fall", done).astype(bool)

        reward_vec = compute_reward_vector(transition, reward_cfg)
        r_balance[t] = reward_vec[:, balance_idx]
        r_progress[t] = reward_vec[:, progress_idx]

    variables = {
        "|pitch|": np.abs(pitch), "|roll|": np.abs(roll), "|pitch_rate|": np.abs(pitch_rate),
        "height": height, "v_z": v_z, "v_x": v_x,
        "action_magnitude": action_mag, "action_saturation": action_sat,
        "torque_magnitude": torque_mag, "torque_saturation": torque_sat,
        "balance_reward": r_balance, "progress_reward": r_progress,
    }
    # These two are booleans-as-float (per-step saturation indicator) --
    # onset detection for them compares FRACTION-of-lanes against a ratio
    # threshold, not a z-score against a continuous distribution.
    fraction_vars = {"action_saturation", "torque_saturation"}

    pre_fall_mask = np.zeros_like(fell)
    for t in range(T):
        if fell[t].any():
            lo = max(0, t - args.window)
            pre_fall_mask[lo:t, fell[t]] = True
    normal_mask = ~pre_fall_mask & ~fell

    n_falls = int(fell.sum())
    print(f"\ntotal fall events across {N} lanes x {T} steps: {n_falls}")
    print(f"fall-cycle window: last {args.window} steps before each fall\n")

    def pctstats(mask: np.ndarray, arr: np.ndarray) -> str:
        vals = arr[mask]
        if vals.size == 0:
            return "n/a"
        p50, p90 = np.percentile(vals, [50, 90])
        return f"mean={np.mean(vals):.4f} p50={p50:.4f} p90={p90:.4f} (n={vals.size})"

    for name, arr in variables.items():
        print(f"{name}:")
        print(f"  pre-fall: {pctstats(pre_fall_mask, arr)}")
        print(f"  normal:   {pctstats(normal_mask, arr)}")

    # Lead-time snapshot: value AT exactly N steps before each fall, not
    # windowed -- same convention as prefall_window_analysis.py.
    print(f"\nlead-time snapshot (value AT exactly N steps before each fall):")
    offsets = list(range(-args.window, 0))
    snapshot_offsets = [o for o in offsets if o in (-15, -12, -10, -8, -6, -5, -4, -3, -2, -1) or o == -args.window]
    snapshot_offsets = sorted(set(snapshot_offsets))
    header = f"{'offset':<8}" + "".join(f"{name:>16}" for name in variables)
    print(header)
    # snapshot[name][offset] = array of values across all (fall_step, lane) pairs
    snapshot = {name: {} for name in variables}
    for offset in snapshot_offsets:
        row_vals = {}
        for t in range(T):
            if not fell[t].any():
                continue
            src_t = t + offset
            if src_t < 0:
                continue
            lanes = fell[t]
            for name, arr in variables.items():
                row_vals.setdefault(name, []).append(arr[src_t, lanes])
        line = f"{offset:<8}"
        for name in variables:
            if name in row_vals:
                vals = np.concatenate(row_vals[name])
                med = float(np.median(vals))
                snapshot[name][offset] = med
                line += f"{med:>16.4f}"
            else:
                line += f"{'n/a':>16}"
        print(line)
    normal_ref = {name: float(np.median(arr[normal_mask])) for name, arr in variables.items()}
    normal_std = {name: float(np.std(arr[normal_mask])) for name, arr in variables.items()}
    normal_frac = {name: float(arr[normal_mask].mean()) for name, arr in variables.items() if name in fraction_vars}
    ref_line = f"{'normal':<8}" + "".join(f"{normal_ref[name]:>16.4f}" for name in variables)
    print(ref_line)

    # First-onset-offset: scanning from -window toward -1, first offset
    # where the pre-fall snapshot median deviates from normal by more than
    # z_thresh std (continuous vars) or exceeds sat_ratio_thresh x the
    # normal fraction (saturation booleans).
    print(f"\nfirst-onset offset per variable (z_thresh={args.z_thresh}, sat_ratio_thresh={args.sat_ratio_thresh}):")
    onsets = {}
    for name in variables:
        onset = None
        for offset in sorted(snapshot[name]):
            val = snapshot[name][offset]
            if name in fraction_vars:
                deviates = normal_frac[name] > 0 and val >= args.sat_ratio_thresh * normal_frac[name]
            else:
                std = normal_std[name] or 1e-6
                deviates = abs(val - normal_ref[name]) / std >= args.z_thresh
            if deviates:
                onset = offset
                break
        onsets[name] = onset
        print(f"  {name:<20}{onset if onset is not None else 'no deviation detected':>10}")

    print("\nranked by earliest onset (most negative offset = deviates soonest):")
    ranked = sorted(((v, k) for k, v in onsets.items() if v is not None), key=lambda x: x[0])
    for offset, name in ranked:
        print(f"  {offset:>5}  {name}")


if __name__ == "__main__":
    main()
