#!/usr/bin/env python3
"""Experiment 2A.4 prep -- empirical activity-ratio statistic, before
designing any anti-passive-hip reward. hip_symmetry_intervention.py gave
directional causal evidence that a collapsed (near-frozen) hip
contributes to falling, not bilateral asymmetry itself (forcing the
passive hip to mirror the active one helped or was neutral; forcing the
active hip to mirror the passive one hurt). Before picking an "activity
floor" threshold from intuition, this measures it from the checkpoint's
own gait statistics directly.

Per inter-fall SEGMENT (from a reset/previous fall to the next fall, or
to the end of the rollout for a lane's final still-running segment --
segment LENGTH itself is the stability proxy here: a lane that falls
again quickly produced a short segment, one that keeps going produced a
long one, so this doesn't need a separately hand-labeled "stable vs
falling" dataset), and per joint type (hip/thigh/calf), computes:

    A_L = std(q_target_L) over the segment
    A_R = std(q_target_R) over the segment
    rho_A = min(A_L, A_R) / max(A_L, A_R)   (1.0 = equal activity, ->0 = one side collapsed)

then reports rho_A grouped by segment-length tercile (short/mid/long) and
the Spearman correlation between segment length and rho_A, per joint
type per seed -- the empirical version of the hypothesis "stable gait:
A_L ~= A_R, falling gait: A_L >> A_R, with a fairly clean separating
threshold". If long segments cluster near rho_A~1 and short segments
cluster low with visible separation, that threshold (not a guessed
constant) is what an activity-floor reward term should target.

    PYTHONPATH="$(pwd):$(pwd)/scripts" python scripts/rl/diagnostics.py gait-activity-ratio \
        runs/<run>/checkpoints/checkpoint.pt 0.7 0.1 0.1 0.1 --seed 0 --steps 200 --min_segment_len 5 \
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


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    def rank(x: np.ndarray) -> np.ndarray:
        return np.argsort(np.argsort(x))
    ra, rb = rank(a).astype(np.float64), rank(b).astype(np.float64)
    if ra.size < 2 or np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("w", type=float, nargs=4)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--min_segment_len", type=int, default=5, help="Skip segments shorter than this (unreliable std estimate)")
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

    action_term = env.action_manager._terms["joint_pos"]
    joint_names = list(action_term._joint_names)
    action_scale = float(action_term.cfg.scale)
    default_joint_pos = action_term._offset[0].cpu().numpy()

    def side_idx(joint_type: str, side: str) -> list[int]:
        legs = ("FL", "RL") if side == "L" else ("FR", "RR")
        return [i for i, n in enumerate(joint_names) if joint_type in n and any(n.startswith(leg) for leg in legs)]

    joint_types = ("hip", "thigh", "calf")
    idx = {(jt, side): side_idx(jt, side) for jt in joint_types for side in ("L", "R")}

    T, N = args.steps, args.num_envs
    q_target = {jt: {"L": np.zeros((T, N), dtype=np.float32), "R": np.zeros((T, N), dtype=np.float32)} for jt in joint_types}
    fell = np.zeros((T, N), dtype=bool)

    trainer.model.eval()
    for t in range(T):
        env.v_command_buf[:] = torch.tensor(args.command, device=env.device)
        action = trainer.act_inference()
        transition, done = env.step(action)
        trainer._last_extrinsics = transition.get("extrinsics") if trainer.encoder else None
        trainer.push_obs(transition["obs"], done_mask=done)

        fell[t] = transition.get("terminal_fall", done).astype(bool)
        target = action * action_scale + default_joint_pos[None, :]
        for jt in joint_types:
            q_target[jt]["L"][t] = target[:, idx[(jt, "L")]].mean(axis=-1)
            q_target[jt]["R"][t] = target[:, idx[(jt, "R")]].mean(axis=-1)

    # Per-lane segments: [seg_start, seg_end) where seg_end is either a
    # fall step or T (the lane's still-running final segment, included so
    # a lane that survives to the end of the rollout contributes a "long"
    # segment rather than being silently dropped).
    segments = []  # (lane, seg_start, seg_end, ended_in_fall)
    for n in range(N):
        fs = [int(f) for f in np.where(fell[:, n])[0]]
        starts = [0] + [f + 1 for f in fs[:-1]]
        ends = list(fs)
        trailing_start = (fs[-1] + 1) if fs else 0
        if trailing_start < T:
            starts.append(trailing_start)
            ends.append(T)
        for seg_start, seg_end in zip(starts, ends):
            if seg_end - seg_start >= args.min_segment_len:
                segments.append((n, seg_start, seg_end, seg_end != T))

    print(f"\n{len(segments)} segments (>= {args.min_segment_len} steps) across {N} lanes")

    for jt in joint_types:
        lengths, rho_As = [], []
        for lane, s, e, _ in segments:
            a_l = float(q_target[jt]["L"][s:e, lane].std())
            a_r = float(q_target[jt]["R"][s:e, lane].std())
            lo, hi = min(a_l, a_r), max(a_l, a_r)
            rho_a = lo / hi if hi > 1e-8 else 1.0  # both sides essentially motionless -> treat as "equal" (0/0), not collapsed
            lengths.append(e - s)
            rho_As.append(rho_a)
        lengths = np.array(lengths)
        rho_As = np.array(rho_As)

        print(f"\n=== {jt} ===")
        rho_corr = _spearman(lengths, rho_As)
        print(f"  Spearman(segment_length, rho_A) = {rho_corr:.4f}  (n={len(segments)})")

        terciles = np.quantile(lengths, [1 / 3, 2 / 3])
        short_mask = lengths <= terciles[0]
        mid_mask = (lengths > terciles[0]) & (lengths <= terciles[1])
        long_mask = lengths > terciles[1]
        for name, mask in (("short", short_mask), ("mid", mid_mask), ("long", long_mask)):
            vals = rho_As[mask]
            len_vals = lengths[mask]
            if vals.size:
                print(f"  {name:<6} segments (n={vals.size}, len {len_vals.min()}-{len_vals.max()}): "
                      f"rho_A mean={vals.mean():.4f} median={np.median(vals):.4f} p10={np.percentile(vals, 10):.4f}")
            else:
                print(f"  {name:<6} segments: n=0")


if __name__ == "__main__":
    main()
