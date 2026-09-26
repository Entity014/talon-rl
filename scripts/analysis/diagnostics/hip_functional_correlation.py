#!/usr/bin/env python3
"""Experiment 2A.4 (continued) -- functional-contribution correlation
analysis, replacing the magnitude-based metrics that didn't explain the
hip_symmetry_intervention.py result (gait_activity_ratio.py's rho_A
didn't separate stable/falling cycles; raw torque/qdot magnitude traces
looked similar across modes despite very different whole-body outcomes).
Asks instead: does hip motion happen in a functionally relevant PHASE
relationship with forward velocity, contact, or vertical recovery --
not just how big it is?

Runs under the SAME action override as hip_symmetry_intervention.py
(--mode none/R_follows_L/L_follows_R), across ALL lanes (not one traced
lane), and reports two analyses:

  1. Segment-phase correlations: each pre-fall window (last `--window`
     steps before a fall) is split into early/middle/late thirds
     (early=[-15,-10], middle=[-9,-4], late=[-3,0] at the default
     window=15) and, pooled across all fall events/lanes within each
     third, computes corr(hip_qdot, v_x), corr(hip_qdot, v_z), and
     corr(|hip_qdot|, contact) separately per hip side (L/R). Splitting
     by cycle-phase matters because pooling a whole 200-step rollout
     (or even a whole pre-fall window) mixes together very different
     dynamical regimes (steady walking-attempt vs the reset transient
     vs the terminal topple) and a raw correlation over the mix can
     easily be an artifact of THAT mixing rather than a real
     within-phase relationship.
  2. Lagged correlation with v_z: corr(hip_qdot(t-tau), v_z(t)) for
     tau in [-lag_max, lag_max] steps, per hip side, pooled over all
     (lane, t) pairs where the WHOLE span between t-tau and t stays
     within one continuous alive stretch (no fall event crossed, so a
     lag never accidentally pairs pre-reset motion with post-reset
     v_z or vice versa). Tests the hypothesized causal chain (hip
     motion -> contact/support -> vertical body response -> v_z) has a
     LAG, not necessarily a same-timestep correlation.

Interpretation caveat this script cannot resolve on its own: correlation
(lagged or not) is association, not proof of causality -- the causal
claim rests on hip_symmetry_intervention.py's actual counterfactual
result; this only characterizes WHAT the functional relationship looks
like assuming that result is real.

    PYTHONPATH="$(pwd):$(pwd)/scripts" python scripts/analysis/run_diagnostics.py hip-functional-correlation \
        runs/<run>/checkpoints/checkpoint.pt 0.7 0.1 0.1 0.1 --seed 0 --steps 200 \
        --mode none --window 15 --lag_max 5 \
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

CONTACT_THRESHOLD_N = 1.0


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("w", type=float, nargs=4)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--window", type=int, default=15, help="Pre-fall window; split into early/middle/late thirds")
    parser.add_argument("--lag_max", type=int, default=5)
    parser.add_argument("--command", type=float, nargs=3, default=(0.5, 0.0, 0.0), metavar=("VX", "VY", "WZ"))
    parser.add_argument("--seed", type=int, default=0, help="Sets cfg.seed BEFORE env creation (not just trainer.rng)")
    parser.add_argument("--mode", choices=["none", "R_follows_L", "L_follows_R"], default="none")
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
    print(f"loaded checkpoint (t={trainer._t}), env seed={args.seed}, mode={args.mode}")

    trainer.w = np.tile(np.array(args.w, dtype=np.float32) / sum(args.w), (args.num_envs, 1)).astype(np.float32)

    action_term = env.action_manager._terms["joint_pos"]
    joint_names = list(action_term._joint_names)
    action_scale = float(action_term.cfg.scale)
    default_joint_pos = action_term._offset[0].cpu().numpy()

    def idx1(leg: str, jt: str) -> int:
        return next(i for i, n in enumerate(joint_names) if n.startswith(leg) and jt in n)

    hip = {leg: idx1(leg, "hip") for leg in ("FL", "FR", "RL", "RR")}

    def apply_symmetry(action: np.ndarray) -> np.ndarray:
        if args.mode == "none":
            return action
        action = action.copy()
        target = action * action_scale + default_joint_pos[None, :]
        pairs = (
            ((hip["FL"], hip["FR"]), (hip["RL"], hip["RR"]))
            if args.mode == "R_follows_L"
            else ((hip["FR"], hip["FL"]), (hip["RR"], hip["RL"]))
        )
        for leader, follower in pairs:
            mirrored_target = -target[:, leader]
            action[:, follower] = (mirrored_target - default_joint_pos[follower]) / action_scale
        return action

    T, N = args.steps, args.num_envs
    v_x = np.zeros((T, N), dtype=np.float32)
    v_z = np.zeros((T, N), dtype=np.float32)
    qdot_L = np.zeros((T, N), dtype=np.float32)
    qdot_R = np.zeros((T, N), dtype=np.float32)
    contact_L = np.zeros((T, N), dtype=np.float32)  # fraction of that side's 2 feet in contact
    contact_R = np.zeros((T, N), dtype=np.float32)
    fell = np.zeros((T, N), dtype=bool)

    # Foot order matches contact_sensor.body_names -- read leg prefixes
    # directly off it rather than assuming FL/FR/RL/RR order, same
    # caution a1_env.py's own foot_vel/foot_contact_force comments take.
    foot_body_ids = env._foot_body_ids
    foot_names = [env.scene["robot"].body_names[i] for i in foot_body_ids]
    left_feet = [i for i, n in enumerate(foot_names) if n.startswith("FL") or n.startswith("RL")]
    right_feet = [i for i, n in enumerate(foot_names) if n.startswith("FR") or n.startswith("RR")]

    trainer.model.eval()
    for t in range(T):
        env.v_command_buf[:] = torch.tensor(args.command, device=env.device)
        action = trainer.act_inference()
        action = apply_symmetry(action)

        joint_vel_pre = env.scene["robot"].data.joint_vel.cpu().numpy()

        transition, done = env.step(action)
        trainer._last_extrinsics = transition.get("extrinsics") if trainer.encoder else None
        trainer.push_obs(transition["obs"], done_mask=done)

        v_x[t] = transition["v_actual"][:, 0]
        v_z[t] = transition["v_z"]
        qdot_L[t] = np.mean([joint_vel_pre[:, hip["FL"]], joint_vel_pre[:, hip["RL"]]], axis=0)
        qdot_R[t] = np.mean([joint_vel_pre[:, hip["FR"]], joint_vel_pre[:, hip["RR"]]], axis=0)
        contact = transition["foot_contact_force"] > CONTACT_THRESHOLD_N  # (N, 4)
        contact_L[t] = contact[:, left_feet].mean(axis=-1)
        contact_R[t] = contact[:, right_feet].mean(axis=-1)
        fell[t] = transition.get("terminal_fall", done).astype(bool)

    # --- 1. segment-phase correlations ---
    print(f"\n=== segment-phase correlations (mode={args.mode}) ===")
    thirds = {
        "early": range(-args.window, -args.window + args.window // 3 or -args.window + 1),
        "middle": range(-args.window + args.window // 3, -args.window + 2 * args.window // 3),
        "late": range(-args.window + 2 * args.window // 3, 1),
    }
    # At window=15: early=[-15,-10), middle=[-10,-5), late=[-5,0] -- close
    # to the user's [-15,-10]/[-9,-4]/[-3,0] spec (window//3 boundaries,
    # not hardcoded to window=15 specifically so --window is honored).
    for phase_name, offsets in thirds.items():
        pooled = {"L": {"vx": [], "vz": [], "contact": [], "qdot": []}, "R": {"vx": [], "vz": [], "contact": [], "qdot": []}}
        for t in range(T):
            if not fell[t].any():
                continue
            lanes = fell[t]
            for offset in offsets:
                src_t = t + offset
                if src_t < 0:
                    continue
                pooled["L"]["vx"].append(v_x[src_t, lanes]); pooled["L"]["vz"].append(v_z[src_t, lanes])
                pooled["L"]["contact"].append(contact_L[src_t, lanes]); pooled["L"]["qdot"].append(qdot_L[src_t, lanes])
                pooled["R"]["vx"].append(v_x[src_t, lanes]); pooled["R"]["vz"].append(v_z[src_t, lanes])
                pooled["R"]["contact"].append(contact_R[src_t, lanes]); pooled["R"]["qdot"].append(qdot_R[src_t, lanes])
        print(f"  {phase_name} (offsets {min(offsets)}..{max(offsets)}):")
        for side in ("L", "R"):
            if not pooled[side]["qdot"]:
                print(f"    {side}: no data")
                continue
            qdot = np.concatenate(pooled[side]["qdot"])
            vx_ = np.concatenate(pooled[side]["vx"])
            vz_ = np.concatenate(pooled[side]["vz"])
            contact_ = np.concatenate(pooled[side]["contact"])
            c_vx = _safe_corr(qdot, vx_)
            c_vz = _safe_corr(qdot, vz_)
            c_contact = _safe_corr(np.abs(qdot), contact_)
            print(f"    hip_{side}: C(vx)={c_vx:.4f}  C(vz)={c_vz:.4f}  C(|qdot|,contact)={c_contact:.4f}  (n={qdot.size})")

    # --- 2. lagged correlation with v_z ---
    print(f"\n=== lagged correlation with v_z (tau in [-{args.lag_max}, {args.lag_max}]) ===")
    # valid_span[a, b] (a<b) = no fall event at any step in [a, b] for that
    # lane -- built once as a per-lane prefix-sum-style "steps since last
    # fall" so an O(1) check per (lane, a, b) is possible instead of
    # rescanning the range every time.
    steps_since_fall = np.zeros((T, N), dtype=np.int32)
    for t in range(1, T):
        steps_since_fall[t] = np.where(fell[t - 1], 0, steps_since_fall[t - 1] + 1)

    for side, qdot in (("L", qdot_L), ("R", qdot_R)):
        line = f"  hip_{side}: "
        for tau in range(-args.lag_max, args.lag_max + 1):
            # corr(qdot(t-tau), v_z(t)) -- valid t range depends on sign of tau.
            if tau >= 0:
                t_range = np.arange(tau, T)
                src_t = t_range - tau
            else:
                t_range = np.arange(0, T + tau)
                src_t = t_range - tau
            # require the WHOLE span [min(src_t,t), max(src_t,t)] fall-free:
            # steps_since_fall at the LATER of the two indices must be
            # large enough to cover the span (i.e. no fall occurred since
            # max(src_t,t) - span_len steps ago).
            span = np.abs(tau)
            later_idx = np.maximum(t_range, src_t)
            valid = steps_since_fall[later_idx, :] > span
            a = qdot[src_t]  # (len(t_range), N)
            b = v_z[t_range]  # (len(t_range), N)
            a_valid, b_valid = a[valid], b[valid]
            rho = _safe_corr(a_valid, b_valid)
            line += f"tau={tau:+d}:{rho:+.3f}  "
        print(line)


if __name__ == "__main__":
    main()
