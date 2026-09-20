#!/usr/bin/env python3
"""Experiment 2A.4 -- causal counterfactual intervention on hip asymmetry,
eval-only (checkpoint unchanged, no retraining, no reward change).
hip_asymmetry_analysis.py found the policy INTENDS asymmetric hip targets
(dq_target present from the start of every cycle, largest of any joint,
shrinking rather than growing toward each fall) -- but that's still only
correlational. This tests whether the asymmetry is a CAUSAL contributor
to falling, or a harmless (if ugly) stylistic choice/compensation, by
forcibly overriding it during rollout and measuring whether locomotion
quality changes.

UNITREE_A1_CFG's own default standing pose already encodes the correct
mirror convention: FL_hip=+0.1, FR_hip=-0.1, RL_hip=+0.1, RR_hip=-0.1
(isaaclab_assets/robots/unitree.py) -- left and right hip abduction/
adduction angles are sign-mirrored about 0 at the symmetric standing
pose, not equal. So "symmetric" here means q_R = -q_L per front/rear
pair (FL<->FR, RL<->RR independently, not an L-average), not q_R = q_L.

Two interventions, both override ONLY the hip actions (thigh/calf and
both legs' own policy output otherwise untouched) by back-solving the
required raw action from the desired mirrored target:

  --mode R_follows_L (Intervention A): FR_hip target forced to
      -FL_hip's own target that step (from the policy's actual FL
      output), RR_hip forced to -RL_hip's. Tests: does constraining the
      hip that was mostly frozen to instead mirror the active one
      improve locomotion, or make it worse/no different?
  --mode L_follows_R (Intervention B, the mirror of A): FL_hip forced
      to -FR_hip's target, RL_hip forced to -RR_hip's -- i.e. force the
      previously-ACTIVE side to instead passively mirror the previously
      -FROZEN side. If results look similar to R_follows_L, the
      asymmetry is about breaking symmetry itself, not about which
      specific side/leg dynamics happens to be favored. If very
      different, there's a side-specific dynamics/observation asymmetry
      at play, not just an arbitrary symmetry-breaking policy quirk.
  --mode none: no override, baseline passthrough (same aggregate metrics
      as attractor_phase_metrics.py, for a same-script apples-to-apples
      comparison rather than re-reading an older log).

Reports the same core locomotion-quality metrics as
attractor_phase_metrics.py's Phase C (falls/lane, v_z, height, v_x
tracking MAE, max |pitch|/|pitch_rate|) so the three modes' numbers sit
directly next to each other with no cross-script formatting drift.

    PYTHONPATH="$(pwd):$(pwd)/scripts" python scripts/rl/diagnostics.py hip-symmetry-intervention \
        runs/<run>/checkpoints/checkpoint.pt 0.7 0.1 0.1 0.1 --seed 0 --steps 200 \
        --mode R_follows_L --progress_std 0.5 --balance_tilt_coef 1.0 --target_height 0.42
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
    parser.add_argument("--command", type=float, nargs=3, default=(0.5, 0.0, 0.0), metavar=("VX", "VY", "WZ"))
    parser.add_argument("--seed", type=int, default=0, help="Sets cfg.seed BEFORE env creation (not just trainer.rng)")
    parser.add_argument("--mode", choices=["none", "R_follows_L", "L_follows_R"], default="R_follows_L")
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
    default_joint_pos = action_term._offset[0].cpu().numpy()  # (12,)

    def hip_idx(leg: str) -> int:
        return next(i for i, n in enumerate(joint_names) if n.startswith(leg) and "hip" in n)

    FL, FR, RL, RR = hip_idx("FL"), hip_idx("FR"), hip_idx("RL"), hip_idx("RR")

    def apply_symmetry(action: np.ndarray) -> np.ndarray:
        """action: (N, 12) raw policy output. Overrides ONLY the follower
        side's hip actions, back-solved from the leader side's OWN target
        this step so the leader is completely untouched (its action is
        exactly what the policy chose)."""
        if args.mode == "none":
            return action
        action = action.copy()
        target = action * action_scale + default_joint_pos[None, :]
        if args.mode == "R_follows_L":
            leader_front, follower_front = FL, FR
            leader_rear, follower_rear = RL, RR
        else:  # L_follows_R
            leader_front, follower_front = FR, FL
            leader_rear, follower_rear = RR, RL
        for leader, follower in ((leader_front, follower_front), (leader_rear, follower_rear)):
            mirrored_target = -target[:, leader]
            action[:, follower] = (mirrored_target - default_joint_pos[follower]) / action_scale
        return action

    T, N = args.steps, args.num_envs
    height = np.zeros((T, N), dtype=np.float32)
    v_z = np.zeros((T, N), dtype=np.float32)
    v_x = np.zeros((T, N), dtype=np.float32)
    pitch = np.zeros((T, N), dtype=np.float32)
    pitch_rate = np.zeros((T, N), dtype=np.float32)
    terminal_fall = np.zeros((T, N), dtype=bool)

    trainer.model.eval()
    for t in range(T):
        env.v_command_buf[:] = torch.tensor(args.command, device=env.device)
        action = trainer.act_inference()
        action = apply_symmetry(action)
        transition, done = env.step(action)
        trainer._last_extrinsics = transition.get("extrinsics") if trainer.encoder else None
        trainer.push_obs(transition["obs"], done_mask=done)

        height[t] = transition["height"]
        v_z[t] = transition["v_z"]
        v_x[t] = transition["v_actual"][:, 0]
        pitch[t] = transition["roll_pitch"][:, 1]
        pitch_rate[t] = transition["roll_pitch_rate"][:, 1]
        terminal_fall[t] = transition.get("terminal_fall", done).astype(bool)

    alive = ~terminal_fall
    v_x_err = v_x - args.command[0]
    print(f"\n=== mode={args.mode}, {T} steps, {N} lanes ===")
    print(f"  falls per lane:            {terminal_fall.sum(axis=0).mean():.4f} / {T} steps")
    print(f"  frac lanes with >=1 fall:  {(terminal_fall.any(axis=0)).mean():.4f}")
    first_fall = np.where(terminal_fall.any(axis=0), terminal_fall.argmax(axis=0), T)
    print(f"  first-fall step:           mean={first_fall.mean():.1f} median={np.median(first_fall):.1f} "
          f"pct_never_fell={(first_fall == T).mean() * 100:.1f}%")
    print(f"  mean height (std):         {height[alive].mean():.4f} ({height[alive].std():.4f})")
    print(f"  mean v_z:                  {v_z[alive].mean():.4f}")
    print(f"  mean v_x (cmd {args.command[0]}):        {v_x[alive].mean():.4f}")
    print(f"  v_x tracking MAE / RMSE:   {np.abs(v_x_err)[alive].mean():.4f} / "
          f"{np.sqrt((v_x_err[alive] ** 2).mean()):.4f}")
    print(f"  mean/max |pitch|:          {np.abs(pitch)[alive].mean():.4f} / {np.abs(pitch).max(axis=0).mean():.4f}")
    print(f"  mean/max |pitch_rate|:     {np.abs(pitch_rate)[alive].mean():.4f} / "
          f"{np.abs(pitch_rate).max(axis=0).mean():.4f}")


if __name__ == "__main__":
    main()
