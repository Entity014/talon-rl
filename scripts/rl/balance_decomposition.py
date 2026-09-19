#!/usr/bin/env python3
"""Breaks balance_reward down into its individual sub-terms (tilt, tilt-rate,
v_z, height, alive_bonus, fall_penalty) over a rollout, rather than just the
combined total -- answers whether a "standing" checkpoint (high survival,
~0% tracking) is actually well-balanced or just avoiding SOME cost while
still paying others (e.g. crouching low, bouncing), which the combined
balance_reward number alone can't distinguish.

    PYTHONPATH="$(pwd):$(pwd)/scripts" python scripts/rl/balance_decomposition.py \
        runs/<run>/checkpoints/checkpoint.pt 0.7 0.1 0.1 0.1 --steps 200
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--progress_std", type=float, default=None, help="Must match what the checkpoint was trained with")
    parser.add_argument("--balance_tilt_coef", type=float, default=None, help="Must match what the checkpoint was trained with")
    parser.add_argument("--balance_tilt_rate_coef", type=float, default=None, help="Must match what the checkpoint was trained with")
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
    reward_cfg = RewardVectorCfg(**reward_cfg_overrides)
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
    print(f"config: tilt_coef={reward_cfg.balance_tilt_coef} tilt_rate_coef={reward_cfg.balance_tilt_rate_coef} "
          f"alive_bonus={reward_cfg.alive_bonus} fall_penalty={reward_cfg.fall_penalty} "
          f"target_height={0.42}")

    trainer.w = np.tile(np.array(args.w, dtype=np.float32) / sum(args.w), (args.num_envs, 1)).astype(np.float32)

    sums = {
        "tilt_penalty": 0.0, "tilt_rate_penalty": 0.0, "v_z_penalty": 0.0,
        "height_penalty": 0.0, "alive_bonus": 0.0, "fall_penalty": 0.0,
        "v_x": 0.0, "height": 0.0, "n": 0,
    }
    trainer.model.eval()
    for _ in range(args.steps):
        env.v_command_buf[:] = torch.tensor(args.command, device=env.device)
        action = trainer.act_inference()
        transition, done = env.step(action)
        trainer._last_extrinsics = transition.get("extrinsics") if trainer.encoder else None
        trainer.push_obs(transition["obs"], done_mask=done)

        roll_pitch = transition["roll_pitch"]
        roll_pitch_rate = transition["roll_pitch_rate"]
        v_z = transition["v_z"]
        height = transition["height"]
        terminal_fall = transition.get("terminal_fall", done).astype(np.float32)

        sums["tilt_penalty"] += float((-reward_cfg.balance_tilt_coef * np.sum(roll_pitch ** 2, axis=-1)).mean())
        sums["tilt_rate_penalty"] += float((-reward_cfg.balance_tilt_rate_coef * np.sum(roll_pitch_rate ** 2, axis=-1)).mean())
        sums["v_z_penalty"] += float((-(v_z ** 2)).mean())
        sums["height_penalty"] += float((-((height - 0.42) ** 2)).mean())
        sums["alive_bonus"] += reward_cfg.alive_bonus
        sums["fall_penalty"] += float((-terminal_fall * reward_cfg.fall_penalty).mean())
        sums["v_x"] += float(transition["v_actual"][:, 0].mean())
        sums["height"] += float(height.mean())
        sums["n"] += 1

    n = sums.pop("n")
    print(f"\nper-step means over {n} steps, w={dict(zip(reward_cfg.term_names, args.w))}:")
    total = 0.0
    for k in ("tilt_penalty", "tilt_rate_penalty", "v_z_penalty", "height_penalty", "alive_bonus", "fall_penalty"):
        v = sums[k] / n
        total += v
        print(f"  {k:<20}{v:>10.4f}")
    print(f"  {'TOTAL balance_reward':<20}{total:>10.4f}")
    print(f"\n  mean v_x: {sums['v_x']/n:.4f}  (command was {args.command[0]})")
    print(f"  mean height: {sums['height']/n:.4f}  (target 0.42)")


if __name__ == "__main__":
    main()
