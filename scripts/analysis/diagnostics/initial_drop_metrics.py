#!/usr/bin/env python3
"""Quantifies the first N steps after spawn (default 15) across ALL lanes --
follow-up to leg_collapse_trace.py's finding that BOTH A (target_height=0.42)
and h025 (target_height=0.25) show applied torque pinning against the A1's
33.5 Nm limit during the initial spawn-to-settle drop, on at least one
single-lane trace each. That was one lane, one (unseeded) launch each --
not enough to say the initial-drop torque saturation is a real, checkpoint-
independent phenomenon (actuator authority insufficient to hold spawn
height, not a reward/policy choice) rather than a coincidence of those two
particular launches.

Sets cfg.seed BEFORE env/gym.make construction (IsaacLabTalonEnvCfg's own
seed field -> ManagerBasedEnv.__init__ -> isaacsim.core.utils.torch.
set_seed, which seeds python random/numpy/torch CPU+CUDA/warp/replicator)
-- the actual Isaac Lab seeding mechanism, not a bare torch.manual_seed()
called after the env (and its terrain/randomization) already exist. Running
the SAME --seed across two checkpoints pairs their stochastic initial
conditions, so a difference between checkpoints isn't confounded with two
independently-random spawns.

Aggregates across all lanes (not just one), over exactly steps [0, window):
peak |torque|, fraction of (step, joint) pairs at >=95% of the torque
limit, height change (height[window-1] - height[0]), min v_z, peak |v_z|,
mean action magnitude.

    PYTHONPATH="$(pwd):$(pwd)/scripts" python scripts/analysis/run_diagnostics.py initial-drop-metrics \
        runs/<run>/checkpoints/checkpoint.pt 0.7 0.1 0.1 0.1 --seed 0 --window 15 \
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
    parser.add_argument("--window", type=int, default=15, help="Steps after spawn to quantify")
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
    # Seeds python random/numpy/torch CPU+CUDA/warp/replicator BEFORE the
    # scene (terrain, initial randomization events) is constructed -- see
    # module docstring. This is the field ManagerBasedEnv.__init__ checks;
    # setting it any later, or only seeding torch by hand after env
    # creation, misses the terrain/randomization RNG streams entirely.
    cfg.seed = args.seed
    env = gym.make("Isaac-Talon-A1-v0", cfg=cfg, render_mode=None).unwrapped

    trainer = MOPPOTrainer(
        env, obs_cfg, reward_cfg, pref_cfg, moppo_cfg=MOPPOConfig(), stack_cfg=stack_cfg,
        extrinsics_cfg=extrinsics_cfg, seed=args.seed,
    )
    trainer.load(args.checkpoint)
    print(f"loaded checkpoint (t={trainer._t}), env seed={args.seed}")

    trainer.w = np.tile(np.array(args.w, dtype=np.float32) / sum(args.w), (args.num_envs, 1)).astype(np.float32)

    height = np.zeros((args.window, args.num_envs), dtype=np.float32)
    v_z = np.zeros((args.window, args.num_envs), dtype=np.float32)
    torque = np.zeros((args.window, args.num_envs, 12), dtype=np.float32)
    action_mag = np.zeros((args.window, args.num_envs), dtype=np.float32)

    trainer.model.eval()
    for t in range(args.window):
        env.v_command_buf[:] = torch.tensor(args.command, device=env.device)
        action = trainer.act_inference()
        transition, done = env.step(action)
        trainer._last_extrinsics = transition.get("extrinsics") if trainer.encoder else None
        trainer.push_obs(transition["obs"], done_mask=done)

        height[t] = transition["height"]
        v_z[t] = transition["v_z"]
        torque[t] = transition["joint_torque"]
        action_mag[t] = np.abs(action).mean(axis=-1)

    peak_torque_per_lane = np.abs(torque).max(axis=(0, 2))  # (num_envs,)
    sat_frac_per_lane = (np.abs(torque) >= 0.95 * A1_TORQUE_LIMIT_NM).mean(axis=(0, 2))
    delta_height_per_lane = height[-1] - height[0]
    min_vz_per_lane = v_z.min(axis=0)
    peak_abs_vz_per_lane = np.abs(v_z).max(axis=0)
    mean_action_mag_per_lane = action_mag.mean(axis=0)

    def stat(name: str, vals: np.ndarray) -> None:
        print(f"  {name:<28}mean={vals.mean():>8.4f}  std={vals.std():>8.4f}")

    print(f"\nfirst {args.window} steps, {args.num_envs} lanes, w={dict(zip(reward_cfg.term_names, args.w))}:")
    stat("peak |torque| (Nm)", peak_torque_per_lane)
    stat("frac steps>=95% limit", sat_frac_per_lane)
    stat("delta height (m)", delta_height_per_lane)
    stat("min v_z (m/s)", min_vz_per_lane)
    stat("peak |v_z| (m/s)", peak_abs_vz_per_lane)
    stat("mean action magnitude", mean_action_mag_per_lane)


if __name__ == "__main__":
    main()
