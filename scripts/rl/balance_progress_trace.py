#!/usr/bin/env python3
"""Per-step trace of pitch, pitch-rate, balance_reward, progress_reward,
v_x, and action magnitude for one lane over a full rollout -- answers
whether balance_reward gives early warning of pitch drift before a fall,
or only reacts once the fall is already underway (2026-09-19 diagnostic:
progress-heavy checkpoints track v_x well but accumulate pitch drift over
20-50 steps before falling -- is that a slow/insensitive balance signal,
or a real Pareto conflict the preference-heavy policy is choosing through?).

Unlike play.py's --analyze (which only records raw transition-dict fields),
this recomputes the actual reward VECTOR every step via
compute_reward_vector so progress/balance can be plotted directly --
Analyzer has no access to per-term reward, only raw physics fields.

    PYTHONPATH="$(pwd):$(pwd)/scripts" python scripts/rl/balance_progress_trace.py \
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
from talon_rl.reward import compute_reward_vector

from rl.core.algorithms import MOPPOConfig, MOPPOTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("w", type=float, nargs=4)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--command", type=float, nargs=3, default=(0.5, 0.0, 0.0), metavar=("VX", "VY", "WZ"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lane", type=int, default=0, help="Which lane to trace (plots are single-lane, like Analyzer)")
    parser.add_argument("--out_dir", type=str, default=None, help="Defaults to <run>/exported")
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
    env.v_command_buf[:] = torch.tensor(args.command, device=env.device)

    progress_idx = reward_cfg.term_names.index("progress")
    balance_idx = reward_cfg.term_names.index("balance")

    v_x, pitch, roll, pitch_rate, r_progress, r_balance, action_mag, fell = [], [], [], [], [], [], [], []
    trainer.model.eval()
    for _ in range(args.steps):
        env.v_command_buf[:] = torch.tensor(args.command, device=env.device)
        action = trainer.act_inference()
        transition, done = env.step(action)
        trainer._last_extrinsics = transition.get("extrinsics") if trainer.encoder else None
        trainer.push_obs(transition["obs"], done_mask=done)

        reward_vec = compute_reward_vector(transition, reward_cfg)
        lane = args.lane
        v_x.append(float(transition["v_actual"][lane, 0]))
        roll.append(float(transition["roll_pitch"][lane, 0]))
        pitch.append(float(transition["roll_pitch"][lane, 1]))
        # Real body-frame angular velocity (root_ang_vel_b), not a
        # finite-difference approximation of pitch across steps.
        pitch_rate.append(float(transition["roll_pitch_rate"][lane, 1]))
        r_progress.append(float(reward_vec[lane, progress_idx]))
        r_balance.append(float(reward_vec[lane, balance_idx]))
        action_mag.append(float(np.abs(action[lane]).mean()))
        fell.append(bool(transition.get("terminal_fall", done)[lane]))

    pitch_rate = np.array(pitch_rate)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.dirname(args.checkpoint)), "exported")
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(6, 1, figsize=(10, 14), sharex=True)
    t = np.arange(args.steps)
    series = [
        ("pitch", pitch, "rad"), ("pitch_rate", pitch_rate, "rad/s"),
        ("balance_reward", r_balance, ""), ("progress_reward", r_progress, ""),
        ("v_x", v_x, "m/s"), ("action_magnitude", action_mag, ""),
    ]
    fall_steps = [i for i, f in enumerate(fell) if f]
    for ax, (name, data, unit) in zip(axes, series):
        ax.plot(t, data)
        for fs in fall_steps:
            ax.axvline(fs, color="red", linestyle="--", alpha=0.5)
        ax.set_ylabel(f"{name}\n({unit})" if unit else name)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("step (red dashed = fall event)")
    fig.suptitle(f"lane {args.lane}, w={dict(zip(reward_cfg.term_names, args.w))}")
    fig.tight_layout()
    out_path = os.path.join(out_dir, "balance_progress_trace.png")
    fig.savefig(out_path, dpi=110)
    print(f"saved {out_path}")
    print(f"fall steps: {fall_steps}")


if __name__ == "__main__":
    main()
