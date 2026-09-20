#!/usr/bin/env python3
"""Per-step trace of height, v_z, pitch, balance_reward, progress_reward,
v_x, and action magnitude for one lane over a full rollout -- answers
whether balance_reward gives early warning of pitch drift before a fall,
or only reacts once the fall is already underway (2026-09-19 diagnostic:
progress-heavy checkpoints track v_x well but accumulate pitch drift over
20-50 steps before falling -- is that a slow/insensitive balance signal,
or a real Pareto conflict the preference-heavy policy is choosing through?).

height/v_z added 2026-09-20 for the h=0.25 vs h=0.42 screening follow-up:
h025's AGGREGATE numbers hit its height target almost exactly but showed a
2x higher mean(v_z^2) and 2x more falls than h042 (A) -- that aggregate
can't say whether height and v_z are actually coupled in time (policy
overshoots toward target, corrects, overshoots again, oscillation grows
until a fall) or unrelated (v_z is just uniformly noisier throughout,
unconnected to any height-correction attempt). Only a per-step trace with
both series plotted together, especially zoomed into the steps
immediately before each fall, can distinguish those two stories -- a
per-rollout mean(v_z^2) collapses away exactly the temporal structure
that would tell them apart.

Unlike play.py's --analyze (which only records raw transition-dict fields),
this recomputes the actual reward VECTOR every step via
compute_reward_vector so progress/balance can be plotted directly --
Analyzer has no access to per-term reward, only raw physics fields.

    PYTHONPATH="$(pwd):$(pwd)/scripts" python scripts/rl/diagnostics.py balance-progress-trace \
        runs/<run>/checkpoints/checkpoint.pt 0.7 0.1 0.1 0.1 --steps 200 \
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
    parser.add_argument("--zoom_window", type=int, default=20, help="Steps before each fall to zoom-plot, 0 disables")
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
    height, v_z = [], []
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
        height.append(float(transition["height"][lane]))
        v_z.append(float(transition["v_z"][lane]))
        r_progress.append(float(reward_vec[lane, progress_idx]))
        r_balance.append(float(reward_vec[lane, balance_idx]))
        action_mag.append(float(np.abs(action[lane]).mean()))
        fell.append(bool(transition.get("terminal_fall", done)[lane]))

    pitch_rate = np.array(pitch_rate)
    height = np.array(height)
    v_z = np.array(v_z)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.dirname(args.checkpoint)), "exported")
    os.makedirs(out_dir, exist_ok=True)

    t = np.arange(args.steps)
    fall_steps = [i for i, f in enumerate(fell) if f]

    # (name, data, unit, reference_line-or-None) -- reference draws a
    # horizontal dashed line (height's target_height, v_x's commanded value)
    # so overshoot/oscillation around the target is visible directly, not
    # just the raw series.
    series = [
        ("height", height, "m", reward_cfg.target_height),
        ("v_z", v_z, "m/s", 0.0),
        ("pitch", pitch, "rad", None),
        ("balance_reward", r_balance, "", None),
        ("progress_reward", r_progress, "", None),
        ("v_x", v_x, "m/s", args.command[0]),
        ("action_magnitude", action_mag, "", None),
    ]

    def plot_series(ax_data, t_range, title_suffix: str):
        fig, axes = plt.subplots(len(ax_data), 1, figsize=(10, 16), sharex=True)
        for ax, (name, data, unit, ref) in zip(axes, ax_data):
            ax.plot(t_range, data)
            if ref is not None:
                ax.axhline(ref, color="green", linestyle=":", alpha=0.7, label=f"target={ref}")
                ax.legend(loc="upper right", fontsize=8)
            for fs in fall_steps:
                if t_range[0] <= fs <= t_range[-1]:
                    ax.axvline(fs, color="red", linestyle="--", alpha=0.5)
            ax.set_ylabel(f"{name}\n({unit})" if unit else name)
            ax.grid(alpha=0.3)
        axes[-1].set_xlabel("step (red dashed = fall event, green dotted = target/command)")
        fig.suptitle(f"lane {args.lane}, w={dict(zip(reward_cfg.term_names, args.w))}{title_suffix}")
        fig.tight_layout()
        return fig

    fig = plot_series(series, t, "")
    out_path = os.path.join(out_dir, "balance_progress_trace.png")
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"saved {out_path}")
    print(f"fall steps: {fall_steps}")

    # Zoom into [-zoom_window, 0] before each fall -- the aggregate mean(v_z^2)
    # over the WHOLE rollout can't distinguish "height/v_z oscillate together
    # right before the fall" from "v_z is just uniformly noisy the whole
    # time" (see module docstring). One zoomed plot per fall event.
    if args.zoom_window > 0:
        for i, fs in enumerate(fall_steps):
            lo = max(0, fs - args.zoom_window)
            t_range = np.arange(lo, fs + 1)
            zoom_series = [(name, np.asarray(data)[lo:fs + 1], unit, ref) for name, data, unit, ref in series]
            fig = plot_series(zoom_series, t_range, f", fall #{i} zoom [{lo},{fs}]")
            out_path = os.path.join(out_dir, f"balance_progress_trace_fall{i}_zoom.png")
            fig.savefig(out_path, dpi=110)
            plt.close(fig)
            print(f"saved {out_path}")


if __name__ == "__main__":
    main()
