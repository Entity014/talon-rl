#!/usr/bin/env python3
"""Experiment 2A.2 -- joint-level gait characterization under the REAL
trained policy (unlike leg_collapse_trace.py's open-loop synthetic trot,
which existed only to test a since-abandoned scripted-controller
feasibility question -- see that script's own docstring history). Ruling
out actuator authority as checkpoint A's primary bottleneck
(torque_authority_ablation.py: 1.0x/1.25x/1.5x torque ceiling made no
meaningful difference to height/v_z/survival) redirects the question to
the policy's own commanded joint trajectory and gait pattern.

Per leg-side group (hip_L/hip_R, thigh_L/thigh_R, calf_L/calf_R -- L =
{FL, RL}, R = {FR, RR}, averaged within each side) and aggregated across
all 12 joints, plots over the full rollout (single lane, like
leg_collapse_trace.py) plus a zoomed window per fall event:

  - height (+ target line), v_z -- for alignment with the fall-cycle
    structure fall_cycle_analysis.py already found (~13-15 step period).
  - q_target per side-group -- tests whether the POLICY's own commanded
    joint targets retract toward a lower-support configuration before
    v_z goes negative (a learned-gait/action-behavior explanation) or
    stay roughly constant (pointing elsewhere).
  - mean |q_target - q_actual| across all 12 joints -- tests whether
    actual joint position lags the commanded target (a control/actuator-
    dynamics explanation, separate from actuator torque AUTHORITY which
    was already ruled out -- lag can exist even with plenty of torque
    headroom, e.g. from Kp/Kd tuning or contact constraints).
  - mean |tau_desired| vs mean |tau_applied| across all 12 joints -- the
    same before/after-clipping comparison torque_authority_ablation.py
    made, but now as a continuous trace instead of a rollout-aggregate,
    to see WHEN clipping is actually occurring relative to the fall
    cycle rather than just how much.

L vs R asymmetry (within each side-group's own plot, both drawn
together) and ~13-step periodicity in q_target over the full 200-step
trace are both visible directly from these plots without needing a
separate statistical test -- this script deliberately stays
visualization-first (matching balance_progress_trace.py/leg_collapse_
trace.py's own approach) rather than building another lead-time-snapshot
table, since the open question here is shape/pattern (does target dip
before v_z drops, is there a fixed period, is one side different), not
another magnitude comparison.

    PYTHONPATH="$(pwd):$(pwd)/scripts" python scripts/rl/diagnostics.py gait-joint-trace \
        runs/<run>/checkpoints/checkpoint.pt 0.7 0.1 0.1 0.1 --steps 200 \
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
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--command", type=float, nargs=3, default=(0.5, 0.0, 0.0), metavar=("VX", "VY", "WZ"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lane", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--zoom_window", type=int, default=20)
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
    print(f"joint order: {joint_names}")
    print(f"Kp={kp} Kd={kd} action_scale={action_scale}")

    def side_idx(joint_type: str, side: str) -> list[int]:
        legs = ("FL", "RL") if side == "L" else ("FR", "RR")
        return [i for i, n in enumerate(joint_names) if joint_type in n and any(n.startswith(leg) for leg in legs)]

    groups = {
        f"{jt}_{side}": side_idx(jt, side)
        for jt in ("hip", "thigh", "calf") for side in ("L", "R")
    }

    T = args.steps
    lane = args.lane
    height, v_z, fell = [], [], []
    q_target_g = {g: [] for g in groups}
    mean_abs_err, mean_abs_tau_desired, mean_abs_tau_applied = [], [], []

    trainer.model.eval()
    for t in range(T):
        env.v_command_buf[:] = torch.tensor(args.command, device=env.device)
        action = trainer.act_inference()

        joint_pos_pre = robot.data.joint_pos.cpu().numpy()
        joint_vel_pre = robot.data.joint_vel.cpu().numpy()

        transition, done = env.step(action)
        trainer._last_extrinsics = transition.get("extrinsics") if trainer.encoder else None
        trainer.push_obs(transition["obs"], done_mask=done)

        height.append(float(transition["height"][lane]))
        v_z.append(float(transition["v_z"][lane]))
        fell.append(bool(transition.get("terminal_fall", done)[lane]))

        target = action[lane] * action_scale + default_joint_pos
        for g, idx in groups.items():
            q_target_g[g].append(float(target[idx].mean()))

        q_err = target - joint_pos_pre[lane]
        mean_abs_err.append(float(np.abs(q_err).mean()))

        tau_desired = kp * q_err - kd * joint_vel_pre[lane]
        tau_applied = transition["joint_torque"][lane]
        mean_abs_tau_desired.append(float(np.abs(tau_desired).mean()))
        mean_abs_tau_applied.append(float(np.abs(tau_applied).mean()))

    height = np.array(height)
    v_z = np.array(v_z)
    fall_steps = [i for i, f in enumerate(fell) if f]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.dirname(args.checkpoint)), "exported")
    os.makedirs(out_dir, exist_ok=True)

    def plot_window(t_range, title_suffix: str):
        fig, axes = plt.subplots(6, 1, figsize=(10, 16), sharex=True)
        sl = slice(t_range[0], t_range[-1] + 1)

        axes[0].plot(t_range, height[sl])
        axes[0].axhline(reward_cfg.target_height, color="green", linestyle=":", label=f"target={reward_cfg.target_height}")
        axes[0].set_ylabel("height (m)")
        axes[0].legend(fontsize=8)

        axes[1].plot(t_range, v_z[sl])
        axes[1].axhline(0.0, color="green", linestyle=":")
        axes[1].set_ylabel("v_z (m/s)")

        for i, jt in enumerate(("hip", "thigh", "calf")):
            ax = axes[2 + i]
            ax.plot(t_range, np.array(q_target_g[f"{jt}_L"])[sl], label=f"{jt}_L")
            ax.plot(t_range, np.array(q_target_g[f"{jt}_R"])[sl], label=f"{jt}_R", linestyle="--")
            ax.set_ylabel(f"{jt} q_target\n(rad)")
            ax.legend(fontsize=8)

        ax = axes[5]
        ax.plot(t_range, np.array(mean_abs_tau_desired)[sl], label="|tau_desired|")
        ax.plot(t_range, np.array(mean_abs_tau_applied)[sl], label="|tau_applied|", linestyle="--")
        ax.set_ylabel("mean |tau|\n(Nm)")
        ax.legend(fontsize=8)

        for ax in axes:
            for fs in fall_steps:
                if t_range[0] <= fs <= t_range[-1]:
                    ax.axvline(fs, color="red", linestyle="--", alpha=0.5)
            ax.grid(alpha=0.3)
        axes[-1].set_xlabel("step (red dashed = fall event)")
        fig.suptitle(f"lane {lane}, w={dict(zip(reward_cfg.term_names, args.w))}{title_suffix}")
        fig.tight_layout()
        return fig

    fig = plot_window(np.arange(T), "")
    out_path = os.path.join(out_dir, "gait_joint_trace.png")
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"saved {out_path}")
    print(f"fall steps: {fall_steps}")

    # Separate mean |q_target - q_actual| plot -- tracking-lag question,
    # kept off the main 6-panel figure (which is already dense) since it's
    # a single aggregate line, not per-side-group.
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(mean_abs_err)
    for fs in fall_steps:
        ax.axvline(fs, color="red", linestyle="--", alpha=0.5)
    ax.set_ylabel("mean |q_target - q_actual|\n(rad)")
    ax.set_xlabel("step (red dashed = fall event)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "gait_joint_trace_tracking_error.png")
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"saved {out_path}")

    if args.zoom_window > 0:
        for i, fs in enumerate(fall_steps):
            lo = max(0, fs - args.zoom_window)
            fig = plot_window(np.arange(lo, fs + 1), f", fall #{i} zoom [{lo},{fs}]")
            out_path = os.path.join(out_dir, f"gait_joint_trace_fall{i}_zoom.png")
            fig.savefig(out_path, dpi=110)
            plt.close(fig)
            print(f"saved {out_path}")


if __name__ == "__main__":
    main()
