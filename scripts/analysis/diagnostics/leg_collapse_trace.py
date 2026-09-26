#!/usr/bin/env python3
"""Per-joint trace (commanded thigh/calf target angle, applied torque) for
one lane, alongside height/v_z for alignment -- follow-up to
balance_progress_trace.py's finding that h025 (target_height=0.25) never
reaches a stable equilibrium: height decays near-monotonically from spawn
(~0.42) down to ~0.08-0.10 every ~14-step episode with v_z staying
continuously negative (no recovery phase), while A (target_height=0.42)
falls twice early then settles into a stable (if stationary/crouched)
equilibrium around height~0.20.

That trace couldn't say WHY h025 collapses: is the policy actively
commanding its legs to fold (thigh/calf targets drifting toward the
crouched end of their range) during the descent, or is it commanding
extension while torque/actuator limits fail to arrest the fall? Neither
transition dict has raw joint_pos, but JointPositionActionCfg's own
contract (a1_env_cfg.py, use_default_offset=True, scale=0.15) means the
commanded target IS recoverable from the raw action: target = action*0.15
+ default_joint_pos. Plotting that alongside applied_torque (joint_torque
in the transition dict, robot.data.applied_torque) answers the "policy
choosing collapse" vs "actuator/torque-limited" question directly.

    PYTHONPATH="$(pwd):$(pwd)/scripts" python scripts/analysis/run_diagnostics.py leg-collapse-trace \
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
    # Seeds python random/numpy/torch CPU+CUDA/warp/replicator BEFORE the
    # scene (terrain, initial randomization events) is constructed --
    # ManagerBasedEnv.__init__ checks this field and calls
    # isaacsim.core.utils.torch.set_seed. --seed alone (passed to
    # MOPPOTrainer below) only seeds its own preference-sampling rng, NOT
    # env creation -- without this, two launches with identical --seed
    # still produce different trajectories (confirmed 2026-09-20: A and
    # h025 traces from separate launches showed qualitatively different
    # fall/settle patterns despite identical CLI args).
    cfg.seed = args.seed
    env = gym.make("Isaac-Talon-A1-v0", cfg=cfg, render_mode=None).unwrapped

    trainer = MOPPOTrainer(
        env, obs_cfg, reward_cfg, pref_cfg, moppo_cfg=MOPPOConfig(), stack_cfg=stack_cfg,
        extrinsics_cfg=extrinsics_cfg, seed=args.seed,
    )
    trainer.load(args.checkpoint)
    print(f"loaded checkpoint (t={trainer._t}), env seed={args.seed}")

    trainer.w = np.tile(np.array(args.w, dtype=np.float32) / sum(args.w), (args.num_envs, 1)).astype(np.float32)

    # Recover the commanded joint-position TARGET from the raw action --
    # transition dict has no raw joint_pos, but JointPositionActionCfg's
    # own affine contract (a1_env_cfg.py: scale=0.15, use_default_offset
    # True) makes target = action*scale + default_joint_pos exactly.
    action_term = env.action_manager._terms["joint_pos"]
    joint_names = list(action_term._joint_names)
    action_scale = float(action_term.cfg.scale)
    default_joint_pos = action_term._offset[0].cpu().numpy()
    print(f"joint order: {joint_names}")
    print(f"default_joint_pos: {default_joint_pos}, action_scale: {action_scale}")

    thigh_idx = [i for i, n in enumerate(joint_names) if "thigh" in n]
    calf_idx = [i for i, n in enumerate(joint_names) if "calf" in n]
    thigh_names = [joint_names[i] for i in thigh_idx]
    calf_names = [joint_names[i] for i in calf_idx]

    height, v_z, fell = [], [], []
    thigh_target, calf_target = [], []  # (steps, 4)
    thigh_torque, calf_torque = [], []

    trainer.model.eval()
    for _ in range(args.steps):
        env.v_command_buf[:] = torch.tensor(args.command, device=env.device)
        action = trainer.act_inference()
        transition, done = env.step(action)
        trainer._last_extrinsics = transition.get("extrinsics") if trainer.encoder else None
        trainer.push_obs(transition["obs"], done_mask=done)

        lane = args.lane
        height.append(float(transition["height"][lane]))
        v_z.append(float(transition["v_z"][lane]))
        fell.append(bool(transition.get("terminal_fall", done)[lane]))

        target = action[lane] * action_scale + default_joint_pos
        thigh_target.append(target[thigh_idx])
        calf_target.append(target[calf_idx])
        torque = transition["joint_torque"][lane]
        thigh_torque.append(torque[thigh_idx])
        calf_torque.append(torque[calf_idx])

    height = np.array(height)
    v_z = np.array(v_z)
    thigh_target = np.array(thigh_target)  # (steps, 4)
    calf_target = np.array(calf_target)
    thigh_torque = np.array(thigh_torque)
    calf_torque = np.array(calf_torque)
    fall_steps = [i for i, f in enumerate(fell) if f]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.dirname(args.checkpoint)), "exported")
    os.makedirs(out_dir, exist_ok=True)

    default_thigh = default_joint_pos[thigh_idx]
    default_calf = default_joint_pos[calf_idx]

    def plot_window(t_range, title_suffix: str):
        sl = slice(t_range[0], t_range[-1] + 1)
        fig, axes = plt.subplots(5, 1, figsize=(10, 14), sharex=True)
        axes[0].plot(t_range, height[sl])
        axes[0].axhline(reward_cfg.target_height, color="green", linestyle=":", label=f"target={reward_cfg.target_height}")
        axes[0].set_ylabel("height (m)")
        axes[0].legend(fontsize=8)

        axes[1].plot(t_range, v_z[sl])
        axes[1].axhline(0.0, color="green", linestyle=":")
        axes[1].set_ylabel("v_z (m/s)")

        for i, name in enumerate(thigh_names):
            axes[2].plot(t_range, thigh_target[sl, i], label=name)
            axes[2].axhline(default_thigh[i], color="gray", linestyle=":", alpha=0.4)
        axes[2].set_ylabel("thigh target\n(rad)")
        axes[2].legend(fontsize=7, ncol=2)

        for i, name in enumerate(calf_names):
            axes[3].plot(t_range, calf_target[sl, i], label=name)
            axes[3].axhline(default_calf[i], color="gray", linestyle=":", alpha=0.4)
        axes[3].set_ylabel("calf target\n(rad)")
        axes[3].legend(fontsize=7, ncol=2)

        for i, name in enumerate(thigh_names):
            axes[4].plot(t_range, thigh_torque[sl, i], label=f"{name} (thigh)", alpha=0.7)
        for i, name in enumerate(calf_names):
            axes[4].plot(t_range, calf_torque[sl, i], label=f"{name} (calf)", linestyle="--", alpha=0.7)
        axes[4].axhline(33.5, color="red", linestyle=":", alpha=0.5, label="A1 torque limit")
        axes[4].axhline(-33.5, color="red", linestyle=":", alpha=0.5)
        axes[4].set_ylabel("applied torque\n(Nm)")
        axes[4].legend(fontsize=6, ncol=3)

        for ax in axes:
            for fs in fall_steps:
                if t_range[0] <= fs <= t_range[-1]:
                    ax.axvline(fs, color="red", linestyle="--", alpha=0.5)
            ax.grid(alpha=0.3)
        axes[-1].set_xlabel("step (red dashed = fall event, gray dotted = default/standing joint angle)")
        fig.suptitle(f"lane {args.lane}, w={dict(zip(reward_cfg.term_names, args.w))}{title_suffix}")
        fig.tight_layout()
        return fig

    fig = plot_window(np.arange(args.steps), "")
    out_path = os.path.join(out_dir, "leg_collapse_trace.png")
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"saved {out_path}")
    print(f"fall steps: {fall_steps}")

    if args.zoom_window > 0:
        for i, fs in enumerate(fall_steps):
            lo = max(0, fs - args.zoom_window)
            fig = plot_window(np.arange(lo, fs + 1), f", fall #{i} zoom [{lo},{fs}]")
            out_path = os.path.join(out_dir, f"leg_collapse_trace_fall{i}_zoom.png")
            fig.savefig(out_path, dpi=110)
            plt.close(fig)
            print(f"saved {out_path}")


if __name__ == "__main__":
    main()
