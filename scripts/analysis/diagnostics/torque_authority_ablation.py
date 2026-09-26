#!/usr/bin/env python3
"""Experiment 2B — eval-only torque/action authority ablation, testing the
hypothesis fall_cycle_analysis.py's findings raised: checkpoint A sinks
(continuous height decay + sustained negative v_z through most of each
~13-15 step fall cycle) because it wants more torque than the A1's
DCMotorCfg actuator model can deliver (effort_limit=saturation_effort=
33.5 Nm, isaaclab_assets.UNITREE_A1_CFG -- the A1's real datasheet peak
torque, unmodified by talon_rl.assets.unitree_a1.a1.TALON_A1_CFG, which
only overrides stiffness/damping to RMA's Kp=55/Kd=0.8).

Deliberately EVAL-ONLY -- checkpoint A is loaded unchanged at every
effort_scale, no retraining, no Kp/Kd/action_scale/reward changes. This
isolates "is the fixed policy's behavior constrained by today's torque
ceiling" from "would a policy trained with more headroom learn
differently" (a separate, not-yet-asked question). Scales BOTH
effort_limit and saturation_effort together (they're equal in the base
config, so the DCMotor's torque-speed curve stays a simple linear derate
from the scaled ceiling at zero velocity to zero at velocity_limit=21
rad/s -- see actuator_pd.py's DCMotor docstring for the exact formula).

The actuator cfg is replaced (not mutated in place) via its own
.replace(), and reassigned into a NEW actuators dict on the per-instance
robot cfg copy IsaacLabTalonEnvCfg.__post_init__ already creates (via
TALON_A1_CFG.replace(spawn=...)) -- mutating the DCMotorCfg instance
in place would risk touching the module-level TALON_A1_CFG singleton
those fields are shallow-copied from (the exact sharing hazard
a1_env_cfg.py's own __post_init__ comments warn about for scene.robot/
scene.terrain).

Logs, per lane then aggregated:
  - height: initial, at t=5/10/15, minimum, and a crude decay rate
    (height[0]-height[15])/15 over the analysis window.
  - vertical dynamics: mean/min v_z, peak |v_z|.
  - torque authority: mean(|tau_applied|/tau_max), fraction of steps
    near the (scaled) limit, peak |tau_applied|.
  - DESIRED torque before clipping: tau_desired = Kp*(q_target - q) -
    Kd*q_dot, computed by hand from the actuator's own Kp/Kd (there is
    no transition-dict field for this -- Isaac Lab's DCMotor clips
    internally and only ever reports the post-clip applied torque) --
    the direct evidence for "does the policy WANT more torque than the
    ceiling allows": mean/peak |tau_desired|, and the fraction of
    (step, joint) pairs where |tau_desired| > tau_max (i.e. would have
    been clipped at whatever the current effort_scale is).
  - failure: first-fall step, falls/steps, max |pitch|/|pitch_rate|.

    PYTHONPATH="$(pwd):$(pwd)/scripts" python scripts/analysis/run_diagnostics.py torque-authority-ablation \
        runs/<run>/checkpoints/checkpoint.pt 0.7 0.1 0.1 0.1 --seed 0 --steps 200 \
        --effort_scale 1.25 --progress_std 0.5 --balance_tilt_coef 1.0 --target_height 0.42
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

A1_BASE_EFFORT_LIMIT_NM = 33.5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("w", type=float, nargs=4)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--command", type=float, nargs=3, default=(0.5, 0.0, 0.0), metavar=("VX", "VY", "WZ"))
    parser.add_argument("--seed", type=int, default=0, help="Sets cfg.seed BEFORE env creation (not just trainer.rng)")
    parser.add_argument("--effort_scale", type=float, default=1.0, help="Multiplies effort_limit/saturation_effort (base 33.5 Nm)")
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
    cfg.seed = args.seed  # see feedback_talon_rl_env_seeding memory -- must be set before gym.make()

    tau_max = A1_BASE_EFFORT_LIMIT_NM * args.effort_scale
    base_actuator = cfg.scene.robot.actuators["base_legs"]
    kp, kd = base_actuator.stiffness, base_actuator.damping
    scaled_actuator = base_actuator.replace(effort_limit=tau_max, saturation_effort=tau_max)
    # Reassign a NEW dict rather than mutating base_actuator in place --
    # see module docstring's sharing-hazard note.
    cfg.scene.robot.actuators = {**cfg.scene.robot.actuators, "base_legs": scaled_actuator}
    print(f"effort_scale={args.effort_scale} -> effort_limit=saturation_effort={tau_max} Nm "
          f"(Kp={kp}, Kd={kd}, unchanged)")

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
    action_scale = float(action_term.cfg.scale)
    default_joint_pos = action_term._offset[0].cpu().numpy()  # (12,)

    T, N = args.steps, args.num_envs
    height = np.zeros((T, N), dtype=np.float32)
    v_z = np.zeros((T, N), dtype=np.float32)
    pitch = np.zeros((T, N), dtype=np.float32)
    pitch_rate = np.zeros((T, N), dtype=np.float32)
    torque_applied = np.zeros((T, N, 12), dtype=np.float32)
    torque_desired = np.zeros((T, N, 12), dtype=np.float32)
    terminal_fall = np.zeros((T, N), dtype=bool)

    trainer.model.eval()
    for t in range(T):
        env.v_command_buf[:] = torch.tensor(args.command, device=env.device)
        action = trainer.act_inference()

        # Snapshot actual joint state BEFORE stepping (the PD error that
        # produces THIS step's applied torque is computed against the
        # pre-step joint_pos/joint_vel, not the post-step one) -- there is
        # no transition-dict field for raw joint_pos (see leg_collapse_
        # trace.py's own docstring), so read it directly off the asset.
        joint_pos_pre = robot.data.joint_pos.cpu().numpy()  # (N, 12)
        joint_vel_pre = robot.data.joint_vel.cpu().numpy()

        transition, done = env.step(action)
        trainer._last_extrinsics = transition.get("extrinsics") if trainer.encoder else None
        trainer.push_obs(transition["obs"], done_mask=done)

        height[t] = transition["height"]
        v_z[t] = transition["v_z"]
        pitch[t] = transition["roll_pitch"][:, 1]
        pitch_rate[t] = transition["roll_pitch_rate"][:, 1]
        torque_applied[t] = transition["joint_torque"]
        terminal_fall[t] = transition.get("terminal_fall", done).astype(bool)

        target_joint_pos = action * action_scale + default_joint_pos[None, :]
        torque_desired[t] = kp * (target_joint_pos - joint_pos_pre) - kd * joint_vel_pre

    print(f"\n=== summary over {T} steps, {N} lanes, effort_scale={args.effort_scale} (tau_max={tau_max} Nm) ===")

    print("\n-- height trajectory --")
    print(f"  initial (t=0):            {height[0].mean():.4f}")
    for snap_t in (5, 10, 15):
        if snap_t < T:
            print(f"  height at t={snap_t}:            {height[snap_t].mean():.4f}")
    print(f"  minimum (per lane, mean):  {height.min(axis=0).mean():.4f}")
    if T > 15:
        decay_rate = (height[0] - height[15]) / 15.0
        print(f"  decay rate (h0-h15)/15:    {decay_rate.mean():.4f} m/step")

    alive = ~terminal_fall  # exclude only the instantaneous fall/reset step
    print("\n-- vertical dynamics (alive steps only) --")
    print(f"  mean v_z:                  {v_z[alive].mean():.4f}")
    print(f"  min v_z (per lane, mean):  {v_z.min(axis=0).mean():.4f}")
    print(f"  peak |v_z| (per lane, mean): {np.abs(v_z).max(axis=0).mean():.4f}")

    applied_mag = np.abs(torque_applied)
    desired_mag = np.abs(torque_desired)
    print("\n-- torque authority --")
    print(f"  mean(|tau_applied|/tau_max):   {(applied_mag[alive] / tau_max).mean():.4f}")
    print(f"  fraction near limit (>=95%):    {(applied_mag[alive] >= 0.95 * tau_max).mean():.4f}")
    print(f"  peak |tau_applied|:              {applied_mag.max():.4f}")
    print(f"  mean |tau_desired|:              {desired_mag[alive].mean():.4f}")
    print(f"  peak |tau_desired|:              {desired_mag.max():.4f}")
    print(f"  frac (step,joint) desired>limit: {(desired_mag[alive] > tau_max).mean():.4f}")
    print(f"  mean (|tau_desired|-|tau_applied|) where desired>limit: "
          f"{(desired_mag[alive][desired_mag[alive] > tau_max] - applied_mag[alive][desired_mag[alive] > tau_max]).mean():.4f}"
          if (desired_mag[alive] > tau_max).any() else "  mean (desired-applied) where desired>limit: n/a (never exceeded)")

    first_fall = np.where(terminal_fall.any(axis=0), terminal_fall.argmax(axis=0), T)
    print("\n-- failure --")
    print(f"  first-fall step: mean={first_fall.mean():.1f} median={np.median(first_fall):.1f} "
          f"pct_never_fell={(first_fall == T).mean() * 100:.1f}%")
    print(f"  falls per lane:  {terminal_fall.sum(axis=0).mean():.2f} / {T} steps")
    print(f"  max |pitch| (per lane, mean):      {np.abs(pitch).max(axis=0).mean():.4f}")
    print(f"  max |pitch_rate| (per lane, mean): {np.abs(pitch_rate).max(axis=0).mean():.4f}")


if __name__ == "__main__":
    main()
