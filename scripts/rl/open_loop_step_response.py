#!/usr/bin/env python3
"""Experiment 2C.4 (part 1) -- open-loop joint step response, no trained
policy involved at all. 2C.3's causal-chain trace found Delta_q_actual
stays near noise level even though Delta_q_target is real and persistent
(command_switch_trace.py's control-subtracted result, all 3 seeds) --
but that leaves two possibilities open: the PD/actuator genuinely can't
track a ~0.02-0.05 rad target change quickly (a controller/dynamics
bottleneck), or the PD/actuator tracks it FINE and the target change
itself (from a +-0.25 command, action_scale=0.15 -> ~0.03-0.2 rad, only
1.7-11 degrees) is just too small to matter against the ongoing gait's
own much larger joint excursions (a perturbation-size problem, not a
controller problem). This test isolates the controller from the gait
entirely: no policy, no ongoing locomotion attempt -- every joint held
at its own default (standing) position except one test joint per leg
group, which gets a step command. Whatever response shows up here is
PURE PD (Kp=55/Kd=0.8) + actuator (DCMotor torque-speed curve) +
passive robot dynamics, nothing else.

Drives FL_hip, FL_thigh, FL_calf (one representative joint per type,
same leg) with independent step schedules while every other joint stays
at cfg.scene.robot's own default_joint_pos the whole time: warmup at
default, then step to default+delta for --step_duration steps, then
back to default for another --step_duration steps. Logs tracking error
(q_target - q_actual), qdot, and applied torque for the 3 test joints
across the step.

If tracking error collapses to near-zero within a few steps and stays
there (controller keeps up fine) -- the earlier command experiment's
near-zero Delta_q_actual is a perturbation-SIZE issue, not a controller
bottleneck: look at whether the target_perturbation_gain.py sweep (2C.4
part 2) shows G_q climbing back toward 1 at larger deltas. If tracking
error stays large/doesn't converge even for a simple isolated step --
the controller/dynamics IS the bottleneck, independent of gait context.

    PYTHONPATH="$(pwd):$(pwd)/scripts" python scripts/rl/open_loop_step_response.py \
        --num_envs 16 --delta 0.03 --step_duration 30
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--step_duration", type=int, default=30)
    parser.add_argument("--delta", type=float, default=0.03, help="Step size, rad -- applied to FL_hip/FL_thigh/FL_calf independently")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher({"headless": True, "enable_cameras": False})
    simulation_app = app_launcher.app  # noqa: F841

    import gymnasium as gym
    import talon_rl.tasks.locomotion.a1_env  # noqa: F401
    from talon_rl.tasks.locomotion.a1_env.a1_env_cfg import IsaacLabTalonEnvCfg

    cfg = IsaacLabTalonEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    env = gym.make("Isaac-Talon-A1-v0", cfg=cfg, render_mode=None).unwrapped
    env.reset()

    action_term = env.action_manager._terms["joint_pos"]
    joint_names = list(action_term._joint_names)
    action_scale = float(action_term.cfg.scale)
    default_joint_pos = action_term._offset[0].cpu().numpy()
    kp = float(cfg.scene.robot.actuators["base_legs"].stiffness)
    kd = float(cfg.scene.robot.actuators["base_legs"].damping)

    test_joints = {"hip": "FL_hip_joint", "thigh": "FL_thigh_joint", "calf": "FL_calf_joint"}
    test_idx = {name: joint_names.index(jn) for name, jn in test_joints.items()}
    print(f"testing joints: {test_joints}, delta={args.delta} rad, Kp={kp} Kd={kd}")

    N = args.num_envs
    T = args.warmup + 2 * args.step_duration
    q_target = {name: np.zeros((T, N), dtype=np.float32) for name in test_joints}
    q_actual = {name: np.zeros((T, N), dtype=np.float32) for name in test_joints}
    qdot = {name: np.zeros((T, N), dtype=np.float32) for name in test_joints}
    tau_applied = {name: np.zeros((T, N), dtype=np.float32) for name in test_joints}

    robot = env.scene["robot"]
    for t in range(T):
        target = np.tile(default_joint_pos, (N, 1))
        if args.warmup <= t < args.warmup + args.step_duration:
            for name, idx in test_idx.items():
                target[:, idx] += args.delta
        elif t >= args.warmup + args.step_duration:
            pass  # back to default

        joint_pos_pre = robot.data.joint_pos.cpu().numpy()

        action = (target - default_joint_pos[None, :]) / action_scale
        transition, done = env.step(action)

        joint_vel_now = transition["joint_vel"]
        for name, idx in test_idx.items():
            q_target[name][t] = target[:, idx]
            q_actual[name][t] = joint_pos_pre[:, idx]
            qdot[name][t] = joint_vel_now[:, idx]
            tau_applied[name][t] = transition["joint_torque"][:, idx]

    for name in test_joints:
        print(f"\n=== {name} ({test_joints[name]}), default={default_joint_pos[test_idx[name]]:+.4f} ===")
        print(f"{'t':<6}{'q_target':>10}{'q_actual':>10}{'track_err':>11}{'qdot':>9}{'tau':>9}")
        for t in range(args.warmup - 2, min(args.warmup + args.step_duration, T)):
            err = q_target[name][t] - q_actual[name][t]
            print(f"{t - args.warmup:<6}{q_target[name][t].mean():>10.4f}{q_actual[name][t].mean():>10.4f}"
                  f"{err.mean():>11.4f}{qdot[name][t].mean():>9.4f}{tau_applied[name][t].mean():>9.4f}")


if __name__ == "__main__":
    main()
