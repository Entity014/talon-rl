#!/usr/bin/env python3
"""Experiment 2D.1 -- static gravity-load / Kp diagnostic, no policy, no
command, no perturbation at all. open_loop_step_response.py found the
calf joint is chronically torque-saturated and >0.6-1.0 rad off target
just holding the DEFAULT standing pose (zero commanded perturbation) --
before touching any controller gain, this establishes exactly which
joints have this problem and how large the static (gravity-load)
tracking error and torque demand are at steady state, then (run again
with --kp) whether raising/lowering Kp changes the picture -- all with
the A1's real 33.5 Nm torque ceiling UNCHANGED, so a "Kp helps but still
saturates" vs "Kp doesn't help, ceiling is the hard limit" distinction
is visible directly.

Holds every joint at cfg.scene.robot's own default_joint_pos (raw
open-loop action=0 every step, no trained policy involved) for --steps,
then reports the LAST --settle_window steps' mean per (joint_type,
side) group (hip/thigh/calf x L/R, same grouping as every other script
in this arc): steady-state tracking error |q_target-q_actual|,
tau_desired (Kp*(target-actual)-Kd*qdot, the PD's own uncapped demand),
tau_applied (post-actuator-clip), qdot, and whether that group is
saturated (|tau_applied| >= 95% of the torque ceiling) in steady state.

    PYTHONPATH="$(pwd):$(pwd)/scripts" python scripts/rl/static_standing_diagnostic.py \
        --num_envs 64 --steps 60 --kp 55
"""

from __future__ import annotations

import argparse
import os

import numpy as np

A1_TORQUE_LIMIT_NM = 33.5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--settle_window", type=int, default=10)
    parser.add_argument("--kp", type=float, default=None, help="Override actuator stiffness (default: TALON_A1_CFG's own 55.0)")
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

    base_actuator = cfg.scene.robot.actuators["base_legs"]
    kp = args.kp if args.kp is not None else base_actuator.stiffness
    kd = base_actuator.damping
    if args.kp is not None:
        # Replace (not mutate in place) -- see torque_authority_ablation.py's
        # own docstring for why: mutating the shared DCMotorCfg instance in
        # place would risk touching the module-level TALON_A1_CFG singleton
        # its fields are shallow-copied from.
        scaled_actuator = base_actuator.replace(stiffness=kp)
        cfg.scene.robot.actuators = {**cfg.scene.robot.actuators, "base_legs": scaled_actuator}

    env = gym.make("Isaac-Talon-A1-v0", cfg=cfg, render_mode=None).unwrapped
    env.reset()
    print(f"Kp={kp} Kd={kd} (torque ceiling unchanged at {A1_TORQUE_LIMIT_NM} Nm)")

    action_term = env.action_manager._terms["joint_pos"]
    joint_names = list(action_term._joint_names)
    default_joint_pos = action_term._offset[0].cpu().numpy()

    def side_idx(joint_type: str, side: str) -> list[int]:
        legs = ("FL", "RL") if side == "L" else ("FR", "RR")
        return [i for i, n in enumerate(joint_names) if joint_type in n and any(n.startswith(leg) for leg in legs)]

    joint_types = ("hip", "thigh", "calf")
    idx = {(jt, side): side_idx(jt, side) for jt in joint_types for side in ("L", "R")}

    N = args.num_envs
    T = args.steps
    track_err = {k: np.zeros((T, N), dtype=np.float32) for k in idx}
    tau_desired = {k: np.zeros((T, N), dtype=np.float32) for k in idx}
    tau_applied = {k: np.zeros((T, N), dtype=np.float32) for k in idx}
    qdot_arr = {k: np.zeros((T, N), dtype=np.float32) for k in idx}

    robot = env.scene["robot"]
    action = np.zeros((N, 12), dtype=np.float32)  # target = default the whole time
    for t in range(T):
        joint_pos_pre = robot.data.joint_pos.cpu().numpy()
        joint_vel_pre = robot.data.joint_vel.cpu().numpy()
        transition, done = env.step(action)

        target_full = np.tile(default_joint_pos, (N, 1))
        err_full = target_full - joint_pos_pre
        tau_d_full = kp * err_full - kd * joint_vel_pre
        tau_a_full = transition["joint_torque"]

        for k, v in idx.items():
            track_err[k][t] = np.abs(err_full[:, v]).mean(axis=-1)
            tau_desired[k][t] = tau_d_full[:, v].mean(axis=-1)
            tau_applied[k][t] = tau_a_full[:, v].mean(axis=-1)
            qdot_arr[k][t] = joint_vel_pre[:, v].mean(axis=-1)

    w = args.settle_window
    print(f"\n=== steady-state (last {w} of {T} steps), {N} lanes ===")
    print(f"{'group':<10}{'|err|(rad)':>12}{'tau_desired':>13}{'tau_applied':>13}{'qdot':>9}{'saturated?':>12}")
    for jt in joint_types:
        for side in ("L", "R"):
            k = (jt, side)
            err = float(track_err[k][-w:].mean())
            td = float(tau_desired[k][-w:].mean())
            ta = float(tau_applied[k][-w:].mean())
            qd = float(qdot_arr[k][-w:].mean())
            sat_frac = float((np.abs(tau_applied[k][-w:]) >= 0.95 * A1_TORQUE_LIMIT_NM).mean())
            flag = f"YES ({sat_frac * 100:.0f}%)" if sat_frac > 0.1 else "no"
            print(f"{jt + '_' + side:<10}{err:>12.4f}{td:>13.4f}{ta:>13.4f}{qd:>9.4f}{flag:>12}")


if __name__ == "__main__":
    main()
