#!/usr/bin/env python3
"""Experiment 2E.6 -- ground-contact timestep validity check. 2E.5 found
dt=0.02 produces a severe, non-decaying period-2 numerical instability
in the FIXED-BASE, no-contact diagnostic (calf qdot pinned at the
velocity_limit, alternating sign every step) -- resolved cleanly by
dt=0.01. But 2D.2/2D.3's "no feasible standing configuration" and
"error not gain-limited" conclusions were measured under ordinary
floating-base, GROUND-CONTACT conditions, also at dt=0.02. This checks
whether that same instability contaminates the ground-contact case
before deciding whether 2D.2/2D.3 need to be redone.

Ordinary reset (floating base, terrain contact allowed, gravity, the A1
falls onto the ground and stands as best it can), calf_target=-1.7,
Kp=25 (matching 2D.2's cleanest-settling point), Kd=0.8, torque ceiling
33.5 Nm -- same as 2D.2/2D.3. Two conditions: G0 (dt=0.02, matches
2D.2/2D.3 exactly) and G1 (dt=0.01). Reports, individually per-step for
the last --n_samples (NOT averaged, same lesson as 2E.2 -- a settle-
window mean can hide an oscillation that cancels in the average): qdot,
qddot, q_actual, tau_applied, saturation flag, height, v_z, pitch,
pitch_rate, n_feet_contact, plus a tail-window oscillation amplitude
(max-min over the reported window) per joint group.

Decision rule (per the user's own framing): if G0 and G1 closely agree
(height/v_z/qdot/contact within noise, no period-2 runaway in either),
2D.2/2D.3 are NOT contaminated by the fixed-base instability -- contact
+ floating-base dynamics change the effective system enough that the
diagnostic-only artifact doesn't apply here, and the existing 2D
conclusions stand. If G0 and G1 diverge substantially, 2D.2/2D.3 need
to be redone at a timestep where behavior is dt-independent.

    PYTHONPATH="$(pwd):$(pwd)/scripts" python scripts/rl/diagnostics.py ground-contact-timestep-check \
        --sim_dt 0.02 --num_envs 32 --steps 150 --n_samples 10
"""

from __future__ import annotations

import argparse

import numpy as np

A1_TORQUE_LIMIT_NM = 33.5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--kp", type=float, default=25.0)
    parser.add_argument("--calf_target", type=float, default=-1.7)
    parser.add_argument("--sim_dt", type=float, default=0.02)
    parser.add_argument("--decimation", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from scripts.rl._diagnostics_impl._common import build_target_action, joint_side_groups, make_diagnostic_env

    simulation_app, env, kp, kd = make_diagnostic_env(  # noqa: F841 (simulation_app kept alive)
        num_envs=args.num_envs, seed=args.seed, kp=args.kp, sim_dt=args.sim_dt, decimation=args.decimation,
    )
    dt = args.sim_dt * args.decimation
    print(f"GROUND-CONTACT (floating base) Kp={args.kp} Kd={kd} calf_target={args.calf_target} "
          f"sim_dt={args.sim_dt} decimation={args.decimation} (controller dt={dt}) torque_ceiling={A1_TORQUE_LIMIT_NM}")

    robot = env.scene["robot"]
    action_term = env.action_manager._terms["joint_pos"]
    joint_names = list(action_term._joint_names)
    default_joint_pos = action_term._offset[0].cpu().numpy()

    idx = joint_side_groups(joint_names)
    calf_idx = idx[("calf", "L")] + idx[("calf", "R")]
    target_pose = default_joint_pos.copy()
    target_pose[calf_idx] = args.calf_target

    N = args.num_envs
    T = args.steps
    action = build_target_action(action_term, target_pose, N)

    qdot_hist = np.zeros((T, N, 12), dtype=np.float32)
    q_hist = np.zeros((T, N, 12), dtype=np.float32)
    tau_applied_hist = np.zeros((T, N, 12), dtype=np.float32)
    height_hist = np.zeros((T, N), dtype=np.float32)
    vz_hist = np.zeros((T, N), dtype=np.float32)
    pitch_hist = np.zeros((T, N), dtype=np.float32)
    contact_hist = np.zeros((T, N), dtype=np.float32)

    for t in range(T):
        joint_pos_pre = robot.data.joint_pos.cpu().numpy()
        joint_vel_pre = robot.data.joint_vel.cpu().numpy()
        transition, done = env.step(action)
        q_hist[t] = joint_pos_pre
        qdot_hist[t] = joint_vel_pre
        tau_applied_hist[t] = transition["joint_torque"]
        height_hist[t] = transition["height"]
        vz_hist[t] = transition["v_z"]
        pitch_hist[t] = transition["roll_pitch"][:, 1]
        if "foot_contact_force" in transition:
            contact_hist[t] = (transition["foot_contact_force"] > 1.0).sum(axis=-1)

    qddot_hist = np.zeros_like(qdot_hist)
    qddot_hist[1:-1] = (qdot_hist[2:] - qdot_hist[:-2]) / (2 * dt)
    pitch_rate_hist = np.zeros_like(pitch_hist)
    pitch_rate_hist[1:-1] = (pitch_hist[2:] - pitch_hist[:-2]) / (2 * dt)

    s = args.n_samples
    sample_range = range(T - s, T)
    print(f"\n=== last {s} individual steps (NOT averaged), {N}-lane mean per step ===")
    for jt in ("hip", "thigh", "calf"):
        for side in ("L", "R"):
            k = (jt, side)
            v = idx[k]
            qdot_win = qdot_hist[T - s:T][:, :, v].mean(axis=-1).mean(axis=-1)
            amp = float(qdot_win.max() - qdot_win.min())
            sat_frac = float((np.abs(tau_applied_hist[T - s:T][:, :, v]) >= 0.95 * A1_TORQUE_LIMIT_NM).mean())
            print(f"\n--- {jt}_{side} (tail qdot oscillation amplitude: {amp:.4f} rad/s, sat_frac: {sat_frac*100:.1f}%) ---")
            print(f"{'t':<5}{'q_actual':>10}{'qdot':>9}{'qddot':>10}{'tau_applied':>12}")
            for t in sample_range:
                qa = float(q_hist[t][:, v].mean())
                qd = float(qdot_hist[t][:, v].mean())
                qdd = float(qddot_hist[t][:, v].mean())
                ta = float(tau_applied_hist[t][:, v].mean())
                print(f"{t - T:<5}{qa:>10.4f}{qd:>9.4f}{qdd:>10.4f}{ta:>12.4f}")

    print(f"\n{'metric':<20}{'mean(last '+str(s)+')':>16}")
    print(f"{'height':<20}{float(height_hist[T-s:T].mean()):>16.4f}")
    print(f"{'|v_z|':<20}{float(np.abs(vz_hist[T-s:T]).mean()):>16.4f}")
    print(f"{'pitch (rad)':<20}{float(pitch_hist[T-s:T].mean()):>16.4f}")
    print(f"{'|pitch_rate|':<20}{float(np.abs(pitch_rate_hist[T-s:T]).mean()):>16.4f}")
    print(f"{'n_feet_contact':<20}{float(contact_hist[T-s:T].mean()):>16.4f}")


if __name__ == "__main__":
    main()
