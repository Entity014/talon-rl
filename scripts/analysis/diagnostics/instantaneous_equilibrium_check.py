#!/usr/bin/env python3
"""Experiment 2E.2 -- instantaneous (no time-averaging) dynamics check, to
tell apart the two remaining explanations for 2E.1's 10-26x tau_applied
vs tau_gravity mismatch on thigh/calf (fixed-base, free-space, calf
target -1.7, Kp=25):

  Case A: thigh/calf are truly at dynamic equilibrium (qdot~0 AND
          qddot~0) when sampled, and tau_PD~tau_gravity there too --
          2E.1's gap was a settle-WINDOW-averaging artifact (mean |qdot|
          over 10 steps can look ~0 even if individual steps still have
          real qddot/oscillation cancelling in the mean).
  Case B: |tau_PD - tau_gravity| stays large even when |qdot|~0 AND
          |qddot|~0 at the SAME instant -- a genuine equilibrium
          violation, i.e. a real model/mapping inconsistency (Newton's
          law says tau_applied should equal tau_gravity(q) exactly at
          zero-acceleration equilibrium in free space with no contact;
          if it doesn't, something in the loaded actuator/URDF model is
          wrong, not just "not settled yet").
  Case C: qdot~0 but qddot still shows real spikes/oscillation -- not
          equilibrium at all, explains why the settle-window mean looked
          deceptively small.

Reports EVERY one of the last --n_samples steps individually (not
averaged) per (joint_type, side) group: qdot, qddot (central finite
difference of qdot, dt=physics step size), tau_PD (hand-computed from
Kp/Kd, matches tau_applied when unsaturated -- 2E.1 already confirmed
this), tau_gravity (PhysX generalized-gravity query, queried at the SAME
sim state as tau_PD's q -- BEFORE stepping, not after, to avoid a
timing/staleness mismatch), residual r=tau_PD-tau_gravity, and the
normalized residual |r|/max(|tau_gravity|, eps) (ratio alone is not
used as the primary judgment call -- explodes when tau_gravity is near
zero).

    PYTHONPATH="$(pwd):$(pwd)/scripts" python scripts/analysis/run_diagnostics.py instantaneous-equilibrium-check \
        --num_envs 32 --steps 80 --n_samples 15
"""

from __future__ import annotations

import argparse

import numpy as np

A1_TORQUE_LIMIT_NM = 33.5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--n_samples", type=int, default=15, help="How many of the last steps to report individually")
    parser.add_argument("--kp", type=float, default=25.0)
    parser.add_argument("--calf_target", type=float, default=-1.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sim_dt", type=float, default=0.02, help="Physics timestep, s -- 2E.5 Condition A: shrink this (and decimation=1, so controller period shrinks with it)")
    parser.add_argument("--decimation", type=int, default=1, help="Physics substeps per controller update -- 2E.5 Condition B: raise this while shrinking --sim_dt proportionally to hold the controller period fixed at 0.02s")
    args = parser.parse_args()

    from scripts.analysis.diagnostics._common import build_target_action, joint_side_groups, make_diagnostic_env

    simulation_app, env, kp, kd = make_diagnostic_env(  # noqa: F841 (simulation_app kept alive)
        num_envs=args.num_envs, seed=args.seed, kp=args.kp, fix_root_link=True,
        sim_dt=args.sim_dt, decimation=args.decimation,
    )
    dt = args.sim_dt * args.decimation
    print(f"Kp={args.kp} Kd={kd} calf_target={args.calf_target} dt={dt} torque_ceiling={A1_TORQUE_LIMIT_NM}")

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
    tau_pd_hist = np.zeros((T, N, 12), dtype=np.float32)
    tau_applied_hist = np.zeros((T, N, 12), dtype=np.float32)
    tau_gravity_hist = np.zeros((T, N, 12), dtype=np.float32)

    n_base_dof = None
    for t in range(T):
        joint_pos_pre = robot.data.joint_pos.cpu().numpy()
        joint_vel_pre = robot.data.joint_vel.cpu().numpy()
        # query gravity at the SAME sim state used for tau_PD's q (before step)
        gravity_full = robot.root_physx_view.get_generalized_gravity_forces().cpu().numpy()
        if n_base_dof is None:
            n_base_dof = gravity_full.shape[1] - len(joint_names)
        tau_gravity_hist[t] = gravity_full[:, n_base_dof:]

        transition, done = env.step(action)

        err_full = np.tile(target_pose, (N, 1)) - joint_pos_pre
        tau_pd_hist[t] = args.kp * err_full - kd * joint_vel_pre
        tau_applied_hist[t] = transition["joint_torque"]
        qdot_hist[t] = joint_vel_pre

    # central-difference qddot from the qdot history (dt = env control step)
    qddot_hist = np.zeros_like(qdot_hist)
    qddot_hist[1:-1] = (qdot_hist[2:] - qdot_hist[:-2]) / (2 * dt)

    s = args.n_samples
    sample_range = range(T - s, T)
    print(f"\n=== last {s} individual steps (NOT averaged), {N}-lane mean per step ===")
    for jt in ("hip", "thigh", "calf"):
        for side in ("L", "R"):
            k = (jt, side)
            v = idx[k]
            print(f"\n--- {jt}_{side} ---")
            print(f"{'t':<5}{'qdot':>9}{'qddot':>10}{'tau_PD':>10}{'tau_grav':>10}{'resid':>9}{'|r|/|g|':>9}")
            for t in sample_range:
                qd = float(qdot_hist[t][:, v].mean())
                qdd = float(qddot_hist[t][:, v].mean())
                tpd = float(tau_pd_hist[t][:, v].mean())
                tg = float(tau_gravity_hist[t][:, v].mean())
                r = tpd - tg
                rn = abs(r) / max(abs(tg), 1e-3)
                print(f"{t - T:<5}{qd:>9.4f}{qdd:>10.4f}{tpd:>10.4f}{tg:>10.4f}{r:>9.4f}{rn:>9.3f}")


if __name__ == "__main__":
    main()
