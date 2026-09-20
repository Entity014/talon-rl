#!/usr/bin/env python3
"""Experiment 2E.1 -- static gravity/actuator torque consistency check, no
RL, no policy, no Kp/Kd tuning. 2D.2/2D.3 found e_ss ~= tau_holding/Kp
holds exactly at every tested point (calf=-1.7, Kp=25: e_ss~0.275,
tau~6.7 Nm, 25*0.275=6.88 -- consistent), and that raising Kp does not
reduce e_ss. That rules out "just needs more gain" but leaves open
WHETHER the ~6.7 Nm the PD demands to hold this pose is itself correct
for this configuration -- if the simulator's own gravity model disagrees
with what the PD is doing, the mismatch is in the model/mapping, not the
controller. This script does not decide that -- it just prints tau_PD
(measured, from the actuator) next to tau_gravity (from PhysX's own
inverse-dynamics gravity-forces query at the same settled q) so a human
can judge whether they reconcile.

One condition per PROCESS (--condition A/B/C), not per-script-run:
recreating an Isaac Lab env a second time inside the same process is
unreliable (hangs / can crash on USD teardown, same class of flakiness
noted for other multi-run diagnostic scripts this session) -- run this
3x from bash instead, exactly like every other script in this arc.

  A. free_space -- robot spawned with articulation_props.fix_root_link=
     True (a real fixed joint at the base, set on the spawn cfg BEFORE
     gym.make(); NOT a per-step teleport -- an earlier version of this
     script tried re-writing root pose every step and that injected
     enough solver-shock velocity that qdot never settled, calf qdot
     stayed ~-4 rad/s after 20 steps -- fix_root_link avoids that
     entirely, it's PhysX's own fixed-base joint, no teardown/rewrite
     each tick) at cfg.scene.robot's own spawn height. No ground contact
     possible (robot bolted in place), gravity acts on the legs normally.
  B. ground_contact -- ordinary reset, terrain contact allowed,
     calf_target=-1.7 (2D.2/2D.3's cleanest equilibrium).
  C. nominal_pose -- ordinary reset, terrain contact allowed,
     calf_target=None (TALON_A1_CFG's own default -1.5, i.e. the pose
     default_joint_pos actually claims is "standing").

Same Kp=25/Kd=0.8(default)/torque ceiling 33.5 Nm in all three. After
--steps of zero policy action (target held fixed) and qdot settled,
reports per (joint_type, side): q_error, qdot, tau_applied (measured),
tau_gravity (PhysX get_generalized_gravity_forces at the final settled
q -- NOT the same thing as tau_applied when contact/coriolis terms are
present, see warning below), plus height, |v_z|, foot contact force.

WARNING (do not skip): tau_gravity from PhysX is the gravity-only
generalized-force term at the queried configuration. At true static
equilibrium with zero ground reaction (condition A), tau_PD should
match tau_gravity closely -- that IS the point of condition A. In B/C,
ground contact injects its own generalized force
(tau_required = tau_gravity + tau_contact + tau_other), so tau_PD and
tau_gravity are NOT expected to match there; the gap (tau_PD -
tau_gravity) in B/C is itself the diagnostic quantity -- it isolates
what the ground contact is contributing.

    PYTHONPATH="$(pwd):$(pwd)/scripts" python scripts/rl/diagnostics.py gravity-actuator-consistency \
        --condition A --num_envs 32 --steps 60 --settle_window 10
"""

from __future__ import annotations

import argparse

import numpy as np

A1_TORQUE_LIMIT_NM = 33.5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["A", "B", "C"], required=True)
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--settle_window", type=int, default=10)
    parser.add_argument("--kp", type=float, default=25.0)
    parser.add_argument("--calf_target", type=float, default=-1.7, help="Used for conditions A/B; ignored (set to None internally) for condition C")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from scripts.rl._diagnostics_impl._common import build_target_action, joint_side_groups, make_diagnostic_env

    calf_target = None if args.condition == "C" else args.calf_target
    simulation_app, env, kp, kd = make_diagnostic_env(  # noqa: F841 (simulation_app kept alive)
        num_envs=args.num_envs, seed=args.seed, kp=args.kp, fix_root_link=(args.condition == "A"),
    )
    print(f"condition={args.condition} Kp={args.kp} Kd={kd} calf_target={calf_target} "
          f"fix_root_link={args.condition == 'A'} torque_ceiling={A1_TORQUE_LIMIT_NM}")

    action_term = env.action_manager._terms["joint_pos"]
    joint_names = list(action_term._joint_names)
    default_joint_pos = action_term._offset[0].cpu().numpy()

    idx = joint_side_groups(joint_names)
    calf_idx = idx[("calf", "L")] + idx[("calf", "R")]

    target_pose = default_joint_pos.copy()
    if calf_target is not None:
        target_pose[calf_idx] = calf_target

    robot = env.scene["robot"]
    N = args.num_envs
    T = args.steps
    action = build_target_action(action_term, target_pose, N)

    track_err, tau_desired, tau_applied, qdot_arr = {}, {}, {}, {}
    for k in idx:
        track_err[k] = np.zeros((T, N), dtype=np.float32)
        tau_desired[k] = np.zeros((T, N), dtype=np.float32)
        tau_applied[k] = np.zeros((T, N), dtype=np.float32)
        qdot_arr[k] = np.zeros((T, N), dtype=np.float32)
    height_arr = np.zeros((T, N), dtype=np.float32)
    v_z_arr = np.zeros((T, N), dtype=np.float32)
    contact_arr = np.zeros((T, N, 4), dtype=np.float32)

    for t in range(T):
        joint_pos_pre = robot.data.joint_pos.cpu().numpy()
        joint_vel_pre = robot.data.joint_vel.cpu().numpy()
        transition, done = env.step(action)

        target_full = np.tile(target_pose, (N, 1))
        err_full = target_full - joint_pos_pre
        tau_d_full = args.kp * err_full - kd * joint_vel_pre
        tau_a_full = transition["joint_torque"]
        for k, v in idx.items():
            track_err[k][t] = np.abs(err_full[:, v]).mean(axis=-1)
            tau_desired[k][t] = tau_d_full[:, v].mean(axis=-1)
            tau_applied[k][t] = tau_a_full[:, v].mean(axis=-1)
            qdot_arr[k][t] = joint_vel_pre[:, v].mean(axis=-1)
        height_arr[t] = transition["height"]
        v_z_arr[t] = transition["v_z"]
        if "foot_contact_force" in transition:
            contact_arr[t] = transition["foot_contact_force"]

    # tau_gravity at the final settled q, from PhysX's own inverse dynamics.
    # Verified empirically (scratch check against a fixed-base run,
    # 2026-09-20): the RAW value from get_generalized_gravity_forces (no
    # sign flip) already matches tau_applied's sign in every joint group --
    # negating it (an earlier version of this script did) was backwards.
    # get_gravity_compensation_forces (the non-deprecated replacement)
    # returns bit-identical values, so this isn't an API-choice issue.
    gravity_full = robot.root_physx_view.get_generalized_gravity_forces().cpu().numpy()
    n_base_dof = gravity_full.shape[1] - len(joint_names)
    tau_gravity_hold = gravity_full[:, n_base_dof:]
    if args.condition != "A":
        print(f"  NOTE: condition {args.condition} is floating-base -- "
              f"get_generalized_gravity_forces is not meaningful here (returns "
              f"~1e-7, numerically zero, verified 2026-09-20 scratch check). "
              f"tau_gravity below should be disregarded; only condition A "
              f"(fixed-base) gives a valid tau_gravity reference.")

    w = args.settle_window
    print(f"\n=== steady-state (last {w} of {T}), {N} lanes ===")
    print(f"{'group':<10}{'|err|':>9}{'tau_PD':>10}{'tau_applied':>13}{'tau_gravity':>13}{'qdot':>9}{'sat?':>8}")
    for jt in ("hip", "thigh", "calf"):
        for side in ("L", "R"):
            k = (jt, side)
            v = idx[k]
            err = float(track_err[k][-w:].mean())
            td = float(tau_desired[k][-w:].mean())
            ta = float(tau_applied[k][-w:].mean())
            tg = float(tau_gravity_hold[:, v].mean())
            qd = float(qdot_arr[k][-w:].mean())
            sat_frac = float((np.abs(tau_applied[k][-w:]) >= 0.95 * A1_TORQUE_LIMIT_NM).mean())
            flag = f"YES({sat_frac*100:.0f}%)" if sat_frac > 0.1 else "no"
            print(f"{jt+'_'+side:<10}{err:>9.4f}{td:>10.4f}{ta:>13.4f}{tg:>13.4f}{qd:>9.4f}{flag:>8}")
    mean_contact = float(contact_arr[-w:].mean()) if contact_arr.any() else float("nan")
    print(f"\n  mean height: {float(height_arr[-w:].mean()):.4f}  mean |v_z|: {float(np.abs(v_z_arr[-w:]).mean()):.4f}  "
          f"mean foot_contact_force: {mean_contact:.4f}")


if __name__ == "__main__":
    main()
