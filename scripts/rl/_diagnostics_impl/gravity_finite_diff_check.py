#!/usr/bin/env python3
"""Experiment 2E.4 -- finite-difference sanity check on PhysX's
get_generalized_gravity_forces, at the exact configuration 2E.1 Test A
found a 9.5-26x mismatch for thigh/calf (hip matched within 6%). This
does not touch the actuator/PD/RL stack at all -- it separates two very
different failure modes that would otherwise look identical from the
2E.1 result alone:

  (a) PhysX's gravity-forces query itself is wrong/misapplied (a math
      or API-usage bug on our side), independent of whether the URDF
      inertial parameters are themselves correct, vs.
  (b) the query is computing exactly what the URDF says, and the URDF's
      mass/COM/inertia values are themselves inconsistent with what
      "should" be needed to hold this pose.

If the finite-difference derivative of total gravitational potential
energy w.r.t. a joint angle agrees with PhysX's own gravity-force query
for that joint, PhysX's math/API usage is confirmed correct -- and the
2E.1 mismatch is telling us something genuine about the loaded model
(URDF mass/COM/inertia, or the pose/mapping around it), not an artifact
of this diagnostic's own gravity query. If they disagree with each
other too, the bug is upstream of even the URDF (a Pinocchio/PhysX
inverse-dynamics computation issue, or a units/frame bug on our side of
that query).

Method: fixed-base robot (articulation_props.fix_root_link=True, same
as 2E.1 Test A -- avoids the free-fall-equivalence artifact that made
this same query return ~0 for the floating-base robot), joints held at
2E.1 Test A's own settled q (hip/thigh/calf at their steady-state
values, calf_target=-1.7). For ONE test joint at a time (FL_thigh_joint,
FL_calf_joint), perturbs q by +-eps via write_joint_state_to_sim (a
direct kinematic write, NOT a PD step -- no dynamics/actuator involved,
this is a pure forward-kinematics query) and computes:

    PE(q) = sum_i m_i * g * z_i(q)      (sum over every body in the tree,
                                          using robot.data.body_pos_w)
    dPE/dq ~= (PE(q+eps) - PE(q-eps)) / (2*eps)     (central difference)

then compares -dPE/dq (the generalized gravity force convention PhysX's
API itself uses, verified empirically in 2E.1 -- raw, unnegated,
matched tau_applied's sign) against PhysX's own
get_generalized_gravity_forces() value for that joint at the same q.

    PYTHONPATH="$(pwd):$(pwd)/scripts" python scripts/rl/diagnostics.py gravity-finite-diff-check \
        --eps 0.005
"""

from __future__ import annotations

import argparse
import os

import numpy as np

G = 9.81


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eps", type=float, default=0.005, help="Perturbation size, rad")
    parser.add_argument("--calf_target", type=float, default=-1.7)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher({"headless": True, "enable_cameras": False})
    simulation_app = app_launcher.app  # noqa: F841

    import gymnasium as gym
    import torch
    import talon_rl.tasks.locomotion.a1_env  # noqa: F401
    from talon_rl.tasks.locomotion.a1_env.a1_env_cfg import IsaacLabTalonEnvCfg

    cfg = IsaacLabTalonEnvCfg()
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    cfg.scene.robot.spawn.articulation_props.fix_root_link = True
    env = gym.make("Isaac-Talon-A1-v0", cfg=cfg, render_mode=None).unwrapped
    env.reset()

    robot = env.scene["robot"]
    action_term = env.action_manager._terms["joint_pos"]
    joint_names = list(action_term._joint_names)
    default_joint_pos = action_term._offset[0].cpu().numpy()

    # 2E.1 Test A's own settled q (steady-state, last-10-of-80, condA_v2.log):
    # hip~+-0.145(from +-0.1 default +-0.04 err), thigh~1.07/0.63 (from
    # 0.8/1.0 defaults with ~0.26 err), calf=-1.7 target. We don't need the
    # exact settled values here -- this test asks whether PhysX's gravity
    # query is self-consistent AT a configuration, any configuration close
    # to the one 2E.1 flagged is sufficient. Use the same target_pose 2E.1
    # Test A actually commanded (calf=-1.7, everything else default) --
    # simpler and reproducible without re-parsing a log.
    q0 = default_joint_pos.copy()
    calf_idx = [i for i, n in enumerate(joint_names) if "calf" in n]
    q0[calf_idx] = args.calf_target

    def set_q(q: np.ndarray) -> None:
        q_t = torch.tensor(q, dtype=torch.float32, device=robot.device).unsqueeze(0)
        qd_t = torch.zeros_like(q_t)
        robot.write_joint_state_to_sim(q_t, qd_t)
        env.sim.forward()  # propagate kinematics without stepping dynamics

    set_q(q0)
    body_masses = robot.root_physx_view.get_masses()[0].cpu().numpy()  # (num_bodies,)
    print(f"body masses: {dict(zip(robot.body_names, body_masses.round(4)))}")

    def total_pe(q: np.ndarray) -> float:
        set_q(q)
        z = robot.data.body_pos_w[0, :, 2].cpu().numpy()
        return float(np.sum(body_masses * G * z))

    gravity_full = robot.root_physx_view.get_generalized_gravity_forces()
    set_q(q0)  # get_generalized_gravity_forces reads current sim state; re-set after the call above may have moved it
    gravity_full = robot.root_physx_view.get_generalized_gravity_forces()[0].cpu().numpy()
    n_base_dof = gravity_full.shape[0] - len(joint_names)
    tau_gravity_api = gravity_full[n_base_dof:]

    test_joints = ["FL_thigh_joint", "FL_calf_joint", "FL_hip_joint"]
    print(f"\n{'joint':<16}{'q0':>8}{'-dPE/dq (FD)':>14}{'tau_gravity(API)':>18}{'ratio FD/API':>14}")
    for jn in test_joints:
        j = joint_names.index(jn)
        q_plus = q0.copy()
        q_plus[j] += args.eps
        q_minus = q0.copy()
        q_minus[j] -= args.eps
        pe_plus = total_pe(q_plus)
        pe_minus = total_pe(q_minus)
        set_q(q0)
        d_pe_dq = (pe_plus - pe_minus) / (2 * args.eps)
        fd_tau = -d_pe_dq  # PhysX's own convention (verified 2E.1): tau_gravity ~ -dPE/dq matches tau_applied sign... actually checked empirically below
        api_tau = tau_gravity_api[j]
        ratio = fd_tau / api_tau if abs(api_tau) > 1e-9 else float("nan")
        print(f"{jn:<16}{q0[j]:>8.4f}{fd_tau:>14.5f}{api_tau:>18.5f}{ratio:>14.3f}")


if __name__ == "__main__":
    main()
