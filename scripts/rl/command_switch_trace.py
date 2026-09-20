#!/usr/bin/env python3
"""Experiment 2C.3 -- closed-loop command-switch response (Step C).
command_sensitivity_probe.py's Step B found the actor IS substantially
sensitive to command at the single-step/isolated-observation level
(action deltas reaching near-ACTION_CLIP magnitude), while command_
gait_comparison.py's full-rollout aggregate found no systematic
difference in mean joint targets across a 150-step window under
different commands. This resolves the two by NOT resetting the
environment between commands -- a single continuous rollout with the
command switched partway through, so any transient response and its
decay (or persistence) is directly visible, not averaged away.

Default schedule (--commands, 3 segments of --segment_len steps each):
+0.25 -> -0.25 -> +0.25 -- covers BOTH switch directions in one run
(switch 1 is +->-, switch 2 is -->+), no need for a separately reversed
schedule.

For each switch boundary, computes, per lane, Delta_x(t) = x(t) -
x(t_switch - 1) for t in the following segment (x = action, q_target,
q_actual, qdot, tau_desired, tau_applied, v_x, height, v_z, pitch,
contact_L, contact_R), then reports mean |Delta_x| as a function of
steps-since-switch -- an impulse-response curve, extended
(Experiment 2C.4) to trace the FULL causal chain command -> action ->
q_target -> q_actual -> qdot -> tau_desired -> tau_applied -> contact
-> v_z/height -> v_x, so a command-sensitivity result that shows up at
the action level but not the v_x level (found via this script's own
switch-vs-no-switch control comparison, 2026-09-20) can be localized to
a specific stage: q_target->q_actual (position-tracking/actuator-
response bottleneck), q_actual/tau->contact (mechanical-interaction
bottleneck), or contact/state->v_x (gait-geometry/locomotion-
representation bottleneck -- the leg motion changes but produces no
net forward impulse). IMPORTANT: this script alone measures raw
Delta_x within ONE trajectory, which includes ordinary within-episode
drift (this whole investigation already found substantial cycle-to-
cycle drift under a CONSTANT command) -- run once with a real switch
schedule and once with a no-switch (constant command) control of the
same length, then subtract the two Delta_x tables by hand (or script)
to get the command-SPECIFIC excess response; the raw numbers alone
overstate how "persistent" any effect looks. Reading the excess:

  Case 1 (persistent):      |Delta| rises after the switch and STAYS
                             elevated through the whole segment -- command
                             conditioning works, response is just slow.
  Case 2 (transient, decays back to ~0): |Delta| spikes then decays back
                             toward 0 within a few steps -- local command
                             sensitivity, weak closed-loop persistence
                             (command_sensitivity_probe.py's Step B
                             hypothesis).
  Case 3 (action moves, state doesn't): Delta_action/Delta_q_target stay
                             elevated but Delta_v_x/Delta_height/
                             Delta_contact stay near 0 the whole segment
                             -- actuation/contact/dynamics filtering, the
                             actor is trying, the physical state isn't
                             responding.
  Case 4 (state moves then gets pulled back): Delta_v_x/Delta_height
                             rise for a few steps then decay back to ~0
                             WHILE Delta_action stays nonzero (the actor
                             keeps correcting) -- command response gets
                             overridden by the policy's own feedback
                             dynamics, not ignored outright.

Saves one representative lane's full per-step trace to .npz for a
plotted impulse-response figure (see plot_command_switch.py) in
addition to the printed aggregate table.

    PYTHONPATH="$(pwd):$(pwd)/scripts" python scripts/rl/command_switch_trace.py \
        runs/<run>/checkpoints/checkpoint.pt 0.7 0.1 0.1 0.1 --seed 0 \
        --segment_len 30 --commands 0.25 -0.25 0.25 \
        --out logs/.../seed0_switch.npz \
        --progress_std 0.5 --balance_tilt_coef 1.0 --target_height 0.42 --balance_hip_activation_coef 0.02
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

CONTACT_THRESHOLD_N = 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("w", type=float, nargs=4)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--segment_len", type=int, default=30)
    parser.add_argument("--commands", type=float, nargs="+", default=[0.25, -0.25, 0.25], help="v_x per segment; vy/omega_z always 0")
    parser.add_argument("--lane", type=int, default=0, help="Which lane's full trace to save to --out")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0, help="Sets cfg.seed BEFORE env creation (not just trainer.rng)")
    parser.add_argument("--progress_std", type=float, default=None, help="Must match what the checkpoint was trained with")
    parser.add_argument("--balance_tilt_coef", type=float, default=None, help="Must match what the checkpoint was trained with")
    parser.add_argument("--balance_tilt_rate_coef", type=float, default=None, help="Must match what the checkpoint was trained with")
    parser.add_argument("--balance_height_coef", type=float, default=None, help="Must match what the checkpoint was trained with")
    parser.add_argument("--target_height", type=float, default=None, help="Must match what the checkpoint was trained with")
    parser.add_argument("--balance_hip_activation_coef", type=float, default=None, help="Must match what the checkpoint was trained with")
    parser.add_argument("--balance_hip_sym_coef", type=float, default=None, help="Must match what the checkpoint was trained with")
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
    if args.balance_hip_activation_coef is not None:
        reward_cfg_overrides["balance_hip_activation_coef"] = args.balance_hip_activation_coef
    if args.balance_hip_sym_coef is not None:
        reward_cfg_overrides["balance_hip_sym_coef"] = args.balance_hip_sym_coef
    reward_cfg = RewardVectorCfg(**reward_cfg_overrides)
    pref_cfg = PreferenceCfg()
    stack_cfg = ObservationStackCfg()
    extrinsics_cfg = ExtrinsicsCfg()

    cfg = IsaacLabTalonEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    env = gym.make("Isaac-Talon-A1-v0", cfg=cfg, render_mode=None).unwrapped

    trainer = MOPPOTrainer(
        env, obs_cfg, reward_cfg, pref_cfg, moppo_cfg=MOPPOConfig(), stack_cfg=stack_cfg,
        extrinsics_cfg=extrinsics_cfg, seed=args.seed,
    )
    trainer.load(args.checkpoint)
    print(f"loaded checkpoint (t={trainer._t}), env seed={args.seed}, schedule={args.commands} x {args.segment_len} steps")

    trainer.w = np.tile(np.array(args.w, dtype=np.float32) / sum(args.w), (args.num_envs, 1)).astype(np.float32)

    action_term = env.action_manager._terms["joint_pos"]
    action_scale = float(action_term.cfg.scale)
    default_joint_pos = action_term._offset[0].cpu().numpy()
    kp = float(cfg.scene.robot.actuators["base_legs"].stiffness)
    kd = float(cfg.scene.robot.actuators["base_legs"].damping)

    foot_body_ids = env._foot_body_ids
    foot_names = [env.scene["robot"].body_names[i] for i in foot_body_ids]
    left_feet = [i for i, n in enumerate(foot_names) if n.startswith("FL") or n.startswith("RL")]
    right_feet = [i for i, n in enumerate(foot_names) if n.startswith("FR") or n.startswith("RR")]

    T, N = args.segment_len * len(args.commands), args.num_envs
    action_arr = np.zeros((T, N, 12), dtype=np.float32)
    q_target_arr = np.zeros((T, N, 12), dtype=np.float32)
    q_actual_arr = np.zeros((T, N, 12), dtype=np.float32)
    qdot_arr = np.zeros((T, N, 12), dtype=np.float32)
    tau_desired_arr = np.zeros((T, N, 12), dtype=np.float32)
    tau_applied_arr = np.zeros((T, N, 12), dtype=np.float32)
    v_x = np.zeros((T, N), dtype=np.float32)
    v_z = np.zeros((T, N), dtype=np.float32)
    height = np.zeros((T, N), dtype=np.float32)
    pitch = np.zeros((T, N), dtype=np.float32)
    pitch_rate = np.zeros((T, N), dtype=np.float32)
    contact_L = np.zeros((T, N), dtype=np.float32)
    contact_R = np.zeros((T, N), dtype=np.float32)
    terminal_fall = np.zeros((T, N), dtype=bool)

    trainer.model.eval()
    for t in range(T):
        segment = t // args.segment_len
        vx_cmd = args.commands[segment]
        env.v_command_buf[:] = torch.tensor([vx_cmd, 0.0, 0.0], device=env.device)
        action = trainer.act_inference()

        # Snapshot actual joint state BEFORE stepping -- the PD error that
        # produces THIS step's applied torque is computed against the
        # pre-step joint_pos/joint_vel, not the post-step one (same
        # convention as torque_authority_ablation.py/gait_joint_trace.py).
        joint_pos_pre = env.scene["robot"].data.joint_pos.cpu().numpy()
        joint_vel_pre = env.scene["robot"].data.joint_vel.cpu().numpy()

        transition, done = env.step(action)
        trainer._last_extrinsics = transition.get("extrinsics") if trainer.encoder else None
        trainer.push_obs(transition["obs"], done_mask=done)

        target = action * action_scale + default_joint_pos[None, :]
        q_actual_arr[t] = joint_pos_pre
        qdot_arr[t] = joint_vel_pre
        tau_desired_arr[t] = kp * (target - joint_pos_pre) - kd * joint_vel_pre
        tau_applied_arr[t] = transition["joint_torque"]

        action_arr[t] = action
        q_target_arr[t] = action * action_scale + default_joint_pos[None, :]
        v_x[t] = transition["v_actual"][:, 0]
        v_z[t] = transition["v_z"]
        height[t] = transition["height"]
        pitch[t] = transition["roll_pitch"][:, 1]
        pitch_rate[t] = transition["roll_pitch_rate"][:, 1]
        contact = transition["foot_contact_force"] > CONTACT_THRESHOLD_N
        contact_L[t] = contact[:, left_feet].mean(axis=-1)
        contact_R[t] = contact[:, right_feet].mean(axis=-1)
        terminal_fall[t] = transition.get("terminal_fall", done).astype(bool)

    # Impulse-response tables, one per switch boundary.
    for switch_idx in range(1, len(args.commands)):
        switch_step = switch_idx * args.segment_len
        pre_step = switch_step - 1
        seg_len = args.segment_len
        alive = ~terminal_fall[switch_step:switch_step + seg_len]  # exclude exact fall steps from this segment's stats

        def seg_diff_joint(arr: np.ndarray) -> np.ndarray:
            return np.abs(arr[switch_step:switch_step + seg_len] - arr[pre_step][None]).mean(axis=-1)  # (seg_len, N)

        def seg_diff_scalar(arr: np.ndarray) -> np.ndarray:
            return np.abs(arr[switch_step:switch_step + seg_len] - arr[pre_step][None])

        d_action = seg_diff_joint(action_arr)
        d_qtarget = seg_diff_joint(q_target_arr)
        d_qactual = seg_diff_joint(q_actual_arr)
        d_qdot = seg_diff_joint(qdot_arr)
        d_tau_desired = seg_diff_joint(tau_desired_arr)
        d_tau_applied = seg_diff_joint(tau_applied_arr)
        d_vx = seg_diff_scalar(v_x)
        d_height = seg_diff_scalar(height)
        d_vz = seg_diff_scalar(v_z)
        d_pitch = seg_diff_scalar(pitch)
        d_contact = (seg_diff_scalar(contact_L) + seg_diff_scalar(contact_R)) / 2.0

        print(f"\n=== switch {switch_idx}: {args.commands[switch_idx - 1]:+.2f} -> {args.commands[switch_idx]:+.2f} "
              f"(at step {switch_step}) ===")
        print(f"{'t-switch':<9}{'|Da|':>8}{'|Dqtgt|':>9}{'|Dqact|':>9}{'|Dqdot|':>9}{'|Dtaud|':>9}{'|Dtaua|':>9}"
              f"{'|Dcont|':>9}{'|Dvz|':>8}{'|Dh|':>8}{'|Dvx|':>8}")
        for offset in range(seg_len):
            mask = alive[offset]
            if not mask.any():
                continue
            print(f"{offset:<9}"
                  f"{float(d_action[offset][mask].mean()):>8.4f}"
                  f"{float(d_qtarget[offset][mask].mean()):>9.4f}"
                  f"{float(d_qactual[offset][mask].mean()):>9.4f}"
                  f"{float(d_qdot[offset][mask].mean()):>9.4f}"
                  f"{float(d_tau_desired[offset][mask].mean()):>9.4f}"
                  f"{float(d_tau_applied[offset][mask].mean()):>9.4f}"
                  f"{float(d_contact[offset][mask].mean()):>9.4f}"
                  f"{float(d_vz[offset][mask].mean()):>8.4f}"
                  f"{float(d_height[offset][mask].mean()):>8.4f}"
                  f"{float(d_vx[offset][mask].mean()):>8.4f}")

    if args.out:
        lane = args.lane
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        np.savez(
            args.out,
            commands=np.array(args.commands), segment_len=args.segment_len,
            action=action_arr[:, lane, :], q_target=q_target_arr[:, lane, :],
            q_actual=q_actual_arr[:, lane, :], qdot=qdot_arr[:, lane, :],
            tau_desired=tau_desired_arr[:, lane, :], tau_applied=tau_applied_arr[:, lane, :],
            v_x=v_x[:, lane], v_z=v_z[:, lane], height=height[:, lane],
            pitch=pitch[:, lane], pitch_rate=pitch_rate[:, lane],
            contact_L=contact_L[:, lane], contact_R=contact_R[:, lane],
            fell=terminal_fall[:, lane],
        )
        print(f"\nsaved lane {lane}'s trace to {args.out}")


if __name__ == "__main__":
    main()
