#!/usr/bin/env python3
"""Experiment 2A.4 -- single-lane, per-step trace under hip_symmetry_
intervention.py's action override, saved to .npz (not plotted directly --
see combine_hip_intervention_traces.py for the 3-mode overlay figure,
kept as a separate pure-matplotlib script since re-creating multiple
Isaac Lab envs in one process is untested/risky here; every other
diagnostic script in this repo creates exactly one env per process).

hip_symmetry_intervention.py's aggregate numbers showed R_follows_L
(force the passive hip to mirror the active one) helped most in seed0
(falls -36%, v_z/pitch_rate/tracking all better) while L_follows_R
(force the active hip to instead mirror the passive one) made seed0
clearly worse. This traces WHAT CHANGES FIRST when the intervention is
applied -- hip torque, hip velocity, foot contact, height, v_z, pitch --
to build the causal chain (hip activity -> leg support -> height held ->
v_z improves -> pitch_rate drops -> fewer falls, or some other order)
rather than just comparing rollout-aggregate numbers.

    PYTHONPATH="$(pwd):$(pwd)/scripts" python scripts/analysis/run_diagnostics.py hip-intervention-trace \
        runs/<run>/checkpoints/checkpoint.pt 0.7 0.1 0.1 0.1 --seed 0 --steps 200 \
        --mode R_follows_L --out logs/eval_bal/hip_intervention_trace/seed0_R_follows_L.npz \
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

CONTACT_THRESHOLD_N = 1.0  # same convention as impact_reward's own foot-slip contact indicator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("w", type=float, nargs=4)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lane", type=int, default=0)
    parser.add_argument("--command", type=float, nargs=3, default=(0.5, 0.0, 0.0), metavar=("VX", "VY", "WZ"))
    parser.add_argument("--seed", type=int, default=0, help="Sets cfg.seed BEFORE env creation (not just trainer.rng)")
    parser.add_argument("--mode", choices=["none", "R_follows_L", "L_follows_R"], default="R_follows_L")
    parser.add_argument("--out", type=str, required=True, help="Output .npz path")
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
    print(f"loaded checkpoint (t={trainer._t}), env seed={args.seed}, mode={args.mode}")

    trainer.w = np.tile(np.array(args.w, dtype=np.float32) / sum(args.w), (args.num_envs, 1)).astype(np.float32)

    robot = env.scene["robot"]
    action_term = env.action_manager._terms["joint_pos"]
    joint_names = list(action_term._joint_names)
    action_scale = float(action_term.cfg.scale)
    default_joint_pos = action_term._offset[0].cpu().numpy()
    kp = float(cfg.scene.robot.actuators["base_legs"].stiffness)
    kd = float(cfg.scene.robot.actuators["base_legs"].damping)

    def idx1(leg: str, jt: str) -> int:
        return next(i for i, n in enumerate(joint_names) if n.startswith(leg) and jt in n)

    hip = {leg: idx1(leg, "hip") for leg in ("FL", "FR", "RL", "RR")}
    thigh = {leg: idx1(leg, "thigh") for leg in ("FL", "FR", "RL", "RR")}
    calf = {leg: idx1(leg, "calf") for leg in ("FL", "FR", "RL", "RR")}

    def apply_symmetry(action: np.ndarray) -> np.ndarray:
        if args.mode == "none":
            return action
        action = action.copy()
        target = action * action_scale + default_joint_pos[None, :]
        pairs = (
            ((hip["FL"], hip["FR"]), (hip["RL"], hip["RR"]))
            if args.mode == "R_follows_L"
            else ((hip["FR"], hip["FL"]), (hip["RR"], hip["RL"]))
        )
        for leader, follower in pairs:
            mirrored_target = -target[:, leader]
            action[:, follower] = (mirrored_target - default_joint_pos[follower]) / action_scale
        return action

    foot_body_ids = env._foot_body_ids
    foot_names = [env.scene["robot"].body_names[i] for i in foot_body_ids]
    left_feet = [i for i, n in enumerate(foot_names) if n.startswith("FL") or n.startswith("RL")]
    right_feet = [i for i, n in enumerate(foot_names) if n.startswith("FR") or n.startswith("RR")]

    T = args.steps
    lane = args.lane
    out = {
        "height": np.zeros(T), "v_z": np.zeros(T), "v_x": np.zeros(T),
        "pitch": np.zeros(T), "pitch_rate": np.zeros(T),
        "hip_L_torque": np.zeros(T), "hip_R_torque": np.zeros(T),
        "hip_L_qdot": np.zeros(T), "hip_R_qdot": np.zeros(T),
        "thigh_dq_target": np.zeros(T), "calf_dq_target": np.zeros(T),
        "n_feet_contact": np.zeros(T), "contact_L": np.zeros(T), "contact_R": np.zeros(T),
        "fell": np.zeros(T, dtype=bool),
    }

    trainer.model.eval()
    for t in range(T):
        env.v_command_buf[:] = torch.tensor(args.command, device=env.device)
        action = trainer.act_inference()
        action = apply_symmetry(action)

        joint_vel_pre = robot.data.joint_vel.cpu().numpy()

        transition, done = env.step(action)
        trainer._last_extrinsics = transition.get("extrinsics") if trainer.encoder else None
        trainer.push_obs(transition["obs"], done_mask=done)

        out["height"][t] = transition["height"][lane]
        out["v_z"][t] = transition["v_z"][lane]
        out["v_x"][t] = transition["v_actual"][lane, 0]
        out["pitch"][t] = transition["roll_pitch"][lane, 1]
        out["pitch_rate"][t] = transition["roll_pitch_rate"][lane, 1]
        torque = transition["joint_torque"][lane]
        out["hip_L_torque"][t] = np.mean([torque[hip["FL"]], torque[hip["RL"]]])
        out["hip_R_torque"][t] = np.mean([torque[hip["FR"]], torque[hip["RR"]]])
        qdot = joint_vel_pre[lane]
        out["hip_L_qdot"][t] = np.mean([qdot[hip["FL"]], qdot[hip["RL"]]])
        out["hip_R_qdot"][t] = np.mean([qdot[hip["FR"]], qdot[hip["RR"]]])

        target = action[lane] * action_scale + default_joint_pos
        out["thigh_dq_target"][t] = abs(
            np.mean([target[thigh["FL"]], target[thigh["RL"]]]) - np.mean([target[thigh["FR"]], target[thigh["RR"]]])
        )
        out["calf_dq_target"][t] = abs(
            np.mean([target[calf["FL"]], target[calf["RL"]]]) - np.mean([target[calf["FR"]], target[calf["RR"]]])
        )

        contact = transition["foot_contact_force"][lane] > CONTACT_THRESHOLD_N
        out["n_feet_contact"][t] = contact.sum()
        out["contact_L"][t] = contact[left_feet].mean()
        out["contact_R"][t] = contact[right_feet].mean()
        out["fell"][t] = bool(transition.get("terminal_fall", done)[lane])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, mode=args.mode, target_height=reward_cfg.target_height, command_vx=args.command[0], **out)
    print(f"saved {args.out}")
    print(f"fall steps: {np.where(out['fell'])[0].tolist()}")


if __name__ == "__main__":
    main()
