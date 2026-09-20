#!/usr/bin/env python3
"""Experiment 2C.1 -- reverse-command mechanism analysis. 2B.2 found
reverse failure holds even at vx=-0.25 (within the trained [-0.3, 1.0]
command distribution, ~23% of episodes) -- negative-command exposure did
not translate into a learned reverse skill. This tests WHY, aggregated
across all lanes (not one traced lane), by comparing the policy's own
commanded joint targets, progress reward, and contact pattern across
command=+0.25/0/-0.25 under the SAME checkpoint+seed:

  Hypothesis A (command ignored): joint targets under -0.25 look like
      the same forward/standing gait as +0.25 or 0 -- the policy never
      developed a command-sign-conditioned gait at all.
  Hypothesis B (gait changes but geometry doesn't reverse): joint
      targets under -0.25 clearly differ from +0.25/0, but contact
      pattern/resulting v_x still doesn't go negative -- the policy
      tries something different, it just doesn't produce backward
      motion (a dynamics/gait-representation/exploration problem).
  Hypothesis C (reward trade-off suppresses reverse): if the raw
      progress_reward achieved under -0.25 is LOW (i.e. the policy
      isn't even collecting the strong reward gradient reverse_command_
      analysis's reward-landscape check already established exists),
      but other objectives (balance/hip_activation) score reasonably --
      suggests a multi-objective conflict pulling away from progress
      specifically under this command, not an inability to move
      backward per se.

Reports, per command, aggregated over all lanes' Phase-C-equivalent
window (same [50,200) convention as attractor_phase_metrics.py):
mean/std q_target per (hip/thigh/calf, L/R) -- 6 values -- mean
progress_reward, mean balance_reward, mean contact_L/contact_R, mean
v_x. Run once per command (this script takes one --command like every
other diagnostic script here); compare the printed tables across the
3 command runs by hand or with a copy of the output side by side, same
convention prefall_window_analysis.py etc. already use for cross-run
comparison (no automatic diffing across processes).

    PYTHONPATH="$(pwd):$(pwd)/scripts" python scripts/rl/diagnostics.py command-gait-comparison \
        runs/<run>/checkpoints/checkpoint.pt 0.7 0.1 0.1 0.1 --seed 0 --steps 200 \
        --command -0.25 0.0 0.0 --progress_std 0.5 --balance_tilt_coef 1.0 --target_height 0.42 \
        --balance_hip_activation_coef 0.02
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from talon_rl.config import (
    ActionSpaceCfg, ExtrinsicsCfg, ObservationSpaceCfg, ObservationStackCfg, PreferenceCfg, RewardVectorCfg,
)
from talon_rl.reward import compute_reward_vector

from rl.core.algorithms import MOPPOConfig, MOPPOTrainer

CONTACT_THRESHOLD_N = 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("w", type=float, nargs=4)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--window", type=int, nargs=2, default=(50, 200), metavar=("LO", "HI"), help="Aggregation window, same convention as attractor_phase_metrics.py's Phase C")
    parser.add_argument("--command", type=float, nargs=3, required=True, metavar=("VX", "VY", "WZ"))
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
    print(f"loaded checkpoint (t={trainer._t}), env seed={args.seed}, command={args.command}")

    trainer.w = np.tile(np.array(args.w, dtype=np.float32) / sum(args.w), (args.num_envs, 1)).astype(np.float32)

    action_term = env.action_manager._terms["joint_pos"]
    joint_names = list(action_term._joint_names)
    action_scale = float(action_term.cfg.scale)
    default_joint_pos = action_term._offset[0].cpu().numpy()

    def side_idx(joint_type: str, side: str) -> list[int]:
        legs = ("FL", "RL") if side == "L" else ("FR", "RR")
        return [i for i, n in enumerate(joint_names) if joint_type in n and any(n.startswith(leg) for leg in legs)]

    joint_types = ("hip", "thigh", "calf")
    idx = {(jt, side): side_idx(jt, side) for jt in joint_types for side in ("L", "R")}
    default_by_group = {
        k: float(default_joint_pos[v].mean()) for k, v in idx.items()
    }

    foot_body_ids = env._foot_body_ids
    foot_names = [env.scene["robot"].body_names[i] for i in foot_body_ids]
    left_feet = [i for i, n in enumerate(foot_names) if n.startswith("FL") or n.startswith("RL")]
    right_feet = [i for i, n in enumerate(foot_names) if n.startswith("FR") or n.startswith("RR")]

    progress_idx = reward_cfg.term_names.index("progress")
    balance_idx = reward_cfg.term_names.index("balance")

    T, N = args.steps, args.num_envs
    q_target = {k: np.zeros((T, N), dtype=np.float32) for k in idx}
    v_x = np.zeros((T, N), dtype=np.float32)
    v_z = np.zeros((T, N), dtype=np.float32)
    height = np.zeros((T, N), dtype=np.float32)
    r_progress = np.zeros((T, N), dtype=np.float32)
    r_balance = np.zeros((T, N), dtype=np.float32)
    contact_L = np.zeros((T, N), dtype=np.float32)
    contact_R = np.zeros((T, N), dtype=np.float32)
    terminal_fall = np.zeros((T, N), dtype=bool)

    trainer.model.eval()
    for t in range(T):
        env.v_command_buf[:] = torch.tensor(args.command, device=env.device)
        action = trainer.act_inference()
        transition, done = env.step(action)
        trainer._last_extrinsics = transition.get("extrinsics") if trainer.encoder else None
        trainer.push_obs(transition["obs"], done_mask=done)

        target = action * action_scale + default_joint_pos[None, :]
        for k, v in idx.items():
            q_target[k][t] = target[:, v].mean(axis=-1)

        v_x[t] = transition["v_actual"][:, 0]
        v_z[t] = transition["v_z"]
        height[t] = transition["height"]
        terminal_fall[t] = transition.get("terminal_fall", done).astype(bool)

        contact = transition["foot_contact_force"] > CONTACT_THRESHOLD_N
        contact_L[t] = contact[:, left_feet].mean(axis=-1)
        contact_R[t] = contact[:, right_feet].mean(axis=-1)

        reward_vec = compute_reward_vector(transition, reward_cfg)
        r_progress[t] = reward_vec[:, progress_idx]
        r_balance[t] = reward_vec[:, balance_idx]

    lo, hi = args.window
    mask = ~terminal_fall[lo:hi]

    def m(arr: np.ndarray) -> float:
        return float(arr[lo:hi][mask].mean())

    def s(arr: np.ndarray) -> float:
        return float(arr[lo:hi][mask].std())

    print(f"\n=== command={args.command}, window=[{lo},{hi}), {N} lanes ===")
    print(f"  mean v_x (actual)         {m(v_x):.4f}")
    print(f"  mean v_z                  {m(v_z):.4f}")
    print(f"  mean height               {m(height):.4f}")
    print(f"  mean progress_reward      {m(r_progress):.4f}")
    print(f"  mean balance_reward       {m(r_balance):.4f}")
    print(f"  mean contact_L / contact_R {m(contact_L):.4f} / {m(contact_R):.4f}")
    print("\n  q_target per (joint_type, side) -- mean (std), default in [brackets]:")
    for jt in joint_types:
        vals = []
        for side in ("L", "R"):
            k = (jt, side)
            vals.append(f"{side}={m(q_target[k]):+.4f} ({s(q_target[k]):.4f}) [default={default_by_group[k]:+.4f}]")
        print(f"    {jt:<6}: " + "  ".join(vals))


if __name__ == "__main__":
    main()
