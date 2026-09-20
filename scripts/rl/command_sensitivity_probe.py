#!/usr/bin/env python3
"""Experiment 2C.2 -- command sensitivity / observation pipeline audit.
command_gait_comparison.py found joint targets essentially unchanged
across command=+0.25/0/-0.25 in all 3 seeds (Hypothesis A: the policy
doesn't condition its gait on v_command at all). Before concluding this
is a LEARNED command-insensitivity (vs an observation-pipeline
construction bug -- wrong index, missing scale, normalization
collapsing the signal), this does two things:

  Step A (static audit, printed once): confirms raw command -> final
  policy-observation-feature mapping is correct. v_command enters the
  observation raw (a1_env_cfg.py's PolicyCfg: `v_command = ObsTerm(func=
  mdp.v_command)`, no ObsTerm scale -- comment there already notes vx is
  "already O(1): [-0.3, 1.0]"), at a fixed index (42:45 in the 51-dim
  raw policy obs -- joint_pos(12)+joint_vel(12)+roll_pitch(2)+
  foot_contact(4)+last_action(12)=42, matching config.py's
  ObservationSpaceCfg field order exactly, and the same order
  a1_env_cfg.py's PolicyCfg ObsTerm list uses), then goes through
  MOPPOTrainer.obs_norm (RunningMeanStd, center=True: (x-mean)/sqrt(var
  +1e-8), clipped to +/-10) -- the SAME per-dimension running stats the
  checkpoint was trained with (restored by trainer.load()). Prints
  raw command -> normalized obs feature for both, confirming sign and
  separation survive the pipeline.

  Step B (single-step action sensitivity, the decisive test): captures
  REAL policy_obs snapshots from a short warmup rollout (so joint_pos/
  joint_vel/roll_pitch/contact/last_action/base_ang_vel/
  projected_gravity all reflect an actual in-episode state, not a reset
  pose), then for each snapshot, overwrites ONLY the command slice
  (rest of the 51-dim observation, plus z_t/w, held fixed) with the
  normalized equivalent of +0.25/0/-0.25 m/s and calls the actor
  directly (MOPPOTrainer._act_inference on a hand-built actor_obs,
  bypassing _actor_obs()/the live stack entirely -- no env step, no
  obs_norm.update(), so this can't corrupt trainer state or be confused
  with an actual rollout). Reports mean/max |action_pos - action_neg|
  and the equivalent |q_target| difference, averaged over
  `--num_snapshots` warmup steps.

If Step A shows raw command mapping to obs feature incorrectly (wrong
sign, collapsed variance, wrong index) -- an implementation bug, fix
that before anything else. If Step A is correct but Step B shows
near-zero action difference despite a clean, well-separated command
input -- learned command insensitivity, not a pipeline bug, consistent
with command_gait_comparison.py's rollout-level finding.

    PYTHONPATH="$(pwd):$(pwd)/scripts" python scripts/rl/command_sensitivity_probe.py \
        runs/<run>/checkpoints/checkpoint.pt 0.7 0.1 0.1 0.1 --seed 0 --num_snapshots 30 \
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

# Offset of v_command_x within the 51-dim raw policy obs -- see module
# docstring for the field-order derivation. Not re-derived from
# ObservationSpaceCfg programmatically here (would need to trust field
# declaration order matches PolicyCfg's ObsTerm order, which is exactly
# the thing being audited) -- hardcoded and printed for a human to verify
# against the "Observation Manager" table Isaac Lab itself prints at env
# construction (every run's own log already shows this).
V_COMMAND_X_OFFSET = 42


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("w", type=float, nargs=4)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=30, help="Real rollout steps before snapshotting starts")
    parser.add_argument("--num_snapshots", type=int, default=20, help="How many post-warmup steps to snapshot and probe")
    parser.add_argument("--command", type=float, nargs=3, default=(0.5, 0.0, 0.0), metavar=("VX", "VY", "WZ"), help="Real command driving the warmup rollout (irrelevant to the probe itself)")
    parser.add_argument("--test_vx", type=float, nargs=3, default=(0.25, 0.0, -0.25), metavar=("POS", "ZERO", "NEG"))
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
    print(f"loaded checkpoint (t={trainer._t}), env seed={args.seed}")

    trainer.w = np.tile(np.array(args.w, dtype=np.float32) / sum(args.w), (args.num_envs, 1)).astype(np.float32)

    action_term = env.action_manager._terms["joint_pos"]
    action_scale = float(action_term.cfg.scale)

    # --- Step A: static observation-pipeline audit ---
    print("\n=== Step A: raw command -> normalized policy-obs feature ===")
    mean_cmd = float(trainer.obs_norm.mean[V_COMMAND_X_OFFSET])
    std_cmd = float(np.sqrt(trainer.obs_norm.var[V_COMMAND_X_OFFSET] + 1e-8))
    print(f"  obs_norm at v_command_x (offset {V_COMMAND_X_OFFSET}): mean={mean_cmd:.4f} std={std_cmd:.4f}")
    for raw_vx in args.test_vx:
        normed = np.clip((raw_vx - mean_cmd) / std_cmd, -10.0, 10.0)
        print(f"  raw v_x={raw_vx:+.2f}  ->  normalized obs feature={normed:+.4f}")

    # --- warmup: real rollout so captured obs reflect an actual in-episode state ---
    trainer.model.eval()
    for _ in range(args.warmup_steps):
        env.v_command_buf[:] = torch.tensor(args.command, device=env.device)
        action = trainer.act_inference()
        transition, done = env.step(action)
        trainer._last_extrinsics = transition.get("extrinsics") if trainer.encoder else None
        trainer.push_obs(transition["obs"], done_mask=done)

    # --- Step B: single-step action sensitivity across num_snapshots real states ---
    print(f"\n=== Step B: action sensitivity to command, {args.num_snapshots} snapshots x {args.num_envs} lanes ===")
    all_diff_pos_neg = []
    all_diff_pos_zero = []
    all_diff_zero_neg = []
    all_qdiff_pos_neg = []

    def build_actor_obs(policy_obs_variant: np.ndarray) -> torch.Tensor:
        parts = [policy_obs_variant]
        if trainer.encoder:
            e_t = trainer.extrinsics_norm.normalize(trainer._last_extrinsics, center=True)
            z_t = trainer.encoder(torch.from_numpy(e_t).float().to(trainer.device)).detach().cpu().numpy()
            parts.append(z_t)
        parts.append(trainer.w)
        actor_obs = np.concatenate(parts, axis=-1).astype(np.float32)
        return torch.from_numpy(actor_obs).to(trainer.device)

    for step in range(args.num_snapshots):
        # Advance the REAL rollout one more step (so each snapshot is a
        # genuinely different, realistic in-episode state) -- normal
        # push_obs updates trainer.obs_norm/stack exactly as any eval
        # script would; the probe below reads a COPY, never mutates them.
        env.v_command_buf[:] = torch.tensor(args.command, device=env.device)
        action = trainer.act_inference()
        transition, done = env.step(action)
        trainer._last_extrinsics = transition.get("extrinsics") if trainer.encoder else None
        trainer.push_obs(transition["obs"], done_mask=done)

        base_policy_obs = trainer.stack.policy_obs.copy()  # (N, policy_obs_dim), already normalized

        actions_by_cmd = {}
        for raw_vx in args.test_vx:
            normed = float(np.clip((raw_vx - mean_cmd) / std_cmd, -10.0, 10.0))
            variant = base_policy_obs.copy()
            # Last stacked frame's command slice -- with num_policy_stacks=1
            # (this repo's default) that's simply [:, V_COMMAND_X_OFFSET],
            # generalized here to the stack's last frame regardless of
            # stack depth. vy/omega_z left at their real (already-near-0
            # under the warmup's own command) normalized values, not
            # forced -- only v_x is the variable under test.
            per_frame_dim = env.obs_dim
            last_frame_start = variant.shape[-1] - per_frame_dim
            variant[:, last_frame_start + V_COMMAND_X_OFFSET] = normed
            with torch.no_grad():
                actor_obs = build_actor_obs(variant)
                action_t = trainer._act_inference(actor_obs)
            actions_by_cmd[raw_vx] = action_t.cpu().numpy()

        pos, zero, neg = args.test_vx
        diff_pos_neg = np.abs(actions_by_cmd[pos] - actions_by_cmd[neg])
        diff_pos_zero = np.abs(actions_by_cmd[pos] - actions_by_cmd[zero])
        diff_zero_neg = np.abs(actions_by_cmd[zero] - actions_by_cmd[neg])
        all_diff_pos_neg.append(diff_pos_neg)
        all_diff_pos_zero.append(diff_pos_zero)
        all_diff_zero_neg.append(diff_zero_neg)
        all_qdiff_pos_neg.append(diff_pos_neg * action_scale)  # action delta -> q_target delta is exact (linear map)

    diff_pos_neg = np.concatenate(all_diff_pos_neg, axis=0)
    diff_pos_zero = np.concatenate(all_diff_pos_zero, axis=0)
    diff_zero_neg = np.concatenate(all_diff_zero_neg, axis=0)
    qdiff_pos_neg = np.concatenate(all_qdiff_pos_neg, axis=0)

    print(f"  |action(+{pos}) - action({zero})|   mean={diff_pos_zero.mean():.5f}  max={diff_pos_zero.max():.5f}")
    print(f"  |action({zero}) - action({neg})|   mean={diff_zero_neg.mean():.5f}  max={diff_zero_neg.max():.5f}")
    print(f"  |action(+{pos}) - action({neg})|   mean={diff_pos_neg.mean():.5f}  max={diff_pos_neg.max():.5f}")
    print(f"  |q_target(+{pos}) - q_target({neg})| mean={qdiff_pos_neg.mean():.5f}  max={qdiff_pos_neg.max():.5f}  (rad)")
    print(f"\n  for reference, ACTION_CLIP={trainer.model.ACTION_CLIP} -- compare the above against that scale")


if __name__ == "__main__":
    main()
