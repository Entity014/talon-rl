#!/usr/bin/env python3
"""Experiment 2A.2's companion diagnostic -- quantifies the pre-fall
progress_reward leak fall_cycle_analysis.py found (v_x, and therefore
progress_reward, consistently RISES in the last few steps before a fall,
across all 3 seeds tested) without touching the reward formula or
retraining anything. The exact-terminal-fall-step zeroing added
2026-09-19 (talon_rl/reward.py's progress_reward, `terminal_fall`
param) only closes the single-step case; the 1-2-3-step lead-up case
was explicitly left open at the time (see that commit / the
2026-09-19 daily note) pending evidence the leak actually mattered
enough to justify a harder, non-causal rollout-buffer fix.

Recomputes the SAME rollout's progress reward under several
counterfactual zero-windows (K steps immediately before AND including
each terminal_fall step get zeroed, K=1 matches today's shipped
behavior exactly) and reports:

  - total/mean progress reward per lane under each K -- how much of
    the reward magnitude the leak window contributes in aggregate.
  - Spearman rank correlation between each lane's total reward at K=1
    (today's shipped formula) vs larger K -- the real question isn't
    "does the number go down" (it must, trivially, as more reward gets
    zeroed) but "does an EPISODE THAT LOOKED GOOD under the leaky
    formula still look good once the leak is closed". A correlation
    near 1.0 means the leak is a roughly uniform tax that doesn't
    change which trajectories the training signal favors -- an
    artifact, not a behavior-shaping bug. A correlation well below 1.0
    means closing the leak would meaningfully re-rank which behavior
    gets reinforced, i.e. the leak is large enough to matter for
    training, not just an end-of-episode rounding error.
  - (2026-09-20, Experiment 2A.3) consecutive-K rank correlation and
    marginal reward removed, K vs K+1 across a finer grid -- locates the
    minimal effective leak window: the smallest K past which extending
    the zero-window further barely changes the ranking or removes much
    more reward. That K, not an arbitrarily round number, is what should
    actually be zeroed if this leak gets fixed and retrained.

Uses the SAME v_actual/v_command/terminal_fall data reward.py's real
progress_reward reads -- this is the exact formula (exp-kernel,
v_x-only, 2026-09-19's fix), just re-run with a wider zero-window, not
a new formula.

    PYTHONPATH="$(pwd):$(pwd)/scripts" python scripts/analysis/run_diagnostics.py progress-leak-counterfactual \
        runs/<run>/checkpoints/checkpoint.pt 0.7 0.1 0.1 0.1 --seed 0 --steps 200 \
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


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank-correlation without a scipy dependency -- argsort-of-argsort
    gives ranks, then Pearson correlation on ranks is Spearman's rho."""
    def rank(x: np.ndarray) -> np.ndarray:
        return np.argsort(np.argsort(x))
    ra, rb = rank(a).astype(np.float64), rank(b).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("w", type=float, nargs=4)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--command", type=float, nargs=3, default=(0.5, 0.0, 0.0), metavar=("VX", "VY", "WZ"))
    parser.add_argument("--seed", type=int, default=0, help="Sets cfg.seed BEFORE env creation (not just trainer.rng)")
    parser.add_argument(
        "--leak_windows", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20],
        help="K values to compare (K=0: no zeroing at all, the pre-2026-09-19 formula; K=1: today's shipped fix). "
             "Fine-grained near the small end to locate the minimal-effective-window knee (Experiment 2A.3).",
    )
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
    cfg.seed = args.seed
    env = gym.make("Isaac-Talon-A1-v0", cfg=cfg, render_mode=None).unwrapped

    trainer = MOPPOTrainer(
        env, obs_cfg, reward_cfg, pref_cfg, moppo_cfg=MOPPOConfig(), stack_cfg=stack_cfg,
        extrinsics_cfg=extrinsics_cfg, seed=args.seed,
    )
    trainer.load(args.checkpoint)
    print(f"loaded checkpoint (t={trainer._t}), env seed={args.seed}")

    trainer.w = np.tile(np.array(args.w, dtype=np.float32) / sum(args.w), (args.num_envs, 1)).astype(np.float32)

    T, N = args.steps, args.num_envs
    v_x = np.zeros((T, N), dtype=np.float32)
    terminal_fall = np.zeros((T, N), dtype=bool)
    std = reward_cfg.progress_std

    trainer.model.eval()
    for t in range(T):
        env.v_command_buf[:] = torch.tensor(args.command, device=env.device)
        action = trainer.act_inference()
        transition, done = env.step(action)
        trainer._last_extrinsics = transition.get("extrinsics") if trainer.encoder else None
        trainer.push_obs(transition["obs"], done_mask=done)

        v_x[t] = transition["v_actual"][:, 0]
        terminal_fall[t] = transition.get("terminal_fall", done).astype(bool)

    # Raw exp-kernel reward, v_x-only (matches reward.py's progress_reward
    # exactly, minus foot_air_time_reward -- not tracked here since it's
    # additive and orthogonal to the fall-proximity question).
    err = (v_x - args.command[0]) ** 2
    raw_reward = np.exp(-err / (std ** 2)).astype(np.float32)

    def zeroed_reward(k: int) -> np.ndarray:
        if k <= 0:
            return raw_reward.copy()
        mask = np.zeros_like(terminal_fall)
        for t in range(T):
            if terminal_fall[t].any():
                lo = max(0, t - (k - 1))
                mask[lo:t + 1, terminal_fall[t]] = True
        return np.where(mask, 0.0, raw_reward)

    totals = {k: zeroed_reward(k).sum(axis=0) for k in args.leak_windows}  # (N,) per lane

    print(f"\nper-lane total progress reward over {T} steps, {N} lanes:")
    for k in args.leak_windows:
        t = totals[k]
        print(f"  K={k:<3} (zero last {k} steps before+incl. each fall): mean={t.mean():.4f} std={t.std():.4f}")

    if 1 in totals:
        print("\nSpearman rank correlation of per-lane totals, K=1 (shipped) vs other K:")
        for k in args.leak_windows:
            if k == 1:
                continue
            rho = _spearman(totals[1], totals[k])
            print(f"  K=1 vs K={k:<3}: rho={rho:.4f}")
    else:
        print("\n(K=1 not in --leak_windows, skipping rank-correlation-vs-shipped-formula section)")

    # Minimal-effective-window knee (Experiment 2A.3): correlation between
    # CONSECUTIVE K values, not just vs K=1 -- finds where extending the
    # zero-window further stops changing the ranking (rho_consecutive -> 1),
    # i.e. the smallest K that already contains most of the leak's effect
    # on which lanes look good. The K=1-vs-K table above shows how far the
    # ranking has drifted from today's shipped formula; this one shows
    # where that drift stops accumulating.
    sorted_ks = sorted(args.leak_windows)
    print("\nconsecutive-K rank correlation (locates the minimal-effective-window knee):")
    for k_lo, k_hi in zip(sorted_ks, sorted_ks[1:]):
        rho = _spearman(totals[k_lo], totals[k_hi])
        marginal_reward = totals[k_lo].mean() - totals[k_hi].mean()
        print(f"  K={k_lo:<3} vs K={k_hi:<3}: rho={rho:.4f}  marginal reward removed={marginal_reward:.4f}")


if __name__ == "__main__":
    main()
