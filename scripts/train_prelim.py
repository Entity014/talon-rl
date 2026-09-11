#!/usr/bin/env python3
"""Prelim entry point — runs MOPPO on DummyTalonEnv and prints per-update stats.

    python scripts/train_prelim.py --updates 50

This exists to eyeball whether the reward-vector terms respond sensibly to
different regions of the preference simplex, NOT to produce a trained policy
worth keeping. Swap `DummyTalonEnv` for the real Isaac Lab env (once written)
to get an actual Phase 1 prelim result — see envs/base_env.py.
"""

from __future__ import annotations

import argparse

from talon_rl.config import ActionSpaceCfg, ObservationSpaceCfg, PreferenceCfg, RewardVectorCfg
from talon_rl.envs.dummy_env import DummyTalonEnv
from talon_rl.training.moppo import MOPPOConfig, MOPPOTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    obs_cfg = ObservationSpaceCfg()
    action_cfg = ActionSpaceCfg()
    reward_cfg = RewardVectorCfg()
    pref_cfg = PreferenceCfg()

    env = DummyTalonEnv(obs_cfg, action_cfg, horizon=200, seed=args.seed)
    trainer = MOPPOTrainer(env, obs_cfg, reward_cfg, pref_cfg, moppo_cfg=MOPPOConfig(), seed=args.seed)

    print(f"reward terms: {reward_cfg.term_names}")
    for i in range(1, args.updates + 1):
        stats = trainer.update()
        r = ", ".join(f"{n}={v:+.3f}" for n, v in zip(reward_cfg.term_names, stats["mean_reward_vec"]))
        print(
            f"update {i:3d} | policy_loss={stats['policy_loss']:+.4f} "
            f"value_loss={stats['value_loss']:.4f} ep_len={stats['mean_episode_len']:.1f} | {r}"
        )


if __name__ == "__main__":
    main()
