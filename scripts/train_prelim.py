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

from moppo.dummy_env import DummyTalonEnv
from moppo.moppo import MOPPOConfig, MOPPOTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env", choices=["dummy", "isaac_lab"], default="dummy")
    parser.add_argument("--num_envs", type=int, default=64)  # CPU-sane default for --env dummy; pass --num_envs 4096 explicitly for --env isaac_lab
    args = parser.parse_args()

    obs_cfg = ObservationSpaceCfg()
    action_cfg = ActionSpaceCfg()
    reward_cfg = RewardVectorCfg()
    pref_cfg = PreferenceCfg()

    if args.env == "dummy":
        env = DummyTalonEnv(obs_cfg, action_cfg, num_envs=args.num_envs, horizon=200, seed=args.seed)
    else:
        # Imported lazily so --env dummy keeps working on machines without
        # Isaac Sim installed (this repo's default 3.12 .venv included).
        #
        # `carb` (and everything else isaaclab imports transitively) is only
        # importable once Isaac Sim's Kit runtime is actually running --
        # SimulationApp must be constructed before the first isaaclab-touching
        # import, same as tests/test_a1_env.py does. Found 2026-09-14 while
        # running this branch for real for the first time (Task 9): without
        # this, `import talon_rl.tasks.locomotion.a1_env` below raises
        # ModuleNotFoundError: No module named 'carb'.
        import os
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
        from isaacsim import SimulationApp
        simulation_app = SimulationApp({"headless": True})  # noqa: F841 — kept alive for the process lifetime

        import gymnasium as gym
        import talon_rl.tasks.locomotion.a1_env  # noqa: F401 — registers Isaac-Talon-A1-v0
        from talon_rl.tasks.locomotion.a1_env.a1_env_cfg import IsaacLabTalonEnvCfg

        cfg = IsaacLabTalonEnvCfg()
        cfg.scene.num_envs = args.num_envs
        env = gym.make("Isaac-Talon-A1-v0", cfg=cfg).unwrapped

    trainer = MOPPOTrainer(env, obs_cfg, reward_cfg, pref_cfg, moppo_cfg=MOPPOConfig(), seed=args.seed)

    print(f"reward terms: {reward_cfg.term_names}")
    for i in range(1, args.updates + 1):
        stats = trainer.update()
        r = ", ".join(f"{n}={v:+.3f}" for n, v in zip(reward_cfg.term_names, stats["mean_reward_vec"]))
        print(
            f"update {i:3d} | policy_loss={stats['policy_loss']:+.4f} "
            f"value_loss={stats['value_loss']:.4f} ep_len={stats['mean_episode_len']:.1f} | {r}"
        )

    if args.env == "isaac_lab":
        import threading
        env.close()
        # simulation_app.close() (raw Kit runtime teardown after a GPU-pipeline
        # scene has been stepped) is the call known to hang on this machine —
        # env.close() above is fast/lightweight and doesn't need the watchdog.
        watchdog = threading.Timer(15.0, lambda: __import__("os")._exit(0))
        watchdog.daemon = True
        watchdog.start()
        simulation_app.close()
        watchdog.cancel()


if __name__ == "__main__":
    main()
