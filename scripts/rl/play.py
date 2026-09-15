#!/usr/bin/env python3
"""Play/export entry point — loads a trained checkpoint, runs deterministic
inference, and optionally exports the policy for deployment/sim2sim.

    python scripts/rl/play.py --checkpoint <path> --steps 200
    python scripts/rl/play.py --checkpoint <path> --export policy.pt

Uses MOPPOTrainer.act_inference() (deterministic mean, no sampling) rather
than the stochastic action update() uses during training — see
core/algorithms/moppo.py and core/modules/actor_critic.py.
"""

from __future__ import annotations

import argparse

import numpy as np

from talon_rl.config import ActionSpaceCfg, ObservationSpaceCfg, PreferenceCfg, RewardVectorCfg
from talon_rl.reward import compute_reward_vector

from rl.core.algorithms import MOPPOConfig, MOPPOTrainer
from rl.core.dummy_env import DummyTalonEnv
from rl.core.wrapper import export_policy_as_jit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env", choices=["dummy", "isaac_lab"], default="dummy")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--export", type=str, default=None, help="Export the loaded policy as TorchScript to this path.")
    args = parser.parse_args()

    obs_cfg = ObservationSpaceCfg()
    action_cfg = ActionSpaceCfg()
    reward_cfg = RewardVectorCfg()
    pref_cfg = PreferenceCfg()

    if args.env == "dummy":
        env = DummyTalonEnv(obs_cfg, action_cfg, num_envs=args.num_envs, horizon=200, seed=args.seed)
    else:
        # Same Isaac Sim launch-ordering requirement as train_prelim.py's
        # --env isaac_lab branch — SimulationApp must exist before any
        # isaaclab-touching import (see that file's comment for why).
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
    trainer.load(args.checkpoint)
    print(f"loaded checkpoint from {args.checkpoint} (t={trainer._t})")

    if args.export:
        export_policy_as_jit(trainer.model, args.export)
        print(f"exported policy to {args.export}")

    trainer.model.eval()
    total_reward = np.zeros(reward_cfg.dim, dtype=np.float32)
    for _ in range(args.steps):
        action = trainer.act_inference()
        transition, done = env.step(action)
        trainer.stack.push(transition["obs"], done_mask=done)
        total_reward += compute_reward_vector(transition, reward_cfg).mean(axis=0)

    mean_reward = total_reward / args.steps
    r = ", ".join(f"{n}={v:+.3f}" for n, v in zip(reward_cfg.term_names, mean_reward))
    print(f"mean reward over {args.steps} steps: {r}")

    if args.env == "isaac_lab":
        import threading
        env.close()
        watchdog = threading.Timer(15.0, lambda: __import__("os")._exit(0))
        watchdog.daemon = True
        watchdog.start()
        simulation_app.close()
        watchdog.cancel()


if __name__ == "__main__":
    main()
