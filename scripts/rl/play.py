#!/usr/bin/env python3
"""Play/export entry point — loads a trained checkpoint, runs deterministic
inference, and optionally exports the policy for deployment/sim2sim.

    python scripts/rl/play.py --checkpoint <path> --steps 200
    python scripts/rl/play.py --load_run last --logs_root logs/talon_rl --export policy.pt
    python scripts/rl/play.py --checkpoint <path> --analyze joint_vel joint_torque --plot

Uses MOPPOTrainer.act_inference() (deterministic mean, no sampling) rather
than the stochastic action update() uses during training — see
core/algorithms/moppo.py and core/modules/actor_critic.py.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from talon_rl.config import ActionSpaceCfg, ObservationSpaceCfg, ObservationStackCfg, PreferenceCfg, RewardVectorCfg
from talon_rl.reward import compute_reward_vector

from rl.core.algorithms import MOPPOConfig, MOPPOTrainer
from rl.core.analyzer import Analyzer
from rl.core.dummy_env import DummyTalonEnv
from rl.core.run_dir import resolve_checkpoint
from rl.core.wrapper import export_policy_as_jit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None, help="Direct path to a checkpoint. Mutually exclusive with --load_run.")
    parser.add_argument("--load_run", type=str, default=None, help="Resolve <logs_root>/<run>/checkpoint.pt — <run> can be an exact run name or \"last\".")
    parser.add_argument("--logs_root", type=str, default="logs/talon_rl", help="Root directory --load_run resolves against.")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env", choices=["dummy", "isaac_lab"], default="dummy")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--num_policy_stacks", type=int, default=1, help="Must match the value used when the checkpoint was trained.")
    parser.add_argument("--num_critic_stacks", type=int, default=1, help="Must match the value used when the checkpoint was trained.")
    parser.add_argument("--export", type=str, default=None, help="Export the loaded policy as TorchScript to this path.")
    parser.add_argument("--analyze", type=str, nargs="+", default=None, help="Transition-dict keys to record per step (e.g. joint_vel joint_torque).")
    parser.add_argument("--plot", action="store_true", help="With --analyze: save one PNG per recorded key next to the checkpoint (or ./exported/ for --checkpoint).")
    args = parser.parse_args()

    if bool(args.checkpoint) == bool(args.load_run):
        raise SystemExit("pass exactly one of --checkpoint or --load_run")
    checkpoint_path = args.checkpoint or resolve_checkpoint(args.logs_root, args.load_run)

    obs_cfg = ObservationSpaceCfg()
    action_cfg = ActionSpaceCfg()
    reward_cfg = RewardVectorCfg()
    pref_cfg = PreferenceCfg()
    stack_cfg = ObservationStackCfg(num_policy_stacks=args.num_policy_stacks, num_critic_stacks=args.num_critic_stacks)

    if args.env == "dummy":
        env = DummyTalonEnv(obs_cfg, action_cfg, num_envs=args.num_envs, horizon=200, seed=args.seed)
    else:
        # Same Isaac Sim launch-ordering requirement as train_prelim.py's
        # --env isaac_lab branch — SimulationApp must exist before any
        # isaaclab-touching import (see that file's comment for why). `os`
        # is already imported at module scope — no local re-import (a local
        # `import os` anywhere in this function would make `os` local to
        # the whole function body, breaking every `os.path`/`os.environ`
        # use above with UnboundLocalError — see train_prelim.py's own note).
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
        from isaacsim import SimulationApp
        simulation_app = SimulationApp({"headless": True})  # noqa: F841 — kept alive for the process lifetime

        import gymnasium as gym
        import talon_rl.tasks.locomotion.a1_env  # noqa: F401 — registers Isaac-Talon-A1-v0
        from talon_rl.tasks.locomotion.a1_env.a1_env_cfg import IsaacLabTalonEnvCfg

        cfg = IsaacLabTalonEnvCfg()
        cfg.scene.num_envs = args.num_envs
        env = gym.make("Isaac-Talon-A1-v0", cfg=cfg).unwrapped

    trainer = MOPPOTrainer(env, obs_cfg, reward_cfg, pref_cfg, moppo_cfg=MOPPOConfig(), stack_cfg=stack_cfg, seed=args.seed)
    trainer.load(checkpoint_path)
    print(f"loaded checkpoint from {checkpoint_path} (t={trainer._t})")

    if args.export:
        export_policy_as_jit(trainer.model, args.export)
        print(f"exported policy to {args.export}")

    analyzer = Analyzer(args.analyze) if args.analyze else None

    trainer.model.eval()
    total_reward = np.zeros(reward_cfg.dim, dtype=np.float32)
    for _ in range(args.steps):
        action = trainer.act_inference()
        transition, done = env.step(action)
        trainer.stack.push(transition["obs"], done_mask=done)
        total_reward += compute_reward_vector(transition, reward_cfg).mean(axis=0)
        if analyzer is not None:
            analyzer.record(transition)

    mean_reward = total_reward / args.steps
    r = ", ".join(f"{n}={v:+.3f}" for n, v in zip(reward_cfg.term_names, mean_reward))
    print(f"mean reward over {args.steps} steps: {r}")

    if analyzer is not None and args.plot:
        plot_dir = os.path.join(os.path.dirname(checkpoint_path), "exported")
        analyzer.save_plots(plot_dir)
        print(f"saved analysis plots to {plot_dir}")

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
