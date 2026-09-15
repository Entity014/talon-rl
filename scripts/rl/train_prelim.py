#!/usr/bin/env python3
"""Prelim entry point — runs MOPPO on DummyTalonEnv and prints per-update stats.

    python scripts/rl/train_prelim.py --updates 50
    python scripts/rl/train_prelim.py --updates 50 --logs_root logs/talon_rl --run_name exp1
    python scripts/rl/train_prelim.py --updates 50 --log_dir runs/exp1 --save_path runs/exp1/ckpt.pt

This exists to eyeball whether the reward-vector terms respond sensibly to
different regions of the preference simplex, NOT to produce a trained policy
worth keeping. Swap `DummyTalonEnv` for the real Isaac Lab env (once written)
to get an actual Phase 1 prelim result — see envs/base_env.py.
"""

from __future__ import annotations

import argparse
import os

from talon_rl.config import ActionSpaceCfg, ObservationSpaceCfg, ObservationStackCfg, PreferenceCfg, RewardVectorCfg

from rl.core.dummy_env import DummyTalonEnv
from rl.core.run_dir import dump_config, make_run_dir

# MOPPOConfig/MOPPOTrainer (rl.core.algorithms.moppo) import torch at module
# scope, and torch touching CUDA before Isaac Sim's SimulationApp owns its
# own CUDA context causes PhysX's GPU pipeline to silently die ~15s into
# scene setup (found 2026-09-15 running --env isaac_lab for real for the
# first time: tests/test_a1_env.py never imports torch, which is why that
# test didn't hit this). So for --env isaac_lab, SimulationApp must be
# constructed before this import — deferred into main() below instead of a
# module-scope import.


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env", choices=["dummy", "isaac_lab"], default="dummy")
    parser.add_argument("--num_envs", type=int, default=64)  # CPU-sane default for --env dummy; pass --num_envs 4096 explicitly for --env isaac_lab
    parser.add_argument("--num_policy_stacks", type=int, default=1, help="History frames the actor sees (Flamingo-style stacking, see obs_stack.py).")
    parser.add_argument("--num_critic_stacks", type=int, default=1, help="History frames the critic sees — can differ from --num_policy_stacks.")
    parser.add_argument("--save_path", type=str, default=None, help="Save a checkpoint here when training finishes (ignored if --logs_root is set).")
    parser.add_argument("--save_every", type=int, default=0, help="Also save a checkpoint every N updates (0 = only at the end) — cheap insurance for a long run that a mid-run OOM/crash doesn't lose everything. Requires --save_path or --logs_root.")
    parser.add_argument("--resume", type=str, default=None, help="Load a checkpoint from this path before training starts.")
    parser.add_argument("--log_dir", type=str, default=None, help="Log per-update scalars to this dir via TensorBoard (ignored if --logs_root is set).")
    parser.add_argument("--logs_root", type=str, default=None, help="Enable run-directory management: creates <logs_root>/<run_name or timestamp>/, dumps config.yaml, logs to its tensorboard/ subdir, and saves checkpoint.pt there — supersedes --log_dir/--save_path when set.")
    parser.add_argument("--run_name", type=str, default=None, help="Run directory name under --logs_root (default: a timestamp).")

    # Peek at --env before the real parse: AppLauncher.add_app_launcher_args
    # needs `isaaclab` importable, which isn't installed in this repo's
    # default --env dummy venv, and it must register its own flags (e.g.
    # --headless) on THIS parser before the real parse_args() below.
    known_args, _ = parser.parse_known_args()
    if known_args.env == "isaac_lab":
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    simulation_app = None
    if args.env == "isaac_lab":
        # Must happen before the torch-importing `moppo` import right below
        # (see module docstring comment above) — and before any other
        # isaaclab-touching import, since `carb` etc. aren't importable
        # until Kit's runtime is actually running.
        #
        # Uses the official AppLauncher (isaaclab.app), not a bare
        # `SimulationApp({"headless": True})` — found 2026-09-15: bare
        # SimulationApp deterministically fails scene construction (PhysX
        # GPU pipeline dies silently, no traceback) when this script is run
        # as `python train_prelim.py` directly, for reasons never fully
        # root-caused despite systematic bisection (ruled out: num_envs
        # scale, torch-import ordering, GPU resource leaks). AppLauncher is
        # what every real Isaac Lab training script (including the
        # jaykorea/Isaac-RL-Two-wheel-Legged-Bot reference project's own
        # scripts/co_rl/train.py) actually uses — it does extra Kit
        # extension/experience-file setup that bare SimulationApp skips,
        # and switching to it fixed the same construction path (verified up
        # to PhysX scene creation; a concurrent run's VRAM usage was the
        # only failure seen after switching, not the earlier silent death).
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
        args.headless = True  # this repo always runs headless; don't require users to remember the flag
        from isaaclab.app import AppLauncher
        app_launcher = AppLauncher(args)
        simulation_app = app_launcher.app

    from rl.core.algorithms.moppo import MOPPOConfig, MOPPOTrainer

    obs_cfg = ObservationSpaceCfg()
    action_cfg = ActionSpaceCfg()
    reward_cfg = RewardVectorCfg()
    pref_cfg = PreferenceCfg()
    stack_cfg = ObservationStackCfg(num_policy_stacks=args.num_policy_stacks, num_critic_stacks=args.num_critic_stacks)
    moppo_cfg = MOPPOConfig()

    run_dir = None
    log_dir = args.log_dir
    save_path = args.save_path
    if args.logs_root:
        run_dir = make_run_dir(args.logs_root, run_name=args.run_name)
        dump_config(run_dir, obs=obs_cfg, action=action_cfg, reward=reward_cfg, preference=pref_cfg, stack=stack_cfg, moppo=moppo_cfg)
        log_dir = os.path.join(run_dir, "tensorboard")
        save_path = os.path.join(run_dir, "checkpoint.pt")
        print(f"run directory: {run_dir}")

    writer = None
    if log_dir:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=log_dir)

    if args.env == "dummy":
        env = DummyTalonEnv(obs_cfg, action_cfg, num_envs=args.num_envs, horizon=200, seed=args.seed)
    else:
        # Imported lazily (not at module scope) so --env dummy keeps working
        # on machines without Isaac Sim installed (this repo's default 3.12
        # .venv included) — `carb` and everything isaaclab imports
        # transitively are only importable once Isaac Sim's Kit runtime is
        # actually running, i.e. after the SimulationApp() constructed above.
        import gymnasium as gym
        import talon_rl.tasks.locomotion.a1_env  # noqa: F401 — registers Isaac-Talon-A1-v0
        from talon_rl.tasks.locomotion.a1_env.a1_env_cfg import IsaacLabTalonEnvCfg

        cfg = IsaacLabTalonEnvCfg()
        cfg.scene.num_envs = args.num_envs
        env = gym.make("Isaac-Talon-A1-v0", cfg=cfg).unwrapped

    trainer = MOPPOTrainer(env, obs_cfg, reward_cfg, pref_cfg, moppo_cfg=moppo_cfg, stack_cfg=stack_cfg, seed=args.seed)

    if args.resume:
        trainer.load(args.resume)
        print(f"resumed from {args.resume} (t={trainer._t})")

    print(f"reward terms: {reward_cfg.term_names}")
    for i in range(1, args.updates + 1):
        stats = trainer.update()
        r = ", ".join(f"{n}={v:+.3f}" for n, v in zip(reward_cfg.term_names, stats["mean_reward_vec"]))
        print(
            f"update {i:3d} | policy_loss={stats['policy_loss']:+.4f} "
            f"value_loss={stats['value_loss']:.4f} ep_len={stats['mean_episode_len']:.1f} | {r}"
        )

        if writer is not None:
            # Tags follow jaykorea/Isaac-RL-Two-wheel-Legged-Bot's
            # rsl_rl-derived OnPolicyRunner convention (Loss/*, Train/*) —
            # Reward/* is our own addition, one tag per reward-vector term,
            # since our reward is a vector (theirs is a pre-summed scalar).
            writer.add_scalar("Loss/policy", stats["policy_loss"], i)
            writer.add_scalar("Loss/value", stats["value_loss"], i)
            writer.add_scalar("Train/mean_episode_length", stats["mean_episode_len"], i)
            for name, value in zip(reward_cfg.term_names, stats["mean_reward_vec"]):
                writer.add_scalar(f"Reward/{name}", value, i)

        if save_path and args.save_every and i % args.save_every == 0:
            trainer.save(save_path)
            print(f"checkpoint saved to {save_path} (update {i})")

    if writer is not None:
        writer.close()

    if save_path:
        trainer.save(save_path)
        print(f"saved checkpoint to {save_path}")

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
