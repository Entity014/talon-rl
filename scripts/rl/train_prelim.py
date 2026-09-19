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
import time

from talon_rl.config import ActionSpaceCfg, ExtrinsicsCfg, ObservationSpaceCfg, ObservationStackCfg, PreferenceCfg, RewardVectorCfg

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


def _format_hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _format_iteration_log(
    i: int, total_updates: int, stats: dict, reward_cfg: RewardVectorCfg,
    timesteps_per_iter: int, total_timesteps: int, iter_time: float, elapsed: float, eta: float,
) -> str:
    # Mirrors rsl_rl's OnPolicyRunner.log() console banner (the convention
    # jaykorea/Isaac-RL-Two-wheel-Legged-Bot's own scripts/co_rl/train.py
    # uses) — adapted to MOPPO's actual stats (policy/value loss, a 5-term
    # reward vector) instead of rsl_rl's AMP-specific loss terms.
    width, pad = 80, 34
    lines = [
        "#" * width,
        f" Learning iteration {i}/{total_updates} ".center(width, " "),
        "",
        f"{'Computation:':>{pad}} {timesteps_per_iter / max(iter_time, 1e-9):.0f} steps/s (iteration {iter_time:.3f}s)",
        f"{'Mean policy loss:':>{pad}} {stats['policy_loss']:.4f}",
        f"{'Mean value loss:':>{pad}} {stats['value_loss']:.4f}",
        f"{'Mean entropy:':>{pad}} {stats['entropy']:.8f}",
        f"{'Mean-magnitude reg (raw actor_mean^2):':>{pad}} {stats.get('mean_reg', 0.0):.4f}",
        f"{'Penalty curriculum k:':>{pad}} {stats['penalty_curriculum_k']:.4f}",
        f"{'log_std ceiling (annealing):':>{pad}} {stats.get('log_std_max', 0.0):.4f}",
        f"{'Mean episode length:':>{pad}} {stats['mean_episode_len']:.2f}",
        f"{'Termination (fall/timeout/obstacle):':>{pad}} "
        f"{stats.get('term_frac_base_contact', 0):.2f}/{stats.get('term_frac_time_out', 0):.2f}/"
        f"{stats.get('term_frac_obstacle_reached', 0):.2f} (n={stats.get('done_count', 0)})",
        "",
    ]
    for name, value in zip(reward_cfg.term_names, stats["mean_reward_vec"]):
        lines.append(f"{f'Episode_Reward/{name}:':>{pad}} {value:.4f}")
    lines += [
        "-" * width,
        f"{'Total timesteps:':>{pad}} {total_timesteps}",
        f"{'Iteration time:':>{pad}} {iter_time:.2f}s",
        f"{'Time elapsed:':>{pad}} {_format_hms(elapsed)}",
        f"{'ETA:':>{pad}} {_format_hms(eta)}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env", choices=["dummy", "isaac_lab"], default="dummy")
    parser.add_argument("--num_envs", type=int, default=64)  # CPU-sane default for --env dummy; pass --num_envs 4096 explicitly for --env isaac_lab
    parser.add_argument(
        "--no_encoder", action="store_true",
        help="Train a deployable actor without the privileged Isaac Lab Env Factor Encoder; required by sim2sim until a student encoder exists.",
    )
    parser.add_argument(
        "--torch_compile", action="store_true",
        help="Compile the CUDA actor-critic with torch.compile to reduce Python/kernel overhead.",
    )
    parser.add_argument(
        "--mean_reg_coef", type=float, default=1e-3,
        help="L2 penalty on the raw (pre-tanh) actor_mean output (MOPPOConfig's own default, the "
             "standard SAC starting point, not swept against this task). Exposed as a CLI override "
             "2026-09-19 to test whether it's too weak to counter persistent action saturation "
             "(~75-95% of joints near ACTION_CLIP across every checkpoint tested that day, unmoved "
             "by any preference-vector fix) -- Loss/mean_reg was still ~7-8 at iteration 1000 with "
             "the 1e-3 default, dwarfed by Loss/value (100s-1000s), so its gradient influence in "
             "practice is likely negligible at that coefficient.",
    )
    parser.add_argument(
        "--progress_std", type=float, default=None,
        help="Override RewardVectorCfg.progress_std (default 0.5). Exposed 2026-09-19 for a "
             "reward-vs-tracking_ratio proxy-mismatch ablation, found alongside restricting "
             "progress_reward's exp-kernel to v_x only (see that function's docstring): at std=0.5, "
             "standing still (v_x error=0.5) scored 0.368 and drifting backward (error~0.54) scored "
             "0.312, both close to a policy that actually tracks (1.0) -- not enough dynamic range "
             "for PPO to strongly prefer real tracking over those. Sweep candidates 0.5/0.35/0.2, "
             "one seed each first (shape test) before committing seeds to a specific value.",
    )
    parser.add_argument(
        "--balance_tilt_coef", type=float, default=None,
        help="Override RewardVectorCfg.balance_tilt_coef (default 1.0). Exposed 2026-09-19 for a "
             "k_theta sweep after a trajectory trace found balance_reward's -sum(roll_pitch^2) term "
             "has too little dynamic range at realistic tilt angles (0.05 vs 0.15 rad differ by only "
             "0.02, dwarfed by alive_bonus's flat +1.0) to distinguish normal walking pitch from "
             "pre-fall pitch. Sweep candidates 5, 10.",
    )
    parser.add_argument(
        "--balance_tilt_rate_coef", type=float, default=None,
        help="Override RewardVectorCfg.balance_tilt_rate_coef (default 0.0 = disabled). A "
             "438-fall-event calibration (prefall_window_analysis.py) found |pitch_rate| separates "
             "pre-fall from normal-walking states ~3.2-3.6x (p50/p90), sharper than |pitch|'s own "
             "~2x -- candidates 0.004, 0.008.",
    )
    parser.add_argument(
        "--balance_height_coef", type=float, default=None,
        help="Override RewardVectorCfg.balance_height_coef (default 1.0). Exposed 2026-09-20: "
             "checkpoints A/B/D (balance_tilt_coef swept 1.0/3.5/2.0) all crouched to "
             "height~0.21-0.25 vs target 0.42 regardless of tilt strength -- height_penalty's own "
             "coefficient is too small (dynamic range ~0.04 at typical error) next to fall_penalty "
             "(-25) and alive_bonus (+1) for the policy to prefer standing tall. Sweep candidates 5, 10.",
    )
    parser.add_argument("--action_scale", type=float, default=0.15, help="Isaac Lab joint target action scale during stabilization curriculum.")
    parser.add_argument(
        "--w_curriculum_updates", type=int, default=0,
        help="Anneal the preference Dirichlet alpha from a balance>progress>smoothness/impact>energy-biased "
             "start to the flat, full-simplex-uniform PreferenceCfg.dirichlet_alpha over this many updates, "
             "instead of sampling uniformly from update 0 (the default, 0 = curriculum disabled). Found "
             "necessary 2026-09-19: fromscratch_v2's uniform-from-start run plateaued ~10k updates with "
             "near-zero forced-command velocity tracking -- progress has no preference floor (unlike "
             "impact/balance) so a Dirichlet draw can dilute it to near-nothing before the policy reliably "
             "learns to track at all. See PreferenceCfg's own docstring.",
    )
    parser.add_argument("--stand_phase_s", type=float, default=2.0, help="Seconds of zero velocity command before locomotion commands are enabled.")
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
    reward_cfg_overrides = {}
    if args.progress_std is not None:
        reward_cfg_overrides["progress_std"] = args.progress_std
    if args.balance_tilt_coef is not None:
        reward_cfg_overrides["balance_tilt_coef"] = args.balance_tilt_coef
    if args.balance_tilt_rate_coef is not None:
        reward_cfg_overrides["balance_tilt_rate_coef"] = args.balance_tilt_rate_coef
    if args.balance_height_coef is not None:
        reward_cfg_overrides["balance_height_coef"] = args.balance_height_coef
    reward_cfg = RewardVectorCfg(**reward_cfg_overrides)
    # Balance > progress > impact/efficiency -- by NAME through
    # reward_cfg.term_names, never a hardcoded position (CLAUDE.md's "single
    # source of truth for reward order" invariant). See --w_curriculum_updates'
    # own help text and PreferenceCfg's docstring for why this ordering.
    # `efficiency` (energy+smoothness merged 2026-09-19) takes over
    # smoothness's old alpha -- it's the same "physical gentleness" family.
    _W_CURRICULUM_ALPHA_START = {"balance": 4.0, "progress": 3.0, "impact": 1.5, "efficiency": 1.5}
    pref_cfg = PreferenceCfg(
        curriculum_alpha_start=(
            tuple(_W_CURRICULUM_ALPHA_START[name] for name in reward_cfg.term_names)
            if args.w_curriculum_updates > 0 else None
        ),
        curriculum_updates=args.w_curriculum_updates,
    )
    stack_cfg = ObservationStackCfg(num_policy_stacks=args.num_policy_stacks, num_critic_stacks=args.num_critic_stacks)
    moppo_cfg = MOPPOConfig(
        device="cuda" if args.env == "isaac_lab" else "cpu",
        torch_compile=args.torch_compile,
        mean_reg_coef=args.mean_reg_coef,
    )
    extrinsics_cfg = ExtrinsicsCfg() if args.env == "isaac_lab" and not args.no_encoder else None

    run_dir = None
    log_dir = args.log_dir
    save_path = args.save_path
    if args.logs_root:
        run_dir = make_run_dir(args.logs_root, run_name=args.run_name)
        dump_config(run_dir, obs=obs_cfg, action=action_cfg, reward=reward_cfg, preference=pref_cfg, stack=stack_cfg, moppo=moppo_cfg)
        log_dir = os.path.join(run_dir, "tensorboard")
        # checkpoints/ subdir, not flat in run_dir -- a long run dumps dozens
        # of checkpoint_t*.pt files (see the save_every block below) that
        # would otherwise clutter run_dir alongside config.yaml/tensorboard/
        # exported. resolve_checkpoint() and play.py's checkpoint_run_dir()
        # know this convention (and fall back to the old flat layout for
        # runs created before it).
        save_path = os.path.join(run_dir, "checkpoints", "checkpoint.pt")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"run directory: {run_dir}")

    writer = None
    if log_dir:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=log_dir)
        # One-time note so a Reward/<term> curve is legible without opening
        # reward.py -- each term is otherwise just a bare, unlabeled scalar.
        writer.add_text(
            "Reward/composition",
            "| term | formula | inputs |\n"
            "|---|---|---|\n"
            "| progress | exp(-(v_x - v_cmd_x)^2 / progress_std^2) | v_x only (2026-09-19, was all 3 axes) |\n"
            "| efficiency | -sum\\|joint_torque * joint_vel\\| - (sum((action - prev_action)^2) + 0.01 * sum(joint_acc^2)) | energy+smoothness merged 2026-09-19 (0.91 correlated) |\n"
            "| impact | -max(0, peak_foot_contact_force - threshold) / threshold | only force above threshold is penalized |\n"
            "| balance | -sum(roll_pitch^2) | dense anti-fall orientation penalty |\n",
            0,
        )

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
        cfg.action_scale = args.action_scale
        cfg.stand_phase_s = args.stand_phase_s
        cfg.actions.joint_pos.scale = args.action_scale
        env = gym.make("Isaac-Talon-A1-v0", cfg=cfg).unwrapped

    trainer = MOPPOTrainer(env, obs_cfg, reward_cfg, pref_cfg, moppo_cfg=moppo_cfg, stack_cfg=stack_cfg, extrinsics_cfg=extrinsics_cfg, seed=args.seed)

    if args.resume:
        trainer.load(args.resume)
        print(f"resumed from {args.resume} (t={trainer._t})")

    print(f"reward terms: {reward_cfg.term_names}")
    timesteps_per_iter = moppo_cfg.num_steps * env.num_envs
    start_time = time.time()
    # Best-so-far checkpoint, by mean_episode_len — found 2026-09-17
    # (phase1_longrun, 20000 updates): the run briefly reached
    # mean_episode_len=70.45 around iteration 14501, then collapsed back to
    # its ~6-8 floor for the remaining 5500 iterations and never recovered.
    # save_path only ever gets overwritten with the LATEST weights, so that
    # brief good policy was unrecoverably lost once training continued past
    # it -- there was no separate "best" file to fall back to.
    best_episode_len = float("-inf")
    best_path = None
    if save_path:
        root, ext = os.path.splitext(save_path)
        best_path = f"{root}_best{ext}"
    for i in range(1, args.updates + 1):
        iter_start = time.time()
        stats = trainer.update()
        iter_time = time.time() - iter_start
        elapsed = time.time() - start_time
        eta = (elapsed / i) * (args.updates - i)
        print(_format_iteration_log(
            i, args.updates, stats, reward_cfg,
            timesteps_per_iter, i * timesteps_per_iter, iter_time, elapsed, eta,
        ))

        if writer is not None:
            # Tags follow jaykorea/Isaac-RL-Two-wheel-Legged-Bot's
            # rsl_rl-derived OnPolicyRunner convention (Loss/*, Train/*) —
            # Reward/* is our own addition, one tag per reward-vector term,
            # since our reward is a vector (theirs is a pre-summed scalar).
            writer.add_scalar("Loss/policy", stats["policy_loss"], i)
            writer.add_scalar("Loss/value", stats["value_loss"], i)
            writer.add_scalar("Loss/entropy", stats["entropy"], i)
            writer.add_scalar("Loss/mean_reg", stats.get("mean_reg", 0.0), i)
            writer.add_scalar("Train/penalty_curriculum_k", stats["penalty_curriculum_k"], i)
            writer.add_scalar("Train/log_std_max", stats.get("log_std_max", 0.0), i)
            writer.add_scalar("Train/mean_episode_length", stats["mean_episode_len"], i)
            writer.add_scalar("Train/done_count", stats.get("done_count", 0), i)
            for name in ("base_contact", "time_out", "obstacle_reached"):
                writer.add_scalar(f"Termination/{name}_frac", stats.get(f"term_frac_{name}", 0.0), i)
            for name, value in zip(reward_cfg.term_names, stats["mean_reward_vec"]):
                writer.add_scalar(f"Reward/{name}", value, i)
            # RAW (pre-normalize_per_objective) mean |advantage| per
            # objective -- see MOPPOTrainer.update()'s adv_mag_per_objective
            # comment for why this is measured before, not after, D3PO's
            # per-objective normalization.
            if "adv_mag_per_objective" in stats:
                for name, value in zip(reward_cfg.term_names, stats["adv_mag_per_objective"]):
                    writer.add_scalar(f"Advantage/{name}", value, i)

        if save_path and args.save_every and i % args.save_every == 0:
            trainer.save(save_path)
            print(f"checkpoint saved to {save_path} (update {i})")
            # A numbered copy per save_every interval, never overwritten --
            # found 2026-09-18 needing to compare checkpoints from different
            # points in a single run (e.g. before/after a plateau) with
            # save_path only ever holding the latest update's weights, any
            # earlier point was already gone by the time it seemed worth
            # comparing against. Named with trainer._t (the persistent,
            # checkpoint-round-tripped step counter), NOT the loop variable
            # `i` -- found the same day, the hard way: `i` restarts at 1
            # every time this script is invoked (a fresh `for i in
            # range(1, args.updates+1)` loop), so a --resume'd run's
            # checkpoint_iter500.pt/_iter1000.pt/etc silently overwrote an
            # earlier run's history files of the same name in the same run
            # directory -- several were lost before this was caught.
            # trainer._t survives save()/load() (see MOPPOTrainer.save's
            # own checkpoint dict) so it keeps climbing across resumes,
            # same convention already used for the "(t=...)" resume log line
            # above and in play.py.
            history_path = f"{root}_t{trainer._t}{ext}"
            trainer.save(history_path)

        if best_path and stats["mean_episode_len"] > best_episode_len:
            best_episode_len = stats["mean_episode_len"]
            trainer.save(best_path)
            print(f"new best mean_episode_len={best_episode_len:.2f} — saved to {best_path} (update {i})")

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
