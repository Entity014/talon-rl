#!/usr/bin/env python3
"""Play/export entry point — loads a trained checkpoint, runs deterministic
inference, and optionally exports the policy for deployment/sim2sim.

    python scripts/rl/play.py --checkpoint <path> --steps 200
    python scripts/rl/play.py --load_run last --logs_root logs/talon_rl --export policy.pt
    python scripts/rl/play.py --checkpoint <path> --analyze joint_vel joint_torque --plot
    python scripts/rl/play.py --checkpoint <path> --env isaac_lab --w 0.2 0.2 0.2 0.2 0.2 \
        --command 0.5 0.0 0.0 --validate --video

--w/--command/--validate formalize a ground-truth physics validation
methodology repeated ad hoc via one-off scratchpad scripts throughout
2026-09-18/19's debugging (see core/physics_validator.py's own docstring):
training-time reward/episode-length numbers repeatedly looked good while
the policy was actually standing nearly still, or gaming a reward term in
some physically-implausible way a raw reward number can't reveal.
--command in particular matters because v_command isn't fixed during
training (mdp/events.py's randomize_velocity_command resamples it per
episode, including a stand_phase_s window of zero command) -- a rollout
without --command is testing under whatever command the env happens to
sample, not necessarily a real sustained walking command.

Uses MOPPOTrainer.act_inference() (deterministic mean, no sampling) rather
than the stochastic action update() uses during training — see
core/algorithms/moppo.py and core/modules/actor_critic.py.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from talon_rl.config import (
    ActionSpaceCfg,
    ExtrinsicsCfg,
    ObservationSpaceCfg,
    ObservationStackCfg,
    PreferenceCfg,
    RewardVectorCfg,
)
from talon_rl.rewards.locomotion import compute_reward_vector

from rl.core.algorithms import MOPPOConfig, MOPPOTrainer
from rl.core.diagnostics.analyzer import Analyzer
from rl.core.envs.dummy import DummyTalonEnv
from rl.core.diagnostics.physics_validator import PhysicsValidator
from rl.core.experiment_io.run_dir import checkpoint_run_dir, resolve_checkpoint
from rl.core.runtime.exporter import export_policy_as_jit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None, help="Direct path to a checkpoint. Mutually exclusive with --load_run.")
    parser.add_argument("--load_run", type=str, default=None, help="Resolve <logs_root>/<run>/checkpoint.pt — <run> can be an exact run name or \"last\".")
    parser.add_argument("--logs_root", type=str, default="logs/talon_rl", help="Root directory --load_run resolves against.")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env", choices=["dummy", "isaac_lab"], default="dummy")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument(
        "--no_encoder", action="store_true",
        help="Load a checkpoint whose actor does not use the privileged Env Factor Encoder; required for sim2sim export.",
    )
    parser.add_argument("--num_policy_stacks", type=int, default=1, help="Must match the value used when the checkpoint was trained.")
    parser.add_argument("--num_critic_stacks", type=int, default=1, help="Must match the value used when the checkpoint was trained.")
    parser.add_argument("--export", type=str, default=None, help="Export the loaded policy as TorchScript to this path.")
    parser.add_argument("--analyze", type=str, nargs="+", default=None, help="Transition-dict keys to record per step (e.g. joint_vel joint_torque).")
    parser.add_argument("--plot", action="store_true", help="With --analyze and/or --validate: save one PNG per recorded key next to the checkpoint (or ./exported/ for --checkpoint).")
    parser.add_argument("--video", action="store_true", help="Record an .mp4 of the rollout (--env isaac_lab only -- DummyTalonEnv has no scene to render).")
    parser.add_argument(
        "--w", type=float, nargs="+", default=None, metavar="W",
        help="Force a fixed preference vector, one value per RewardVectorCfg.term_names entry in order, "
             "overriding the trainer's random Dirichlet init -- does NOT go through preference.floor_clip, so "
             "pass floor-respecting values yourself if that matters for your test.",
    )
    parser.add_argument(
        "--command", type=float, nargs=3, default=None, metavar=("VX", "VY", "WZ"),
        help="Force v_command to this value every step, overriding whatever the env resamples on reset "
             "(mdp/events.py's randomize_velocity_command, including its stand_phase_s zero-command window). "
             "Without this, a rollout tests under whatever command the env happens to sample, not a real "
             "sustained command.",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Print a ground-truth physics validation summary (survival, action saturation, torque vs the A1's "
             "limit, roll/pitch, velocity tracking ratio, undesired contact) instead of trusting Episode_Reward/* "
             "alone -- see core/physics_validator.py's docstring.",
    )
    parser.add_argument("--progress_std", type=float, default=None, help="Must match what the checkpoint was trained with")
    parser.add_argument("--balance_tilt_coef", type=float, default=None, help="Must match what the checkpoint was trained with")
    parser.add_argument("--balance_tilt_rate_coef", type=float, default=None, help="Must match what the checkpoint was trained with")
    parser.add_argument("--balance_height_coef", type=float, default=None, help="Must match what the checkpoint was trained with")
    parser.add_argument(
        "--target_height", type=float, default=None,
        help="Must match what the checkpoint was trained with -- also fed into PhysicsValidator so "
             "--validate's mean_height_error is measured against the right target, not always 0.42.",
    )
    args = parser.parse_args()

    if bool(args.checkpoint) == bool(args.load_run):
        raise SystemExit("pass exactly one of --checkpoint or --load_run")
    if args.video and args.env == "dummy":
        raise SystemExit("--video needs a real scene to render -- pass --env isaac_lab")
    if (args.command is not None or args.validate) and args.env == "dummy":
        # DummyTalonEnv has no locomotion physics at all (CLAUDE.md: "numbers
        # mean nothing about locomotion") -- a ground-truth physics check
        # against it would validate nothing real. It also doesn't expose
        # v_command per-env the way IsaacLabTalonEnv's v_command_buf does
        # (DummyTalonEnv wraps N single-env instances, each with its own
        # scalar v_command -- forcing it uniformly needs its own plumbing
        # this repo has no use for, since the real target is always
        # isaac_lab).
        raise SystemExit("--command/--validate need real physics -- pass --env isaac_lab")
    checkpoint_path = args.checkpoint or resolve_checkpoint(args.logs_root, args.load_run)

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
    stack_cfg = ObservationStackCfg(num_policy_stacks=args.num_policy_stacks, num_critic_stacks=args.num_critic_stacks)

    if args.w is not None and len(args.w) != reward_cfg.dim:
        raise SystemExit(f"--w needs {reward_cfg.dim} values (RewardVectorCfg.term_names order), got {len(args.w)}")

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
        # AppLauncher, not a bare SimulationApp -- see train_prelim.py's own
        # comment on this exact class (bare SimulationApp was found to
        # silently fail PhysX GPU scene construction some of the time, no
        # traceback; AppLauncher does extra Kit extension/experience-file
        # setup that fixed it there). Also required for --video: passing
        # enable_cameras straight into a bare SimulationApp's config dict is
        # silently ignored -- AppLauncher is what actually picks the
        # camera-enabled Kit experience file (isaaclab.python.headless.
        # rendering.kit) that makes env.render() produce real frames instead
        # of raising "Cannot render 'rgb_array' when ... 'NO_GUI_OR_RENDERING'".
        from isaaclab.app import AppLauncher
        app_launcher = AppLauncher({"headless": True, "enable_cameras": args.video})
        simulation_app = app_launcher.app  # noqa: F841 — kept alive for the process lifetime

        import gymnasium as gym
        import talon_rl.tasks.locomotion.a1_env  # noqa: F401 — registers Isaac-Talon-A1-v0
        from talon_rl.tasks.locomotion.a1_env.a1_env_cfg import IsaacLabTalonEnvCfg

        cfg = IsaacLabTalonEnvCfg()
        cfg.scene.num_envs = args.num_envs
        # render_mode="rgb_array" is what makes env.render() return frames instead
        # of None -- gym.wrappers.RecordVideo doesn't work here (it expects the
        # standard 5-tuple step() contract; IsaacLabTalonEnv.step() returns this
        # repo's own (transition_dict, done_array) shape, see a1_env.py's
        # docstring), so frames are captured by hand in the play loop below.
        env = gym.make("Isaac-Talon-A1-v0", cfg=cfg, render_mode="rgb_array" if args.video else None).unwrapped

    # Must match train_prelim.py's --env isaac_lab branch exactly: a checkpoint
    # trained with an encoder has an actor/critic sized for z_t and an "encoder"
    # key in its state dict -- building the trainer without extrinsics_cfg here
    # would load_state_dict() into a wrongly-shaped model.
    extrinsics_cfg = ExtrinsicsCfg() if args.env == "isaac_lab" and not args.no_encoder else None
    trainer = MOPPOTrainer(
        env, obs_cfg, reward_cfg, pref_cfg, moppo_cfg=MOPPOConfig(), stack_cfg=stack_cfg,
        extrinsics_cfg=extrinsics_cfg, seed=args.seed,
    )
    trainer.load(checkpoint_path)
    print(f"loaded checkpoint from {checkpoint_path} (t={trainer._t})")

    if args.w is not None:
        trainer.w = np.tile(np.array(args.w, dtype=np.float32), (args.num_envs, 1))
        print(f"forcing w = {args.w}")

    def force_command() -> None:
        # v_command_buf only resamples on reset -- force it every step, not
        # just once, or a lane that falls and auto-resets mid-rollout
        # drifts back to whatever the env samples on its own
        # (mdp/events.py's randomize_velocity_command, including its
        # stand_phase_s zero-command window). (--env dummy is rejected
        # above, before this closure is ever called.)
        env.v_command_buf[:] = torch.tensor(args.command, device=env.device)

    if args.command is not None:
        force_command()
        print(f"forcing v_command = {args.command}")

    analyzer = Analyzer(args.analyze) if args.analyze else None
    validator = PhysicsValidator(
        args.num_envs, trainer.model.ACTION_CLIP, target_height=reward_cfg.target_height
    ) if args.validate else None

    video_writer = None
    video_path = None
    if args.video:
        import imageio

        video_path = os.path.join(checkpoint_run_dir(checkpoint_path), "exported", "play.mp4")
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        fps = round(1.0 / env.step_dt)
        video_writer = imageio.get_writer(video_path, fps=fps)
        # First call after creating the render product comes back black --
        # a warm-up quirk of Omniverse Replicator's annotator pipeline, not a
        # real frame. Discard it, then capture the actual post-reset frame.
        env.render()
        video_writer.append_data(env.render())

    trainer.model.eval()
    total_reward = np.zeros(reward_cfg.dim, dtype=np.float32)
    for _ in range(args.steps):
        if args.command is not None:
            force_command()
        action = trainer.act_inference()
        transition, done = env.step(action)
        # This loop drives the env directly instead of MOPPOTrainer._collect_rollout,
        # which is the only other place _last_extrinsics normally advances -- without
        # this line z_t would stay frozen at the initial reset's value for every step.
        trainer._last_extrinsics = transition.get("extrinsics") if trainer.encoder else None
        trainer.push_obs(transition["obs"], done_mask=done)
        total_reward += compute_reward_vector(transition, reward_cfg).mean(axis=0)
        if analyzer is not None:
            analyzer.record(transition)
        if validator is not None:
            validator.record(transition, action, done)
        if video_writer is not None:
            video_writer.append_data(env.render())

    if video_writer is not None:
        video_writer.close()
        print(f"saved video to {video_path}")

    if validator is not None:
        validator.print_summary()

    mean_reward = total_reward / args.steps
    r = ", ".join(f"{n}={v:+.3f}" for n, v in zip(reward_cfg.term_names, mean_reward))
    print(f"mean reward over {args.steps} steps: {r}")

    if analyzer is not None and args.plot:
        plot_dir = os.path.join(checkpoint_run_dir(checkpoint_path), "exported")
        analyzer.save_plots(plot_dir)
        print(f"saved analysis plots to {plot_dir}")

    if validator is not None and args.plot:
        plot_dir = os.path.join(checkpoint_run_dir(checkpoint_path), "exported")
        validator.save_plots(plot_dir)
        print(f"saved validation plots to {plot_dir}")

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
