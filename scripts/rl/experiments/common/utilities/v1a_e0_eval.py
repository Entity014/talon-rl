#!/usr/bin/env python3
"""V1A-E0 paired stock A1 deterministic evaluator.

The evaluator opens only terminal M0.1/V1-A checkpoints supplied by the
caller.  It evaluates each seed pair after identical seeded resets, records
initial-observation hashes, and refuses a verdict on integrity failure.
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]


def obs_tensor(value):
    if isinstance(value, dict):
        value = value.get("policy", next(iter(value.values())))
    return value if torch.is_tensor(value) else torch.as_tensor(value)


def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_base():
    from rsl_rl.modules import ActorCritic

    return ActorCritic(
        obs={"policy": torch.zeros(1, 48), "critic": torch.zeros(1, 48)},
        obs_groups={"policy": ["policy"], "critic": ["critic"]},
        num_actions=12,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type="scalar",
    )


def load_policy(path: Path, kind: str):
    from rl.core.integration.rsl_rl.v1a_wrapper import RslRlV1AWrapper

    state = torch.load(path, map_location="cuda", weights_only=False)
    raw = state.get("model_state_dict", state.get("model", state))
    base = build_base().cuda()
    if kind == "m01":
        base.load_state_dict(raw)
        model = base
    elif kind == "v1a":
        model = RslRlV1AWrapper(base, bottleneck_dim=8).cuda()
        result = model.load_state_dict(raw)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(f"V1-A state mismatch: {result}")
    else:
        raise ValueError(kind)
    model.eval()
    return model


def snapshot(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def run_one(env, model, kind: str, reset_seed: int, steps: int, mark):
    obs, _ = env.reset(seed=reset_seed)
    obs = obs_tensor(obs).cuda()
    mark("RESET_OK", {"kind": kind, "reset_seed": reset_seed})
    initial_hash = tensor_hash(obs)
    before = snapshot(model)
    terminated_any = np.zeros(obs.shape[0], dtype=bool)
    velocity_error = []
    tilt = []
    contacts = []
    terminations = []
    finite = True
    with torch.no_grad():
        for step in range(steps):
            if kind == "m01":
                # rsl_rl's public actor path expects grouped observations.
                # Keep the raw backend call; adapt only the input container.
                action = model.act_inference({"policy": obs, "critic": obs})
            else:
                action = model.act_inference(obs, torch.full((obs.shape[0], 5), 0.2, device=obs.device))
            nxt, _, term, trunc, _ = env.step(action)
            done = term | trunc
            terminated_any |= done.detach().cpu().numpy()
            terminations.append(term.detach().cpu().numpy().astype(np.float32))
            data = env.unwrapped.scene["robot"].data
            command = env.unwrapped.command_manager.get_command("base_velocity")
            vel = data.root_lin_vel_b[:, 0]
            velocity_error.append((vel - command[:, 0]).abs().detach().cpu().numpy())
            q = data.root_quat_w
            roll = torch.atan2(2 * (q[:, 0] * q[:, 1] + q[:, 2] * q[:, 3]), 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2))
            pitch = torch.asin(torch.clamp(2 * (q[:, 0] * q[:, 2] - q[:, 3] * q[:, 1]), -1, 1))
            tilt.append(torch.rad2deg(torch.maximum(roll.abs(), pitch.abs())).detach().cpu().numpy())
            base_contact = env.unwrapped.termination_manager.get_term("base_contact")
            contacts.append(base_contact.detach().cpu().numpy().astype(np.float32))
            obs = obs_tensor(nxt).cuda()
            finite &= bool(torch.isfinite(obs).all() and torch.isfinite(action).all())
            if step == 0:
                mark("STEP_1_OK", {"kind": kind})
            elif (step + 1) % 100 == 0:
                mark("STEP_PROGRESS", {"kind": kind, "step": step + 1})
    after = snapshot(model)
    nonmutation = all(torch.equal(before[name], after[name]) for name in before)
    all_tilt = np.concatenate(tilt)
    result = {
        "initial_observation_sha256": initial_hash,
        "survival": float(1.0 - terminated_any.mean()),
        "velocity_tracking_error": float(np.mean(np.concatenate(velocity_error))),
        "tilt_p95_deg": float(np.percentile(all_tilt, 95)),
        "max_tilt_deg": float(np.max(all_tilt)),
        "termination_rate": float(np.mean(np.concatenate(terminations))),
        "base_contact_rate": float(np.mean(np.concatenate(contacts))),
        "finite_state": bool(finite),
        "evaluator_non_mutation": bool(nonmutation),
        "deterministic_actor_mean": True,
    }
    mark("ROLLOUT_DONE", {"kind": kind, "steps": steps})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, choices=(0, 1, 2))
    parser.add_argument("--m01-checkpoint", type=Path, required=True)
    parser.add_argument("--v1a-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--num-envs", type=int, default=64)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lifecycle_path = args.output.with_name(args.output.stem + ".lifecycle.jsonl")
    error_path = args.output.with_name(args.output.stem + ".ERROR.json")
    state = {"artifact_written": False, "failed": False}

    def mark(event: str, extra=None):
        record = {"event": event, "unix": time.time(), "seed": args.seed}
        if extra:
            record.update(extra)
        with lifecycle_path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()

    def write_error(reason: str, exc=None):
        state["failed"] = True
        payload = {
            "schema": "v1a_e0_error_v1",
            "status": "ERROR",
            "reason": reason,
            "seed": args.seed,
            "output": str(args.output),
            "lifecycle": str(lifecycle_path),
        }
        if exc is not None:
            payload["error"] = str(exc)
            payload["traceback"] = traceback.format_exc()
        error_path.write_text(json.dumps(payload, indent=2) + "\n")
        mark("ERROR", {"reason": reason, "error": str(exc) if exc is not None else None})

    def atexit_guard():
        if not state["artifact_written"] and not state["failed"]:
            write_error("PROCESS_EXIT_BEFORE_ARTIFACT")

    atexit.register(atexit_guard)
    mark("RUN_STARTED", {"output": str(args.output)})
    for path in (args.m01_checkpoint, args.v1a_checkpoint):
        if not path.exists():
            write_error("CHECKPOINT_MISSING", FileNotFoundError(path))
            raise FileNotFoundError(path)
    from isaaclab.app import AppLauncher

    # AppLauncher owns a separate CLI parser.  Keep the evaluator's checkpoint
    # arguments away from it; otherwise Isaac can terminate cleanly before the
    # evaluator reaches the first reset.
    saved_argv = sys.argv[:]
    sys.argv = [sys.argv[0]]
    app = AppLauncher({"headless": True, "enable_cameras": False}).app
    sys.argv = saved_argv
    mark("APP_INIT_OK")
    env = None
    try:
        import gymnasium as gym
        import isaaclab_tasks
        from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg

        cfg = UnitreeA1FlatEnvCfg()
        cfg.scene.num_envs = args.num_envs
        cfg.seed = 17001 + args.seed
        env = gym.make("Isaac-Velocity-Flat-Unitree-A1-v0", cfg=cfg)
        mark("ENV_CREATED", {"num_envs": args.num_envs})
        m01 = load_policy(args.m01_checkpoint, "m01")
        v1a = load_policy(args.v1a_checkpoint, "v1a")
        mark("CKPT_LOADED")
        reset_seed = 17001 + args.seed
        baseline = run_one(env, m01, "m01", reset_seed, args.steps, mark)
        candidate = run_one(env, v1a, "v1a", reset_seed, args.steps, mark)
        paired_reset = baseline["initial_observation_sha256"] == candidate["initial_observation_sha256"]
        result = {
            "schema": "v1a_e0_pair_eval_v1",
            "protocol": "V1A-E0",
            "seed": args.seed,
            "steps": args.steps,
            "num_envs": args.num_envs,
            "reset_seed": reset_seed,
            "m01_checkpoint": str(args.m01_checkpoint),
            "m01_checkpoint_sha256": file_hash(args.m01_checkpoint),
            "v1a_checkpoint": str(args.v1a_checkpoint),
            "v1a_checkpoint_sha256": file_hash(args.v1a_checkpoint),
            "paired_reset_hash_equal": paired_reset,
            "m01": baseline,
            "v1a": candidate,
            "integrity_pass": paired_reset and baseline["evaluator_non_mutation"] and candidate["evaluator_non_mutation"],
            "verdict": "PENDING_THRESHOLD_APPLICATION",
        }
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        state["artifact_written"] = True
        mark("ARTIFACT_WRITTEN", {"path": str(args.output)})
        mark("RUN_DONE", {"status": "RUN_DONE"})
        print(json.dumps(result, indent=2))
    except BaseException as exc:
        write_error("EVALUATION_EXCEPTION", exc)
        raise
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
