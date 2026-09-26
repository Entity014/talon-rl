#!/usr/bin/env python3
"""V1-B S6 baseline-only objective decomposition audit.

This is a read-only audit of stock M0.1 reward terms under the corrected
deployment path.  It does not train, regroup, or alter reward coefficients.
"""
from __future__ import annotations

# scripts/ on sys.path so the absolute rl.experiments.* imports below
# resolve when this file is run directly, as these scripts always are.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def corr_matrix(rows: np.ndarray) -> np.ndarray:
    """Pearson correlation with a stable zero-variance convention."""
    rows = np.asarray(rows, dtype=np.float64)
    out = np.zeros((rows.shape[1], rows.shape[1]), dtype=np.float64)
    for i in range(rows.shape[1]):
        xi = rows[:, i]
        for j in range(rows.shape[1]):
            xj = rows[:, j]
            if np.std(xi) > 0 and np.std(xj) > 0:
                out[i, j] = float(np.corrcoef(xi, xj)[0, 1])
            elif i == j:
                out[i, j] = 1.0
    return out


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def snapshot(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--reset-seed", type=int, default=61001)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lifecycle = args.output.with_name(args.output.stem + ".lifecycle.jsonl")
    error_path = args.output.with_name(args.output.stem + ".ERROR.json")
    state = {"artifact_written": False, "failed": False}

    def mark(event: str, **extra):
        record = {"event": event, "unix": time.time()}
        record.update(extra)
        with lifecycle.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()

    def fail(reason: str, exc=None):
        state["failed"] = True
        payload = {"schema": "v1b_s6_error_v1", "status": "ERROR", "reason": reason,
                   "output": str(args.output), "lifecycle": str(lifecycle)}
        if exc is not None:
            payload["error"] = str(exc)
            payload["traceback"] = traceback.format_exc()
        error_path.write_text(json.dumps(payload, indent=2) + "\n")
        mark("ERROR", reason=reason, error=str(exc) if exc is not None else None)

    def guard():
        if not state["artifact_written"] and not state["failed"]:
            fail("PROCESS_EXIT_BEFORE_ARTIFACT")

    atexit.register(guard)
    mark("RUN_STARTED", protocol="V1-B-S6", output=str(args.output))
    if not args.checkpoint.exists():
        fail("CHECKPOINT_MISSING", FileNotFoundError(args.checkpoint))
        raise FileNotFoundError(args.checkpoint)

    from isaaclab.app import AppLauncher

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
        from rl.experiments.common.utilities.v1a_e0_eval import build_base, load_policy

        cfg = UnitreeA1FlatEnvCfg()
        cfg.scene.num_envs = args.num_envs
        cfg.seed = args.reset_seed
        env = gym.make("Isaac-Velocity-Flat-Unitree-A1-v0", cfg=cfg)
        mark("ENV_CREATED", num_envs=args.num_envs, reset_seed=args.reset_seed)

        # Load the raw rsl_rl M0.1 actor.  The environment receives the exact
        # stock deployment action: deterministic actor mean clipped to [-1, 1].
        model = load_policy(args.checkpoint, "m01")
        model.eval()
        before = snapshot(model)
        mark("CKPT_LOADED", checkpoint=str(args.checkpoint), checkpoint_sha256=sha256(args.checkpoint))

        obs, _ = env.reset(seed=args.reset_seed)
        obs = obs_tensor(obs).cuda()
        mark("RESET_OK", initial_observation_shape=list(obs.shape))

        term_names = None
        term_rows = []
        physical_rows = []
        returned_rewards = []
        reconstructed_rewards = []
        finite = True
        done_any = np.zeros(args.num_envs, dtype=bool)

        with torch.no_grad():
            for step in range(args.steps):
                action = model.act_inference({"policy": obs, "critic": obs})
                action = torch.clamp(action, -1.0, 1.0)
                previous_action = env.unwrapped.action_manager.prev_action.clone()
                action_rate = (action - previous_action).square().mean(dim=-1)
                nxt, reward, terminated, truncated, _ = env.step(action)
                if step == 0:
                    mark("STEP_1_OK")
                elif (step + 1) % 100 == 0:
                    mark("STEP_PROGRESS", step=step + 1)

                manager = env.unwrapped.reward_manager
                names = list(manager.active_terms)
                if term_names is None:
                    term_names = names
                    mark("REWARD_TERMS_DISCOVERED", terms=term_names)
                elif names != term_names:
                    raise RuntimeError(f"Reward term order changed: {term_names} -> {names}")
                raw = manager._step_reward.detach().cpu().numpy().astype(np.float64)
                returned = reward.detach().cpu().numpy().astype(np.float64)
                reconstructed = raw.sum(axis=1) * float(env.unwrapped.step_dt)
                term_rows.append(raw)
                returned_rewards.append(returned)
                reconstructed_rewards.append(reconstructed)

                data = env.unwrapped.scene["robot"].data
                command = env.unwrapped.command_manager.get_command("base_velocity")
                quat = data.root_quat_w
                roll = torch.atan2(2 * (quat[:, 0] * quat[:, 1] + quat[:, 2] * quat[:, 3]),
                                   1 - 2 * (quat[:, 1].square() + quat[:, 2].square()))
                pitch = torch.asin(torch.clamp(2 * (quat[:, 0] * quat[:, 2] - quat[:, 3] * quat[:, 1]), -1, 1))
                base_contact = env.unwrapped.termination_manager.get_term("base_contact")
                joint_limits = data.joint_pos_limits
                joint_pos = data.joint_pos
                joint_limit_violation = torch.relu(joint_limits[..., 0] - joint_pos) + torch.relu(joint_pos - joint_limits[..., 1])
                physical_rows.append(np.stack([
                    (data.root_lin_vel_b[:, 0] - command[:, 0]).abs().cpu().numpy(),
                    (data.root_ang_vel_b[:, 2] - command[:, 2]).abs().cpu().numpy(),
                    data.root_lin_vel_b[:, 2].abs().cpu().numpy(),
                    torch.linalg.vector_norm(data.root_ang_vel_b[:, :2], dim=-1).cpu().numpy(),
                    torch.rad2deg(torch.maximum(roll.abs(), pitch.abs())).cpu().numpy(),
                    action.abs().mean(dim=-1).cpu().numpy(),
                    data.applied_torque.square().mean(dim=-1).cpu().numpy(),
                    action_rate.cpu().numpy(),
                    joint_limit_violation.mean(dim=-1).cpu().numpy(),
                    base_contact.cpu().numpy().astype(np.float64),
                ], axis=1))
                done_any |= (terminated | truncated).detach().cpu().numpy()
                finite &= bool(torch.isfinite(obs).all() and torch.isfinite(action).all() and torch.isfinite(reward).all())
                obs = obs_tensor(nxt).cuda()

        terms = np.concatenate(term_rows, axis=0)
        physical = np.concatenate(physical_rows, axis=0)
        returned_flat = np.concatenate(returned_rewards)
        reconstructed_flat = np.concatenate(reconstructed_rewards)
        physical_names = ["vx_error", "wz_error", "abs_lin_vel_z", "abs_ang_vel_xy", "tilt_deg",
                          "mean_abs_action", "torque_l2", "action_rate_l2", "joint_limit_violation", "base_contact"]
        term_corr = corr_matrix(terms)
        term_physical_corr = np.array([[safe_corr(terms[:, i], physical[:, j]) for j in range(physical.shape[1])]
                                       for i in range(terms.shape[1])])

        # A descriptive candidate grouping, not a new formulation.  The
        # assignment is based only on the strongest observed physical
        # association and explicitly marks degenerate terms as unresolved.
        target_group = {
            "vx_error": "progress",
            "wz_error": "progress",
            "mean_abs_action": "efficiency_or_limits",
            "torque_l2": "efficiency",
            "action_rate_l2": "efficiency_or_limits",
            "joint_limit_violation": "limits",
            "abs_lin_vel_z": "balance",
            "abs_ang_vel_xy": "balance",
            "tilt_deg": "balance",
            "base_contact": "contact",
        }
        dominant = {}
        for i, name in enumerate(term_names):
            j = int(np.argmax(np.abs(term_physical_corr[i])))
            strength = float(abs(term_physical_corr[i, j]))
            dominant[name] = {"metric": physical_names[j], "candidate_group": target_group[physical_names[j]],
                              "abs_correlation": strength, "degenerate": bool(np.std(terms[:, i]) == 0)}

        after = snapshot(model)
        nonmutation = all(torch.equal(before[name], after[name]) for name in before)
        reconstruction_error = float(np.max(np.abs(returned_flat - reconstructed_flat)))
        result = {
            "schema": "v1b_s6_objective_decomposition_audit_v1",
            "protocol": "V1-B-S6",
            "status": "READ_ONLY_COMPLETE",
            "verdict": "OBJECTIVE_DECOMPOSITION_AUDIT_COMPLETE",
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256(args.checkpoint),
            "num_envs": args.num_envs,
            "steps": args.steps,
            "reset_seed": args.reset_seed,
            "stock_clip_actions": 1.0,
            "reward_term_order": term_names,
            "physical_metric_order": physical_names,
            "term_statistics": {
                name: {"mean": float(terms[:, i].mean()), "std": float(terms[:, i].std()),
                       "nonzero_fraction": float(np.mean(np.abs(terms[:, i]) > 0))}
                for i, name in enumerate(term_names)
            },
            "term_term_pearson": term_corr.tolist(),
            "term_physical_pearson": term_physical_corr.tolist(),
            "dominant_behavioral_alignment": dominant,
            "scalar_reward_reconstruction": {"max_abs_error": reconstruction_error,
                                              "mean_abs_error": float(np.mean(np.abs(returned_flat - reconstructed_flat))),
                                              "pass": reconstruction_error <= 1e-5},
            "rollout_sanity": {"survival": float(1.0 - done_any.mean()), "finite_state": bool(finite),
                                "evaluator_non_mutation": bool(nonmutation)},
            "interpretation": "Descriptive baseline-only evidence; no regrouping or training was performed.",
        }
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        state["artifact_written"] = True
        mark("ARTIFACT_WRITTEN", path=str(args.output), rows=int(terms.shape[0]))
        mark("RUN_DONE", status="READ_ONLY_COMPLETE")
        print(json.dumps(result, indent=2))
    except BaseException as exc:
        fail("EVALUATION_EXCEPTION", exc)
        raise
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
