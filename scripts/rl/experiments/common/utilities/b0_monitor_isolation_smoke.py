#!/usr/bin/env python3
"""Isolation and determinism acceptance smoke for B0's actor-mean monitor."""
from __future__ import annotations

import hashlib
import io
import json
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts" / "b0_smoke"
STARTED = OUT / "b0-monitor-isolation_RUN_STARTED.json"
ERROR = OUT / "b0-monitor-isolation_ERROR.json"
COMPLETE = OUT / "b0-monitor-isolation_RUN_DONE.json"
ARTIFACT = OUT / "b0-monitor-isolation.json"
FROZEN = OUT / "b0-monitor-frozen-reset-states.npz"
MODEL_STATE = OUT / "b0-monitor-isolation-model.pt"
WORKER_RESULT = OUT / "b0-monitor-isolation-worker.json"
WORKER_RESULT_REPEAT = OUT / "b0-monitor-isolation-worker-repeat.json"
WORKER_LOG = OUT / "b0-monitor-isolation-worker.log"
WORKER_PROGRESS = OUT / "b0-monitor-isolation-worker-progress.json"
SMOKE_STATES, SMOKE_HORIZON = 4, 32
ACCEPTANCE_STATES, ACCEPTANCE_HORIZON = 64, 500


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _hash(value) -> str:
    """Stable content hash for state dictionaries, tensors, and RNG snapshots."""
    def cpu(item):
        if isinstance(item, torch.Tensor): return item.detach().cpu()
        if isinstance(item, dict): return {key: cpu(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)): return type(item)(cpu(val) for val in item)
        return item
    stream = io.BytesIO(); torch.save(cpu(value), stream)
    return hashlib.sha256(stream.getvalue()).hexdigest()


def _global_rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_global_rng(state: dict) -> None:
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"])
    if state["torch_cuda"]: torch.cuda.set_rng_state_all(state["torch_cuda"])


def _nominalize_cfg(cfg) -> None:
    from talon_rl.assets.unitree_a1.a1 import TALON_A1_CFG
    cfg.scene.terrain.terrain_type = "plane"; cfg.scene.terrain.terrain_generator = None; cfg.scene.robot = TALON_A1_CFG.replace()
    cfg.events.randomize_payload_mass.params["mass_distribution_params"] = (0.0, 0.0)
    cfg.events.randomize_payload_com.params["com_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}
    cfg.events.randomize_friction.params.update(static_friction_range=(1.0, 1.0), dynamic_friction_range=(1.0, 1.0), restitution_range=(0.0, 0.0))
    cfg.events.randomize_motor_power.params.update(stiffness_distribution_params=(1.0, 1.0), damping_distribution_params=(1.0, 1.0))
    cfg.events.randomize_joint_range.params["scale_range"] = (1.0, 1.0)
    cfg.events.push_robot = None; cfg.curriculum.terrain_levels = None; cfg.terminations.obstacle_reached = None
    cfg.episode_length_s = 22.0; cfg.stand_phase_s = 0.0


def _capture_states(base) -> dict[str, np.ndarray]:
    robot = base.scene["robot"]
    return {"root_state": robot.data.root_state_w.detach().cpu().numpy().copy(),
            "joint_pos": robot.data.joint_pos.detach().cpu().numpy().copy(),
            "joint_vel": robot.data.joint_vel.detach().cpu().numpy().copy()}


def _install_states(base, states: dict[str, np.ndarray], lanes: int) -> None:
    robot = base.scene["robot"]
    root = torch.from_numpy(states["root_state"][:lanes]).to(base.device)
    pos = torch.from_numpy(states["joint_pos"][:lanes]).to(base.device)
    vel = torch.from_numpy(states["joint_vel"][:lanes]).to(base.device)
    env_ids = torch.arange(lanes, device=base.device)
    robot.write_root_pose_to_sim(root[:, :7], env_ids=env_ids); robot.write_root_velocity_to_sim(root[:, 7:], env_ids=env_ids)
    robot.write_joint_state_to_sim(pos, vel, env_ids=env_ids); base.scene.write_data_to_sim(); base.sim.forward()
    # These are evaluator-owned lanes. Reset stateful reward bookkeeping so
    # no previous monitor rollout crosses a frozen reset boundary.
    base._foot_air_time[env_ids] = 0.0; base._foot_last_contact[env_ids] = False; base._dp_just_reset[env_ids] = True


def _training_env_state(base) -> dict:
    robot = base.scene["robot"]
    return {"common_step_counter": base.common_step_counter,
            "root_state": robot.data.root_state_w, "joint_pos": robot.data.joint_pos,
            "joint_vel": robot.data.joint_vel, "action": base.action_manager.action,
            "prev_action": base.action_manager.prev_action, "command": base.v_command_buf,
            "obstacle": base.obstacle_ahead_buf, "foot_air_time": base._foot_air_time,
            "foot_last_contact": base._foot_last_contact, "dp_origin_xy": base._dp_origin_xy,
            "dp_origin_heading": base._dp_origin_heading, "dp_origin_command_x": base._dp_origin_command_x,
            "dp_just_reset": base._dp_just_reset}


def _run_monitor(env, model, states: dict[str, np.ndarray], lanes: int, horizon: int) -> dict:
    base = env.env
    assert lanes <= base.num_envs and env.command == (0.5, 0.0, 0.0)
    base.reset(); env._command(); _install_states(base, states, lanes)
    transition = env._scalar_transition(base._transition(base.observation_manager.compute()), np.zeros(base.num_envs, bool))
    obs = transition["obs"]
    initial_x = transition["fields"]["root_pos_w"][:lanes, 0].copy()
    first_fall = np.full(lanes, horizon, dtype=np.int32); alive = np.ones(lanes, dtype=bool)
    base_contacts = np.zeros(lanes, dtype=np.int32); vx_error = []; tilt = []; height_error = []
    action_hash = hashlib.sha256()
    with torch.no_grad():
        for step in range(horizon):
            actor_obs = torch.as_tensor(obs, device=base.device, dtype=torch.float32)
            action = model.act_inference(actor_obs)
            # This is the monitor contract, not an equivalent hand-written
            # actor path: act_inference is precisely tanh(actor_mean).
            expected = torch.tanh(model.raw_mean(actor_obs)) * model.ACTION_CLIP
            assert torch.equal(action, expected)
            action_np = action.cpu().numpy().astype(np.float32); action_hash.update(action_np[:lanes].tobytes())
            transition = env.step(action_np); fields = transition["fields"]
            newly = alive & transition["done"][:lanes]
            first_fall[newly] = step + 1; alive &= ~newly
            base_contacts += (alive | newly) & transition["term_base_contact"][:lanes]
            live = alive | newly
            if live.any():
                vx_error.append(np.abs(fields["v_actual"][:lanes, 0][live] - 0.5))
                tilt.append(np.max(np.abs(fields["roll_pitch"][:lanes][live]), axis=1))
                height_error.append(np.abs(fields["height"][:lanes][live] - env.z_nominal))
            obs = transition["obs"]
    final_x = fields["root_pos_w"][:lanes, 0]
    flat = lambda values: np.concatenate(values) if values else np.zeros(0, np.float32)
    return {"horizon": horizon, "first_fall": first_fall, "base_contacts": base_contacts, "displacement": final_x - initial_x,
            "survived": first_fall == horizon, "mean_abs_vx_error": float(flat(vx_error).mean()),
            "tilt_p95": float(np.percentile(flat(tilt), 95)), "tilt_max": float(flat(tilt).max()),
            "mean_height_error": float(flat(height_error).mean()), "action_hash": action_hash.hexdigest()}


def _compact(metrics: dict) -> dict:
    return {"lanes": int(len(metrics["first_fall"])), "horizon": int(metrics["horizon"]),
            "survival_rate": float(metrics["survived"].mean()), "first_fall": metrics["first_fall"].tolist(),
            "base_contact_count": metrics["base_contacts"].tolist(), "mean_displacement": float(metrics["displacement"].mean()),
            "mean_abs_vx_error": metrics["mean_abs_vx_error"], "tilt_p95_rad": metrics["tilt_p95"],
            "tilt_max_rad": metrics["tilt_max"], "mean_height_error": metrics["mean_height_error"], "action_hash": metrics["action_hash"]}


def _validate_metrics(metrics: dict, lanes: int, horizon: int) -> None:
    """Reject a monitor result that is shaped but numerically meaningless."""
    assert metrics["first_fall"].shape == metrics["base_contacts"].shape == metrics["displacement"].shape == (lanes,)
    assert metrics["survived"].shape == (lanes,) and metrics["horizon"] == horizon
    assert np.isfinite(metrics["displacement"]).all()
    assert all(np.isfinite(metrics[key]) for key in ("mean_abs_vx_error", "tilt_p95", "tilt_max", "mean_height_error"))
    assert np.all((metrics["first_fall"] >= 1) & (metrics["first_fall"] <= horizon))
    assert np.all(metrics["base_contacts"] >= 0)


def worker(model_path: Path, result_path: Path, reuse_frozen: bool = False, actor_dir: Path | None = None) -> None:
    """Run the monitor in a separate process: Isaac Lab permits one context."""
    from isaaclab.app import AppLauncher
    app = base = None
    _write(WORKER_PROGRESS, {"status": "WORKER_STARTED", "unix": time.time()})
    try:
        app = AppLauncher({"headless": True, "enable_cameras": False}).app
        _write(WORKER_PROGRESS, {"status": "APP_STARTED", "unix": time.time()})
        import gymnasium as gym
        import talon_rl.tasks.locomotion.a1_env
        from talon_rl.wrappers.scalar_reward_env import B0TalonEnv
        from talon_rl.tasks.locomotion.a1_env.a1_env_cfg import IsaacLabTalonEnvCfg
        from rl.core.modules.actor_critic import ActorCritic

        cfg = IsaacLabTalonEnvCfg(); cfg.scene.num_envs = ACCEPTANCE_STATES; cfg.seed = 17; cfg.sim.dt = .01; cfg.decimation = 1; _nominalize_cfg(cfg)
        base = gym.make("Isaac-Talon-A1-v0", cfg=cfg, render_mode=None).unwrapped
        _write(WORKER_PROGRESS, {"status": "ENV_CREATED", "unix": time.time()})
        env = B0TalonEnv(base); env.reset()
        if reuse_frozen:
            with np.load(FROZEN) as saved:
                frozen = {key: saved[key].copy() for key in saved.files}
        else:
            frozen = _capture_states(base); np.savez_compressed(FROZEN, **frozen)
        checkpoint = torch.load(model_path, map_location=base.device)
        state_dict = checkpoint.get("model", checkpoint)
        has_reconstruction = "reconstruction_head.weight" in state_dict
        model = ActorCritic(base.obs_dim, base.obs_dim, base.action_dim, 1, [64, 64], reconstruction_dim=3 if has_reconstruction else 0).to(base.device)
        model.load_state_dict(state_dict); model.eval()
        emit_obs = os.environ.get("B0_EMIT_OBS")
        if emit_obs:
            base.reset(); env._command(); _install_states(base, frozen, ACCEPTANCE_STATES)
            np.save(emit_obs, base._transition(base.observation_manager.compute())["obs"])
        if actor_dir is not None:
            base.reset(); env._command(); _install_states(base, frozen, ACCEPTANCE_STATES)
            actor_obs = torch.as_tensor(base._transition(base.observation_manager.compute())["obs"], device=base.device, dtype=torch.float32)
            rows = []; previous = None
            for checkpoint in sorted(actor_dir.glob("update_*.pt"), key=lambda p: int(p.stem.split("_")[-1])):
                state = torch.load(checkpoint, map_location=base.device); model.load_state_dict(state["model"])
                with torch.no_grad(): mean = model.raw_mean(actor_obs).cpu(); action = model.act_inference(actor_obs).cpu()
                params = _flat_model_state(state["model"])
                row = {"update": int(checkpoint.stem.split("_")[-1]), "action_saturation": float((action.abs() >= 2.9).float().mean()), "mean_norm": float(mean.norm()), "param_norm": float(params.norm())}
                if previous is not None:
                    pm, pp = previous; row.update({"cos_mu_prev": float(torch.nn.functional.cosine_similarity(mean.reshape(1, -1), pm.reshape(1, -1)).item()), "delta_mu_norm": float((mean-pm).norm()), "delta_param_norm": float((params-pp).norm()), "relative_param_delta": float((params-pp).norm()/(pp.norm()+1e-12))})
                else: row.update({"cos_mu_prev": None, "delta_mu_norm": None, "delta_param_norm": None, "relative_param_delta": None})
                previous = (mean, params); rows.append(row)
            _write(result_path, {"pass": True, "actor_rows": rows}); return
        smoke_a = _run_monitor(env, model, frozen, SMOKE_STATES, SMOKE_HORIZON)
        _validate_metrics(smoke_a, SMOKE_STATES, SMOKE_HORIZON)
        _write(WORKER_PROGRESS, {"status": "SEMANTIC_A_DONE", "unix": time.time()})
        acceptance = _run_monitor(env, model, frozen, ACCEPTANCE_STATES, ACCEPTANCE_HORIZON)
        _validate_metrics(acceptance, ACCEPTANCE_STATES, ACCEPTANCE_HORIZON)
        _write(result_path, {"pass": True, "semantic_smoke": _compact(smoke_a), "acceptance": _compact(acceptance)})
        _write(WORKER_PROGRESS, {"status": "WORKER_DONE", "unix": time.time()})
    except BaseException as exc:
        _write(WORKER_PROGRESS, {"status": "WORKER_ERROR", "unix": time.time(), "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})
        raise
    finally:
        if base is not None: base.close()
        if app is not None: app.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for path in (ERROR, COMPLETE, ARTIFACT, FROZEN, MODEL_STATE, WORKER_RESULT, WORKER_RESULT_REPEAT, WORKER_LOG, WORKER_PROGRESS): path.unlink(missing_ok=True)
    started = {"status": "RUN_STARTED", "pid": os.getpid(), "started_unix": time.time(),
               "semantic_smoke": [SMOKE_STATES, SMOKE_HORIZON], "acceptance": [ACCEPTANCE_STATES, ACCEPTANCE_HORIZON]}
    _write(STARTED, started); app = train_base = None
    try:
        from isaaclab.app import AppLauncher
        app = AppLauncher({"headless": True, "enable_cameras": False}).app
        import gymnasium as gym
        import talon_rl.tasks.locomotion.a1_env
        from talon_rl.wrappers.scalar_reward_env import B0TalonEnv
        from talon_rl.tasks.locomotion.a1_env.a1_env_cfg import IsaacLabTalonEnvCfg
        from rl.core.algorithms.scalar_ppo import B0PPOConfig, B0PPOTrainer
        from rl.core.modules.actor_critic import ActorCritic

        torch.manual_seed(0); np.random.seed(0); random.seed(0)
        train_cfg = IsaacLabTalonEnvCfg(); train_cfg.scene.num_envs = 4; train_cfg.seed = 0; train_cfg.sim.dt = .01; train_cfg.decimation = 1; _nominalize_cfg(train_cfg)
        train_base = gym.make("Isaac-Talon-A1-v0", cfg=train_cfg, render_mode=None).unwrapped
        train_env = B0TalonEnv(train_base); train_env.reset()
        model = ActorCritic(train_base.obs_dim, train_base.obs_dim, train_base.action_dim, 1, [64, 64]).to(train_base.device)
        trainer = B0PPOTrainer(model, B0PPOConfig()); trainer.update_idx = 25; trainer.begin_update(); model.train(True)
        before = {"model": _hash(model.state_dict()), "model_training": model.training, "optimizer": _hash(trainer.optim.state_dict()),
                  "rng": _hash(_global_rng_state()), "env": _hash(_training_env_state(train_base)), "update_idx": trainer.update_idx,
                  "scheduled_std": trainer._std_for_update, "exploration_mode": model.exploration_mode, "normalizers": None}
        torch.save(model.state_dict(), MODEL_STATE)
        worker_env = dict(os.environ, PYTHONPATH=f"{ROOT}:{ROOT / 'scripts'}")
        with WORKER_LOG.open("w") as stream:
            for result, reuse in ((WORKER_RESULT, False), (WORKER_RESULT_REPEAT, True)):
                cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", str(MODEL_STATE), str(result)]
                if reuse: cmd.append("--reuse-frozen")
                subprocess.run(cmd, cwd=ROOT, env=worker_env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        worker_result = json.loads(WORKER_RESULT.read_text()); repeat_result = json.loads(WORKER_RESULT_REPEAT.read_text())
        assert worker_result["pass"] and repeat_result["pass"]
        a, b = worker_result["semantic_smoke"], repeat_result["semantic_smoke"]
        assert a["first_fall"] == b["first_fall"] and a["base_contact_count"] == b["base_contact_count"] and a["action_hash"] == b["action_hash"]
        for key in ("mean_displacement", "mean_abs_vx_error", "tilt_p95_rad", "tilt_max_rad", "mean_height_error"):
            assert np.isclose(a[key], b[key], atol=1e-6, rtol=0.0), f"non-reproducible {key}"
        after = {"model": _hash(model.state_dict()), "model_training": model.training, "optimizer": _hash(trainer.optim.state_dict()),
                 "rng": _hash(_global_rng_state()), "env": _hash(_training_env_state(train_base)), "update_idx": trainer.update_idx,
                 "scheduled_std": trainer._std_for_update, "exploration_mode": model.exploration_mode, "normalizers": None}
        assert before == after
        _write(ARTIFACT, {"pass": True, "checks": {"separate_evaluation_environment": True, "actor_mean_tanh_only": True,
               "fixed_command_vx_0_5": True, "frozen_reset_states": True, "semantic_smoke_reproducible": True,
               "acceptance_64_by_500": True, "training_state_hashes_unchanged": True, "no_optimizer_or_normalizer_mutation": True,
               "scheduled_std_and_exploration_unchanged": True, "metrics_finite_and_shaped": True}, "before_hashes": before,
               "after_hashes": after, "frozen_states": str(FROZEN), "worker_log": str(WORKER_LOG),
               "semantic_smoke": worker_result["semantic_smoke"], "acceptance": worker_result["acceptance"]})
        _write(COMPLETE, dict(started, status="RUN_DONE", ended_unix=time.time(), exit_code=0, artifact=str(ARTIFACT), complete=True)); print(ARTIFACT)
    except BaseException as exc:
        _write(ERROR, dict(started, status="ERROR", ended_unix=time.time(), error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())); raise
    finally:
        if train_base is not None: train_base.close()
        if app is not None: app.close()


def _flat_model_state(state):
    return torch.cat([v.detach().float().cpu().reshape(-1) for v in state.values() if torch.is_tensor(v) and v.is_floating_point()])

if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--worker":
        actor = Path(sys.argv[sys.argv.index("--actor-audit") + 1]) if "--actor-audit" in sys.argv else None
        reuse = "--reuse-frozen" in sys.argv[4:]
        worker(Path(sys.argv[2]), Path(sys.argv[3]), reuse, actor)
    else: main()
