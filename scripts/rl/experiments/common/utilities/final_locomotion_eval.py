#!/usr/bin/env python3
"""Run one training checkpoint/reset-seed shard of the frozen final protocol.

One process evaluates all 25 command/preference cells for a checkpoint and
evaluation reset seed. Run nine shards (3 checkpoints x 3 reset seeds); use
``final_locomotion_score.py`` to combine their episode CSV files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
import yaml

from talon_rl.config import (
    ActionSpaceCfg,
    ExtrinsicsCfg,
    ObservationSpaceCfg,
    ObservationStackCfg,
    PreferenceCfg,
    RewardVectorCfg,
)
from rl.core.algorithms import MOPPOConfig, MOPPOTrainer
from rl.core.preferences.functional import floor_clip_terms


COMMANDS = (-0.25, 0.0, 0.25, 0.50, 0.75)
PREFERENCES = {
    "uniform": (0.25, 0.25, 0.25, 0.25),
    "progress_heavy": (0.55, 0.15, 0.15, 0.15),
    "efficiency_heavy": (0.15, 0.55, 0.10, 0.20),
    "impact_heavy": (0.15, 0.10, 0.55, 0.20),
    "balance_heavy": (0.15, 0.15, 0.10, 0.60),
}
EVAL_SEEDS = (1001, 1002, 1003)
CONTACT_N = 1.0
TORQUE_LIMIT_NM = 33.5


def _select(cls, values: dict) -> dict:
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in values.items() if k in names}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _yaw_wxyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = (q[:, i] for i in range(4))
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def _terminal_frame(transition: dict, done: np.ndarray) -> dict:
    if not done.any() or "reward_transition" not in transition:
        return transition
    terminal = transition["reward_transition"]
    out = dict(transition)
    for key in (
        "root_pos_w", "root_quat_w", "root_lin_vel_w", "v_actual", "roll_pitch",
        "v_z", "height", "roll_pitch_rate", "hip_qdot_L", "hip_qdot_R",
        "hip_q_L", "hip_q_R", "foot_air_time_reward",
        "foot_contact_force", "foot_vel", "undesired_contact_count",
        "undesired_contact_force", "joint_torque", "joint_vel", "action", "prev_action",
    ):
        if key in terminal:
            value = np.array(transition[key], copy=True)
            value[done] = terminal[key][done]
            out[key] = value
    return out


def _nominalize_cfg(cfg) -> None:
    """Apply the protocol's flat, nominal, no-disturbance environment."""
    from talon_rl.assets.unitree_a1.a1 import TALON_A1_CFG

    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.scene.robot = TALON_A1_CFG.replace()
    cfg.events.randomize_payload_mass.params["mass_distribution_params"] = (0.0, 0.0)
    cfg.events.randomize_payload_com.params["com_range"] = {
        "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)
    }
    cfg.events.randomize_friction.params.update(
        static_friction_range=(1.0, 1.0),
        dynamic_friction_range=(1.0, 1.0),
        restitution_range=(0.0, 0.0),
    )
    cfg.events.randomize_motor_power.params.update(
        stiffness_distribution_params=(1.0, 1.0), damping_distribution_params=(1.0, 1.0)
    )
    cfg.events.randomize_joint_range.params["scale_range"] = (1.0, 1.0)
    cfg.events.push_robot = None
    cfg.curriculum.terrain_levels = None
    cfg.terminations.obstacle_reached = None
    cfg.episode_length_s = 22.0
    cfg.stand_phase_s = 0.0


def _set_initial_state(env, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    robot = env.scene["robot"]
    state = robot.data.default_root_state.clone()
    offsets = rng.uniform(-0.05, 0.05, size=(env.num_envs, 2)).astype(np.float32)
    yaw = rng.uniform(-0.05, 0.05, size=env.num_envs).astype(np.float32)
    state[:, :3] += env.scene.env_origins
    state[:, 0:2] += torch.from_numpy(offsets).to(env.device)
    half = torch.from_numpy(yaw * 0.5).to(env.device)
    state[:, 3] = torch.cos(half)
    state[:, 4:6] = 0.0
    state[:, 6] = torch.sin(half)
    state[:, 7:] = 0.0
    robot.write_root_pose_to_sim(state[:, :7])
    robot.write_root_velocity_to_sim(state[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    env.scene.write_data_to_sim()
    env.sim.forward()
    return offsets, yaw


def _run_cell(env, trainer, reward_cfg, vx: float, pref_name: str, eval_seed: int,
              settling_steps: int, command_steps: int, body_mass_kg: np.ndarray) -> list[dict]:
    transition = env.reset()
    rng = np.random.default_rng(eval_seed)
    _, initial_yaw = _set_initial_state(env, rng)
    obs_dict = env.observation_manager.compute()
    transition = env._transition(obs_dict)
    trainer.push_obs(transition["obs"], update_normalizer=False)
    trainer._last_extrinsics = transition.get("extrinsics") if trainer.encoder else None

    w = np.asarray(PREFERENCES[pref_name], dtype=np.float32)[None]
    floors = {
        "progress": reward_cfg.progress_floor_eps,
        "impact": reward_cfg.impact_floor_eps,
        "balance": reward_cfg.balance_floor_eps,
    }
    w = floor_clip_terms(w, reward_cfg.term_names, floors)
    if not np.allclose(w[0], PREFERENCES[pref_name], atol=1e-7):
        raise RuntimeError(f"floor projection changed frozen preference {pref_name}: {w[0]}")
    trainer.w = np.repeat(w, env.num_envs, axis=0)

    n, total = env.num_envs, settling_steps + command_steps
    alive = np.ones(n, dtype=bool)
    failed = np.zeros(n, dtype=bool)
    base_contact = np.zeros(n, dtype=bool)
    numerical_failure = np.zeros(n, dtype=bool)
    fail_step = np.full(n, total, dtype=np.int32)
    command_start_pos = np.zeros((n, 3), dtype=np.float32)
    last_pos = np.zeros((n, 3), dtype=np.float32)
    horizontal_path = np.zeros(n, dtype=np.float64)
    vx_abs_err = np.zeros(n); abs_vz = np.zeros(n); power = np.zeros(n)
    action_sat = np.zeros(n); torque_sat = np.zeros(n); undesired_steps = np.zeros(n)
    slip = np.zeros(n); valid_steps = np.zeros(n, dtype=np.int32)
    max_lateral = np.zeros(n); max_yaw = np.zeros(n); max_foot_force = np.zeros(n)
    max_undesired_force = np.zeros(n); max_all_air_run = np.zeros(n, dtype=np.int32)
    all_air_run = np.zeros(n, dtype=np.int32)
    tilt_series = np.full((command_steps, n), np.nan, dtype=np.float32)
    abs_vz_series = np.full((command_steps, n), np.nan, dtype=np.float32)
    foot_force_series = np.full((command_steps, n), np.nan, dtype=np.float32)
    contact_count_sum = np.zeros(n); contact_samples = np.zeros(n)
    contact_per_foot_sum = np.zeros((n, 4), dtype=np.float64)
    previous_contact = np.zeros((n, 4), dtype=bool)
    touchdown_window = np.zeros((n, 4), dtype=np.int32)
    touchdown_peak = np.zeros((n, 4), dtype=np.float32)
    touchdown_events: list[list[float]] = [[] for _ in range(n)]

    norm_before = trainer.obs_norm.state_dict()
    trainer.model.eval()
    for step in range(total):
        command_vx = 0.0 if step < settling_steps else vx
        env.v_command_buf[:] = torch.tensor((command_vx, 0.0, 0.0), device=env.device)
        action = trainer.act_inference()
        transition, done = env.step(action)
        frame = _terminal_frame(transition, done)
        trainer._last_extrinsics = transition.get("extrinsics") if trainer.encoder else None
        trainer.push_obs(transition["obs"], done_mask=done, update_normalizer=False)

        finite = np.isfinite(action).all(axis=1)
        for key in ("root_pos_w", "root_quat_w", "joint_torque", "v_actual"):
            finite &= np.isfinite(frame[key]).reshape(n, -1).all(axis=1)
        just_numerical = alive & ~finite
        numerical_failure |= just_numerical
        time_out = frame.get("term_time_out", np.zeros(n, dtype=bool)).astype(bool)
        protocol_timeout = time_out & (step == total - 1)
        effective_done = done | ~finite
        failure_done = (done & ~protocol_timeout) | ~finite
        just_failed = alive & failure_done
        fail_step[just_failed] = step + 1
        failed |= just_failed
        base_contact |= alive & frame.get("term_base_contact", np.zeros(n, bool))

        if step == settling_steps - 1:
            command_start_pos[:] = frame["root_pos_w"]
            last_pos[:] = command_start_pos

        if step >= settling_steps:
            k = step - settling_steps
            use = alive & finite
            pos = frame["root_pos_w"]
            q = frame["root_quat_w"]
            yaw = _yaw_wxyz(q)
            delta = pos - command_start_pos
            c, s = np.cos(initial_yaw), np.sin(initial_yaw)
            forward = delta[:, 0] * c + delta[:, 1] * s
            lateral = -delta[:, 0] * s + delta[:, 1] * c
            segment = np.linalg.norm(pos[:, :2] - last_pos[:, :2], axis=1)
            horizontal_path[use] += segment[use]
            last_pos[use] = pos[use]
            max_lateral[use] = np.maximum(max_lateral[use], np.abs(lateral[use]))
            max_yaw[use] = np.maximum(max_yaw[use], np.abs(_wrap(yaw[use] - initial_yaw[use])))
            rp = frame["roll_pitch"]
            tilt = np.max(np.abs(rp), axis=1)
            vz = np.abs(frame["root_lin_vel_w"][:, 2])
            force = frame["foot_contact_force"]
            peak_force = np.max(force, axis=1)
            contact = force > CONTACT_N
            new_contact = contact & ~previous_contact & use[:, None]
            touchdown_window[new_contact] = 10
            touchdown_peak[new_contact] = force[new_contact]
            active_touchdown = (touchdown_window > 0) & use[:, None]
            touchdown_peak[active_touchdown] = np.maximum(
                touchdown_peak[active_touchdown], force[active_touchdown]
            )
            finishing = touchdown_window == 1
            for lane_idx, foot_idx in np.argwhere(finishing):
                touchdown_events[int(lane_idx)].append(float(touchdown_peak[lane_idx, foot_idx]))
            touchdown_window[active_touchdown] -= 1
            previous_contact[use] = contact[use]
            all_air_run[use & ~contact.any(axis=1)] += 1
            all_air_run[use & contact.any(axis=1)] = 0
            max_all_air_run[use] = np.maximum(max_all_air_run[use], all_air_run[use])
            torque = frame["joint_torque"]
            joint_vel = frame["joint_vel"]
            vx_abs_err[use] += np.abs(frame["v_actual"][:, 0][use] - vx)
            abs_vz[use] += vz[use]
            power[use] += np.sum(np.abs(torque[use] * joint_vel[use]), axis=1)
            action_sat[use] += (np.abs(action[use]) >= 0.95 * trainer.model.ACTION_CLIP).mean(axis=1)
            torque_sat[use] += (np.abs(torque[use]) >= 0.95 * TORQUE_LIMIT_NM).mean(axis=1)
            undesired_steps[use] += (frame["undesired_contact_count"][use] > 0)
            foot_speed = np.linalg.norm(frame["foot_vel"], axis=2)
            slip[use] += np.where(contact[use], foot_speed[use], 0.0).sum(axis=1) / np.maximum(contact[use].sum(axis=1), 1)
            contact_count_sum[use] += contact[use].sum(axis=1)
            contact_per_foot_sum[use] += contact[use]
            contact_samples[use] += 1
            max_foot_force[use] = np.maximum(max_foot_force[use], peak_force[use])
            undesired_force = np.max(frame["undesired_contact_force"], axis=1)
            max_undesired_force[use] = np.maximum(max_undesired_force[use], undesired_force[use])
            tilt_series[k, use] = tilt[use]
            abs_vz_series[k, use] = vz[use]
            foot_force_series[k, use] = peak_force[use]
            valid_steps[use] += 1

        alive &= ~effective_done

    for lane_idx, foot_idx in np.argwhere(touchdown_window > 0):
        touchdown_events[int(lane_idx)].append(float(touchdown_peak[lane_idx, foot_idx]))

    norm_after = trainer.obs_norm.state_dict()
    for key in norm_before:
        if not np.array_equal(norm_before[key], norm_after[key]):
            raise RuntimeError(f"observation normalizer mutated during evaluation: {key}")

    final_pos = last_pos
    delta = final_pos - command_start_pos
    c, s = np.cos(initial_yaw), np.sin(initial_yaw)
    directed = np.sign(vx) * (delta[:, 0] * c + delta[:, 1] * s) if vx else np.zeros(n)
    duration = command_steps * env.step_dt
    rows = []
    for lane in range(n):
        count = max(int(valid_steps[lane]), 1)
        tilt_values = tilt_series[:, lane][np.isfinite(tilt_series[:, lane])]
        vz_values = abs_vz_series[:, lane][np.isfinite(abs_vz_series[:, lane])]
        force_values = foot_force_series[:, lane][np.isfinite(foot_force_series[:, lane])]
        row = {
            "eval_seed": eval_seed, "lane": lane, "command_vx": vx, "preference": pref_name,
            "survived_22s": int(not failed[lane]), "fell": int(base_contact[lane]),
            "numerical_failure": int(numerical_failure[lane]), "failure_step": int(fail_step[lane]),
            "valid_command_steps": int(valid_steps[lane]), "vx_mae": vx_abs_err[lane] / count,
            "directed_displacement_m": directed[lane], "horizontal_path_m": horizontal_path[lane],
            "max_lateral_m": max_lateral[lane], "max_heading_rad": max_yaw[lane],
            "tilt_p95_rad": float(np.percentile(tilt_values, 95)) if len(tilt_values) else np.nan,
            "tilt_max_rad": float(np.max(tilt_values)) if len(tilt_values) else np.nan,
            "tilt_rms_rad": float(np.sqrt(np.mean(tilt_values ** 2))) if len(tilt_values) else np.nan,
            "mean_abs_vz_mps": abs_vz[lane] / count,
            "abs_vz_p95_mps": float(np.percentile(vz_values, 95)) if len(vz_values) else np.nan,
            "undesired_contact_step_frac": undesired_steps[lane] / count,
            "max_all_feet_air_s": max_all_air_run[lane] * env.step_dt,
            "action_sat_joint_step_frac": action_sat[lane] / count,
            "torque_sat_joint_step_frac": torque_sat[lane] / count,
            "mean_mechanical_power_w": power[lane] / count,
            "mechanical_energy_j": power[lane] * env.step_dt,
            "energy_per_directed_m_jpm": power[lane] * env.step_dt / directed[lane] if directed[lane] > 0 else np.nan,
            "mechanical_cot": (
                power[lane] * env.step_dt / (body_mass_kg[lane] * 9.81 * directed[lane])
                if directed[lane] > 0 else np.nan
            ),
            "foot_force_p95_n": float(np.percentile(force_values, 95)) if len(force_values) else np.nan,
            "foot_force_max_n": max_foot_force[lane], "undesired_force_max_n": max_undesired_force[lane],
            "touchdown_peak_p95_n": (
                float(np.percentile(touchdown_events[lane], 95)) if touchdown_events[lane] else np.nan
            ),
            "touchdown_peak_max_n": (
                float(np.max(touchdown_events[lane])) if touchdown_events[lane] else np.nan
            ),
            "mean_contact_count": contact_count_sum[lane] / max(contact_samples[lane], 1),
            "mean_contact_slip_mps": slip[lane] / count,
        }
        for foot in range(4):
            row[f"foot{foot}_duty_factor"] = contact_per_foot_sum[lane, foot] / max(contact_samples[lane], 1)
        moving_ok = (0.70 * abs(vx) * duration <= directed[lane] <= 1.30 * abs(vx) * duration) if vx else (
            horizontal_path[lane] <= 1.0 and np.linalg.norm(delta[lane, :2]) <= 0.20
        )
        row["episode_success"] = int(
            not failed[lane]
            and row["vx_mae"] <= max(0.05, 0.20 * abs(vx))
            and moving_ok and row["max_lateral_m"] <= 0.50
            and row["max_heading_rad"] <= np.deg2rad(20)
            and row["tilt_p95_rad"] <= np.deg2rad(15) and row["tilt_max_rad"] <= np.deg2rad(30)
            and row["mean_abs_vz_mps"] <= 0.15 and row["abs_vz_p95_mps"] <= 0.40
            and row["undesired_contact_step_frac"] <= 0.01 and row["max_all_feet_air_s"] <= 0.20
            and row["torque_sat_joint_step_frac"] <= 0.05 and row["action_sat_joint_step_frac"] <= 0.05
        )
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--training-seed", type=int, required=True, choices=(0, 1, 2))
    parser.add_argument("--eval-seed", type=int, required=True, choices=EVAL_SEEDS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="4 lanes, one full-duration uniform/vx=0.5 cell")
    args = parser.parse_args()

    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher({"headless": True, "enable_cameras": False})
    simulation_app = app_launcher.app
    import gymnasium as gym
    import isaaclab
    import talon_rl.tasks.locomotion.a1_env  # noqa: F401
    from talon_rl.tasks.locomotion.a1_env.a1_env_cfg import IsaacLabTalonEnvCfg

    run_dir = args.checkpoint.resolve().parents[1]
    stored = yaml.safe_load((run_dir / "config.yaml").read_text())
    obs_cfg = ObservationSpaceCfg(**_select(ObservationSpaceCfg, stored["obs"]))
    action_cfg = ActionSpaceCfg(**_select(ActionSpaceCfg, stored["action"]))
    reward_cfg = RewardVectorCfg(**_select(RewardVectorCfg, stored["reward"]))
    pref_cfg = PreferenceCfg(**_select(PreferenceCfg, stored["preference"]))
    stack_cfg = ObservationStackCfg(**_select(ObservationStackCfg, stored["stack"]))
    moppo_cfg = MOPPOConfig(**_select(MOPPOConfig, stored["moppo"]))
    cfg = IsaacLabTalonEnvCfg()
    cfg.scene.num_envs = 4 if args.smoke else 64
    cfg.seed = args.eval_seed
    cfg.sim.dt = 0.01
    cfg.decimation = 1
    cfg.sim.render_interval = 1
    _nominalize_cfg(cfg)
    env = gym.make("Isaac-Talon-A1-v0", cfg=cfg, render_mode=None).unwrapped
    trainer = MOPPOTrainer(
        env, obs_cfg, reward_cfg, pref_cfg, moppo_cfg=moppo_cfg, stack_cfg=stack_cfg,
        extrinsics_cfg=ExtrinsicsCfg(), seed=args.eval_seed,
    )
    trainer.load(str(args.checkpoint))
    if trainer._t // moppo_cfg.num_steps != 500:
        raise RuntimeError(f"expected 500 training updates, got t={trainer._t}/{moppo_cfg.num_steps}")
    if env.obs_dim != trainer.obs_norm.mean.shape[0] or env.action_dim != action_cfg.dim or reward_cfg.dim != 4:
        raise RuntimeError("checkpoint/environment observation, action, or objective dimension mismatch")
    masses = env.scene["robot"].root_physx_view.get_masses().detach().cpu().numpy().sum(axis=1)

    cells = [(0.5, "uniform")] if args.smoke else [(v, p) for v in COMMANDS for p in PREFERENCES]
    settling_steps, command_steps = (200, 2000)
    all_rows = []
    for vx, preference in cells:
        print(f"running cell vx={vx:g} preference={preference}", flush=True)
        cell_rows = _run_cell(
            env, trainer, reward_cfg, vx, preference, args.eval_seed, settling_steps, command_steps, masses
        )
        for row in cell_rows:
            row["training_seed"] = args.training_seed
        all_rows.extend(cell_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if args.smoke else "episodes"
    csv_path = args.output_dir / f"seed{args.training_seed}_eval{args.eval_seed}_{suffix}.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader(); writer.writerows(all_rows)
    preflight = {
        "training_seed": args.training_seed, "eval_seed": args.eval_seed,
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": _sha256(args.checkpoint),
        "checkpoint_env_steps": trainer._t, "training_updates": trainer._t // moppo_cfg.num_steps,
        "sim_dt": env.step_dt, "num_envs": env.num_envs, "obs_dim": env.obs_dim,
        "action_dim": env.action_dim, "objective_dim": reward_cfg.dim,
        "objective_names": reward_cfg.term_names, "action_clip": trainer.model.ACTION_CLIP,
        "torque_limit_nm": TORQUE_LIMIT_NM, "isaaclab_version": getattr(isaaclab, "__version__", None),
        "body_mass_kg_min": float(masses.min()), "body_mass_kg_max": float(masses.max()),
        "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": str(trainer.device), "smoke": args.smoke, "rows": len(all_rows),
    }
    (args.output_dir / f"seed{args.training_seed}_eval{args.eval_seed}_{suffix}.json").write_text(
        json.dumps(preflight, indent=2, default=list) + "\n"
    )
    print(csv_path)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
