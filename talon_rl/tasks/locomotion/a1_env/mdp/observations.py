# talon_rl/tasks/locomotion/a1_env/mdp/observations.py
"""Observation terms for IsaacLabTalonEnv. Reuses Isaac Lab's own
isaaclab.envs.mdp.joint_pos, .joint_vel, .last_action for those fields (see
mdp/__init__.py's re-export) — roll_pitch, foot_contact_binary, and
v_command have no Isaac Lab builtin matching ObservationSpaceCfg's exact
shape/meaning, so they're written here.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def roll_pitch(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """(roll, pitch) in radians from the root quaternion (w, x, y, z),
    standard aerospace convention — same formula as the single-env
    IsaacLabTalonEnv's _quat_to_roll_pitch helper (2026-09-13), ported to
    batched torch. (N, 2)."""
    asset: Articulation = env.scene[asset_cfg.name]
    quat = asset.data.root_quat_w
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = 2.0 * (w * y - z * x)
    pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))
    return torch.stack([roll, pitch], dim=-1)


def yaw(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Heading about the world z-axis, radians, from the root quaternion
    (w, x, y, z) -- standard aerospace-convention yaw, same quat layout
    roll_pitch above already uses. Added for progress_reward's
    directed-progress term (R1, 2026-09-20, artifacts/r1_freeze/FREEZE.md)
    -- that term projects displacement onto heading AT COMMAND ONSET, so
    a1_env.py snapshots this once per window-reset, not every step. (N,)."""
    asset: Articulation = env.scene[asset_cfg.name]
    quat = asset.data.root_quat_w
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def foot_contact_binary(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor"), threshold: float = 1.0
) -> torch.Tensor:
    """(N, 4) binary contact per foot, matching ObservationSpaceCfg.foot_contact_dim."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force_mag = torch.norm(sensor.data.net_forces_w, dim=-1)
    return (force_mag > threshold).float()


def v_command(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Scripted constant forward-velocity command — no real CommandManager
    in this prelim (matches the single-env IsaacLabTalonEnv's
    self.v_command, 2026-09-13). (N, 3)."""
    return env.v_command_buf


# --- Privileged extrinsics e_t (Adaptation Module Phase 1, chapter3.tex
# §3.2.1) — only consumed by ObservationsCfg.PrivilegedCfg, never PolicyCfg.
# See a1_env_cfg.py for the conditional payload wiring.


def payload_extrinsics(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Payload mass + CoM offset on the trunk (extrinsics factor 1+2).
    Only wired into PrivilegedCfg when payload_treatment != noise_only —
    see a1_env_cfg.py."""
    asset: Articulation = env.scene[asset_cfg.name]
    trunk_id = asset.find_bodies("trunk")[0][0]
    # root_physx_view getters return CPU tensors regardless of sim device
    # (confirmed live against installed isaaclab 0.48.0) — every other
    # extrinsics term is CUDA, and ObservationManager.compute_group's
    # torch.cat over the whole privileged group fails across mixed devices.
    mass = asset.root_physx_view.get_masses()[:, trunk_id : trunk_id + 1].to(asset.device)
    com = asset.root_physx_view.get_coms()[:, trunk_id, :3].to(asset.device)
    return torch.cat([mass, com], dim=-1)


def friction_extrinsic(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    # get_material_properties() is CPU-backed, same caveat as get_masses/get_coms above.
    materials = asset.root_physx_view.get_material_properties().to(asset.device)
    return materials[:, 0, 0:1]  # static friction of the first shape, as a per-env scalar


def motor_power_extrinsic(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Kp (stiffness) and Kd (damping), matching randomize_motor_power's
    randomization of both -- an earlier version only observed stiffness,
    leaving Kd randomized but never fed back to the encoder."""
    actuator = next(iter(env.scene[asset_cfg.name].actuators.values()))
    return torch.stack([actuator.stiffness.mean(dim=-1), actuator.damping.mean(dim=-1)], dim=-1)


def leg_length_extrinsic(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reads back the legScale custom attribute Task 5's generator script
    wrote onto each spawned variant's root prim. InteractiveScene has no
    .stage attribute (confirmed against installed isaaclab 0.48.0) — the
    stage lives on the SimulationContext instead.

    legScale is fixed at spawn time and never changes for an env's whole
    lifetime (leg length isn't reset-randomized, only variant-selected at
    spawn — see a1_env_cfg.py), so the per-env USD prim lookup below only
    needs to run once per env instance, not once per observation step: at
    this repo's configured default of 4096 envs, re-reading on every call
    would be 4096 USD attribute reads every single step for a value that
    never changes. Cached on the env object itself, the same pattern
    a1_env.py's load_managers() already uses for v_command_buf/
    obstacle_ahead_buf (per-env-instance state with no other natural home)."""
    if not hasattr(env, "_leg_scale_cache"):
        asset: Articulation = env.scene[asset_cfg.name]
        scales = []
        for i in range(env.num_envs):
            prim = asset._root_physx_view.prim_paths[i]  # noqa: SLF001 — no public per-env prim accessor
            attr = env.sim.stage.GetPrimAtPath(prim).GetAttribute("legScale")
            scales.append(attr.Get() if attr.IsValid() else 1.0)
        env._leg_scale_cache = torch.tensor(scales, device=asset.device).unsqueeze(-1)
    return env._leg_scale_cache


def joint_range_extrinsic(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    limits = asset.data.joint_pos_limits
    ranges = (limits[..., 1] - limits[..., 0]).mean(dim=-1, keepdim=True)
    default_ranges = (asset.data.default_joint_pos_limits[..., 1] - asset.data.default_joint_pos_limits[..., 0]).mean(dim=-1, keepdim=True)
    return ranges / default_ranges  # current/default ratio — this env's scale factor


def local_terrain_height(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terrain height needs no new randomization — it's already
    effectively randomized by which sub-terrain cell a lane spawns on
    (A1_ROUGH_TERRAINS_CFG). Reads the terrain's own tracked level per env."""
    terrain = env.scene.terrain
    levels = terrain.terrain_levels.float() if hasattr(terrain, "terrain_levels") else torch.zeros(env.num_envs, device=env.device)
    return levels.unsqueeze(-1)
