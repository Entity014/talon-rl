"""Custom domain-randomization event functions this A1 task needs beyond
Isaac Lab's stock isaaclab.envs.mdp.events — see
docs/superpowers/specs/2026-09-15-adaptation-module-phase1-design.md.
"""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils


def randomize_joint_range(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    scale_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Scales each env's joint position limits by a per-env factor around
    the default limits' midpoint (extrinsics factor: joint angle range,
    chapter3.tex §3.2.1). Written via Articulation.write_joint_position_
    limit_to_sim — a plain property write on the already-spawned
    Articulation, unlike leg-length (a geometry change Isaac Lab blocks
    for Articulations at runtime — see the spec)."""
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    default_limits = asset.data.default_joint_pos_limits[env_ids]  # (E, J, 2)
    mean = default_limits.mean(dim=-1)  # (E, J)
    half_range = (default_limits[..., 1] - default_limits[..., 0]) / 2  # (E, J)

    scale = math_utils.sample_uniform(
        scale_range[0], scale_range[1], (len(env_ids), 1), device=asset.device
    )  # (E, 1), broadcasts over J

    new_half_range = half_range * scale
    new_limits = torch.stack([mean - new_half_range, mean + new_half_range], dim=-1)  # (E, J, 2)

    asset.write_joint_position_limit_to_sim(new_limits, env_ids=env_ids)


def randomize_passive_joint(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    blend_range: tuple[float, float] = (0.0, 1.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Passive joint damping and armature, sampled on the same one-parameter
    family as the Phase-5 plant ensemble (wrappers/plant_ensemble.py): blend
    lp gives hip damping lp, thigh/calf damping 2*lp and armature 0.01*lp.
    Same family, so e_t's joint_damping channel means the same thing whether
    this event or the ensemble set it. Writes the PhysX view (CPU-backed)
    directly, like the ensemble does."""
    asset: Articulation = env.scene[asset_cfg.name]
    view = asset.root_physx_view
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs)
    ids = env_ids.cpu()
    names = asset.data.joint_names
    hips = [i for i, n in enumerate(names) if "_hip_joint" in n]
    flex = [i for i, n in enumerate(names) if "_thigh_joint" in n or "_calf_joint" in n]
    lp = math_utils.sample_uniform(blend_range[0], blend_range[1], (len(ids), 1), device="cpu")
    damp = view.get_dof_dampings()
    arm = view.get_dof_armatures()
    damp[ids.unsqueeze(-1), torch.tensor(hips)] = lp
    damp[ids.unsqueeze(-1), torch.tensor(flex)] = 2.0 * lp
    arm[ids] = 0.01 * lp
    view.set_dof_dampings(damp, indices=ids.to(torch.int32))
    view.set_dof_armatures(arm, indices=ids.to(torch.int32))


def randomize_velocity_command(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    lin_vel_x_range: tuple[float, float],
    lin_vel_y_range: tuple[float, float],
    ang_vel_z_range: tuple[float, float],
):
    """Resamples this task's scripted v_command_buf (v_x, v_y, omega_z) on
    reset. Without this the command was fixed at [0.5, 0, 0] for the whole
    run, so progress_reward's tracking term (chapter3.tex's Progress) could
    never actually reward following a lateral or turning command -- see
    a1_env.py's load_managers() for where v_command_buf is first allocated
    (this event only resamples it, mode="reset")."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n = len(env_ids)
    if getattr(env.cfg, "g1_command_exposure", False):
        # G1 owns command sampling/reset in the environment reset hook, before
        # ManagerBasedEnv computes the post-reset observation.
        return
    if getattr(env.cfg, "stand_phase_s", 0.0) > 0.0 and env.common_step_counter * env.step_dt < env.cfg.stand_phase_s:
        env.v_command_buf[env_ids] = 0.0
        return
    vx = math_utils.sample_uniform(*lin_vel_x_range, (n,), device=env.device)
    vy = math_utils.sample_uniform(*lin_vel_y_range, (n,), device=env.device)
    wz = math_utils.sample_uniform(*ang_vel_z_range, (n,), device=env.device)
    env.v_command_buf[env_ids] = torch.stack([vx, vy, wz], dim=-1)
