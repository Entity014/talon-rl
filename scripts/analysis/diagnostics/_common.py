"""Shared setup helpers for the fixed/floating-base static & instantaneous
PD/actuator diagnostics (2D/2E arc, 2026-09-20). Pulled out after several
scripts (static_standing_diagnostic, gravity_actuator_consistency,
instantaneous_equilibrium_check, ground_contact_timestep_check) had grown
near-identical copies of: boot Isaac Lab, build IsaacLabTalonEnvCfg with an
optional Kp override / fixed-root-link / custom sim dt+decimation, gym.make
+ reset, then group the 12 actuated joints into (joint_type, side) buckets
and build a constant-target action array. Kept as plain functions, not a
class -- each diagnostic script runs once and exits, there's no state to
carry between calls that would justify an object.

Do not change this file's behavior without re-running at least one caller
against its own last-known-good log (see each script's own 2D/2E log under
logs/eval_bal/) -- these are the scripts thesis conclusions were drawn from.
"""

from __future__ import annotations

import os

import numpy as np


def make_diagnostic_env(
    num_envs: int,
    seed: int,
    kp: float | None = None,
    fix_root_link: bool = False,
    sim_dt: float = 0.02,
    decimation: int = 1,
):
    """Boots Isaac Lab headless, builds the A1 env with these overrides,
    gym.make()s + reset()s it. Returns (simulation_app, env, kp_value, kd).

    simulation_app must be kept alive by the caller (assign it to a local,
    same as every diagnostic script's own `simulation_app = app_launcher.app
    # noqa: F841` idiom) -- letting it get garbage-collected tears down the
    sim.
    """
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher({"headless": True, "enable_cameras": False})
    simulation_app = app_launcher.app

    import gymnasium as gym
    import talon_rl.tasks.locomotion.a1_env  # noqa: F401
    from talon_rl.tasks.locomotion.a1_env.a1_env_cfg import IsaacLabTalonEnvCfg

    cfg = IsaacLabTalonEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    cfg.sim.dt = sim_dt
    cfg.decimation = decimation
    cfg.sim.render_interval = cfg.decimation
    if fix_root_link:
        cfg.scene.robot.spawn.articulation_props.fix_root_link = True

    base_actuator = cfg.scene.robot.actuators["base_legs"]
    kp_value = kp if kp is not None else base_actuator.stiffness
    kd = base_actuator.damping
    if kp is not None:
        # Replace (not mutate in place) -- mutating the shared DCMotorCfg
        # instance in place would risk touching the module-level
        # TALON_A1_CFG singleton its fields are shallow-copied from.
        scaled_actuator = base_actuator.replace(stiffness=kp)
        cfg.scene.robot.actuators = {**cfg.scene.robot.actuators, "base_legs": scaled_actuator}

    env = gym.make("Isaac-Talon-A1-v0", cfg=cfg, render_mode=None).unwrapped
    env.reset()
    return simulation_app, env, kp_value, kd


def joint_side_groups(joint_names: list[str]) -> dict[tuple[str, str], list[int]]:
    """(joint_type, side) -> indices into a 12-long joint-space array, e.g.
    ("calf", "L") -> [FL_calf index, RL_calf index]."""

    def side_idx(joint_type: str, side: str) -> list[int]:
        legs = ("FL", "RL") if side == "L" else ("FR", "RR")
        return [i for i, n in enumerate(joint_names) if joint_type in n and any(n.startswith(leg) for leg in legs)]

    return {(jt, side): side_idx(jt, side) for jt in ("hip", "thigh", "calf") for side in ("L", "R")}


def build_target_action(action_term, target_pose: np.ndarray, num_envs: int) -> np.ndarray:
    """Converts an absolute joint-target pose into the raw action array
    JointPositionActionCfg expects: action*scale + default_joint_pos = target."""
    default_joint_pos = action_term._offset[0].cpu().numpy()
    action_scale = float(action_term.cfg.scale)
    return ((target_pose - default_joint_pos) / action_scale)[None, :].repeat(num_envs, axis=0).astype(np.float32)
