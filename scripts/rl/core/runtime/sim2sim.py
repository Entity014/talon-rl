"""A1 sim2sim mechanism: build talon-rl's ObservationSpaceCfg-shaped
observation from MuJoCo state, and roll a TorchScript-exported policy out
in MuJoCo. See scripts/rl/sim2sim.py for the CLI entry point.

00_Proposal §3.4's Sim-to-Sim Validation (Isaac Gym -> MuJoCo) is [CORE]
scope for this thesis, but the actual *validation* (comparing metrics
against an Isaac Sim rollout) needs a real trained A1 policy to compare
against, which doesn't exist yet (no GPU in this environment, A1 hasn't
been trained for real). This module is scoped to the mechanism only:
proving a policy exported from talon-rl's training pipeline can drive the
vendored A1 MuJoCo model end to end without crashing/NaN-ing, using
whatever policy is available (including a meaningless dummy-env-trained
one) — see scripts/rl/core/wrapper/exporter.py for how a policy gets
exported.

Checked against a live Isaac Sim 2026-09-18 (this machine now has one) --
one assumption below was WRONG and is now fixed; the rest remain open:
- Joint order: this module used to ASSUME MuJoCo's a1.xml actuator order
  (FR_hip, FR_thigh, FR_calf, FL_*, RR_*, RL_*, confirmed correct for
  MuJoCo itself -- data.qpos/data.ctrl both use this leg-grouped order)
  matched Isaac Lab's action order. Queried directly
  (env.action_manager.get_term("joint_pos")._joint_names against a live
  IsaacLabTalonEnv): Isaac Lab's actual order is TYPE-grouped, not
  leg-grouped -- [FL_hip, FR_hip, RL_hip, RR_hip, FL_thigh, FR_thigh,
  RL_thigh, RR_thigh, FL_calf, FR_calf, RL_calf, RR_calf]. A policy
  exported from Isaac Lab training and driven straight into MuJoCo's
  data.ctrl (or fed MuJoCo's data.qpos straight as joint_pos) would send
  every action to the wrong joint and read every joint's own state as a
  different joint's. Fixed via ISAAC_TO_MUJOCO_PERM /
  MUJOCO_TO_ISAAC_PERM below, built from the two joint-name lists rather
  than hardcoded indices, so a real name mismatch fails loudly (.index()
  raises) instead of silently permuting wrong.
- Sign convention: a first live-env attempt (one joint's action held at
  +1.0 for 20 steps, checking data.joint_pos's sign of change) was
  inconclusive -- whole-body contact/gravity dynamics confound an isolated
  actuator reading that short. Re-tested isolated (base pinned rigid every
  step, MuJoCo-only, see test_sim2sim.py's
  test_mujoco_position_actuators_move_every_joint_in_the_commanded_sign):
  all 12 of MuJoCo's own position actuators move their joint in the
  commanded sign, cleanly. This confirms MuJoCo's own convention is sane;
  it does NOT independently re-confirm Isaac Lab's side uses the same
  sign for the same physical direction (both presumably derive joint axes
  from the same source URDF, so they should agree, but that hasn't been
  checked with an equally isolated Isaac Lab test).
- Foot contact: a1.xml has no named foot geoms (only an unnamed `class="foot"`
  default) — approximated here as "does any contact touch this leg's *_calf
  body," not the actual foot geom specifically. Isaac Lab's real
  `contact_sensor` (a1_env_cfg.py) presumably reads named foot bodies
  directly; this is a coarser proxy.
- Only num_policy_stacks=1 is supported (no frame-history replication of
  ObservationStack here) — export/train the policy being tested with
  `ObservationStackCfg(num_policy_stacks=1)` or this obs vector's dim won't
  match what the policy expects.
"""

from __future__ import annotations

import numpy as np
import torch

from talon_rl.deployment.phase1 import CANONICAL_DEFAULT_Q, root_com_velocity_b_from_freejoint

_LEG_CALF_BODIES = ("FR_calf", "FL_calf", "RR_calf", "RL_calf")

# MuJoCo's own order (confirmed live, 2026-09-18: both data.qpos[7:19] and
# data.ctrl follow this -- a1.xml declares/actuates joints leg-by-leg).
_MUJOCO_JOINT_ORDER = (
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
)
# Isaac Lab's order (confirmed live against IsaacLabTalonEnv, 2026-09-18:
# env.action_manager.get_term("joint_pos")._joint_names, same order
# robot.data.joint_pos/joint_names use) -- type-grouped, not leg-grouped.
# This is the order talon-rl's trained policies actually expect/produce.
_ISAAC_LAB_JOINT_ORDER = (
    "FL_hip", "FR_hip", "RL_hip", "RR_hip",
    "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh",
    "FL_calf", "FR_calf", "RL_calf", "RR_calf",
)
# obs[MUJOCO_TO_ISAAC_PERM] reorders a MuJoCo-order array (e.g. data.qpos's
# joint slice) into Isaac Lab order, for building the policy's observation.
MUJOCO_TO_ISAAC_PERM = [_MUJOCO_JOINT_ORDER.index(name) for name in _ISAAC_LAB_JOINT_ORDER]
# action[ISAAC_TO_MUJOCO_PERM] reorders a policy's Isaac-Lab-order action
# into MuJoCo order, for writing to data.ctrl.
ISAAC_TO_MUJOCO_PERM = [_ISAAC_LAB_JOINT_ORDER.index(name) for name in _MUJOCO_JOINT_ORDER]


def quat_to_roll_pitch(quat_wxyz: np.ndarray) -> tuple[float, float]:
    """MuJoCo quaternion convention is [w, x, y, z]. Standard roll/pitch
    (radians) from a unit quaternion; yaw is not part of
    the reward's roll_pitch input so it's not computed here."""
    w, x, y, z = quat_wxyz
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    return float(roll), float(pitch)


def quat_rotate_inverse_wxyz(quat_wxyz: np.ndarray, vec_world: np.ndarray) -> np.ndarray:
    """Rotates a world-frame vector into the body frame given the body's
    world-orientation quaternion (MuJoCo [w, x, y, z] convention) — the
    standard Isaac Gym/Lab `quat_rotate_inverse` formula, used here to build
    projected_gravity_b (added 2026-09-18, see ObservationSpaceCfg) from
    MuJoCo state the same way isaaclab.envs.mdp.projected_gravity derives it
    from Isaac Lab's own root orientation."""
    w, q_vec = float(quat_wxyz[0]), quat_wxyz[1:4].astype(np.float64)
    a = vec_world * (2.0 * w**2 - 1.0)
    b = np.cross(q_vec, vec_world) * w * 2.0
    c = q_vec * (q_vec @ vec_world) * 2.0
    return (a - b + c).astype(np.float32)


def get_a1_foot_contacts(model, data) -> np.ndarray:
    """Binarized (4,) contact flag per leg, order (FR, FL, RR, RL) — matches
    the actuator order in a1.xml. See module docstring's foot-contact
    caveat: approximated via *_calf body contact, not a named foot geom."""
    calf_body_ids = [mj_id(model, name) for name in _LEG_CALF_BODIES]
    contacts = np.zeros(4, dtype=np.float32)
    for i in range(data.ncon):
        contact = data.contact[i]
        body1 = model.geom_bodyid[contact.geom1]
        body2 = model.geom_bodyid[contact.geom2]
        for leg_idx, calf_id in enumerate(calf_body_ids):
            if body1 == calf_id or body2 == calf_id:
                contacts[leg_idx] = 1.0
    return contacts


def mj_id(model, body_name: str) -> int:
    import mujoco
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)


def build_a1_actor_obs(
    model, data, prev_action: np.ndarray, command: np.ndarray, preference: np.ndarray
) -> np.ndarray:
    """Builds the (53,) actor_obs_w vector — the canonical 48-D layout
    (ObservationSpaceCfg field order) with the preference vector appended,
    exactly matching MOPPOTrainer._collect_rollout's
    `np.concatenate([stack.policy_obs, w])` composition with
    num_policy_stacks=1 (see module docstring)."""
    # data.qpos/qvel's joint slice is MuJoCo order (leg-grouped); the
    # policy expects Isaac Lab order (type-grouped) -- see module
    # docstring's 2026-09-18 joint-order finding.
    joint_pos = data.qpos[7:19][MUJOCO_TO_ISAAC_PERM].astype(np.float32) - CANONICAL_DEFAULT_Q  # skip the 7-dim free joint (pos+quat)
    joint_vel = data.qvel[6:18][MUJOCO_TO_ISAAC_PERM].astype(np.float32)  # skip the 6-dim free joint (linvel+angvel)
    quat_wxyz = data.qpos[3:7]
    # Free joint's qvel[3:6] is angular velocity already in the body frame
    # under MuJoCo's convention (unlike qvel[0:3], the linear velocity,
    # which is world-frame) -- matches Isaac Lab's root_ang_vel_b directly.
    base_ang_vel = data.qvel[3:6].astype(np.float32)
    # Isaac's base_lin_vel is the root COM velocity, not the free-joint
    # origin's -- the D3-A interface verdict's correction.
    base_lin_vel = root_com_velocity_b_from_freejoint(
        quat_wxyz, data.qvel[0:3], data.qvel[3:6], model.body_ipos[mj_id(model, "trunk")]
    ).astype(np.float32)
    projected_gravity = quat_rotate_inverse_wxyz(quat_wxyz, np.array([0.0, 0.0, -1.0]))

    obs = np.concatenate([
        base_lin_vel, base_ang_vel, projected_gravity, command,
        joint_pos, joint_vel, prev_action,
    ])
    return np.concatenate([obs, preference]).astype(np.float32)


def rollout(
    policy: torch.jit.ScriptModule,
    model,
    data,
    steps: int,
    command: np.ndarray,
    preference: np.ndarray,
) -> dict:
    """Steps MuJoCo `steps` times, feeding the policy's output straight to
    `data.ctrl` (a1.xml's actuators are `<position>` type — the policy's
    target-joint-angle action IS the ctrl value directly, no separate PD
    conversion needed on this side, matching ActionSpaceCfg's own "target
    joint angle, PD-converted downstream" contract). Returns simple
    finite/height stats proving the loop ran, not a locomotion result."""
    import mujoco

    action_dim = model.nu
    prev_action = np.zeros(action_dim, dtype=np.float32)
    heights = []

    for _ in range(steps):
        obs = build_a1_actor_obs(model, data, prev_action, command, preference)
        with torch.no_grad():
            action = policy(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()
        # action is in Isaac Lab order (what the policy natively produces
        # and what prev_action must stay in, to match next step's
        # build_a1_actor_obs / the policy's own training-time convention);
        # reorder only the copy written to MuJoCo's actuators.
        data.ctrl[:] = action[ISAAC_TO_MUJOCO_PERM]
        mujoco.mj_step(model, data)
        prev_action = action
        heights.append(float(data.qpos[2]))

    return {
        "all_finite": bool(np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))),
        "final_height": heights[-1],
        "min_height": min(heights),
        "max_height": max(heights),
    }
