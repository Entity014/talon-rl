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

Unverified assumptions (this machine has no Isaac Sim to cross-check
against — flag and confirm before trusting sim2sim results for real):
- Joint order: MuJoCo's a1.xml actuator order (FR_hip, FR_thigh, FR_calf,
  FL_*, RR_*, RL_*) is ASSUMED to match Isaac Lab's UNITREE_A1_CFG /
  talon_rl.config.ActionSpaceCfg's 12-dim action order. If Isaac Lab uses a
  different joint ordering, actions will drive the wrong joints.
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

_LEG_CALF_BODIES = ("FR_calf", "FL_calf", "RR_calf", "RL_calf")


def quat_to_roll_pitch(quat_wxyz: np.ndarray) -> tuple[float, float]:
    """MuJoCo quaternion convention is [w, x, y, z]. Standard roll/pitch
    (radians) from a unit quaternion; yaw is not part of
    ObservationSpaceCfg.roll_pitch_dim so it's not computed here."""
    w, x, y, z = quat_wxyz
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    return float(roll), float(pitch)


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
    """Builds the (50,) actor_obs_w vector — ObservationSpaceCfg's field
    order (joint_pos, joint_vel, roll_pitch, foot_contact, prev_action,
    command), then the preference vector appended, exactly matching
    MOPPOTrainer._collect_rollout's `np.concatenate([stack.policy_obs, w])`
    composition with num_policy_stacks=1 (see module docstring)."""
    joint_pos = data.qpos[7:19].astype(np.float32)  # skip the 7-dim free joint (pos+quat)
    joint_vel = data.qvel[6:18].astype(np.float32)  # skip the 6-dim free joint (linvel+angvel)
    roll, pitch = quat_to_roll_pitch(data.qpos[3:7])
    roll_pitch = np.array([roll, pitch], dtype=np.float32)
    foot_contact = get_a1_foot_contacts(model, data)

    obs = np.concatenate([joint_pos, joint_vel, roll_pitch, foot_contact, prev_action, command])
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
        data.ctrl[:] = action
        mujoco.mj_step(model, data)
        prev_action = action
        heights.append(float(data.qpos[2]))

    return {
        "all_finite": bool(np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))),
        "final_height": heights[-1],
        "min_height": min(heights),
        "max_height": max(heights),
    }
