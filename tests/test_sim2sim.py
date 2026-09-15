import math

import mujoco
import numpy as np
import torch

from talon_rl.config import ActionSpaceCfg, ObservationSpaceCfg, RewardVectorCfg

from rl.core.modules import ActorCritic
from rl.core.sim2sim import build_a1_actor_obs, get_a1_foot_contacts, quat_to_roll_pitch, rollout
from rl.core.wrapper import export_policy_as_jit

A1_SCENE_XML = "talon_rl/assets/data/Robots/unitree_a1/mujoco/scene.xml"


def test_quat_to_roll_pitch_identity_quaternion_is_zero():
    roll, pitch = quat_to_roll_pitch(np.array([1.0, 0.0, 0.0, 0.0]))
    assert math.isclose(roll, 0.0, abs_tol=1e-6)
    assert math.isclose(pitch, 0.0, abs_tol=1e-6)


def test_quat_to_roll_pitch_90deg_roll():
    # 90 deg rotation about x-axis: w=cos(45deg), x=sin(45deg)
    half = math.pi / 4
    quat = np.array([math.cos(half), math.sin(half), 0.0, 0.0])
    roll, pitch = quat_to_roll_pitch(quat)
    assert math.isclose(roll, math.pi / 2, abs_tol=1e-5)
    assert math.isclose(pitch, 0.0, abs_tol=1e-5)


def test_get_a1_foot_contacts_all_zero_when_standing_in_air():
    """With the robot spawned well above the ground plane, no contacts
    should register — every leg's flag stays 0."""
    model = mujoco.MjModel.from_xml_path(A1_SCENE_XML)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)  # the "home" keyframe
    data.qpos[2] = 5.0  # lift well clear of the ground plane
    mujoco.mj_forward(model, data)

    contacts = get_a1_foot_contacts(model, data)
    assert contacts.shape == (4,)
    assert np.all(contacts == 0.0)


def test_build_a1_actor_obs_matches_observation_space_total_dim():
    model = mujoco.MjModel.from_xml_path(A1_SCENE_XML)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    obs_cfg = ObservationSpaceCfg()
    prev_action = np.zeros(12, dtype=np.float32)
    command = np.array([0.5, 0.0, 0.0], dtype=np.float32)
    preference = np.full(5, 0.2, dtype=np.float32)

    obs = build_a1_actor_obs(model, data, prev_action, command, preference)

    assert obs.shape == (obs_cfg.total_dim,)  # num_policy_stacks=1 case
    assert np.all(np.isfinite(obs))


def test_rollout_runs_without_crashing_or_nans(tmp_path):
    """Mechanism-only smoke test (see sim2sim.py's module docstring) — an
    untrained ActorCritic's weights are meaningless, this only proves the
    export -> MuJoCo -> policy -> ctrl -> mj_step loop runs end to end."""
    obs_cfg = ObservationSpaceCfg()
    action_cfg = ActionSpaceCfg()
    reward_cfg = RewardVectorCfg()

    model_nn = ActorCritic(
        actor_obs_dim=obs_cfg.total_dim,
        critic_obs_dim=obs_cfg.total_dim,
        action_dim=action_cfg.dim,
        reward_dim=reward_cfg.dim,
        hidden_dim=16,
    )
    export_path = str(tmp_path / "policy.pt")
    export_policy_as_jit(model_nn, export_path)
    policy = torch.jit.load(export_path)

    mj_model = mujoco.MjModel.from_xml_path(A1_SCENE_XML)
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)

    stats = rollout(
        policy, mj_model, mj_data, steps=20,
        command=np.array([0.5, 0.0, 0.0], dtype=np.float32),
        preference=np.full(5, 0.2, dtype=np.float32),
    )

    assert stats["all_finite"]
    assert stats["min_height"] < stats["max_height"] or stats["min_height"] == stats["max_height"]
