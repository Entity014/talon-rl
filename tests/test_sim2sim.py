import math

import mujoco
import numpy as np
import torch

from talon_rl.config import ActionSpaceCfg, ObservationSpaceCfg, RewardVectorCfg

from rl.core.modules import ActorCritic
from rl.core.runtime.sim2sim import (
    ISAAC_TO_MUJOCO_PERM,
    MUJOCO_TO_ISAAC_PERM,
    _ISAAC_LAB_JOINT_ORDER,
    _MUJOCO_JOINT_ORDER,
    build_a1_actor_obs,
    get_a1_foot_contacts,
    quat_rotate_inverse_wxyz,
    quat_to_roll_pitch,
    rollout,
)
from rl.core.runtime.exporter import export_policy_as_jit

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


def test_quat_rotate_inverse_identity_quaternion_leaves_vector_unchanged():
    world_gravity = np.array([0.0, 0.0, -1.0])
    body_gravity = quat_rotate_inverse_wxyz(np.array([1.0, 0.0, 0.0, 0.0]), world_gravity)
    assert np.allclose(body_gravity, world_gravity, atol=1e-6)


def test_quat_rotate_inverse_90deg_roll_rotates_gravity_into_y():
    """Matches test_quat_to_roll_pitch_90deg_roll's same quaternion — after a
    90deg roll (rotation about x), a robot lying on its side should see
    world-down gravity appear along its body y-axis, not z, in
    projected_gravity_b (added 2026-09-18, see ObservationSpaceCfg)."""
    half = math.pi / 4
    quat = np.array([math.cos(half), math.sin(half), 0.0, 0.0])
    body_gravity = quat_rotate_inverse_wxyz(quat, np.array([0.0, 0.0, -1.0]))
    assert np.allclose(body_gravity, [0.0, -1.0, 0.0], atol=1e-5)


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
    preference = np.full(obs_cfg.preference_dim, 1.0 / obs_cfg.preference_dim, dtype=np.float32)

    obs = build_a1_actor_obs(model, data, prev_action, command, preference)

    assert obs.shape == (obs_cfg.total_dim,)  # num_policy_stacks=1 case


def test_joint_order_permutations_are_self_consistent_inverses():
    """MUJOCO_TO_ISAAC_PERM and ISAAC_TO_MUJOCO_PERM must round-trip: taking
    a MuJoCo-order array to Isaac Lab order and back must recover the
    original, and vice versa -- catches a typo'd permutation before it
    ever touches real joint data."""
    mujoco_order_array = np.arange(12)
    isaac_order = mujoco_order_array[MUJOCO_TO_ISAAC_PERM]
    back_to_mujoco_order = isaac_order[ISAAC_TO_MUJOCO_PERM]
    np.testing.assert_array_equal(back_to_mujoco_order, mujoco_order_array)


def test_joint_order_permutation_matches_confirmed_live_isaac_lab_order():
    """Found 2026-09-18: this module assumed MuJoCo's leg-grouped actuator
    order (FR_hip, FR_thigh, FR_calf, FL_*, RR_*, RL_*) matched Isaac Lab's
    own action order -- queried live against a real IsaacLabTalonEnv
    (env.action_manager.get_term("joint_pos")._joint_names) and found Isaac
    Lab actually uses a TYPE-grouped order instead ([FL_hip, FR_hip,
    RL_hip, RR_hip, FL_thigh, ..., FL_calf, ...]). This locks in that
    confirmed order as a regression test -- if talon-rl's Isaac Lab config
    ever changes joint_names ordering, this must be re-verified live, not
    just have this constant edited to match."""
    assert _ISAAC_LAB_JOINT_ORDER == (
        "FL_hip", "FR_hip", "RL_hip", "RR_hip",
        "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh",
        "FL_calf", "FR_calf", "RL_calf", "RR_calf",
    )


def test_build_a1_actor_obs_reorders_mujoco_joint_state_into_isaac_lab_order():
    """Sets each MuJoCo joint to a distinct, identifiable angle (its
    _MUJOCO_JOINT_ORDER index) and checks build_a1_actor_obs's joint_pos
    slice comes out permuted into _ISAAC_LAB_JOINT_ORDER, not left in raw
    MuJoCo order -- a regression test for the 2026-09-18 joint-order fix
    that would fail immediately (silently sending wrong angles for wrong
    joints) if the MUJOCO_TO_ISAAC_PERM application were ever reverted."""
    model = mujoco.MjModel.from_xml_path(A1_SCENE_XML)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    # qpos[7:19] is the 12 leg joints in MuJoCo order -- give each a unique,
    # identifiable value equal to its MuJoCo-order index.
    data.qpos[7:19] = np.arange(12, dtype=np.float64)
    mujoco.mj_forward(model, data)

    obs_cfg = ObservationSpaceCfg()
    prev_action = np.zeros(12, dtype=np.float32)
    command = np.zeros(3, dtype=np.float32)
    preference = np.full(obs_cfg.preference_dim, 1.0 / obs_cfg.preference_dim, dtype=np.float32)
    obs = build_a1_actor_obs(model, data, prev_action, command, preference)

    joint_pos_in_obs = obs[:12]
    expected = np.arange(12, dtype=np.float32)[MUJOCO_TO_ISAAC_PERM]
    np.testing.assert_array_equal(joint_pos_in_obs, expected)
    # Concretely: MuJoCo's joint 0 (FR_hip) must land at Isaac Lab's
    # obs index 1 (FR_hip is _ISAAC_LAB_JOINT_ORDER[1]), not index 0.
    assert joint_pos_in_obs[1] == 0.0
    assert joint_pos_in_obs[0] != 0.0


def test_mujoco_position_actuators_move_every_joint_in_the_commanded_sign():
    """Found 2026-09-18: a live 20-step Isaac Lab test (one joint held at
    action=+1.0) showed a NEGATIVE joint_pos delta, initially suspicious of
    an inverted sign convention -- but that test wasn't isolated (whole-body
    gravity/contact dynamics over only 20 steps confound a single actuator's
    own reading). This is the isolated version, MuJoCo-only: the free joint
    (base) is pinned rigid every physics step so ONLY the one actuator being
    tested can move anything, each of the 12 position actuators gets +0.3
    added to its home-pose target and 300 steps to settle, one at a time.
    Every joint's qpos delta comes out positive for a positive ctrl delta --
    confirms MuJoCo's own actuator sign convention is sane and consistent
    across all 12 joints, isolating the earlier live-test result down to a
    whole-body-dynamics artifact, not a real inverted joint."""
    model = mujoco.MjModel.from_xml_path(A1_SCENE_XML)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    base_qpos = data.qpos[:7].copy()
    default_ctrl = data.qpos[7:19].copy()

    for i in range(model.nu):
        mujoco.mj_resetDataKeyframe(model, data, 0)
        data.ctrl[:] = default_ctrl
        data.ctrl[i] = default_ctrl[i] + 0.3
        for _ in range(300):
            data.qpos[:7] = base_qpos
            data.qvel[:6] = 0.0
            mujoco.mj_step(model, data)
        delta = data.qpos[7 + i] - default_ctrl[i]
        assert delta > 0, f"actuator {i} ({_MUJOCO_JOINT_ORDER[i]}) moved the wrong direction: delta={delta}"


def test_rollout_writes_ctrl_in_mujoco_order_from_an_isaac_lab_order_action():
    """A policy that always outputs a fixed, per-dimension-distinct action
    (Isaac Lab order) must drive data.ctrl reordered into MuJoCo order --
    checks the OTHER direction of the same 2026-09-18 fix (build_a1_actor_obs
    covers the observation side, this covers the action side)."""
    model = mujoco.MjModel.from_xml_path(A1_SCENE_XML)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    class _FixedActionPolicy:
        def __call__(self, obs_t):
            # Isaac-Lab-order action, one distinct value per dimension.
            return torch.arange(12, dtype=torch.float32).unsqueeze(0)

    rollout(
        _FixedActionPolicy(), model, data, steps=1,
        command=np.zeros(3, dtype=np.float32),
        preference=np.full(5, 0.2, dtype=np.float32),
    )
    expected_ctrl = np.arange(12, dtype=np.float32)[ISAAC_TO_MUJOCO_PERM]
    np.testing.assert_array_equal(np.asarray(data.ctrl), expected_ctrl)


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
        hidden_dims=[16, 16],
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
        preference=np.full(obs_cfg.preference_dim, 1.0 / obs_cfg.preference_dim, dtype=np.float32),
    )

    assert stats["all_finite"]
    assert stats["min_height"] < stats["max_height"] or stats["min_height"] == stats["max_height"]
