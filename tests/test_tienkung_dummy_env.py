import numpy as np

from talon_rl.envs.base_env import BaseTalonEnv
from talon_rl.tasks.manipulation.tienkung_env.config import ActionSpaceCfg, ObservationSpaceCfg
from talon_rl.tasks.manipulation.tienkung_env.dummy_env import DummyTalonEnv


def _make_env(num_envs=4, horizon=5, seed=0):
    obs_cfg = ObservationSpaceCfg()
    action_cfg = ActionSpaceCfg()
    env = DummyTalonEnv(obs_cfg, action_cfg, num_envs=num_envs, horizon=horizon, seed=seed)
    return env, obs_cfg, action_cfg


def test_satisfies_base_contract_and_shapes():
    env, obs_cfg, action_cfg = _make_env(num_envs=4)
    assert isinstance(env, BaseTalonEnv)
    assert env.num_envs == 4
    assert env.obs_dim == obs_cfg.total_dim - obs_cfg.preference_dim
    assert env.action_dim == action_cfg.dim

    transition = env.reset()
    assert transition["obs"].shape == (4, env.obs_dim)
    for key in (
        "box_height", "target_height", "obstacle_dist", "joint_torque", "joint_vel",
        "joint_acc", "arm_contact_force", "action", "prev_action",
    ):
        assert key in transition
        assert transition[key].shape[0] == 4

    action = np.zeros((4, env.action_dim), dtype=np.float32)
    transition, done = env.step(action)
    assert transition["obs"].shape == (4, env.obs_dim)
    assert done.shape == (4,)
    assert done.dtype == bool


def test_lanes_auto_reset_independently():
    """horizon=1 forces every lane to truncate every single step — after
    enough steps, each lane must have resampled its own obstacle_dist
    (proves auto-reset runs per-lane, not just once globally)."""
    env, _, action_cfg = _make_env(num_envs=8, horizon=1, seed=0)
    env.reset()
    action = np.zeros((8, action_cfg.dim), dtype=np.float32)

    obstacle_dists_seen = set()
    for _ in range(10):
        transition, done = env.step(action)
        assert np.all(done)  # horizon=1: every lane truncates every step
        for val in transition["obstacle_dist"]:
            obstacle_dists_seen.add(round(float(val), 4))

    assert len(obstacle_dists_seen) > 5


def test_smoothness_uses_correct_prev_action_ordering():
    """joint_acc = (action - prev_action) / dt must use the PREVIOUS
    step's action, not this step's — same regression concern the A1's
    DummyEnv guards against."""
    env, _, action_cfg = _make_env(num_envs=1, horizon=200, seed=0)
    env.reset()

    a1 = np.full((1, action_cfg.dim), 0.5, dtype=np.float32)
    t1, _ = env.step(a1)
    assert np.allclose(t1["prev_action"], 0.0)
    assert np.allclose(t1["action"], 0.5)

    a2 = np.full((1, action_cfg.dim), -0.3, dtype=np.float32)
    t2, _ = env.step(a2)
    assert np.allclose(t2["prev_action"], 0.5)
    assert np.allclose(t2["action"], -0.3)


def test_positive_action_0_raises_box_height_toward_target():
    env, _, action_cfg = _make_env(num_envs=1, horizon=200, seed=0)
    t0 = env.reset()
    action = np.zeros((1, action_cfg.dim), dtype=np.float32)
    action[0, 0] = 1.0  # max lift drive
    t1, _ = env.step(action)
    assert t1["box_height"][0] > t0["box_height"][0]


def test_action_1_drives_arm_contact_force():
    env, _, action_cfg = _make_env(num_envs=1, horizon=200, seed=0)
    env.reset()
    gentle = np.zeros((1, action_cfg.dim), dtype=np.float32)
    firm = np.zeros((1, action_cfg.dim), dtype=np.float32)
    firm[0, 1] = 1.0
    t_gentle, _ = env.step(gentle)
    env.reset()
    t_firm, _ = env.step(firm)
    assert np.all(t_firm["arm_contact_force"] > t_gentle["arm_contact_force"])
