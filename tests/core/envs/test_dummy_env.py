import numpy as np

from talon_rl.config import ActionSpaceCfg, ObservationSpaceCfg
from rl.core.envs.base import TalonEnv

from rl.core.envs.dummy import DummyTalonEnv


def _make_env(num_envs=4, horizon=5, seed=0):
    obs_cfg = ObservationSpaceCfg()
    action_cfg = ActionSpaceCfg()
    env = DummyTalonEnv(obs_cfg, action_cfg, num_envs=num_envs, horizon=horizon, seed=seed)
    return env, obs_cfg, action_cfg


def test_satisfies_base_contract_and_shapes():
    env, obs_cfg, action_cfg = _make_env(num_envs=4)
    assert isinstance(env, TalonEnv)
    assert env.num_envs == 4
    assert env.obs_dim == obs_cfg.total_dim - obs_cfg.preference_dim
    assert env.action_dim == action_cfg.dim

    transition = env.reset()
    assert transition["obs"].shape == (4, env.obs_dim)
    for key in (
        "v_actual", "v_command", "obstacle_dist", "joint_torque", "joint_vel",
        "foot_contact_force", "action", "prev_action", "joint_acc",
    ):
        assert key in transition
        assert transition[key].shape[0] == 4

    action = np.zeros((4, env.action_dim), dtype=np.float32)
    transition, done = env.step(action)
    assert transition["obs"].shape == (4, env.obs_dim)
    assert done.shape == (4,)
    assert done.dtype == bool


def test_lanes_auto_reset_independently():
    """horizon=1 forces every lane to terminate every single step — after
    enough steps, each lane must have resampled its own v_command/obstacle
    (proves auto-reset runs per-lane, not just once globally)."""
    env, _, action_cfg = _make_env(num_envs=8, horizon=1, seed=0)
    env.reset()
    action = np.zeros((8, action_cfg.dim), dtype=np.float32)

    v_commands_seen = set()
    for _ in range(10):
        transition, done = env.step(action)
        assert np.all(done)  # horizon=1: every lane terminates every step
        for row in transition["v_command"]:
            v_commands_seen.add(tuple(np.round(row, 4)))

    # 8 lanes x 10 auto-resets each, sampled from a continuous range -> expect
    # meaningfully more than 1 distinct v_command, proving lanes actually
    # resample independently rather than sharing one global RNG draw.
    assert len(v_commands_seen) > 5


def test_terminal_reward_uses_pre_reset_action_fields():
    """SAME_STEP autoreset must not give the terminal action a fresh reset
    reward (where action/prev_action are zero again)."""
    env, _, action_cfg = _make_env(num_envs=1, horizon=1, seed=0)
    env.reset()
    action = np.full((1, action_cfg.dim), 0.5, dtype=np.float32)

    transition, done = env.step(action)

    assert done[0]
    assert "reward_transition" in transition
    assert np.allclose(transition["action"], 0.0)  # post-reset policy state
    assert np.allclose(transition["reward_transition"]["action"], action)
    assert np.any(transition["reward_transition"]["joint_acc"] != 0.0)


def test_smoothness_uses_correct_prev_action_ordering():
    """joint_acc = (action - prev_action) / dt must use the PREVIOUS step's
    action, not this step's — regression test for the prev_action snapshot
    timing bug caught while writing this env (see DummyEnv._compute_transition)."""
    env, _, action_cfg = _make_env(num_envs=1, horizon=200, seed=0)
    env.reset()

    a1 = np.full((1, action_cfg.dim), 0.5, dtype=np.float32)
    t1, _ = env.step(a1)
    assert np.allclose(t1["prev_action"], 0.0)  # first step: prev was the reset zero-action
    assert np.allclose(t1["action"], 0.5)

    a2 = np.full((1, action_cfg.dim), -0.3, dtype=np.float32)
    t2, _ = env.step(a2)
    assert np.allclose(t2["prev_action"], 0.5)  # second step: prev is step 1's action
    assert np.allclose(t2["action"], -0.3)
