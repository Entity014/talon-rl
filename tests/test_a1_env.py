# tests/test_a1_env.py
"""Structural contract test for IsaacLabTalonEnv — skips entirely on a venv
without Isaac Sim installed (e.g. the repo's default 3.12 .venv). The real
proof this env works at scale is the manual GPU smoke run documented in
README, not this test.
"""

import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

import pytest

pytest.importorskip("isaacsim")

import numpy as np

from talon_rl.config import ActionSpaceCfg, ObservationSpaceCfg
from talon_rl.envs.base_env import BaseTalonEnv


def test_isaac_lab_env_implements_base_contract():
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        import gymnasium as gym
        import talon_rl.tasks.locomotion.a1_env  # noqa: F401 — registers Isaac-Talon-A1-v0

        from talon_rl.tasks.locomotion.a1_env.a1_env_cfg import IsaacLabTalonEnvCfg

        cfg = IsaacLabTalonEnvCfg()
        cfg.scene.num_envs = 4  # small N for a fast structural check
        env = gym.make("Isaac-Talon-A1-v0", cfg=cfg).unwrapped

        obs_cfg = ObservationSpaceCfg()
        action_cfg = ActionSpaceCfg()
        assert isinstance(env, BaseTalonEnv)
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
        assert np.all(np.isfinite(transition["obs"]))
        assert np.all(np.isfinite(transition["reward_vec"]))
    finally:
        import threading
        watchdog = threading.Timer(15.0, lambda: os._exit(0))
        watchdog.daemon = True
        watchdog.start()
        app.close()
        watchdog.cancel()
