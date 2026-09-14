"""Structural contract test for IsaacLabTalonEnv — skips entirely on a venv
without Isaac Sim installed (e.g. the repo's default 3.12 .venv). The real
proof this env works is the manual GPU smoke run documented in README, not
this test — see docs/superpowers/specs/2026-09-13-isaac-lab-env-setup-design.md.
"""

import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

import pytest

pytest.importorskip("isaacsim")

from talon_rl.config import ActionSpaceCfg, ObservationSpaceCfg
from talon_rl.envs.base_env import BaseTalonEnv
from talon_rl.envs.isaac_lab_env import IsaacLabTalonEnv


def test_isaac_lab_env_implements_base_contract():
    obs_cfg = ObservationSpaceCfg()
    action_cfg = ActionSpaceCfg()
    env = IsaacLabTalonEnv(obs_cfg, action_cfg, horizon=10, headless=True)

    assert isinstance(env, BaseTalonEnv)
    assert env.num_envs == 1
    assert env.obs_dim == obs_cfg.total_dim - obs_cfg.preference_dim
    assert env.action_dim == action_cfg.dim

    transition = env.reset()
    assert transition["obs"].shape == (env.obs_dim,)
    for key in (
        "v_actual", "v_command", "obstacle_dist", "joint_torque", "joint_vel",
        "foot_contact_force", "action", "prev_action", "joint_acc",
    ):
        assert key in transition

    import numpy as np
    action = np.zeros(env.action_dim, dtype=np.float32)
    transition, done = env.step(action)
    assert transition["obs"].shape == (env.obs_dim,)
    assert isinstance(done, bool)

    env.close()
