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

from talon_rl.config import ActionSpaceCfg, ObservationSpaceCfg, RewardVectorCfg
from talon_rl.envs.base_env import BaseTalonEnv
from talon_rl.reward import compute_reward_vector


def test_isaac_lab_env_implements_base_contract():
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    ok = False
    try:
        import gymnasium as gym
        import talon_rl.tasks.locomotion.a1_env  # noqa: F401 — registers Isaac-Talon-A1-v0

        from talon_rl.tasks.locomotion.a1_env.a1_env_cfg import IsaacLabTalonEnvCfg
        from talon_rl.tasks.locomotion.a1_env.terrain_config import A1_ROUGH_TERRAINS_CFG

        cfg = IsaacLabTalonEnvCfg()
        cfg.scene.num_envs = 4  # small N for a fast structural check
        env = gym.make("Isaac-Talon-A1-v0", cfg=cfg).unwrapped

        # terrain: confirm the scene uses the generator config, not a flat
        # ground plane. NOT an `is` identity check (pre-existing bug found
        # while wiring Task 6: Isaac Lab clones/replaces cfg objects on the
        # way into the scene, e.g. IsaacLabTalonEnvCfg.__post_init__ already
        # does this for scene.robot via TALON_A1_CFG.replace(), so the
        # terrain_generator living on the built scene is never the same
        # object as the module-level A1_ROUGH_TERRAINS_CFG — `is` silently
        # failed every run, masked because the finally block's watchdog
        # always exits 0 before pytest can report the AssertionError. Compare
        # the structural fingerprint that actually distinguishes "our rough
        # generator" from a flat plane or a different generator instead.
        terrain_generator = env.scene["terrain"].cfg.terrain_generator
        assert terrain_generator.num_rows == A1_ROUGH_TERRAINS_CFG.num_rows
        assert terrain_generator.num_cols == A1_ROUGH_TERRAINS_CFG.num_cols
        assert set(terrain_generator.sub_terrains) == set(A1_ROUGH_TERRAINS_CFG.sub_terrains)

        # Adaptation Module Phase 1: privileged extrinsics group must exist
        # alongside policy — proves the teacher's 7-factor e_t is actually
        # wired into the ObservationManager, not just defined in cfg.
        assert "privileged" in env.observation_manager.active_terms
        assert "payload" in env.observation_manager.active_terms["privileged"]
        # reset_scene (pre-existing, load-bearing per-episode reset) plus the
        # 5 new DR terms — a regression here means EventCfg's merge (Ruling 1)
        # silently dropped one or the other.
        assert len(env.event_manager.active_terms["reset"]) == 6

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
        from talon_rl.config import ExtrinsicsCfg
        assert "extrinsics" in transition
        assert transition["extrinsics"].shape == (4, ExtrinsicsCfg().dim)

        action = np.zeros((4, env.action_dim), dtype=np.float32)
        transition, done = env.step(action)
        assert transition["obs"].shape == (4, env.obs_dim)
        assert done.shape == (4,)
        assert done.dtype == bool
        assert np.all(np.isfinite(transition["obs"]))
        assert "extrinsics" in transition
        assert transition["extrinsics"].shape == (4, ExtrinsicsCfg().dim)
        reward_vec = compute_reward_vector(transition, RewardVectorCfg())
        assert reward_vec.shape == (4, RewardVectorCfg().dim)
        assert np.all(np.isfinite(reward_vec))
        ok = True
    finally:
        # app.close() reliably hangs past this point in this Isaac Sim
        # install (confirmed independent of this file's own changes — see
        # task-6-report.md), so this watchdog force-exits instead of hanging
        # the whole suite forever. It must NOT always exit 0: that previously
        # masked a real AssertionError (the terrain `is`-identity bug) for
        # multiple prior tasks, since a failure inside try still reaches this
        # finally block. `ok` is only True once every assertion above passed.
        import threading
        watchdog = threading.Timer(15.0, lambda: os._exit(0 if ok else 1))
        watchdog.daemon = True
        watchdog.start()
        app.close()
        watchdog.cancel()
