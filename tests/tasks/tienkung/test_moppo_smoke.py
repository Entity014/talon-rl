import numpy as np

from talon_rl.tasks.manipulation.tienkung_env.config import ActionSpaceCfg, ObservationSpaceCfg, RewardVectorCfg
from talon_rl.tasks.manipulation.tienkung_env.dummy_env import DummyTalonEnv
from talon_rl.tasks.manipulation.tienkung_env.reward import compute_reward_vector


def test_dummy_env_transition_is_accepted_by_reward_vector():
    """Pins the DummyTalonEnv <-> reward.py key contract end-to-end — each
    module's own tests only check their own side with a hand-written dict;
    this is the one place that would have caught a key-name typo or a
    missing RewardVectorCfg field (e.g. impact_floor_eps) before it broke
    at MOPPOTrainer construction time. Mirrors tests/test_moppo_smoke.py's
    role on the A1 side."""
    obs_cfg = ObservationSpaceCfg()
    action_cfg = ActionSpaceCfg()
    reward_cfg = RewardVectorCfg()
    env = DummyTalonEnv(obs_cfg, action_cfg, num_envs=4, horizon=10, seed=0)

    env.reset()
    action = np.random.default_rng(0).uniform(-1.0, 1.0, size=(4, env.action_dim)).astype(np.float32)
    transition, done = env.step(action)

    reward_vec = compute_reward_vector(transition, reward_cfg)
    assert reward_vec.shape == (4, reward_cfg.dim)
    assert np.all(np.isfinite(reward_vec))
