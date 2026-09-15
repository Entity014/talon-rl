import numpy as np

from talon_rl.config import ActionSpaceCfg, ObservationSpaceCfg, PreferenceCfg, RewardVectorCfg

from rl.core.algorithms import MOPPOConfig, MOPPOTrainer
from rl.core.dummy_env import DummyTalonEnv


def test_trainer_act_inference_is_deterministic_and_correctly_shaped():
    """play.py's inference loop depends on this: same internal state must
    give the same action every call (unlike update()'s stochastic act())."""
    obs_cfg = ObservationSpaceCfg()
    action_cfg = ActionSpaceCfg()
    reward_cfg = RewardVectorCfg()
    pref_cfg = PreferenceCfg()

    env = DummyTalonEnv(obs_cfg, action_cfg, num_envs=4, horizon=40, seed=0)
    trainer = MOPPOTrainer(
        env, obs_cfg, reward_cfg, pref_cfg, moppo_cfg=MOPPOConfig(num_steps=5, epochs_per_update=1), seed=0
    )

    action1 = trainer.act_inference()
    action2 = trainer.act_inference()

    assert action1.shape == (4, action_cfg.dim)
    assert np.array_equal(action1, action2)
    assert np.all(np.isfinite(action1))


def test_trainer_act_inference_usable_in_an_env_step_loop():
    """The actual play.py pattern: act_inference() -> env.step() -> push
    onto stack -> repeat, without ever calling the stochastic update()."""
    obs_cfg = ObservationSpaceCfg()
    action_cfg = ActionSpaceCfg()
    reward_cfg = RewardVectorCfg()
    pref_cfg = PreferenceCfg()

    env = DummyTalonEnv(obs_cfg, action_cfg, num_envs=2, horizon=40, seed=0)
    trainer = MOPPOTrainer(
        env, obs_cfg, reward_cfg, pref_cfg, moppo_cfg=MOPPOConfig(num_steps=5, epochs_per_update=1), seed=0
    )

    for _ in range(10):
        action = trainer.act_inference()
        transition, done = env.step(action)
        trainer.stack.push(transition["obs"], done_mask=done)
        assert np.all(np.isfinite(transition["obs"]))
