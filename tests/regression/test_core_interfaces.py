import numpy as np
import torch

from talon_rl.config import PreferenceCfg, RewardVectorCfg
from rl.core.objectives.losses import d3po_actor_loss
from rl.core.normalization import RunningNormalizer
from rl.core.objectives import D3POObjective
from rl.core.preferences.functional import floor_clip_terms, sample_preference_vector
from rl.core.preferences import DirichletPreferenceSampler, FloorPreferenceTransform
from rl.core.rollout import PerObjectiveGAE
from rl.core.rollout.gae_functional import gae_per_objective


def test_dirichlet_sampler_matches_existing_function():
    reward_cfg = RewardVectorCfg()
    pref_cfg = PreferenceCfg()
    seed = 17
    sampler = DirichletPreferenceSampler(reward_cfg, pref_cfg, seed=seed)
    expected = sample_preference_vector(
        np.random.default_rng(seed), reward_cfg, pref_cfg, 8, step=3
    )
    np.testing.assert_array_equal(sampler.sample(8, step=3), expected)


def test_floor_transform_matches_existing_function():
    names = ("progress", "balance", "energy")
    floors = {"balance": 0.2}
    w = np.array([[0.9, 0.05, 0.05]], dtype=np.float32)
    transform = FloorPreferenceTransform(names, floors)
    np.testing.assert_array_equal(transform.transform(w), floor_clip_terms(w, names, floors))


def test_per_objective_gae_matches_existing_function():
    rewards = np.arange(12, dtype=np.float32).reshape(2, 2, 3) / 10
    values = np.zeros((3, 2, 3), dtype=np.float32)
    dones = np.array([[False, False], [True, False]])
    estimator = PerObjectiveGAE(gamma=0.99, lam=0.95)
    adv, returns = estimator.compute(rewards, values, dones)
    expected = gae_per_objective(rewards, values, dones, 0.99, 0.95)
    np.testing.assert_allclose(adv, expected)
    np.testing.assert_allclose(returns, expected + values[:-1])


def test_running_normalizer_round_trip_state():
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    normalizer = RunningNormalizer(2, center=True)
    normalizer.update(x)
    y = normalizer.transform(x)
    restored = RunningNormalizer(2, center=True)
    restored.load_state_dict(normalizer.state_dict())
    np.testing.assert_allclose(restored.transform(x), y)


def test_d3po_objective_delegates_to_existing_loss():
    ratio = torch.tensor([0.9, 1.1])
    adv = torch.tensor([[1.0, -0.5], [0.2, 0.8]])
    w = torch.tensor([[0.7, 0.3], [0.4, 0.6]])
    batch = {"ratio": ratio, "advantages": adv, "preferences": w}
    expected = d3po_actor_loss(ratio, adv, w, 0.2)
    actual = D3POObjective(clip_eps=0.2).loss(batch)
    torch.testing.assert_close(actual, expected)
