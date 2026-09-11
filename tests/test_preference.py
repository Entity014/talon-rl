import numpy as np

from talon_rl.config import PreferenceCfg, RewardVectorCfg
from talon_rl.preference import floor_clip, rate_limit, sample_preference_vector


def test_sample_preference_vector_sums_to_one_and_matches_dim():
    rng = np.random.default_rng(0)
    reward_cfg = RewardVectorCfg()
    pref_cfg = PreferenceCfg()
    w = sample_preference_vector(rng, reward_cfg, pref_cfg)
    assert w.shape == (reward_cfg.dim,)
    assert np.isclose(w.sum(), 1.0, atol=1e-5)
    assert np.all(w >= 0.0)


def test_rate_limit_caps_step_size():
    w_prev = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    w_target = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    w_next = rate_limit(w_prev, w_target, max_delta=0.05)
    assert np.linalg.norm(w_next - w_prev) <= 0.05 + 1e-6


def test_rate_limit_passes_through_when_within_cap():
    w_prev = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    w_target = np.array([0.21, 0.2, 0.2, 0.2, 0.19])
    w_next = rate_limit(w_prev, w_target, max_delta=1.0)
    assert np.allclose(w_next, w_target)


def test_floor_clip_enforces_minimum_and_renormalizes():
    term_names = ("progress", "clearance", "energy", "impact", "smoothness")
    w = np.array([0.5, 0.3, 0.2, 0.0, 0.0], dtype=np.float32)  # impact = 0, violates floor
    w_clipped = floor_clip(w, term_names, floor_eps=0.05, floored_term="impact")
    idx = term_names.index("impact")
    assert w_clipped[idx] >= 0.05 - 1e-6
    assert np.isclose(w_clipped.sum(), 1.0, atol=1e-5)


def test_floor_clip_noop_when_already_above_floor():
    term_names = ("progress", "clearance", "energy", "impact", "smoothness")
    w = np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)
    w_clipped = floor_clip(w, term_names, floor_eps=0.05, floored_term="impact")
    assert np.allclose(w_clipped, w, atol=1e-5)
