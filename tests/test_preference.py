import numpy as np

from talon_rl.config import PreferenceCfg, RewardVectorCfg
from rl.core.preference import floor_clip, floor_clip_terms, rate_limit, sample_preference_vector


def test_sample_preference_vector_sums_to_one_and_matches_shape():
    rng = np.random.default_rng(0)
    reward_cfg = RewardVectorCfg()
    pref_cfg = PreferenceCfg()
    w = sample_preference_vector(rng, reward_cfg, pref_cfg, num_envs=5)
    assert w.shape == (5, reward_cfg.dim)
    assert np.allclose(w.sum(axis=-1), 1.0, atol=1e-5)
    assert np.all(w >= 0.0)


def test_rate_limit_caps_step_size_per_row():
    w_prev = np.tile([0.2, 0.2, 0.2, 0.2, 0.2], (3, 1))
    w_target = np.tile([1.0, 0.0, 0.0, 0.0, 0.0], (3, 1))
    w_next = rate_limit(w_prev, w_target, max_delta=0.05)
    norms = np.linalg.norm(w_next - w_prev, axis=-1)
    assert np.all(norms <= 0.05 + 1e-6)


def test_rate_limit_passes_through_rows_within_cap():
    w_prev = np.tile([0.2, 0.2, 0.2, 0.2, 0.2], (2, 1))
    w_target = np.array([[0.21, 0.2, 0.2, 0.2, 0.19], [0.2, 0.2, 0.2, 0.2, 0.2]])
    w_next = rate_limit(w_prev, w_target, max_delta=1.0)
    assert np.allclose(w_next, w_target)


def test_rate_limit_handles_mixed_rows_independently():
    """One row needs capping, the other doesn't — each row's cap decision
    must not affect the other (the original bug this guards: a naive
    scalar-norm implementation applied uniformly across the whole batch)."""
    w_prev = np.tile([0.2, 0.2, 0.2, 0.2, 0.2], (2, 1))
    w_target = np.array([[1.0, 0.0, 0.0, 0.0, 0.0], [0.21, 0.2, 0.2, 0.2, 0.19]])
    w_next = rate_limit(w_prev, w_target, max_delta=0.05)
    assert np.linalg.norm(w_next[0] - w_prev[0]) <= 0.05 + 1e-6
    assert np.allclose(w_next[1], w_target[1])  # row 1 was within cap, passes through


def test_floor_clip_enforces_minimum_and_renormalizes():
    term_names = ("progress", "clearance", "energy", "impact", "smoothness")
    w = np.array([[0.5, 0.3, 0.2, 0.0, 0.0], [0.2, 0.2, 0.2, 0.2, 0.2]], dtype=np.float32)
    w_clipped = floor_clip(w, term_names, floor_eps=0.05, floored_term="impact")
    idx = term_names.index("impact")
    assert np.all(w_clipped[:, idx] >= 0.05 - 1e-6)
    assert np.allclose(w_clipped.sum(axis=-1), 1.0, atol=1e-5)
    assert np.allclose(w_clipped[1], w[1], atol=1e-5)  # row already above floor: no-op


def test_floor_clip_terms_keeps_balance_signal_present():
    names = ("progress", "energy", "impact", "smoothness", "balance")
    w = np.array([[0.98, 0.01, 0.0, 0.01, 0.0]], dtype=np.float32)

    clipped = floor_clip_terms(w, names, {"impact": 0.05, "balance": 0.15})

    assert clipped[0, names.index("impact")] >= 0.05 - 1e-6
    assert clipped[0, names.index("balance")] >= 0.15 - 1e-6
    assert np.allclose(clipped.sum(axis=-1), 1.0, atol=1e-5)
