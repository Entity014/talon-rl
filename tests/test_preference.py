import numpy as np

from talon_rl.config import PreferenceCfg, RewardVectorCfg
from rl.core.preference import curriculum_alpha, floor_clip, floor_clip_terms, rate_limit, sample_preference_vector


def test_sample_preference_vector_sums_to_one_and_matches_shape():
    rng = np.random.default_rng(0)
    reward_cfg = RewardVectorCfg()
    pref_cfg = PreferenceCfg()
    w = sample_preference_vector(rng, reward_cfg, pref_cfg, num_envs=5)
    assert w.shape == (5, reward_cfg.dim)
    assert np.allclose(w.sum(axis=-1), 1.0, atol=1e-5)
    assert np.all(w >= 0.0)


def test_curriculum_alpha_disabled_by_default_matches_flat_dirichlet_alpha():
    # Default PreferenceCfg() has curriculum_updates=0 -- must be a no-op
    # regardless of step, so every existing (pre-curriculum) caller sees
    # byte-identical behavior.
    reward_cfg = RewardVectorCfg()
    pref_cfg = PreferenceCfg()
    expected = np.full(reward_cfg.dim, pref_cfg.dirichlet_alpha)
    assert np.array_equal(curriculum_alpha(reward_cfg, pref_cfg, step=0), expected)
    assert np.array_equal(curriculum_alpha(reward_cfg, pref_cfg, step=99999), expected)


def test_curriculum_alpha_at_step_zero_equals_start():
    reward_cfg = RewardVectorCfg()
    start = tuple(2.0 for _ in reward_cfg.term_names)
    pref_cfg = PreferenceCfg(curriculum_alpha_start=start, curriculum_updates=100)
    assert np.allclose(curriculum_alpha(reward_cfg, pref_cfg, step=0), start)


def test_curriculum_alpha_interpolates_linearly_then_clamps_to_flat_end():
    reward_cfg = RewardVectorCfg()
    start = tuple(0.0 for _ in reward_cfg.term_names)  # end (dirichlet_alpha=1.0) minus start=0 -> frac == value
    pref_cfg = PreferenceCfg(dirichlet_alpha=1.0, curriculum_alpha_start=start, curriculum_updates=100)
    assert np.allclose(curriculum_alpha(reward_cfg, pref_cfg, step=50), 0.5)
    assert np.allclose(curriculum_alpha(reward_cfg, pref_cfg, step=100), 1.0)
    assert np.allclose(curriculum_alpha(reward_cfg, pref_cfg, step=1_000_000), 1.0)  # past curriculum_updates: stays flat


def test_curriculum_alpha_orders_balance_above_progress_above_smoothness_impact_above_energy():
    """The ordering this curriculum exists to encode (2026-09-19 diagnosis:
    balance must come first or the robot never survives long enough for
    progress to matter, energy is the least urgent per legged_gym/IsaacLab's
    own reference config's negligible torque-penalty weight)."""
    reward_cfg = RewardVectorCfg()
    by_name = {"balance": 4.0, "progress": 3.0, "impact": 1.5, "smoothness": 1.5, "energy": 0.5}
    start = tuple(by_name[name] for name in reward_cfg.term_names)
    idx = {name: i for i, name in enumerate(reward_cfg.term_names)}
    assert start[idx["balance"]] > start[idx["progress"]]
    assert start[idx["progress"]] > start[idx["smoothness"]]
    assert start[idx["progress"]] > start[idx["impact"]]
    assert start[idx["smoothness"]] > start[idx["energy"]]
    assert start[idx["impact"]] > start[idx["energy"]]


def test_sample_preference_vector_still_valid_simplex_under_curriculum():
    reward_cfg = RewardVectorCfg()
    by_name = {"balance": 4.0, "progress": 3.0, "impact": 1.5, "smoothness": 1.5, "energy": 0.5}
    pref_cfg = PreferenceCfg(
        curriculum_alpha_start=tuple(by_name[name] for name in reward_cfg.term_names), curriculum_updates=100,
    )
    rng = np.random.default_rng(0)
    for step in (0, 50, 100, 500):
        w = sample_preference_vector(rng, reward_cfg, pref_cfg, num_envs=8, step=step)
        assert w.shape == (8, reward_cfg.dim)
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


def test_floor_clip_terms_never_produces_negative_weights():
    """Found 2026-09-18 sampling 200k real Dirichlet(1,1,1,1,1) draws
    through this exact pipeline: some came out with negative components.
    Root cause: impact already sitting well above its own floor (so
    untouched, and excluded from the "adjustable" pool that pays for
    raising OTHER floored terms) can starve that pool of mass -- here
    impact=0.90 (way over its 0.05 floor) leaves progress+energy+smoothness
    only ~0.10 combined, but raising balance (0.01 -> 0.15 floor) needs
    ~0.14, driving all three adjustable terms negative under the
    unclamped version of this function. A negative preference weight is
    nonsensical under any interpretation and breaks every downstream
    consumer that assumes a preference simplex."""
    names = ("progress", "energy", "impact", "smoothness", "balance")
    w = np.array([[0.05, 0.02, 0.90, 0.02, 0.01]], dtype=np.float32)

    clipped = floor_clip_terms(w, names, {"impact": 0.05, "balance": 0.15})

    assert np.all(clipped >= 0.0), f"negative weight produced: {clipped}"
    assert np.allclose(clipped.sum(axis=-1), 1.0, atol=1e-5)


def test_floor_clip_terms_is_never_negative_across_many_random_draws():
    """Broader sweep, not just the one hand-built adversarial case --
    samples real Dirichlet(1,1,1,1,1) draws (matching PreferenceCfg's own
    alpha) through the real floors this repo actually uses
    (impact_floor_eps=0.05, balance_floor_eps=0.15, progress_floor_eps=0.15,
    RewardVectorCfg's defaults) and checks none of 50,000 draws ever go
    negative."""
    names = ("progress", "energy", "impact", "smoothness", "balance")
    rng = np.random.default_rng(42)
    w = rng.dirichlet(np.ones(5), size=50_000).astype(np.float32)

    clipped = floor_clip_terms(w, names, {"impact": 0.05, "balance": 0.15, "progress": 0.15})

    assert np.all(clipped >= 0.0), f"found {np.sum(clipped < 0)} negative entries"
    assert np.allclose(clipped.sum(axis=-1), 1.0, atol=1e-4)


def test_floor_clip_terms_never_undershoots_its_own_floor():
    """Regression for the exact bug fixed 2026-09-19: with only impact+
    balance floored, a floored term ending up slightly BELOW its own floor
    after the final renormalize was rare (~0.01% per draw, only 3 adjustable
    terms funding 2 floors). Adding progress_floor_eps left only 2
    adjustable terms (energy, smoothness) to fund 3 floors, and the OLD
    "only shrink adjustable terms, proportional to their own value"
    algorithm's undershoot rate jumped to ~1.8% of draws (worst case 0.118
    vs a 0.15 floor) -- a real, frequent invariant violation, not a rare
    edge case. The water-filling rewrite (floored terms with slack above
    their own floor also give theirs up) must hold every floor exactly,
    every draw."""
    names = ("progress", "energy", "impact", "smoothness", "balance")
    floors = {"impact": 0.05, "balance": 0.15, "progress": 0.15}
    rng = np.random.default_rng(1)
    w = rng.dirichlet(np.ones(5), size=200_000).astype(np.float32)

    clipped = floor_clip_terms(w, names, floors)

    for name, eps in floors.items():
        idx = names.index(name)
        assert np.all(clipped[:, idx] >= eps - 1e-4), (
            f"{name} undershot its {eps} floor: worst={clipped[:, idx].min():.6f}"
        )
    assert np.allclose(clipped.sum(axis=-1), 1.0, atol=1e-4)

    # the adversarial case from floor_clip_terms's own history: one term
    # (impact) holds most of the simplex, well above its own floor, and
    # must give up its slack to fund the other two floors' rise.
    w_adversarial = np.array([[0.05, 0.02, 0.90, 0.02, 0.01]], dtype=np.float32)
    c = floor_clip_terms(w_adversarial, names, floors)
    assert c[0, names.index("impact")] >= floors["impact"] - 1e-4
    assert c[0, names.index("balance")] >= floors["balance"] - 1e-4
    assert c[0, names.index("progress")] >= floors["progress"] - 1e-4
    assert np.isclose(c.sum(), 1.0, atol=1e-4)
