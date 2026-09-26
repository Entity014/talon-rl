import numpy as np
import pytest

from rl.core.preferences.v1c_curriculum import evaluation_grid, load_manifest, sample_preferences, stage_for_update


def test_manifest_stage_boundaries_are_exact():
    assert stage_for_update(1)["name"] == "stage_0_reference"
    assert stage_for_update(75)["name"] == "stage_0_reference"
    assert stage_for_update(76)["name"] == "stage_1_objective_bias"
    assert stage_for_update(150)["name"] == "stage_1_objective_bias"
    assert stage_for_update(151)["name"] == "stage_2_coverage_expansion"
    assert stage_for_update(225)["name"] == "stage_2_coverage_expansion"
    assert stage_for_update(226)["name"] == "stage_3_full_simplex"
    assert stage_for_update(300)["name"] == "stage_3_full_simplex"
    with pytest.raises(ValueError):
        stage_for_update(0)
    with pytest.raises(ValueError):
        stage_for_update(301)


def test_manifest_concentrations_and_grid_are_frozen():
    manifest = load_manifest()
    assert manifest["sampler_regions"]["reference"]["concentration"] == [20.0, 20.0, 20.0]
    assert manifest["sampler_regions"]["progress_heavy"]["concentration"] == [8.0, 2.0, 2.0]
    assert manifest["sampler_regions"]["balance_heavy"]["concentration"] == [2.0, 8.0, 2.0]
    assert manifest["sampler_regions"]["efficiency_heavy"]["concentration"] == [2.0, 2.0, 8.0]
    assert manifest["sampler_regions"]["full_simplex"]["concentration"] == [1.0, 1.0, 1.0]
    grid = evaluation_grid(manifest)
    assert grid.shape == (10, 3)
    assert np.allclose(grid.sum(axis=1), 1.0)
    assert np.all(grid >= 0)


def test_samples_are_finite_and_on_simplex_for_all_stages_and_conditions():
    for condition in ("curriculum", "full_simplex_control"):
        for update in (1, 75, 76, 150, 151, 225, 226, 300):
            w, labels = sample_preferences(np.random.default_rng(update), update, 128, condition)
            assert w.shape == (128, 3)
            assert labels.shape == (128,)
            assert np.isfinite(w).all()
            assert np.all(w >= 0)
            assert np.allclose(w.sum(axis=1), 1.0, atol=1e-6)


def test_control_is_full_simplex_at_every_update():
    for update in (1, 75, 150, 225, 300):
        _, labels = sample_preferences(np.random.default_rng(update), update, 256, "full_simplex_control")
        assert set(labels.tolist()) == {"full_simplex"}


def test_stage_mixture_proportions_are_empirically_respected():
    manifest = load_manifest()
    rng = np.random.default_rng(123)
    _, labels = sample_preferences(rng, 76, 100000, "curriculum", manifest)
    frequencies = {name: float(np.mean(labels == name)) for name in ("reference", "progress_heavy", "balance_heavy", "efficiency_heavy")}
    assert abs(frequencies["reference"] - 0.40) < 0.01
    assert abs(frequencies["progress_heavy"] - 0.20) < 0.01
    assert abs(frequencies["balance_heavy"] - 0.20) < 0.01
    assert abs(frequencies["efficiency_heavy"] - 0.20) < 0.01
