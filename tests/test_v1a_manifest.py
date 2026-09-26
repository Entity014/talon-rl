import pytest

from rl.core.checkpoint.v1a_manifest import V1AManifest, V1A_EXCLUSIONS


def _manifest(**overrides):
    values = dict(
        manifest_version="v1a-0.1",
        baseline_checkpoint="baseline.pt",
        baseline_normalization="norm.json",
        baseline_optimizer_state="optimizer.pt",
        baseline_learned_std_state="learned_std.pt",
        baseline_actor_architecture={"hidden_dims": [128, 128], "action_dim": 12},
        action_distribution_semantics="tanh_normal_scale_3.0",
        adapter_insertion_point="after_actor_body_final_hidden",
        adapter_activation="sigmoid",
        objective_order=("progress", "efficiency", "contact", "balance", "limits"),
        w_ref=(0.2,) * 5,
        bottleneck_dim=8,
        parameter_budget_fraction=0.30,
        inference_overhead_budget=1.5,
        initialization="up_projection_zero;down_projection_normal",
        evaluator_hash="evaluator-sha",
        reset_suite_hash="reset-sha",
        checkpoint_schema="v1a_checkpoint_v1",
        exclusion_flags=dict(V1A_EXCLUSIONS),
    )
    values.update(overrides)
    return V1AManifest(**values)


def test_manifest_hashes_are_separate_and_deterministic():
    first = _manifest().with_hashes()
    second = _manifest().with_hashes()
    assert first.source_baseline_hash == second.source_baseline_hash
    assert first.v1a_formulation_hash == second.v1a_formulation_hash
    changed = _manifest(bottleneck_dim=9).with_hashes()
    assert changed.source_baseline_hash == first.source_baseline_hash
    assert changed.v1a_formulation_hash != first.v1a_formulation_hash


def test_manifest_rejects_non_simplex_reference_or_enabled_later_stage():
    with pytest.raises(ValueError, match="simplex"):
        _manifest(w_ref=(1.0, 0.0, 0.0, 0.0, 0.1))
    flags = dict(V1A_EXCLUSIONS)
    flags["learned_gate"] = True
    with pytest.raises(ValueError, match="disabled"):
        _manifest(exclusion_flags=flags)
