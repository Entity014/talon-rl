# `architectures/preference_architectures/semantic_diagnostics`

<!-- nav:start -->
[Experiments](../../../README.md) · [Architectures](../../README.md) · [Preference Architectures](../README.md)
<!-- nav:end -->

7 scripts. One line each, taken from the file's own docstring — edit the docstring, not this file.

| file | lines | tags | description |
|---|---:|---|---|
| `causal_semantics.py` | 4726 | isaac trains | Consolidated thematic experiment stages. |
| `critic_diagnosis.py` | 1890 | isaac trains | Consolidated thematic experiment stages. |
| `critic_representation.py` | 2452 | isaac trains | Consolidated thematic experiment stages. |
| `gradient_geometry.py` | 2096 | isaac | Consolidated thematic experiment stages. |
| `policy_authority.py` | 4645 | isaac trains | Consolidated thematic experiment stages. |
| `ppo_consistency.py` | 378 | isaac trains | Consolidated thematic experiment stages. |
| `stochastic_objective.py` | 576 | isaac | Consolidated thematic experiment stages. |

## Run directories these touch

- `runs/post_v1_d1-2026-09-22`
- `runs/post_v2_a-2026-09-23`
- `runs/post_v2_r21_shared_residual-2026-09-23`
- `runs/post_v2_r23a_coeff_guidance-2026-09-23`
- `runs/post_v2_r23c_normalized_guidance-2026-09-23`
- `runs/post_v2_t0_trajectory_credit-2026-09-23`
- `runs/post_v2_t1_specialist_semantics-2026-09-23`
- `runs/post_v2_t3a2_atomic_coherence-2026-09-23`
- `runs/post_v2_t3a3_objective_selection-2026-09-23`
- `runs/post_v2_t3a4_controllability-2026-09-23`
- `runs/post_v2_t3a_regrouping-2026-09-23`
- `runs/post_v2_t5_c0_zero_critic-2026-09-23`
- `runs/post_v2_t5_c10_h32_control-2026-09-23`
- `runs/post_v2_t5_c10_h32_mc-2026-09-23`
- `runs/post_v2_t5_c11_loss_geometry-2026-09-23`
- `runs/post_v2_t5_c12_moving_batch-2026-09-23`
- `runs/post_v2_t5_c13_conditioning-2026-09-23`
- `runs/post_v2_t5_c14_representation_drift-2026-09-23`
- `runs/post_v2_t5_c15_spectral_update-2026-09-23`
- `runs/post_v2_t5_c16_head_progress-2026-09-23`
- `runs/post_v2_t5_c17_head_tracking-2026-09-23`
- `runs/post_v2_t5_c18_control-2026-09-23`
- `runs/post_v2_t5_c18_ridge3-2026-09-23`
- `runs/post_v2_t5_c19_actor_frozen-2026-09-23`
- `runs/post_v2_t5_c19_actor_moving-2026-09-23`
- `runs/post_v2_t5_c1_zero_critic-2026-09-23`
- `runs/post_v2_t5_c20_coverage-2026-09-23`
- `runs/post_v2_t5_c21_recent12-2026-09-23`
- `runs/post_v2_t5_c21_recent12warm-2026-09-23`
- `runs/post_v2_t5_c21_recent3-2026-09-23`
- `runs/post_v2_t5_c22_diverse12-2026-09-23`
- `runs/post_v2_t5_c22_reset12-2026-09-23`
- `runs/post_v2_t5_c23_reset_diverse-2026-09-23`
- `runs/post_v2_t5_c23_reset_diverse12-2026-09-23`
- `runs/post_v2_t5_c23_reset_recent12-2026-09-23`
- `runs/post_v2_t5_c24_phase_balanced12-2026-09-23`
- `runs/post_v2_t5_c24_tracking_residual-2026-09-23`
- `runs/post_v2_t5_c25_actor_updating-2026-09-23`
- `runs/post_v2_t5_c25_actor_updating25-2026-09-23`
- `runs/post_v2_t5_c25_phase_balanced24-2026-09-23`
- `runs/post_v2_t5_c26_actor_coupled-2026-09-23`
- `runs/post_v2_t5_c26_angular_credit-2026-09-23`
- `runs/post_v2_t5_c27_smoothness_safety-2026-09-23`
- `runs/post_v2_t5_c28_angular_temporal_credit-2026-09-23`
- `runs/post_v2_t5_c29_angular_aggregation-2026-09-23`
- `runs/post_v2_t5_c2_gradient_physical-2026-09-23`
- `runs/post_v2_t5_c30_angular_timestep_window-2026-09-23`
- `runs/post_v2_t5_c31_dynamics_source-2026-09-23`
- `runs/post_v2_t5_c31_dynamics_source_u10-2026-09-23`
- `runs/post_v2_t5_c32_state_intervention-2026-09-23`
- `runs/post_v2_t5_c33_tangent_preservation-2026-09-23`
- `runs/post_v2_t5_c34_action_credit-2026-09-23`
- `runs/post_v2_t5_c35_policy_tangent-2026-09-23`
- `runs/post_v2_t5_c36_score_attribution-2026-09-23`
- `runs/post_v2_t5_c36_score_attribution_matched-2026-09-23`
- `runs/post_v2_t5_c36_score_controls-2026-09-23`
- `runs/post_v2_t5_c37_stochasticity-2026-09-23`
- `runs/post_v2_t5_c38_expected_score_path-2026-09-23`
- `runs/post_v2_t5_c39_variance_curvature-2026-09-23`
- `runs/post_v2_t5_c3_value_learning-2026-09-23`
- `runs/post_v2_t5_c40_derivative_equivalence-2026-09-23`
- `runs/post_v2_t5_c41_corrected_gradient_compare-2026-09-23`
- `runs/post_v2_t5_c42_highsample_confidence-2026-09-23`
- `runs/post_v2_t5_c43_corrected_variance_limit-2026-09-23`
- `runs/post_v2_t5_c4_critic_lr1e4-2026-09-23`
- `runs/post_v2_t5_c5_h16-2026-09-23`
- `runs/post_v2_t5_c6_critic_target_repr-2026-09-23`
- `runs/post_v2_t5_c7_critic_updates10-2026-09-23`
- `runs/post_v2_t5_c8_lambda1-2026-09-23`
- `runs/post_v2_t5_c9_delayed_target10-2026-09-23`
- `runs/post_v2_t5_consistency_repair-2026-09-23`
- `runs/post_v2_t5_gradient_separability-2026-09-23`
- `runs/post_v2_t5_repaired-2026-09-23`
- `runs/v1c_confirmatory_seed0-2026-09-22`
