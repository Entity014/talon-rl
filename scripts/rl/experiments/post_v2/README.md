# `post_v2`

177 scripts. One line each, taken from the file's own docstring — edit the docstring, not this file.

| file | lines | tags | description |
|---|---:|---|---|
| `post_v2_a_offline_latent_anchor_audit.py` | 136 | — | V2-A offline latent-anchor and simplex-continuity audit. |
| `post_v2_b1_latent_learning_audit.py` | 49 | isaac trains | Instrumented matched V2-B rerun for latent-learning diagnosis. |
| `post_v2_b_isaac_smoke.py` | 46 | isaac trains | V2-B function-preserving and real-Isaac smoke; no optimizer step. |
| `post_v2_b_short_screen.py` | 56 | isaac trains | V2-B one-seed short screen: only PPO + the frozen manifold loss. |
| `post_v2_b_terminal_sensitivity.py` | 31 | isaac | Read-only terminal sensitivity audit for the V2-B short-screen checkpoint. |
| `post_v2_c1_action_behavior_causal_audit.py` | 134 | isaac | V2-C1 read-only action-to-behavior causal audit. |
| `post_v2_c2_projection_audit.py` | 198 | isaac | V2-C2 read-only semantic direction projection/orthogonal causal audit. |
| `post_v2_c3_trajectory_divergence_audit.py` | 191 | isaac | V2-C3 read-only short-horizon closed-loop semantic divergence audit. |
| `post_v2_c4_saturation_authority_audit.py` | 212 | isaac | V2-C4 read-only saturation-aware trajectory authority audit. |
| `post_v2_c5_raw_output_authority_audit.py` | 163 | isaac | V2-C5 read-only raw-output authority audit. |
| `post_v2_c_semantic_response_audit.py` | 146 | isaac | V2-C read-only semantic preference-response audit on frozen V2-B checkpoint. |
| `post_v2_r1_authority_audit.py` | 163 | isaac | V2-R1 read-only raw-output authority audit for fixed FiLM alpha=0.5. |
| `post_v2_r1_authority_compare.py` | 78 | isaac | Read-only V2 control vs V2-R1 authority comparison on matched states. |
| `post_v2_r1_isaac_smoke.py` | 66 | isaac | V2-R1 real-Isaac function-preserving smoke; no optimizer step. |
| `post_v2_r1_semantic_response_audit.py` | 146 | isaac | V2-R1 read-only semantic preference-response audit for fixed FiLM alpha=0.5. |
| `post_v2_r1_short_authority_screen.py` | 56 | isaac trains | V2-R1 one-seed short authority screen: V2 with fixed FiLM alpha=0.5 only. |
| `post_v2_r21_shared_residual_geometry_audit.py` | 172 | isaac | V2-R2.1 read-only shared/residual hidden-geometry audit. |
| `post_v2_r22_coefficient_gradient_audit.py` | 146 | isaac trains | V2-R2.2 diagnostic-only coefficient gradient orientation audit. |
| `post_v2_r23a_coeff_guidance_gradient_audit.py` | 179 | isaac trains | V2-R2.3-A read-only semantic coefficient-guidance gradient compatibility audit. |
| `post_v2_r23c_normalized_guidance_audit.py` | 126 | class | How much normalised guidance is needed to orient the pairwise semantics? |
| `post_v2_r23d_min_norm_constraint_audit.py` | 142 | — | V2-R2.3-D read-only minimum-norm pairwise semantic orientation correction audit. |
| `post_v2_r2_freeze_basis.py` | 76 | isaac | Freeze provenance-backed R2 h1 shared/residual basis from D1 specialists. |
| `post_v2_r2_semantic_geometry_audit.py` | 182 | isaac | V2-R2 read-only semantic hidden-geometry audit. |
| `post_v2_r2_short_coefficient_screen.py` | 172 | isaac trains | V2-R2 one-seed short coefficient/authority screen. No semantic verdict. |
| `post_v2_t0_c3_rollout32.py` | 191 | isaac | V2-C3 read-only short-horizon closed-loop semantic divergence audit. |
| `post_v2_t0_offline_crossrank.py` | 29 | — | **no docstring** |
| `post_v2_t1_specialist_reward_semantics_audit.py` | 108 | — | **no docstring** |
| `post_v2_t3a2_atomic_coherence_audit.py` | 153 | — | T3-A2 read-only atomic semantic-coordinate/coherence audit. |
| `post_v2_t3a3_objective_set_selection.py` | 177 | — | T3-A3 read-only objective-set selection/compression review. |
| `post_v2_t3a4_controllability_audit.py` | 163 | — | T3-A4 read-only controllability / reachable-set audit. |
| `post_v2_t3a5_effort_controllability_screen.py` | 140 | isaac trains | T3-A5 targeted effort controllability screen. |
| `post_v2_t3a_regrouping_audit.py` | 162 | isaac | T3-A read-only candidate reward regrouping/separability audit on matched D1 specialists. |
| `post_v2_t3b_scaling_audit.py` | 175 | isaac | T3-B read-only scaling/normalization audit for frozen 4D MORL vector. |
| `post_v2_t4_init_smoke.py` | 34 | — | **no docstring** |
| `post_v2_t4_specialists.py` | 161 | isaac trains | T4: generate four fixed-preference specialists on frozen normalized 4D objectives and validate endpoints. |
| `post_v2_t5_adam_geometry_smoke.py` | 52 | isaac | **no docstring** |
| `post_v2_t5_advantage_decomposition_smoke.py` | 56 | isaac | **no docstring** |
| `post_v2_t5_c0_zero_critic_audit.py` | 111 | isaac | **no docstring** |
| `post_v2_t5_c10_endpoint_compare.py` | 57 | isaac | **no docstring** |
| `post_v2_t5_c10_exact_h32_target_gap.py` | 64 | isaac | **no docstring** |
| `post_v2_t5_c10_value_compare.py` | 95 | isaac | **no docstring** |
| `post_v2_t5_c11_critic_loss_geometry_audit.py` | 142 | isaac trains | **no docstring** |
| `post_v2_t5_c12_moving_batch_audit.py` | 134 | isaac trains | **no docstring** |
| `post_v2_t5_c12_optimal_head_transfer.py` | 70 | isaac | **no docstring** |
| `post_v2_t5_c12_ridge_head_transfer.py` | 77 | isaac | **no docstring** |
| `post_v2_t5_c13_collect_features.py` | 47 | isaac | **no docstring** |
| `post_v2_t5_c13_conditioning_regularization_audit.py` | 197 | isaac trains | **no docstring** |
| `post_v2_t5_c13_offline_analyze.py` | 118 | trains | **no docstring** |
| `post_v2_t5_c13_pcr_audit.py` | 88 | class | Principal-component regression conditioning of the C13 critic features. |
| `post_v2_t5_c14_collect_matched_features.py` | 52 | isaac | **no docstring** |
| `post_v2_t5_c14_offline_analyze.py` | 94 | — | **no docstring** |
| `post_v2_t5_c15_sequential_projection.py` | 75 | trains | **no docstring** |
| `post_v2_t5_c15_spectral_update_audit.py` | 143 | trains | **no docstring** |
| `post_v2_t5_c16_function_space_progress.py` | 50 | — | **no docstring** |
| `post_v2_t5_c16_head_progress_audit.py` | 153 | class | Is online head learning moving toward the solution a stable fit would pick? |
| `post_v2_t5_c17_controlled_head_tracking_audit.py` | 166 | trains class | Can a head fitted on a recent window predict the next batch? |
| `post_v2_t5_c17_per_head_progress.py` | 39 | — | **no docstring** |
| `post_v2_t5_c18_horizon_value_compare.py` | 59 | isaac | **no docstring** |
| `post_v2_t5_c18_online_head_tracker.py` | 134 | isaac trains | **no docstring** |
| `post_v2_t5_c18_terminal_compare.py` | 108 | isaac | **no docstring** |
| `post_v2_t5_c19_actor_coupling_train.py` | 117 | isaac trains | **no docstring** |
| `post_v2_t5_c19_replay_audit.py` | 90 | isaac | **no docstring** |
| `post_v2_t5_c20_collect_pool.py` | 58 | isaac | **no docstring** |
| `post_v2_t5_c20_coverage_analyze.py` | 85 | — | **no docstring** |
| `post_v2_t5_c20_mc64_eval.py` | 50 | isaac | **no docstring** |
| `post_v2_t5_c21_replay_audit.py` | 22 | — | Value replay for C21 recent3 versus recent12 support. |
| `post_v2_t5_c21_support_width_train.py` | 117 | isaac trains | **no docstring** |
| `post_v2_t5_c21_warm_replay_audit.py` | 22 | — | Value replay for C21 recent3 versus warm-started recent12. |
| `post_v2_t5_c21_warm_support_train.py` | 120 | isaac trains | **no docstring** |
| `post_v2_t5_c22_diverse_support_train.py` | 115 | isaac | **no docstring** |
| `post_v2_t5_c22_freeze.py` | 66 | class | Freeze C22: support-topology diversity. |
| `post_v2_t5_c22_replay_audit.py` | 22 | — | Value replay for C22 consecutive versus diverse support. |
| `post_v2_t5_c22_reset_replay_audit.py` | 22 | — | Value replay for C22 consecutive versus reset-seeded support. |
| `post_v2_t5_c22_reset_support_train.py` | 122 | isaac trains | **no docstring** |
| `post_v2_t5_c22_reset_terminal_multisuite.py` | 73 | isaac | **no docstring** |
| `post_v2_t5_c22_terminal_multisuite.py` | 65 | isaac | **no docstring** |
| `post_v2_t5_c23_freeze.py` | 55 | class | Freeze C23: the reset-diverse support principle. |
| `post_v2_t5_c23_paired_terminal_multisuite.py` | 73 | isaac | **no docstring** |
| `post_v2_t5_c23_phase_split_audit.py` | 62 | isaac | **no docstring** |
| `post_v2_t5_c23_replay_audit.py` | 22 | — | Value replay for C23 reset-recent versus reset-diverse support. |
| `post_v2_t5_c23_reset_diverse_pilot.py` | 137 | isaac | **no docstring** |
| `post_v2_t5_c23_reset_diverse_train.py` | 116 | isaac | **no docstring** |
| `post_v2_t5_c23_reset_recent_paired_train.py` | 122 | isaac trains | **no docstring** |
| `post_v2_t5_c23_terminal_multisuite.py` | 73 | isaac | **no docstring** |
| `post_v2_t5_c24_freeze.py` | 60 | class | Freeze C24: the Tracking-head residual under the reset-diverse repair. |
| `post_v2_t5_c24_phase_balanced_train.py` | 120 | isaac | **no docstring** |
| `post_v2_t5_c24_phase_eval.py` | 69 | isaac | **no docstring** |
| `post_v2_t5_c24_reset_breakdown.py` | 76 | isaac | **no docstring** |
| `post_v2_t5_c24_tracking_residual_audit.py` | 128 | isaac | **no docstring** |
| `post_v2_t5_c25_actor_updating_reset_support.py` | 108 | isaac trains | **no docstring** |
| `post_v2_t5_c25_actor_updating_reset_support_25.py` | 108 | isaac trains | **no docstring** |
| `post_v2_t5_c25_angular_credit_confirm.py` | 58 | isaac | **no docstring** |
| `post_v2_t5_c25_angular_u0_control.py` | 9 | — | **no docstring** |
| `post_v2_t5_c25_causal_semantics.py` | 82 | isaac | **no docstring** |
| `post_v2_t5_c25_causal_u10_confirm.py` | 58 | isaac | **no docstring** |
| `post_v2_t5_c25_causal_u25_confirm.py` | 58 | isaac | **no docstring** |
| `post_v2_t5_c25_eval.py` | 121 | isaac | **no docstring** |
| `post_v2_t5_c25_freeze.py` | 89 | class | Freeze C25: the critic repair under actor learning. |
| `post_v2_t5_c25_mid_causal_semantics.py` | 82 | isaac | **no docstring** |
| `post_v2_t5_c25_phase_eval.py` | 69 | isaac | **no docstring** |
| `post_v2_t5_c25_replay25.py` | 65 | isaac | **no docstring** |
| `post_v2_t5_c25_replay25_audit.py` | 65 | isaac | **no docstring** |
| `post_v2_t5_c25_replay_audit.py` | 65 | isaac | **no docstring** |
| `post_v2_t5_c25_replay_audit_25.py` | 65 | isaac | **no docstring** |
| `post_v2_t5_c25_semantic_perturb.py` | 85 | isaac | **no docstring** |
| `post_v2_t5_c25_semantic_u25.py` | 85 | isaac | **no docstring** |
| `post_v2_t5_c25_tracking_confirm.py` | 49 | isaac | **no docstring** |
| `post_v2_t5_c25_u10_perturb_confirm.py` | 30 | isaac | **no docstring** |
| `post_v2_t5_c25_u10_safety_confirm.py` | 34 | isaac | **no docstring** |
| `post_v2_t5_c25_u25_causal_semantics.py` | 82 | isaac | **no docstring** |
| `post_v2_t5_c25_u25_safety_confirm.py` | 34 | isaac | **no docstring** |
| `post_v2_t5_c26_actor_coupled_train.py` | 109 | isaac trains | **no docstring** |
| `post_v2_t5_c26_angular_consensus.py` | 60 | isaac | **no docstring** |
| `post_v2_t5_c26_angular_credit_stability.py` | 68 | isaac | **no docstring** |
| `post_v2_t5_c26_conditional_decomp.py` | 99 | isaac | **no docstring** |
| `post_v2_t5_c26_freeze.py` | 55 | class | Freeze C26: angular-credit stability under actor learning. |
| `post_v2_t5_c26_fresh_eval.py` | 51 | isaac | **no docstring** |
| `post_v2_t5_c26_local_bin_causal.py` | 91 | isaac | **no docstring** |
| `post_v2_t5_c27_freeze.py` | 49 | class | Freeze C27: the smoothness-branch trajectory safety risk. |
| `post_v2_t5_c27_smoothness_counterfactual.py` | 44 | isaac | **no docstring** |
| `post_v2_t5_c27_smoothness_safety_audit.py` | 108 | class | Why does the smoothness-heavy policy lose a lane, and does it get worse? |
| `post_v2_t5_c28_temporal_credit.py` | 97 | isaac | **no docstring** |
| `post_v2_t5_c29_aggregation_temporal.py` | 117 | isaac | **no docstring** |
| `post_v2_t5_c2_gradient_physical_audit.py` | 148 | isaac trains | **no docstring** |
| `post_v2_t5_c30_timestep_window.py` | 117 | isaac | **no docstring** |
| `post_v2_t5_c30_window_core.py` | 63 | isaac | **no docstring** |
| `post_v2_t5_c31_combo_single.py` | 54 | isaac | **no docstring** |
| `post_v2_t5_c31_dynamics_source.py` | 95 | isaac | **no docstring** |
| `post_v2_t5_c31_stratify_single.py` | 59 | isaac | **no docstring** |
| `post_v2_t5_c31_u10.py` | 95 | isaac | **no docstring** |
| `post_v2_t5_c32_state_intervention.py` | 114 | isaac | **no docstring** |
| `post_v2_t5_c33_fd_worker.py` | 39 | isaac | **no docstring** |
| `post_v2_t5_c33_one.py` | 96 | isaac | **no docstring** |
| `post_v2_t5_c33_policy_jvp_worker.py` | 55 | isaac | **no docstring** |
| `post_v2_t5_c33_policy_projection_worker.py` | 43 | isaac | **no docstring** |
| `post_v2_t5_c33_reset_fd.py` | 68 | isaac | **no docstring** |
| `post_v2_t5_c33_tangent_preservation.py` | 96 | isaac | **no docstring** |
| `post_v2_t5_c33_vectorized.py` | 56 | isaac | **no docstring** |
| `post_v2_t5_c34_action_credit_worker.py` | 43 | isaac | **no docstring** |
| `post_v2_t5_c35_p_column_worker.py` | 52 | isaac | **no docstring** |
| `post_v2_t5_c36_aggregate.py` | 61 | — | **no docstring** |
| `post_v2_t5_c36_controls_aggregate.py` | 34 | — | **no docstring** |
| `post_v2_t5_c36_matched_aggregate.py` | 56 | — | **no docstring** |
| `post_v2_t5_c36_matched_draw.py` | 86 | isaac | **no docstring** |
| `post_v2_t5_c36_score_attribution.py` | 91 | isaac | **no docstring** |
| `post_v2_t5_c36_score_controls.py` | 93 | isaac | **no docstring** |
| `post_v2_t5_c37_aggregate.py` | 58 | — | **no docstring** |
| `post_v2_t5_c37_stochasticity_audit.py` | 34 | — | **no docstring** |
| `post_v2_t5_c37_worker.py` | 86 | isaac | **no docstring** |
| `post_v2_t5_c38_expected_score_path.py` | 131 | isaac | **no docstring** |
| `post_v2_t5_c39_variance_curvature.py` | 123 | isaac | **no docstring** |
| `post_v2_t5_c3_optimizer_counterfactual.py` | 73 | isaac trains | **no docstring** |
| `post_v2_t5_c3_value_learning_diagnosis.py` | 231 | isaac trains | C3 diagnostic-only audit of four-head critic/value learning. |
| `post_v2_t5_c40_derivative_equivalence.py` | 165 | isaac | **no docstring** |
| `post_v2_t5_c41_corrected_gradient_compare.py` | 146 | isaac | **no docstring** |
| `post_v2_t5_c42_highsample_confidence.py` | 126 | isaac | **no docstring** |
| `post_v2_t5_c43_corrected_variance_limit.py` | 114 | isaac | **no docstring** |
| `post_v2_t5_c4_samebatch_check.py` | 43 | isaac trains | **no docstring** |
| `post_v2_t5_c4_value_semantics_audit.py` | 92 | isaac | **no docstring** |
| `post_v2_t5_c5_h16_endpoint_eval.py` | 57 | isaac | **no docstring** |
| `post_v2_t5_c5_h16_value_endpoint_audit.py` | 115 | isaac | **no docstring** |
| `post_v2_t5_c5_h16_value_semantics_audit.py` | 86 | isaac | **no docstring** |
| `post_v2_t5_c6_critic_target_representation_audit.py` | 177 | isaac trains | **no docstring** |
| `post_v2_t5_c6_timeout_target_audit.py` | 99 | class | What target does the critic get when an episode times out rather than fails? |
| `post_v2_t5_c7_endpoint_eval.py` | 57 | isaac | **no docstring** |
| `post_v2_t5_c7_target_gap_audit.py` | 21 | — | Target gap for C5 at one critic update against C7 at ten. |
| `post_v2_t5_c7_value_semantics_audit.py` | 86 | isaac | **no docstring** |
| `post_v2_t5_c8_endpoint_eval.py` | 57 | isaac | **no docstring** |
| `post_v2_t5_c8_exact_h16_target_gap.py` | 59 | isaac | **no docstring** |
| `post_v2_t5_c8_target_gap_audit.py` | 21 | — | Target gap at GAE lambda 0.95 against lambda 1. |
| `post_v2_t5_c8_value_semantics_audit.py` | 86 | isaac | **no docstring** |
| `post_v2_t5_c9_endpoint_eval.py` | 57 | isaac | **no docstring** |
| `post_v2_t5_c9_exact_h16_target_gap.py` | 63 | isaac | **no docstring** |
| `post_v2_t5_c9_value_semantics_audit.py` | 89 | isaac | **no docstring** |
| `post_v2_t5_consistency_repair_smoke.py` | 71 | isaac | **no docstring** |
| `post_v2_t5_gradient_separability_audit.py` | 131 | isaac trains | **no docstring** |
| `post_v2_t5_ratio_consistency_smoke.py` | 45 | isaac | **no docstring** |

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

## Still undescribed (126)

These have no module docstring, so there is nothing to put in the table above.

- `post_v2_t0_offline_crossrank.py`
- `post_v2_t1_specialist_reward_semantics_audit.py`
- `post_v2_t4_init_smoke.py`
- `post_v2_t5_adam_geometry_smoke.py`
- `post_v2_t5_advantage_decomposition_smoke.py`
- `post_v2_t5_c0_zero_critic_audit.py`
- `post_v2_t5_c10_endpoint_compare.py`
- `post_v2_t5_c10_exact_h32_target_gap.py`
- `post_v2_t5_c10_value_compare.py`
- `post_v2_t5_c11_critic_loss_geometry_audit.py`
- `post_v2_t5_c12_moving_batch_audit.py`
- `post_v2_t5_c12_optimal_head_transfer.py`
- `post_v2_t5_c12_ridge_head_transfer.py`
- `post_v2_t5_c13_collect_features.py`
- `post_v2_t5_c13_conditioning_regularization_audit.py`
- `post_v2_t5_c13_offline_analyze.py`
- `post_v2_t5_c14_collect_matched_features.py`
- `post_v2_t5_c14_offline_analyze.py`
- `post_v2_t5_c15_sequential_projection.py`
- `post_v2_t5_c15_spectral_update_audit.py`
- `post_v2_t5_c16_function_space_progress.py`
- `post_v2_t5_c17_per_head_progress.py`
- `post_v2_t5_c18_horizon_value_compare.py`
- `post_v2_t5_c18_online_head_tracker.py`
- `post_v2_t5_c18_terminal_compare.py`
- `post_v2_t5_c19_actor_coupling_train.py`
- `post_v2_t5_c19_replay_audit.py`
- `post_v2_t5_c20_collect_pool.py`
- `post_v2_t5_c20_coverage_analyze.py`
- `post_v2_t5_c20_mc64_eval.py`
- `post_v2_t5_c21_support_width_train.py`
- `post_v2_t5_c21_warm_support_train.py`
- `post_v2_t5_c22_diverse_support_train.py`
- `post_v2_t5_c22_reset_support_train.py`
- `post_v2_t5_c22_reset_terminal_multisuite.py`
- `post_v2_t5_c22_terminal_multisuite.py`
- `post_v2_t5_c23_paired_terminal_multisuite.py`
- `post_v2_t5_c23_phase_split_audit.py`
- `post_v2_t5_c23_reset_diverse_pilot.py`
- `post_v2_t5_c23_reset_diverse_train.py`
- `post_v2_t5_c23_reset_recent_paired_train.py`
- `post_v2_t5_c23_terminal_multisuite.py`
- `post_v2_t5_c24_phase_balanced_train.py`
- `post_v2_t5_c24_phase_eval.py`
- `post_v2_t5_c24_reset_breakdown.py`
- `post_v2_t5_c24_tracking_residual_audit.py`
- `post_v2_t5_c25_actor_updating_reset_support.py`
- `post_v2_t5_c25_actor_updating_reset_support_25.py`
- `post_v2_t5_c25_angular_credit_confirm.py`
- `post_v2_t5_c25_angular_u0_control.py`
- `post_v2_t5_c25_causal_semantics.py`
- `post_v2_t5_c25_causal_u10_confirm.py`
- `post_v2_t5_c25_causal_u25_confirm.py`
- `post_v2_t5_c25_eval.py`
- `post_v2_t5_c25_mid_causal_semantics.py`
- `post_v2_t5_c25_phase_eval.py`
- `post_v2_t5_c25_replay25.py`
- `post_v2_t5_c25_replay25_audit.py`
- `post_v2_t5_c25_replay_audit.py`
- `post_v2_t5_c25_replay_audit_25.py`
- `post_v2_t5_c25_semantic_perturb.py`
- `post_v2_t5_c25_semantic_u25.py`
- `post_v2_t5_c25_tracking_confirm.py`
- `post_v2_t5_c25_u10_perturb_confirm.py`
- `post_v2_t5_c25_u10_safety_confirm.py`
- `post_v2_t5_c25_u25_causal_semantics.py`
- `post_v2_t5_c25_u25_safety_confirm.py`
- `post_v2_t5_c26_actor_coupled_train.py`
- `post_v2_t5_c26_angular_consensus.py`
- `post_v2_t5_c26_angular_credit_stability.py`
- `post_v2_t5_c26_conditional_decomp.py`
- `post_v2_t5_c26_fresh_eval.py`
- `post_v2_t5_c26_local_bin_causal.py`
- `post_v2_t5_c27_smoothness_counterfactual.py`
- `post_v2_t5_c28_temporal_credit.py`
- `post_v2_t5_c29_aggregation_temporal.py`
- `post_v2_t5_c2_gradient_physical_audit.py`
- `post_v2_t5_c30_timestep_window.py`
- `post_v2_t5_c30_window_core.py`
- `post_v2_t5_c31_combo_single.py`
- `post_v2_t5_c31_dynamics_source.py`
- `post_v2_t5_c31_stratify_single.py`
- `post_v2_t5_c31_u10.py`
- `post_v2_t5_c32_state_intervention.py`
- `post_v2_t5_c33_fd_worker.py`
- `post_v2_t5_c33_one.py`
- `post_v2_t5_c33_policy_jvp_worker.py`
- `post_v2_t5_c33_policy_projection_worker.py`
- `post_v2_t5_c33_reset_fd.py`
- `post_v2_t5_c33_tangent_preservation.py`
- `post_v2_t5_c33_vectorized.py`
- `post_v2_t5_c34_action_credit_worker.py`
- `post_v2_t5_c35_p_column_worker.py`
- `post_v2_t5_c36_aggregate.py`
- `post_v2_t5_c36_controls_aggregate.py`
- `post_v2_t5_c36_matched_aggregate.py`
- `post_v2_t5_c36_matched_draw.py`
- `post_v2_t5_c36_score_attribution.py`
- `post_v2_t5_c36_score_controls.py`
- `post_v2_t5_c37_aggregate.py`
- `post_v2_t5_c37_stochasticity_audit.py`
- `post_v2_t5_c37_worker.py`
- `post_v2_t5_c38_expected_score_path.py`
- `post_v2_t5_c39_variance_curvature.py`
- `post_v2_t5_c3_optimizer_counterfactual.py`
- `post_v2_t5_c40_derivative_equivalence.py`
- `post_v2_t5_c41_corrected_gradient_compare.py`
- `post_v2_t5_c42_highsample_confidence.py`
- `post_v2_t5_c43_corrected_variance_limit.py`
- `post_v2_t5_c4_samebatch_check.py`
- `post_v2_t5_c4_value_semantics_audit.py`
- `post_v2_t5_c5_h16_endpoint_eval.py`
- `post_v2_t5_c5_h16_value_endpoint_audit.py`
- `post_v2_t5_c5_h16_value_semantics_audit.py`
- `post_v2_t5_c6_critic_target_representation_audit.py`
- `post_v2_t5_c7_endpoint_eval.py`
- `post_v2_t5_c7_value_semantics_audit.py`
- `post_v2_t5_c8_endpoint_eval.py`
- `post_v2_t5_c8_exact_h16_target_gap.py`
- `post_v2_t5_c8_value_semantics_audit.py`
- `post_v2_t5_c9_endpoint_eval.py`
- `post_v2_t5_c9_exact_h16_target_gap.py`
- `post_v2_t5_c9_value_semantics_audit.py`
- `post_v2_t5_consistency_repair_smoke.py`
- `post_v2_t5_gradient_separability_audit.py`
- `post_v2_t5_ratio_consistency_smoke.py`
