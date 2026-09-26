# `common/utilities`

<!-- nav:start -->
[Architecture](../../../../../docs/methods/architecture/teacher-architecture.md) · [Train and run](../../../README.md) · [Experiments](../../README.md) · [Research](../../../../../docs/README.md) · [RL core](../../../core/README.md) · [Package](../../../../../talon_rl/README.md)

[TALON RL](../../../../../README.md) · [RL runner](../../../README.md) · [Experiments](../../README.md) · [Common](../README.md) · [Utilities](README.md)
<!-- nav:end -->

30 scripts. One line each, taken from the file's own docstring — edit the docstring, not this file.

| file | lines | tags | description |
|---|---:|---|---|
| `authority_isolated_adversarial_vulnerability_audit.py` | 152 | isaac | **no docstring** |
| `authority_isolated_classB_invariant_audit.py` | 116 | isaac | **no docstring** |
| `authority_isolated_h1_screen.py` | 339 | isaac trains | **no docstring** |
| `authority_isolated_h2a_u30_semantic_validity.py` | 224 | isaac | **no docstring** |
| `authority_isolated_simplex_edge_noregression_eval.py` | 91 | isaac | **no docstring** |
| `authority_retention.py` | 99 | class | How much preference authority a checkpoint keeps against a reference critic. |
| `b0_monitor_isolation_smoke.py` | 285 | isaac | Isolation and determinism acceptance smoke for B0's actor-mean monitor. |
| `final_locomotion_eval.py` | 427 | isaac class | Run one training checkpoint/reset-seed shard of the frozen final protocol. |
| `foundation_v2_semantic_eval.py` | 212 | isaac | **no docstring** |
| `isaac_plant_snapshot.py` | 60 | class | Every plant parameter the Isaac A1 scene resolves to, as one record. |
| `objective_set_g1_c0_critic_substrate.py` | 110 | isaac | **no docstring** |
| `objective_set_g1_credit_vs_mc_audit.py` | 240 | isaac | **no docstring** |
| `objective_set_g1_endpoint_screen.py` | 145 | isaac | **no docstring** |
| `objective_set_g1_evaluate.py` | 206 | isaac | **no docstring** |
| `objective_set_g1_train.py` | 185 | isaac trains | **no docstring** |
| `preference_behavior_ordering_audit.py` | 171 | isaac | **no docstring** |
| `projected_endpoint_audit.py` | 97 | class | Endpoint survival of a tail-projected wide critic, per preference and seed. |
| `rank_stats.py` | 34 | — | Rank correlation without a scipy dependency. |
| `replay_audit.py` | 154 | class | Fresh-rollout value replay: how well each snapshot's critic predicts return. |
| `simplex_edge_endpoint.py` | 89 | class | Simplex-edge endpoint audit: authority kept by a narrow/wide pair at u10. |
| `target_gap_audit.py` | 110 | class | How far the GAE target sits from the Monte-Carlo return it approximates. |
| `train_b0.py` | 139 | isaac | Frozen B0.1 scalar-PPO training entrypoint; intentionally no MOPPO path. |
| `trajectory_information_attribution_audit.py` | 277 | isaac | **no docstring** |
| `update_discriminator.py` | 75 | — | Shared scoring for the update-effect audits. |
| `v1a_e0_eval.py` | 248 | isaac | V1A-E0 paired stock A1 deterministic evaluator. |
| `v1b_s6_objective_decomposition_audit.py` | 277 | isaac | V1-B S6 baseline-only objective decomposition audit. |
| `v2b2_semantic_eval.py` | 212 | isaac | **no docstring** |
| `v2b_bounded_deltaa_multiseed.py` | 403 | isaac | **no docstring** |
| `v2b_fixed_stream_multiupdate_audit.py` | 203 | isaac trains | **no docstring** |
| `v2b_lambda_training_pilot.py` | 312 | isaac trains | **no docstring** |

## Run directories these touch

- `runs/authority_isolated_adversarial_vulnerability_audit-2026-09-25`
- `runs/authority_isolated_classB_invariant_audit-2026-09-25`
- `runs/authority_isolated_coverage2_control-2026-09-25`
- `runs/authority_isolated_h0-2026-09-25`
- `runs/authority_isolated_h1-2026-09-25`
- `runs/authority_isolated_h2a_u30_semantic_validity-2026-09-25`
- `runs/authority_isolated_projected_tail_descent-2026-09-25`
- `runs/authority_isolated_simplex_edge_noregression_eval-2026-09-25`
- `runs/authority_isolated_simplex_edge_retain-2026-09-25`
- `runs/authority_isolated_tail_budget_calibration-2026-09-25`
- `runs/authority_isolated_u20_authority_support-2026-09-25`
- `runs/objective_set_g0_structural_parity-2026-09-25`
- `runs/objective_set_g1_c0_critic_substrate-2026-09-25`
- `runs/objective_set_g1_credit_vs_mc_audit-2026-09-25`
- `runs/preference_behavior_ordering_audit-2026-09-24`
- `runs/relational_persistence_heldout-2026-09-24`
- `runs/rv1_a_implementation_gate-2026-09-23`
- `runs/trajectory_information_attribution_audit-2026-09-24`
- `runs/update_functional_effect_audit-2026-09-24`
- `runs/v2a0_function_preserving_gate-2026-09-23`
- `runs/v2b0_function_preserving_gate-2026-09-23`
- `runs/v2b1_film_authority-2026-09-23`
- `runs/v2b_adam_continuous-2026-09-24`
- `runs/v2b_bounded_deltaa_multiseed-2026-09-24`
- `runs/v2b_competence_floor_final_floor-2026-09-24`
- `runs/v2b_competence_floor_paired_pilot-2026-09-24`
- `runs/v2b_competence_floor_semantic_path-2026-09-24`
- `runs/v2b_fixed_stream_multiupdate_audit-2026-09-24`
- `runs/v2b_lambda095_pilot-2026-09-23`
- `runs/v2b_lambda095_semantic-2026-09-23`
- `runs/v2b_lambda100_pilot-2026-09-23`
- `runs/v2b_lambda100_semantic-2026-09-23`
- `runs/v2b_semantic_path_095_u10-2026-09-24`
- `runs/v2b_semantic_path_095_u25-2026-09-24`
- `runs/v2b_semantic_path_095_u50-2026-09-24`
- `runs/v2b_semantic_path_100_u10-2026-09-24`
- `runs/v2b_semantic_path_100_u25-2026-09-24`
- `runs/v2b_semantic_path_100_u50-2026-09-24`

## Still undescribed (17)

These have no module docstring, so there is nothing to put in the table above.

- `authority_isolated_adversarial_vulnerability_audit.py`
- `authority_isolated_classB_invariant_audit.py`
- `authority_isolated_h1_screen.py`
- `authority_isolated_h2a_u30_semantic_validity.py`
- `authority_isolated_simplex_edge_noregression_eval.py`
- `foundation_v2_semantic_eval.py`
- `objective_set_g1_c0_critic_substrate.py`
- `objective_set_g1_credit_vs_mc_audit.py`
- `objective_set_g1_endpoint_screen.py`
- `objective_set_g1_evaluate.py`
- `objective_set_g1_train.py`
- `preference_behavior_ordering_audit.py`
- `trajectory_information_attribution_audit.py`
- `v2b2_semantic_eval.py`
- `v2b_bounded_deltaa_multiseed.py`
- `v2b_fixed_stream_multiupdate_audit.py`
- `v2b_lambda_training_pilot.py`
