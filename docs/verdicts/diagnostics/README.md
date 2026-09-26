# Diagnostics

<!-- nav:start -->
[Architecture](../../methods/architecture/teacher-architecture.md) · [Train and run](../../../scripts/rl/README.md) · [Experiments](../../../scripts/rl/experiments/README.md) · [Research](../../README.md) · [RL core](../../../scripts/rl/core/README.md) · [Package](../../../talon_rl/README.md)

[TALON RL](../../../README.md) · [Documentation](../../README.md) · [Verdicts](../README.md) · [Diagnostics](README.md)
<!-- nav:end -->


## Purpose

Conclusions from cross-cutting diagnostic branches.

## Scope

These verdicts decide whether failures are explained by relational objectives, semantic gates, trajectory information, visitation/update interactions, state-manifold localization, or retention mechanisms.

## Research progression

1. identify a candidate mechanism
2. run controlled diagnostic
3. reject non-discriminative explanations
4. retain the narrowest explanation consistent with evidence

## Documents

- context-incremental-retention-verdict.md
- relational-objective-design-audit-verdict.md
- relational-persistence-heldout-verdict.md
- robust-collapse-mechanism-audit-verdict.md
- semantic-gate-factorization-verdict.md
- semantic-gate-robustness-audit-verdict.md
- state-manifold-localization-audit-verdict.md
- trajectory-information-attribution-verdict.md
- trajectory-objective-sufficiency-verdict.md
- update-functional-effect-audit-verdict.md
- update-visitation-interaction-audit-verdict.md
- v2b-o-trajectory-realization-verdict.md
- v2b-paired-joint-update-geometry-verdict.md
- v2b-repeated-update-composition-verdict.md
- v2b-trajectory-semantic-retention-gate-verdict.md

## Maintenance

Keep provenance-bearing experiment IDs in document filenames. Move documents by role/domain and update repository references when paths change.
