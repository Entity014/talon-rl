# Stability Robustness

<!-- nav:start -->
[Architecture](../../methods/architecture/teacher-architecture.md) · [Train and run](../../../scripts/rl/README.md) · [Experiments](../../../scripts/rl/experiments/README.md) · [Research](../../README.md) · [RL core](../../../scripts/rl/core/README.md) · [Package](../../../talon_rl/README.md)

[TALON RL](../../../README.md) · [Documentation](../../README.md) · [Baselines](../README.md) · [Stability Robustness](README.md)
<!-- nav:end -->


## Purpose

Design notes for stability, observability, recovery, and policy-preservation interventions.

## Scope

Tracks near-fall diagnostics, observability/memory questions, controlled recovery screens, reward scaling, KL preservation, and adaptive KL control.

## Research progression

1. diagnose failure observability
2. screen controlled recovery
3. calibrate stability reward scaling
4. add policy-preservation constraints
5. progress from fixed target-KL to adaptive KL control

## Documents

- b1-deterministic-policy-preservation-stability-margin-design.md
- b1-p1-target-kl-early-stopping-freeze-draft.md
- b1-p2-adaptive-kl-lr-design-draft.md
- b1-r0-2-information-value-audit.md
- b1-r0-observability-memory-diagnostic.md
- b1-r1-srm-style-controlled-screen.md
- b1-s1-acaps-stability-margin-design-draft.md

## Maintenance

Keep provenance-bearing experiment IDs in document filenames. Move documents by role/domain and update repository references when paths change.
