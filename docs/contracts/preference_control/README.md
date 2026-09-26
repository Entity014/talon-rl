# Preference Control

<!-- nav:start -->
[Architecture](../../methods/architecture/teacher-architecture.md) · [Train and run](../../../scripts/rl/README.md) · [Experiments](../../../scripts/rl/experiments/README.md) · [Research](../../README.md) · [RL core](../../../scripts/rl/core/README.md) · [Package](../../../talon_rl/README.md)

[TALON RL](../../../README.md) · [Documentation](../../README.md) · [Contracts](../README.md) · [Preference Control](README.md)
<!-- nav:end -->


## Purpose

Contracts for explicit preference-to-behavior control mechanisms.

## Scope

Covers preference-behavior ordering, prospective slope control, and separated-anchor control.

## Research progression

1. verify preference ordering
2. measure local response slope
3. test separated anchors or control interventions
4. confirm that control semantics survive rollout dynamics

## Documents

- preference-behavior-ordering-audit-contract.md
- prospective-slope-validation-contract.md
- separated-anchor-policy-control-contract.md

## Maintenance

Keep provenance-bearing experiment IDs in document filenames. Move documents by role/domain and update repository references when paths change.
