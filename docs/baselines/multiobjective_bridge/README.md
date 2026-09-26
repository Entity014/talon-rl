# Multiobjective Bridge

<!-- nav:start -->
[Architecture](../../methods/architecture/teacher-architecture.md) · [Train and run](../../../scripts/rl/README.md) · [Experiments](../../../scripts/rl/experiments/README.md) · [Research](../../README.md) · [RL core](../../../scripts/rl/core/README.md) · [Package](../../../talon_rl/README.md)

[TALON RL](../../../README.md) · [Documentation](../../README.md) · [Baselines](../README.md) · [Multiobjective Bridge](README.md)
<!-- nav:end -->


## Purpose

Bridge from scalar locomotion to multi-objective and preference-conditioned training.

## Scope

Documents reward-preserving vectorization, preference-conditioned MOPPO redesign, AMOR baselines, command exposure, and reward-objective revision.

## Research progression

1. vectorize rewards without changing scalar semantics
2. introduce preference conditioning
3. repair function preservation
4. compare faithful AMOR baselines
5. revise objectives when semantic evidence requires it

## Documents

- g1-command-exposure-design.md
- m0-1-reward-preserving-vectorization.md
- m0-2-preference-conditioned-moppo-design-draft.md
- m0-2a-function-preserving-staged-redesign.md
- m0-3a-amor-faithful-baseline.md
- m0-3b-amor-fidelity-scale-test.md
- reward-objectives-revision-r1-design.md

## Maintenance

Keep provenance-bearing experiment IDs in document filenames. Move documents by role/domain and update repository references when paths change.
