# Preference Architectures

<!-- nav:start -->
[Architecture](../../methods/architecture/teacher-architecture.md) · [Train and run](../../../scripts/rl/README.md) · [Experiments](../../../scripts/rl/experiments/README.md) · [Research](../../README.md) · [RL core](../../../scripts/rl/core/README.md) · [Package](../../../talon_rl/README.md)

[TALON RL](../../../README.md) · [Documentation](../../README.md) · [Contracts](../README.md) · [Preference Architectures](README.md)
<!-- nav:end -->


## Purpose

Contracts for preference-conditioned architecture families and policy-family authority.

## Scope

Covers the V2 architecture branches, competence-floor gates, and one-model policy-family requirements while preserving legacy experiment IDs in filenames.

## Research progression

1. establish function preservation
2. measure preference authority
3. test alternative conditioning mechanisms
4. audit competence and retention under training
5. retain only architectures that preserve semantic controllability

## Documents

- competence-floor-maxmin-contract.md
- competence-floor-pilot-contract.md
- one-model-policy-family-contract.md
- v2b-contract.md
- v2c-contract.md
- v2h-contract.md
- v2k-contract.md
- v2pf-h1-authority-contract.md

## Maintenance

Keep provenance-bearing experiment IDs in document filenames. Move documents by role/domain and update repository references when paths change.
