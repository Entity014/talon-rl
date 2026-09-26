# Freeze

<!-- nav:start -->
[Architecture](../../methods/architecture/teacher-architecture.md) · [Train and run](../../../scripts/rl/README.md) · [Experiments](../../../scripts/rl/experiments/README.md) · [Research](../../README.md) · [RL core](../../../scripts/rl/core/README.md) · [Package](../../../talon_rl/README.md)

[TALON RL](../../../README.md) · [Documentation](../../README.md) · [Protocols](../README.md) · [Freeze](README.md)
<!-- nav:end -->


## Purpose

Canonical freeze/provenance manifests.

## Scope

Defines exactly which implementation/checkpoint/configuration becomes the reference for downstream experiments.

## Research progression

1. identify accepted implementation
2. record hashes/configuration
3. prevent silent drift
4. use the frozen artifact downstream

## Documents

- phase1-d0-canonical-freeze-manifest.md

## Maintenance

Keep provenance-bearing experiment IDs in document filenames. Move documents by role/domain and update repository references when paths change.
