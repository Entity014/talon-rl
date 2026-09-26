# Checkpoint

<!-- nav:start -->
[Architecture](../../../../docs/methods/architecture/teacher-architecture.md) · [Train and run](../../README.md) · [Experiments](../../experiments/README.md) · [Research](../../../../docs/README.md) · [RL core](../README.md) · [Package](../../../../talon_rl/README.md)

[TALON RL](../../../../README.md) · [RL runner](../../README.md) · [RL core](../README.md) · [Checkpoint](README.md)
<!-- nav:end -->


## Purpose

Owns checkpoint persistence, manifest metadata, compatibility checks, and future schema
migration. It must not define training semantics.

## Files

| File | Role | Status |
| --- | --- | --- |
| `base.py` | `CheckpointManager` Protocol | Stable extension point |

## Planned Migration

`checkpoint/v1a_manifest.py` and save/load logic embedded in trainers are candidates for
migration here. Existing checkpoint keys and manifest hashes must remain reproducible.

## Adding Checkpoint Logic

Implement `CheckpointManager`. Keep policy state naming stable unless an explicit
migration tool and compatibility test accompany the change.
