# rsl_rl Integration

<!-- nav:start -->
[RL runner](../../../README.md) · [RL core](../../README.md) · [Integration](../README.md)
<!-- nav:end -->


## Purpose

Canonical destination for wrappers that adapt live `rsl_rl` actor-critic objects to Talon
policy extensions.

## Current State

Existing implementations remain at the old import paths for compatibility:
- `core/rsl_v1a_wrapper.py`
- `core/rsl_shared_residual_wrapper.py`
- `core/rsl_v1a_integration.py`

## Migration Rule

Move implementations here only after all external imports and state-dict expectations are
covered. Keep compatibility re-exports at the old paths during migration.
