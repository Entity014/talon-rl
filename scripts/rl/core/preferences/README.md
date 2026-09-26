# Preferences

<!-- nav:start -->
[Architecture](../../../../docs/methods/architecture/teacher-architecture.md) · [Train and run](../../README.md) · [Experiments](../../experiments/README.md) · [Research](../../../../docs/README.md) · [RL core](../README.md) · [Package](../../../../talon_rl/README.md)

[TALON RL](../../../../README.md) · [RL runner](../../README.md) · [RL core](../README.md) · [Preferences](README.md)
<!-- nav:end -->


## Purpose

Owns preference-vector generation and transformations independently from trainers.

## Files

| File | Role | Status |
| --- | --- | --- |
| `core.py` | Preference Protocols plus Dirichlet sampler and floor transform | Stable extension point + tested adapters |

## Legacy Compatibility

`functional.py` contains the established functional implementation:
Dirichlet curriculum sampling, rate limiting, and floor projection. The OO classes delegate
to those functions so existing imports and numerical behavior stay unchanged.

## Adding a Preference Strategy

Implement a sampler for generation or a transform for projection/constraints. Keep reward
semantics and optimizer behavior outside this layer.
