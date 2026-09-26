# Normalization

<!-- nav:start -->
[Architecture](../../../../docs/methods/architecture/teacher-architecture.md) · [Train and run](../../README.md) · [Experiments](../../experiments/README.md) · [Research](../../../../docs/README.md) · [RL core](../README.md) · [Package](../../../../talon_rl/README.md)

[TALON RL](../../../../README.md) · [RL runner](../../README.md) · [RL core](../README.md) · [Normalization](README.md)
<!-- nav:end -->


## Purpose

Reusable observation, reward, return, and objective normalization components.

## Files

| File | Role | Status |
| --- | --- | --- |
| `running.py` | `RunningNormalizer` stateful facade | Stable adapter |

## Legacy Compatibility

`stats.py::RunningMeanStd` is the established implementation.
`running.py::RunningNormalizer` composes it rather than duplicating numerical logic.

## Rules

Normalization components own statistics and transforms only. They must not know about
experiment chronology, checkpoint selection policy, or rollout scheduling.
