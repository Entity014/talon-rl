# Normalization

<!-- nav:start -->
[RL runner](../../README.md) · [RL core](../README.md)
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
