# Integration

<!-- nav:start -->
[Architecture](../../../../docs/methods/architecture/teacher-architecture.md) · [Train and run](../../README.md) · [Experiments](../../experiments/README.md) · [Research](../../../../docs/README.md) · [RL core](../README.md) · [Package](../../../../talon_rl/README.md)

[TALON RL](../../../../README.md) · [RL runner](../../README.md) · [RL core](../README.md) · [Integration](README.md)
<!-- nav:end -->


## Purpose

Adapters that connect the RL core to third-party training/runtime frameworks while keeping
framework-specific assumptions out of policy and objective abstractions.

## Subfolders

| Folder | Role | Status |
| --- | --- | --- |
| [`rsl_rl/`](rsl_rl/README.md) | rsl_rl compatibility wrappers and integration helpers | Transitional |

## Rules

Integration code may translate interfaces and state formats. It should not define new
research objectives or silently change model behavior.
