# Integration

<!-- nav:start -->
[RL runner](../../README.md) · [RL core](../README.md)
<!-- nav:end -->


## Purpose

Adapters that connect the RL core to third-party training/runtime frameworks while keeping
framework-specific assumptions out of policy and objective abstractions.

## Subfolders

| Folder | Role | Status |
| --- | --- | --- |
| `rsl_rl/` | rsl_rl compatibility wrappers and integration helpers | Transitional |

## Rules

Integration code may translate interfaces and state formats. It should not define new
research objectives or silently change model behavior.
