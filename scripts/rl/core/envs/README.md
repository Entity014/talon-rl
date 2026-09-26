# Environments

<!-- nav:start -->
[Architecture](../../../../docs/methods/architecture/teacher-architecture.md) · [Train and run](../../README.md) · [Experiments](../../experiments/README.md) · [Research](../../../../docs/README.md) · [RL core](../README.md) · [Package](../../../../talon_rl/README.md)

[TALON RL](../../../../README.md) · [RL runner](../../README.md) · [RL core](../README.md) · [Envs](README.md)
<!-- nav:end -->


## Purpose

Defines the structural environment contract consumed by RL trainers and provides a
lightweight physics-free smoke-test implementation.

## Files

| File | Role | Status |
| --- | --- | --- |
| `base.py` | Runtime-checkable `TalonEnv` Protocol | Stable extension point |
| `dummy.py` | Physics-free Gymnasium vector smoke-test environment | Stable test utility |

## Contract

Real environments do not need to inherit `TalonEnv`. They satisfy the contract
structurally by exposing `num_envs`, `obs_dim`, `action_dim`, `reset()`, and
`step()`.

## Dependency Rule

Task implementations under `talon_rl/tasks/` must not import `scripts/rl/core`.
The trainer may type against `TalonEnv`, while task environments remain independent.
