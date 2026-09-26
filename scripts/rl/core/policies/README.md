# Policies

<!-- nav:start -->
[RL runner](../../README.md) · [RL core](../README.md)
<!-- nav:end -->


## Purpose

Defines the stable policy-facing API used by trainers, rollout collectors, runtime code,
and future policy architectures.

## Files

| File | Role | Status |
| --- | --- | --- |
| `base.py` | `Policy` and optional `ActionEvaluator` Protocols | Stable extension point |

## Adding a Policy

Implement `Policy` without requiring trainer inheritance. PPO-compatible policies should
also expose `ActionEvaluator`. Keep architecture-specific internals in their own module
and avoid importing concrete algorithms.

## Dependency Rules

Policies may depend on torch and reusable neural modules. They must not depend on
experiment scripts or concrete trainer implementations.
