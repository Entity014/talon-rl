# Optimization

<!-- nav:start -->
[Architecture](../../docs/methods/architecture/teacher-architecture.md) · [Train and run](../../scripts/rl/README.md) · [Experiments](../../scripts/rl/experiments/README.md) · [Research](../../docs/README.md) · [RL core](../../scripts/rl/core/README.md) · [Package](../README.md)

[TALON RL](../../README.md) · [Talon Rl](../README.md) · [Optimization](README.md)
<!-- nav:end -->


Reusable optimization primitives grouped by function rather than pivot/experiment ID.

- `scalarization.py`: early-vs-late scalarization comparison helpers.
- `scalar_critic.py`: scalar-critic, scalar/vector GAE, and PPO helpers used by the former Pivot-P1 branch.

Experiment identifiers such as P0/P1 remain in class/function names or experiment scripts where needed for thesis traceability.
