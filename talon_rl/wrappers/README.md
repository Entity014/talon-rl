# Wrappers

<!-- nav:start -->
[Architecture](../../docs/methods/architecture/teacher-architecture.md) · [Train and run](../../scripts/rl/README.md) · [Experiments](../../scripts/rl/experiments/README.md) · [Research](../../docs/README.md) · [RL core](../../scripts/rl/core/README.md) · [Package](../README.md)

[TALON RL](../../README.md) · [Talon Rl](../README.md) · [Wrappers](README.md)
<!-- nav:end -->


Reusable environment/model wrappers.

- `scalar_reward_env.py`: scalar-reward wrapper around the unchanged A1 environment (former B0 environment wrapper).
- `plant_ensemble.py`: Phase-5 plant-ensemble environment wrapper.

Wrappers should adapt existing environments/models without becoming task definitions themselves.
