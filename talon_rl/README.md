# talon_rl

<!-- nav:start -->
[Architecture](../docs/methods/architecture/teacher-architecture.md) · [Train and run](../scripts/rl/README.md) · [Experiments](../scripts/rl/experiments/README.md) · [Research](../docs/README.md) · [RL core](../scripts/rl/core/README.md) · [Package](README.md)

[TALON RL](../README.md) · [Talon Rl](README.md)
<!-- nav:end -->

The reusable Python package behind TALON's robot tasks, models, rewards, curricula, deployment helpers, optimization utilities, and wrappers.

Use this page when you are changing **what the task or model is**. Training orchestration lives under [`scripts/rl/`](../scripts/rl/README.md).

## Start here

| area | what you will find |
|---|---|
| [Models](models/README.md) | actor/critic foundations, conditioning mechanisms, authority models, and auxiliary heads |
| [Rewards](rewards/README.md) | locomotion reward terms and objective-vector semantics |
| [Curricula](curricula/README.md) | command schedules and curriculum logic |
| [Optimization](optimization/README.md) | reusable scalarization and critic-optimization helpers |
| [Wrappers](wrappers/README.md) | environment/model adapters such as scalar-reward and plant-ensemble wrappers |
| [Unitree A1 task](tasks/locomotion/a1_env/README.md) | Isaac Lab locomotion task definition |
| [Deployment](deployment/README.md) | deployment/runtime helpers and simulator adapters |

## Package layout

```text
talon_rl/
├── assets/
├── curricula/
├── deployment/
├── isaaclab/
├── models/
├── optimization/
├── rewards/
├── tasks/
└── wrappers/
```

## Boundary

Keep task/model semantics in this package and trainer-specific behavior out of it.

- task-specific Isaac Lab wiring → `tasks/`
- reusable model architecture → `models/`
- reward/objective semantics → `rewards/`
- reusable optimization helper → `optimization/`
- training algorithm / rollout machinery → [`scripts/rl/core/`](../scripts/rl/core/README.md)
- scientific workflow → [`scripts/rl/experiments/`](../scripts/rl/experiments/README.md)

## Current architecture

The active model direction is the [Phase 1 Teacher Architecture](../docs/methods/architecture/teacher-architecture.md). Model implementations should follow that design without embedding experiment chronology into package structure.
