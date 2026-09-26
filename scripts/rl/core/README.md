# RL Core

<!-- nav:start -->
[Architecture](../../../docs/methods/architecture/teacher-architecture.md) · [Train and run](../README.md) · [Experiments](../experiments/README.md) · [Research](../../../docs/README.md) · [RL core](README.md) · [Package](../../../talon_rl/README.md)

[TALON RL](../../../README.md) · [RL runner](../README.md) · [RL core](README.md)
<!-- nav:end -->


Reusable reinforcement-learning infrastructure for the Talon thesis codebase.

## Layout

```text
algorithms/      trainer orchestration and algorithm variants
checkpoint/      checkpoint contracts and formulation manifests
diagnostics/     reusable analysis, audit, and physics-validation tools
envs/            lightweight core-facing test environments
integration/     third-party framework adapters (currently rsl_rl)
modules/         shared neural-network building blocks
normalization/   running statistics and normalization facades
objectives/      PPO/MORL losses and objective-gradient routing
policies/        stable policy interfaces
preferences/     preference sampling, projection, and curricula
rollout/         observation stacking, GAE, and rollout interfaces
experiment_io/   run directories, reports, provenance, and freeze utilities
runtime/         deployment, export, and sim-to-sim runtime code
```

## Design Rules

1. Prefer composition over trainer inheritance.
2. Keep experiment chronology out of generic core abstractions when practical.
3. Core modules must not import experiment scripts.
4. Split files by responsibility, not by class count; small related contracts and implementations may share one module.
5. Preserve checkpoint/state-dict compatibility during structural refactors.
6. A structural move must not change numerical behavior unless explicitly documented.

## Adding Functionality

Choose the responsibility first, add the implementation to the matching package, update that
package's README, and add or update parity tests. Avoid placing new implementation files at
the `core/` top level.
