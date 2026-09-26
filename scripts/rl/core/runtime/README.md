# Runtime

<!-- nav:start -->
[Architecture](../../../../docs/methods/architecture/teacher-architecture.md) · [Train and run](../../README.md) · [Experiments](../../experiments/README.md) · [Research](../../../../docs/README.md) · [RL core](../README.md) · [Package](../../../../talon_rl/README.md)

[TALON RL](../../../../README.md) · [RL runner](../../README.md) · [RL core](../README.md) · [Runtime](README.md)
<!-- nav:end -->


## Purpose

Future home for inference-time policy execution, export, and deployment-facing adapters.

## Current State

Files:
- `deployment.py` — guarded inference runtime.
- `exporter.py` — TorchScript/JIT policy export.
- `sim2sim.py` — MuJoCo sim-to-sim rollout utilities.

## Rules

Runtime code may consume stable policy/checkpoint interfaces but must not own training
updates or research experiment logic.
