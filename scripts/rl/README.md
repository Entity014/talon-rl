# RL Runner

<!-- nav:start -->
[TALON RL](../../README.md)
<!-- nav:end -->

This folder is the executable RL front end for TALON. It owns training, playback, simulator-transfer entry points, reusable RL infrastructure, and the research workflows built on top of that infrastructure.

## Start here

| page / command | use it for |
|---|---|
| `python scripts/rl/train.py` | start or resume training with the current trainer/configuration path |
| `python scripts/rl/play.py` | deterministic checkpoint rollout, inspection, and playback |
| `python scripts/rl/sim2sim.py` | simulator-to-simulator policy rollout and transfer checks |
| [RL core](core/README.md) | reusable algorithms, rollout, objectives, preferences, normalization, checkpoints, and runtime |
| [Experiments](experiments/README.md) | scientific workflows, diagnostics, evaluation, and transfer experiments |

## Structure

- `core/` — reusable training infrastructure; new generic RL code belongs here.
- `experiments/` — research questions and one-off/controlled workflows; reusable discoveries should graduate out of here.
- `runs/` — run-oriented local outputs used by the experiment workflows.
- `assets/` — RL-side asset generation helpers.
- `_diagnostics_impl/` — implementation modules behind the diagnostic entry points.

## Boundary

`scripts/rl/` is driver/research code. Reusable robot-task and model definitions live in `talon_rl/`.

The practical dependency direction is:

`talon_rl task/model package -> consumed by scripts/rl core -> exercised by experiments`

## Current direction

The next active training target is the [Phase 1 Teacher Architecture](../../docs/methods/architecture/teacher-architecture.md), which combines robot state, privileged environment context, and a variable-cardinality objective set.
