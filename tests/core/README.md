# RL core tests

[Tests](../README.md)

Reusable RL infrastructure: algorithms, rollout, objectives, preferences, normalization, checkpoints, runtime, diagnostics, environment protocols, and integrations.

## Subfolders

- [`algorithms/`](algorithms/README.md) — 5 test modules
- [`checkpoint/`](checkpoint/README.md) — 1 test modules
- [`diagnostics/`](diagnostics/README.md) — 3 test modules
- [`envs/`](envs/README.md) — 1 test modules
- [`experiment_io/`](experiment_io/README.md) — 2 test modules
- [`integration/`](integration/README.md) — 2 test modules
- [`modules/`](modules/README.md) — 3 test modules
- [`normalization/`](normalization/README.md) — 1 test modules
- [`objectives/`](objectives/README.md) — 2 test modules
- [`preferences/`](preferences/README.md) — 2 test modules
- [`rollout/`](rollout/README.md) — 1 test modules
- [`runtime/`](runtime/README.md) — 3 test modules

## Rule

Name tests after the responsibility or module under test. Keep historical experiment IDs inside test names only when they are part of the behavior being preserved, not as the directory structure.
