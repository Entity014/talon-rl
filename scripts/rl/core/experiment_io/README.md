# Runs

<!-- nav:start -->
[Architecture](../../../../docs/methods/architecture/teacher-architecture.md) · [Train and run](../../README.md) · [Experiments](../../experiments/README.md) · [Research](../../../../docs/README.md) · [RL core](../README.md) · [Package](../../../../talon_rl/README.md)

[TALON RL](../../../../README.md) · [RL runner](../../README.md) · [RL core](../README.md) · [Experiment Io](README.md)
<!-- nav:end -->


## Purpose

Owns run-directory creation, report helpers, provenance hashing, and freeze/finalization
utilities shared across experiments.

## Files

| File | Role | Status |
| --- | --- | --- |
| `run_dir.py` | Create/resolve run directories and checkpoint paths | Stable utility |
| `run_report.py` | Shared report paths, JSON writing, and hashing helpers | Stable utility |
| `freeze.py` | Base class for immutable experiment freeze/provenance reports | Stable utility |

## Rules

This package manages experiment artifacts, not learning semantics. Algorithms may use run
utilities, but run/report code must not depend on concrete trainers.
