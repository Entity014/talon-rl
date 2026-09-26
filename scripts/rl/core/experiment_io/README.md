# Runs

<!-- nav:start -->
[RL runner](../../README.md) · [RL core](../README.md)
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
