# Diagnostics

<!-- nav:start -->
[RL runner](../../README.md) · [RL core](../README.md)
<!-- nav:end -->


## Purpose

Future home for reusable audits, validation helpers, and analysis utilities that inspect
training/runtime behavior without defining the learning algorithm.

## Current State

Existing diagnostic code remains at:
- `analyzer.py`
- `physics_validator.py`
- `isaac_audit.py`
- `offline_audit.py`

Run/report infrastructure now lives under `core/experiment_io/`.

## Rules

Diagnostics should be read-only whenever possible. Experimental one-off diagnostics belong
under `scripts/rl/experiments/` rather than becoming permanent core dependencies.
