# Analysis

[TALON RL](../../README.md)

Post-hoc analysis, audit, trace, probe, and diagnostic tooling for TALON.

This subtree is intentionally separate from `scripts/rl/`: training/runtime code should not depend on research-analysis implementations.

## Start here

| area | what you will find |
|---|---|
| [Diagnostic runner](run_diagnostics.py) | Single CLI entry point for eval-only diagnostics and probes. |
| [Diagnostic implementations](diagnostics/README.md) | Decompositions, rollout traces, interventions, counterfactuals, physical consistency checks, and update diagnostics. |
| [Audit helpers](audits.py) | Shared analysis helpers whose outputs are used by retained research evidence. |

## Boundary

Use `scripts/analysis/` for code whose primary purpose is to **inspect, compare, decompose, or explain** an already-defined policy/training process.

Use `scripts/rl/core/diagnostics/` for reusable diagnostic functionality that is part of the RL infrastructure itself.

Use `scripts/rl/experiments/diagnostics/` for controlled scientific workflows that answer a preregistered research question.

In short:

```text
core/diagnostics        reusable diagnostic mechanism
experiments/diagnostics scientific diagnostic experiment
scripts/analysis        analysis CLI + post-hoc tooling
```

## Running diagnostics

List available commands:

```bash
PYTHONPATH="$(pwd):$(pwd)/scripts" python scripts/analysis/run_diagnostics.py --list
```

Run one diagnostic:

```bash
PYTHONPATH="$(pwd):$(pwd)/scripts" \
python scripts/analysis/run_diagnostics.py static-standing --help
```

Each subcommand forwards its remaining arguments to the corresponding implementation module.
