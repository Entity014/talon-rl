# Experiments

<!-- nav:start -->
[Architecture](../../../docs/methods/architecture/teacher-architecture.md) · [Train and run](../README.md) · [Experiments](README.md) · [Research](../../../docs/README.md) · [RL core](../core/README.md) · [Package](../../../talon_rl/README.md)

[TALON RL](../../../README.md) · [RL runner](../README.md) · [Experiments](README.md)
<!-- nav:end -->

Executable research workflows for TALON.

This folder answers **scientific questions**. Reusable mechanisms should live in `talon_rl/` or `scripts/rl/core/` instead.

## Choose a path

| area | use it for |
|---|---|
| [`architectures/`](architectures/README.md) | policy/critic architecture, preference authority, policy-family, and semantic-control studies |
| [`baselines/`](baselines/README.md) | scalar/reference baselines, stability work, and MORL bridge experiments |
| [`diagnostics/`](diagnostics/README.md) | read-only causal, semantic, trajectory, and update diagnostics |
| [`evaluation/`](evaluation/README.md) | frozen task-level evaluation workflows |
| [`transfer/`](transfer/README.md) | deployment equivalence, plant alignment, controller robustness, and ensemble robustness |
| [`common/`](common/README.md) | experiment-side utilities reused by multiple workflows |

## Research IDs vs. filesystem names

Folder and file names describe **responsibility**. Historical experiment IDs such as B0, V2B, C25, AI-C2, and Phase5-E2 stay in stage names, run directories, contracts, verdicts, and artifacts.

## How an experiment connects to the research record

```text
contract
  ↓
experiment workflow
  ↓
run / artifact evidence
  ↓
verdict
  ↓
closure / thesis synthesis
```

Start from [Research docs](../../../docs/README.md) if you are reconstructing why an experiment exists.

## README generation

Leaf experiment READMEs are generated from module docstrings:

```bash
python scripts/rl/experiments/generate_readmes.py --apply
```

Edit a script's module docstring when you want to improve its generated description.
