# Experiments

<!-- nav:start -->
[TALON RL](../../../README.md) · [RL runner](../README.md)
<!-- nav:end -->


Research workflows grouped by responsibility rather than experiment chronology.

## Layout

```text
architectures/   model architecture, preference authority, critic, and policy-family research
baselines/       scalar/reference/MORL bridge and robustness baselines
common/          reusable experiment-side utilities
diagnostics/     cross-cutting read-only audits and causal diagnostics
evaluation/      final task-level evaluation workflows
transfer/        deployment, plant alignment, controller robustness, and ensemble robustness
```

Experiment IDs such as B0, V2B, C25, and Phase 5 remain in stage names, run directories,
schemas, and artifacts for thesis provenance. Folder and file names describe responsibility.

Run `python scripts/rl/experiments/generate_readmes.py --apply` after changing leaf scripts.
