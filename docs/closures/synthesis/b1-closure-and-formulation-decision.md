# B1 closure and formulation decision

## Status

**B1 CLOSED — no seed-robust deterministic locomotion formulation found.**

B1-P2, B1-S1, and B1-R1 were frozen, run across seeds 0–2, and evaluated with the same update-500 deterministic gate. Each intervention produced a distinct partial improvement, but no branch passed all seeds simultaneously:

- P2 primarily improved policy preservation.
- S1 improved local stability in selected basins.
- R1 improved representation/stability and was strongest overall, with one passing seed and substantial gains on the difficult seed, but retained tracking and catastrophic-tail failures.

## Interpretation

The remaining failure is not a single isolated bottleneck. It presents as seed-dependent tracking loss, stability-tail collapse, and basin selection. Further composition of P2/S1/R1 would therefore lose causal clarity and is not authorized under this phase.

## Next decision gate

Before opening B0.2 or returning to MOPPO, choose exactly one path:

1. **Thesis consolidation:** treat B1 as an evidence-backed negative/partial-result contribution.
2. **New architecture-level baseline phase:** define and freeze a new seed-robust locomotion formulation, with a new manifest, acceptance criteria, and single causal question. It must not be named B1-P3/S2/R2 or be an untracked patch composition.

Until that choice is recorded, B0.2, MOPPO, KL sweeps, ACAPS sweeps, and additional reconstruction-coefficient sweeps remain closed.
