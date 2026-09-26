# Documentation

<!-- nav:start -->
[Architecture](methods/architecture/teacher-architecture.md) · [Train and run](../scripts/rl/README.md) · [Experiments](../scripts/rl/experiments/README.md) · [Research](README.md) · [RL core](../scripts/rl/core/README.md) · [Package](../talon_rl/README.md)

[TALON RL](../README.md) · [Documentation](README.md)
<!-- nav:end -->

The research record for TALON.

Use this tree to answer **why a method exists, what was tested, what the evidence supported, and what conclusions were carried into the thesis**.

## Choose a path

| section | use it for |
|---|---|
| [`methods/`](methods/README.md) | active method descriptions and reusable design notes |
| [`contracts/`](contracts/README.md) | questions, treatments, controls, invariants, and gates fixed before evaluation |
| [`verdicts/`](verdicts/README.md) | retained conclusions from completed experiments |
| [`closures/`](closures/README.md) | branch-level and method-selection decisions |
| [`protocols/`](protocols/README.md) | frozen evaluation, pipeline, and provenance procedures |
| [`baselines/`](baselines/README.md) | baseline and bridge design documents |
| [`thesis/`](thesis/README.md) | thesis-facing method, results, discussion, conclusion, and traceability |
| [`superpowers/`](superpowers/README.md) | historical implementation plans/specifications |

## Research progression

```text
contract
  → experiment implementation
  → run / artifact evidence
  → verdict
  → closure
  → thesis synthesis
```

## Document counts

- [`contracts/`](contracts/README.md) — 68 documents
- [`verdicts/`](verdicts/README.md) — 121 documents
- [`baselines/`](baselines/README.md) — 21 documents
- [`closures/`](closures/README.md) — 6 documents
- [`methods/`](methods/README.md) — 6 documents
- [`protocols/`](protocols/README.md) — 3 documents
- [`thesis/`](thesis/README.md) — 7 documents
- [`superpowers/`](superpowers/README.md) — 12 historical planning/spec documents

## Current architecture

The active method specification is the **[Phase 1 Teacher Architecture](methods/architecture/teacher-architecture.md)**.

## Maintenance

Document folders are organized by **role + research domain**. Filenames intentionally retain provenance-bearing experiment IDs. When a document moves, update repository references to its new `docs/...` path.
