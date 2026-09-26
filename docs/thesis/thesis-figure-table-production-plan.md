# Thesis Figure and Table Production Plan

Status: READY FOR PRODUCTION

## Figures

### Figure 1 — Architecture progression

Purpose: show that conditioning expressivity improves from RV1 to V2-B while complete semantic endpoint success does not.

Panel A:
- RV1 → V2-A → V2-B architecture ladder.

Panel B:
- continuum monotonicity: 0.60938 → 0.62500 → 0.64583;
- endpoint-between: 0.40278 → 0.47917 → 0.52083.

Panel C:
- endpoint passes: 0/4 → 1/4 (Tracking) → 1/4 (Angular).

Caption message:

> Increasing preference-conditioning expressivity improved continuum behavior and causal action authority, but endpoint semantic correctness did not accumulate across all objectives.

### Figure 2 — Causal/evidence map

Purpose: compress the full research path without C-number chronology.

Flow:
historical failures → foundation repair → RV1 → V2-A → V2-B → semantic forgetting → retention ladder → literature-guided round → final no-retention reference.

The source layout is already specified in `docs/thesis/thesis-results-consolidation.md`.

### Figure 3 — Semantic acquisition/forgetting timeline

Purpose: distinguish acquisition failure from retention failure.

Recommended format:
- x-axis: training update/checkpoint;
- y-axis: four semantic axes T/A/O/S;
- marker: endpoint PASS at each checkpoint;
- optional background: semantic score.

Use the V2-B path and retention-gate checkpoints that demonstrate axes becoming valid and later disappearing.

Caption message:

> Semantic competencies can be acquired transiently but are not preserved reliably under continued shared-policy optimization.

### Figure 4 — Retention ladder comparison

Purpose: show the mechanism transition rather than every experiment.

Suggested columns:
- retained object;
- optimization stability;
- plasticity;
- semantic retention;
- final decision.

Rows:
gradient → hard action → hard trajectory → fixed soft Δa → bounded Δa → semantic outcome → DER → Policy Consolidation.

### Figure 5 — Bounded rehearsal budget behavior

Purpose: isolate the successful mechanism finding.

Primary plot:
- per-update weighted auxiliary/mixed gradient ratio;
- horizontal reference at ρ = 0.25.

Secondary plot:
- mixed-total gradient cosine over updates/seeds.

Key summary:
- mean cosine = 0.9745;
- minimum cosine = 0.9698.

Caption message:

> Explicit gradient budgeting prevents auxiliary rehearsal from taking control of the current-learning update without collapsing preference authority.

### Figure 6 — Final method-selection flow

Purpose: final summary figure for Discussion/Conclusion.

Decision blocks:
- foundation valid?
- architecture authority?
- semantic completeness?
- retention efficacy?
- final reference.

End state:
V2-B + Foundation V2 + GAE λ = 0.95 + no retention intervention.

## Tables

### Table 1 — Foundation validation

Columns:
- component;
- original confound;
- repair;
- validation criterion;
- final status.

Rows:
objectives, PPO action semantics, critic heads, critic support/freshness, semantic evaluation.

### Table 2 — Architecture comparison

Use the exact architecture table in `docs/thesis/thesis-chapter-results.md`.

### Table 3 — Retention candidates

Columns:
- family;
- retained object;
- optimization effect;
- semantic effect;
- evidence status.

### Table 4 — Literature-guided adaptations

Rows:
- DER-style diverse replay;
- multi-timescale Policy Consolidation.

Columns:
- source principle;
- PPO-safe adaptation;
- short-gate PASS events;
- mean semantic score;
- decision.

### Table 5 — Contributions and limitations

Two sections:
- supported contribution;
- scope / limitation.

This table should be placed near the end of Discussion to distinguish evidence-supported claims from open problems.

## Production rule

Main text figures and tables should use scientific labels, not internal experiment identifiers.

C-number and run-path references belong in `docs/thesis/thesis-appendix-traceability.md`.
