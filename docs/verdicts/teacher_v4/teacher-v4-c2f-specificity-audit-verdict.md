# Teacher V4 — V4-C2F Objective Specificity / Identifiability Verdict

Status: **FROZEN — A and O are distinct objectives (specificity realized in only ~half the checkpoints); the redundant pair is A–S, not A–O: S is subsumed by A. T is fully specific.**
Date: 2026-09-27
Branch: `v4-c2-semantic-preservation`
Contract: [teacher-v4-c2f-specificity-audit-contract.md](../../contracts/teacher_v4/teacher-v4-c2f-specificity-audit-contract.md) (frozen at `5319ad9`)
Data: `specificity_audit.npz` in the 16 V4-C run directories (`specificity_audit.py`, `b04093c`); aggregate `runs/teacher_v4_c2f_specificity_audit-2026-09-27/aggregate.json` (`specificity_aggregate.py`).

## Layers 1–2 (10 primary checkpoints; medians)

| pair | Spearman per step | Spearman 32-step | off-diagonal mass per step (0.5 = independent) | PC1 share, snapshots (0.5 = 2-D) | PC1 share, policies |
|---|---:|---:|---:|---:|---:|
| TA | −0.04 | −0.01 | 0.52 | 0.56 | 0.96 |
| TO | −0.06 | −0.11 | 0.51 | 0.53 | 0.72 |
| TS | −0.05 | −0.03 | 0.52 | 0.56 | 0.95 |
| **AO** | **0.16** | **0.23** | **0.46** | **0.68** | **0.77** |
| **AS** | **0.84** | **0.92** | **0.14** | **0.96** | **0.99** |
| OS | 0.21 | 0.29 | 0.44 | 0.61 | 0.71 |

A–S is the most redundant pair on every layer-1 and layer-2 measure. A–O is
not: it ranks 3rd on Spearman and off-diagonal mass and 2nd on PC1, and its
per-step off-diagonal mass (0.46) is close to that of independent
variables. Upright-but-rotating and tilted-but-still states both occur
often in the feasible distribution.

## Layer 3 — specificity S_i(i⁺) − S_i(j⁺), 95% CI sign over checkpoints

| | primary (10): pos / zero / neg | all (16) |
|---|---|---|
| TT−TA, TT−TO, TT−TS | 10/0/0 each | 16/0/0 each |
| AA−AT | 10/0/0 | 16/0/0 |
| **AA−AO** | **6/0/4** | 8/0/8 |
| AA−AS | 9/0/1 | 13/0/3 |
| OO−OT | 8/2/0 | 14/2/0 |
| **OO−OA** | **4/1/5** | 10/1/5 |
| OO−OS | 6/1/3 | 12/1/3 |
| SS−ST | 10/0/0 | 15/1/0 |
| **SS−SA** | **4/0/6** | 6/0/10 |
| SS−SO | 7/0/3 | 10/1/5 |

## Reading (contract rule; applied in both directions of each pair)

The contract's B/A/C rule was written for "A subsumed by O" and said the same
reading is reported for A–S. It is applied here to both directions of each
pair; that symmetric application is stated openly.

| direction | most redundant pair? | own specificity pos | other's specificity pos | reading |
|---|---|---:|---:|---|
| A vs O | no | 6/10 | 4/10 | **C: regime-dependent** |
| O vs A | no | 4/10 | 6/10 | **A: O distinct from A but not realized** |
| A vs S | yes | 9/10 | 4/10 | C (A *is* specific against S) |
| S vs A | yes | 4/10 | 9/10 | **B: S subsumed by A** |

## Conclusions

1. **A–O is not a formulation problem.** In the feasible locomotion
   distribution, angular rate and tilt are only weakly related (step ρ 0.16,
   near-independent quadrant mass). They are controllably distinct
   behaviors, and specificity between them is realized in about half the
   checkpoints (AA > AO in 6/10, OO > OA in 4/10). This is the case in which a
   semantic-specificity loss is justified for the A/O pair.
2. **A–S is the real redundancy.** Action smoothness (S, ‖Δa‖) and body
   angular rate (A, ‖ω_xy‖) move almost identically (ρ 0.84 per step, 0.92
   over windows; PC1 0.96–0.99). A-heavy is better at A than S-heavy (9/10),
   but S-heavy is *not* better at S than A-heavy (6/10 negative): asking for
   A already buys S. Per the rule: **S is subsumed by A.** A specificity loss
   forcing SS > SA would fight the task's physics.
3. **T is fully specific** against every other objective in every
   checkpoint.
4. This revises the earlier reading. The A–O entanglement suspected in the V4-C
   A decomposition is partial. The A–S entanglement it also flagged is the
   dominant one.

## Decision options for V4-C2 (not taken here)

- S: merge into a stability objective with A, or redefine S so it is
  orthogonal to A (for example, penalize actuator effort or jerk in
  joint-space components that do not drive body rotation).
- A/O: keep both. A specificity loss (pairwise margin, AA > AO and OO > OA)
  is justified by this audit.
