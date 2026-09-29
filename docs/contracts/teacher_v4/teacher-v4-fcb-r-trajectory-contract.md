# Teacher V4 — FC-B R Semantic Trajectory Audit (read-only)

Status: **FROZEN 2026-09-29, with `fcb_trajectory.py`, before any FC-B trace was read.**

*Timing disclosure:* the first freeze commit failed on a syntax error in
the analysis script's print line, and the trace collection had already
launched in the same shell call. This commit fixes only that print
formatting. The analysis logic is unchanged. No trace had been read, and
the analysis runs only after collection ends.
Branch: `v4-c2-semantic-preservation`
Follows: [FC-A verdict](../../verdicts/teacher_v4/teacher-v4-fca-r-credit-audit-verdict.md). Single-update mechanisms do not explain the R failure. At checkpoint 300, T dominated credit where λ was large.

## Question

When does R lose its semantic direction during FB-2a training, and how
does that timing relate to the dual (λ) trajectory?

No training, no formulation change, no per-checkpoint credit audit.

## Evidence types (kept separate)

- **Historical** (`metrics.jsonl`, online): λ per region and online tl per
  region, every iteration.
- **Checkpoint replay**: what the learned policy at checkpoint u can do,
  under the same reset suite and evaluator (F2-A protocol with `--traces`,
  every checkpoint 50 … 600, conditions R vertex, R⁺, C, O⁺, O vertex). It
  is **not** the training rollout.

Seeds: FB-2a 79101–79103.

## Measures per checkpoint u

Window medians of the FC-0 features (`fc0_audit.windows`): F_rate
(‖ω_xy‖²), F_osc, F_O. Replay tl.

- Δ_R(u) = F_rate(R⁺)/F_rate(C) − 1 (F_osc reported alongside);
- Δtl_R(u) = tl(R⁺) − tl(C);
- Δ_O(u) = F_O(O⁺)/F_O(C) − 1: the O positive control, expected negative;
- λ window: the mean of λ_R⁺ and λ_C over the 50 iterations before u.
  Phase: **high** if ≥ 3.0 (about α 0.75 and above), **low** if ≤ 0.1, mid
  otherwise.

R state, read only at **viable** checkpoints (tl(R⁺) ≥ 0.40 and tl(C) ≥
0.40, so standing is never compared with locomotion): **correct** if Δ_R ≤
−5 %, **flat** if |Δ_R| < 5 %, **inverted** if Δ_R ≥ +5 %.

## Pre-declared case (per seed)

Let u* be the first viable inverted checkpoint.

| case | condition | reading |
|---|---|---|
| A | u* in the high-λ phase, and the last viable endpoint checkpoint (≥ 550) still inverted | consistent with early task-dominated shaping / path dependence |
| B | no inversion at high-λ checkpoints; u* after λ fell | high-λ is not the main cause; points to accumulated preference updates / long-horizon drift |
| C | u* before any high-λ checkpoint (or no high-λ phase) | high λ does not explain the onset |
| D | correct at a low-λ checkpoint, then inverted later | later across-update drift |
| — | fewer than 3 viable checkpoints | unresolved |
| — | never inverted at viable checkpoints | no inversion in this metric |

"Consistent with" is deliberate: timing alone does not establish
causality. Cases are checked in the order C, A, D, B (as coded). The
per-seed cases are reported. No majority rule is imposed.

Next-step mapping (not decided by FC-B): case A leads to a formulation
intervention (R credit protected while λ is high). B or D lead to a
multi-update probe started from the checkpoint before u*.

Scripts: `f2a_bifurcation.py --traces` (collection, every checkpoint),
`fcb_trajectory.py` (analysis). Output: `runs/teacher_v4_fc-2026-09-29/fcb/`.
