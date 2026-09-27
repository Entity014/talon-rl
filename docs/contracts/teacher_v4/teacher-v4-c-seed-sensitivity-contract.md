# Teacher V4 — V4-C Seed-Sensitivity Characterization Contract

Status: **PREDECLARED — FROZEN 2026-09-27, before any of the new runs were trained**

## Question

How reproducible is the m = 4 A semantic direction (and T/O/S, and A/S critic
validity) across stochastic training seeds, under the unchanged V4-C recipe?
This is not an attempt to make G1 pass. The V4-C G1 and G1-R verdicts stay
FAIL.

## Runs

- Same folds, architecture, trainer (`train_v4c.py` at `822cbcb`), env,
  contracts and 300-iteration budget as V4-C G1.
- New seeds: 73104, 73105, 73106, 73107, 73108 for each of G1-2 and G1-3 = 10
  new runs.
- Checkpoint: iteration 300 only. No checkpoint selection, no tuning, no
  seed replaced, and no run excluded for any outcome (a run stopped by a gate
  counts as a failure on every metric).
- No seed from this set is ever picked as a "final model".

## Per-run measurements (m = 4 set, G1 m = 4 suite seeds, iteration 300)

Script `anchor_evaluate.py`. Before it is used on new runs, it must reproduce
the stored G1 m = 4 endpoint values of the six existing runs.

- Endpoint pass for T, A, O, S: the G1 rule (objective-correct fraction ≥ 0.75,
  physical-correct fraction ≥ 0.75, endpoint survival ≥ 0.95), heavy vs center,
  4 suites × 64 steps.
- A-heavy and center mean ‖ω_xy‖, ΔJ_A, O-heavy ‖ω_xy‖.
- Authority pairwise retention vs V3 G0 (as in G1).
- Minimum endpoint survival.
- Critic validity for A and for S: MC256 target (G1-R form) on the same
  rollouts, registered pooling per objective (EV per rollout × segment),
  valid if mean EV > 0 and negative fraction ≤ 0.25.

## Aggregates

Pass counts for A, T, O and S endpoints, and for A and S critic validity.
Reported for the 10 new runs (primary) and for all 16 runs (the 6 existing
plus the 10 new), per fold and pooled.

## Interpretation of the A pass count on the 10 new runs (fixed now)

- 8–10 / 10: A largely reproducible, seed-sensitive at the margin
  (10/10: reproducible).
- 3–7 / 10: A semantic correctness is unstable under the current formulation.
- 0–2 / 10: effectively a systematic A failure.

The next decision (accept the limitation, or open a V4-D / V5 redesign) is
made only after this characterization is frozen.
