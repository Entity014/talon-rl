# V2-B Optimizer-History vs Visitation Isolation Verdict

Status: FROZEN — ADAM FIRST-MOMENT HISTORY CONTRIBUTES TO PATH DIVERGENCE BUT DOES NOT EXPLAIN OR PREVENT LATE ANGULAR COLLAPSE

Date: 2026-09-24

## Question

Is the u50->u75 semantic collapse under lambda=1 primarily caused by accumulated Adam optimizer memory, or does it persist when directional optimizer history is removed?

## Clean causal treatment

Control:
- lambda=1.0
- continuous Adam

Treatment:
- same lambda, architecture, seed, foundation, PPO, critic, support, evaluator
- same parameters through u50 (bit-identical checkpoint)
- at u51, zero only Adam exp_avg (first moment)
- preserve exp_avg_sq (second moment)
- preserve Adam step counter / bias-correction state

This isolates directional momentum history without restarting the optimizer or removing the preconditioner.

## Why earlier reset treatments were invalid or blocked

Full optimizer-state clear reset the Adam step counter and caused an artificial restart transient.
Resetting both exp_avg and exp_avg_sq while preserving step caused large effective updates (u51 delta ~0.147 vs control ~0.050), >2x late average parameter delta, and critic-foundation failure.
Those treatments are not used for the causal verdict.

## Optimizer-path effect of m1-only reset

At u51:
- control parameter delta ~0.0497
- m1-reset delta ~0.0212
- m1 before update = 0 by construction
- second moment remains continuous (~0.001388)

Over u51->u75:
- control mean parameter delta: 0.0470
- m1-reset mean parameter delta: 0.0410
- raw actor gradient norm mean is nearly unchanged: 4.210 vs 4.201

Thus the treatment changes the realized optimizer path without changing raw gradient scale materially.

## Parameter-path displacement u50->u75

| block | continuous norm | m1-reset norm | ratio | displacement cosine |
|---|---:|---:|---:|---:|
| shared body | 0.724 | 0.611 | 0.844 | 0.671 |
| direct input | 0.316 | 0.265 | 0.840 | 0.660 |
| embedding | 0.0465 | 0.0385 | 0.828 | 0.811 |
| FiLM | 0.141 | 0.131 | 0.927 | 0.747 |
| actor head | 0.166 | 0.156 | 0.937 | 0.735 |
| log_std | 0.00893 | 0.00554 | 0.620 | 0.545 |

So Adam first-moment history clearly contributes to optimization-path direction and travel distance.

## Semantic outcome

| metric | continuous Adam | m1-reset at u50 |
|---|---:|---:|
| Tracking obj/phys | .25/.50 | .50/.50 |
| Angular obj/phys | .00/.00 | .25/.25 |
| Orientation obj/phys | .50/.50 | .25/.50 |
| Smoothness obj/phys | .75/.75 (PASS) | .25/.25 |
| endpoint passes | 1/4 (S) | 0/4 |
| continuum monotonicity | .635 | .568 |
| endpoint-between | .417 | .319 |
| critic H32 EV | .309 | .354 |
| critic negative fraction | .1125 | .0625 |
| survival | 1.0 | 1.0 |

Angular improves slightly but remains a clear failure. Overall semantic controllability becomes worse.

Because critic freshness and survival remain clean, this is a valid behavioral comparison.

## Visitation summary

Late u51->u75 coarse visitation summaries remain similar:
- observation mean norm: 2.596 (control) vs 2.538 (m1-reset)
- observation std mean: 1.204 vs 1.226
- objective means remain close across all four objectives

This does not prove visitation equivalence at the full state-distribution level, but it shows no gross regime collapse caused by the m1 reset.

## Interpretation

Adam directional memory has a causal effect on the late parameter path, but removing that memory is not sufficient to preserve Angular semantics or improve the overall MORL solution.

Therefore:
> optimizer memory is a contributor to path divergence, not the primary sufficient cause of late semantic collapse.

The strongest remaining explanation is on-policy shared-policy path dependence: policy updates alter the visited state/action distribution and representation manifold, so locally valid gradients continue to act on a moving closed-loop system whose semantic realization changes over time.

## Causal status

- Static local credit failure: rejected.
- Static mixed-gradient suppression: rejected.
- Simple parameter-step overshoot: rejected.
- Adam first-moment history contributes to path divergence: established.
- Adam first-moment history as sufficient cause of Angular collapse: rejected.
- Removing first-moment memory as a useful method: not supported.
- On-policy visitation / shared-manifold evolution: primary remaining suspect.

## Decision

Do not adopt optimizer resets as a training method.
Do not reopen lambda, architecture, LR, or vector-lambda branches.

If diagnosis continues, the next clean gate should isolate policy-distribution feedback directly, for example by comparing fresh-gradient geometry on replayed matched state distributions from u50 versus u75 while holding policy parameters fixed, or by training with externally frozen/matched visitation support for a short controlled interval. The question is whether changing on-policy state distribution, rather than optimizer memory, causes the semantic realization to drift.