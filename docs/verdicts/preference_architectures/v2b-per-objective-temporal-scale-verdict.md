# V2-B Per-Objective Temporal-Scale Audit

Status: FROZEN — OBJECTIVE-SPECIFIC LAMBDA HETEROGENEITY REJECTED UNDER MC32 GEOMETRY

Date: 2026-09-24

## Question

Do Tracking, Angular, Orientation, and Smoothness prefer different temporal credit scales when each is compared against its own long-horizon MC32 gradient?

## Design

Read-only frozen V2-B audit.

For each objective T/A/O/S:
- state sources = center visitation + corresponding heavy-objective visitation
- phases = t8 / t20 / t36 / t52
- seeds = 4
- one common-random-number 32-step continuation per frozen state
- lambda sweep = 0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 1.0
- n-step sweep = 1, 2, 4, 8, 16, 32
- reference = that objective's own MC32 score-gradient direction

Metrics:
- cosine to objective-specific MC32
- norm ratio to MC32
- estimator SNR
- source/phase consistency

## Pooled lambda results

| Objective | lambda=.95 cosine | lambda=.98 cosine | lambda=1.0 cosine | lambda=1 norm ratio | SNR at lambda=1 |
|---|---:|---:|---:|---:|---:|
| Tracking | 0.8733 | 0.9361 | **0.9595** | 0.9244 | 1.3309 |
| Angular | 0.8251 | 0.9037 | **0.9382** | 0.9469 | 1.3249 |
| Orientation | 0.8376 | 0.8991 | **0.9255** | 0.9630 | 1.3107 |
| Smoothness | 0.8954 | 0.9462 | **0.9670** | 0.9586 | 1.3492 |

Lambda=1.0 is the best tested lambda for all four objectives.

Cell-wise consistency:
- Tracking: lambda=1.0 wins 8/8 source x phase cells
- Angular: lambda=1.0 wins 8/8
- Orientation: lambda=1.0 wins 8/8
- Smoothness: lambda=1.0 wins 8/8

Thus there is no evidence that Angular intrinsically prefers lambda~0.95 while Orientation/Smoothness prefer lambda~1.0 under the tested long-horizon MC32 criterion.

## Effective-horizon result

All four objectives show the same qualitative trend: cosine to MC32 rises strongly with n-step horizon.

At n=32:
- Tracking: cosine 0.9595, norm ratio 0.9244
- Angular: cosine 0.9382, norm ratio 0.9469
- Orientation: cosine 0.9255, norm ratio 0.9630
- Smoothness: cosine 0.9670, norm ratio 0.9586

This reinforces that long-horizon geometry is preferred by every objective in the estimator-level audit.

## Interpretation relative to the paired training pilot

The paired training pilot showed:
- lambda=.95: Angular endpoint passes; continuum is better
- lambda=1.0: Orientation/Smoothness improve; Angular collapses; continuum worsens

This audit shows that the behavioral trade-off cannot be explained by different per-objective lambda optima relative to MC32.

Angular also prefers lambda=1.0 in its own estimator geometry, even though Angular behavior becomes worse after joint training with lambda=1.0.

Therefore the remaining failure occurs downstream of per-objective temporal-estimator fidelity.

Most likely remaining loci include:
- joint parameter-space interference during multi-objective updates
- unequal learning-rate / gradient-magnitude effects induced by lambda=1.0
- objective interaction through shared actor parameters over repeated updates
- preference-conditioned visitation changes caused by the jointly trained policy
- optimization-path dependence rather than static per-objective credit geometry

## Causal status

- Global lambda tuning as a final system fix: CLOSED / insufficient.
- Objective-specific fixed lambda heterogeneity: NOT SUPPORTED by MC32 alignment.
- Vector lambda [lambda_T,lambda_A,lambda_O,lambda_S]: NOT AUTHORIZED.
- Adaptive/state-dependent lambda: NOT AUTHORIZED.
- Long-horizon fidelity preference: SHARED across T/A/O/S.
- Behavioral redistribution under lambda=1.0: remains real and must be explained downstream of estimator geometry.

## Decision

Do not open vector-lambda training.

The next diagnostic should target joint optimization dynamics directly. A clean next gate is to compare per-objective actor-gradient magnitudes, cosine conflicts, and update contribution shares under lambda=.95 versus lambda=1.0 on the same frozen batches, then track whether lambda=1.0 changes which objective dominates shared actor parameter updates despite improving every objective's individual MC32 fidelity.