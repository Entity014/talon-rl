# V2-B Lambda / Effective-Horizon Audit

Status: FROZEN — LAMBDA=1.0 CONSISTENTLY TRACKS MC32 BETTER THAN CURRENT 0.95; TEMPORAL-CREDIT INTERVENTION HAS DECISION VALUE

Date: 2026-09-23

## Design

Read-only audit on the frozen V2-B checkpoint. No optimizer steps.

Primary objective: Orientation. Controls: Angular and Smoothness.

For every frozen state/source/phase, one 32-step common-random-number continuation was generated. All lambda and n-step estimators were recomputed from the same rewards, values, actions, terminations, and score samples. The target comparator was the corresponding 32-step Monte-Carlo objective-return score direction.

Lambda sweep:
- 0.0
- 0.5
- 0.8
- 0.95 (current training setting)
- 1.0

Effective n-step sweep:
- 1, 2, 4, 8, 16, 32

Metrics:
- cosine to MC32
- gradient norm ratio to MC32
- per-cell consistency over visitation source and phase
- estimator SNR

## Orientation primary result

Pooled over center/O-heavy state sources, t8/t20/t36/t52, and four seeds:

| lambda | cos to MC32 | norm ratio to MC32 | SNR |
|---:|---:|---:|---:|
| 0.0 | 0.050 | 0.275 | 1.460 |
| 0.5 | 0.221 | 0.279 | 1.353 |
| 0.8 | 0.520 | 0.375 | 1.334 |
| 0.95 | 0.838 | 0.637 | 1.331 |
| 1.0 | **0.926** | **0.963** | 1.311 |

Lambda=1.0 beats lambda=0.95 in all 8/8 source x phase cells.
Mean cosine gain = +0.0879; range = +0.0545 to +0.1323.

Per-cell lambda=1.0 cosine to MC32 stays high:
- Center: 0.876 / 0.946 / 0.960 / 0.926 at t8/t20/t36/t52
- O-heavy: 0.881 / 0.950 / 0.934 / 0.931

The SNR cost relative to lambda=0.95 is small and inconsistent in sign by phase; there is no evidence of a large variance penalty.

## Effective-horizon result

Orientation n-step pooled cosine to MC32 rises systematically with horizon:
- n=1: 0.050
- n=2: 0.205
- n=4: 0.394
- n=8: 0.521
- n=16: 0.811
- n=32: 0.926

This independently reproduces the horizon-dependence diagnosis.

## Angular control

Pooled lambda results:
- lambda=0.95: cosine 0.825, norm ratio 0.586, SNR 1.352
- lambda=1.0: cosine 0.938, norm ratio 0.947, SNR 1.325

Lambda=1.0 beats 0.95 in all 8/8 center/heavy-source x phase cells.
Mean cosine gain = +0.1131.

## Smoothness control

Pooled lambda results:
- lambda=0.95: cosine 0.895, norm ratio 0.573, SNR 1.407
- lambda=1.0: cosine 0.967, norm ratio 0.959, SNR 1.349

Lambda=1.0 beats 0.95 in all 8/8 center/heavy-source x phase cells.
Mean cosine gain = +0.0716.

## Interpretation

The current lambda=0.95 setting is already strongly long-horizon oriented, but it is not the closest tested estimator to the 32-step MC geometry.

Across Orientation and both controls, lambda=1.0 consistently:
- increases cosine alignment to MC32
- restores gradient magnitude toward the MC32 scale
- does so without a large observed SNR penalty
- wins every tested source x phase cell against lambda=0.95

This makes the effect broader than an Orientation-only artifact.

The result does NOT show that lambda=1.0 will improve trained behavior. It establishes intervention value: a lambda-only training comparison is now experimentally justified.

## Causal status

- Conditioning architecture insufficiency as primary remaining cause: NOT SUPPORTED.
- Horizon-dependent objective geometry: STRONGLY SUPPORTED.
- Current lambda=0.95 catastrophically short-sighted: REJECTED.
- Lambda=0.95 optimal among tested temporal estimators: REJECTED.
- Lambda=1.0 closer to MC32 geometry: STRONGLY SUPPORTED across O/A/S.
- Large SNR penalty at lambda=1.0: NOT OBSERVED in this audit.

## Decision

V2-C remains OFF.

A minimal temporal-credit intervention pilot is now authorized in principle: keep Foundation V2 and V2-B architecture frozen and change only GAE lambda from 0.95 to 1.0. The comparison must retain the same seed, support, critic, PPO, objectives, evaluation suites, thresholds, and training budget.

No architecture escalation, reward change, critic redesign, or simultaneous PPO modification is authorized by this result.