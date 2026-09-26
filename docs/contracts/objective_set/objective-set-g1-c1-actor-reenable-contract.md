# V3-G1-C1 — Actor Re-enable with Validated Critic Substrate

Status: **PREDECLARED — 10-UPDATE CAUSAL SCREEN**
Date: 2026-09-25

## Motivation

Initial G1 training preserved preference authority but collapsed fresh critic validity and T/A/O semantics by u10. G1-C0 showed that a Phase-1-style 64-step support + ridge(lambda=1) token-query critic restores strong fresh value validity in both folds with zero actor drift.

## Sole change from failed G1 trainer

Replace online critic SGD with current-policy support/ridge refresh.

Actor architecture, objective-set sampler, PPO scalarization, tail control, edge retention, learning rate, and fold leakage rules remain unchanged.

## Screen

Run seed 73101 independently for both folds:

    G1-2 train cardinalities {3,4}
    G1-3 train cardinalities {2,4}

Budget:

    10 actor updates

At every update, before computing GAE:
- collect one deterministic 64-step support batch for each allowed cardinality;
- each support batch contains only that allowed cardinality, with lane-wise active subsets/preferences sampled deterministically;
- fit only the token-query value basis with ridge lambda=1.0;
- critic body remains frozen;
- held-out cardinality is never queried or fabricated.

No critic optimizer is used.

## u10 gate

Evaluate the same frozen m=4 anchor suites used in the temporal audit.

Require:

    fresh critic EV mean > 0
    negative EV fraction <= 0.25
    T endpoint PASS
    A endpoint PASS
    O endpoint PASS
    endpoint survival >= 0.95
    preference authority retention >= 0.75
    permutation drift <= 1e-6

S is reported only.

## Decision

Both folds PASS at u10:
- authorize full 30-update C1 runs and final G1 evaluation.

Either fold FAIL:
- do not return to held-out-cardinality testing;
- localize remaining actor/credit interaction first.
