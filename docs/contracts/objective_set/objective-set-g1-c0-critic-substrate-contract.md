# V3-G1-C0 — Token-Query Critic Substrate Parity Contract

Status: **PREDECLARED — FIXED POLICY / CRITIC-ONLY**
Date: 2026-09-25

## Question

Before interpreting G1 training failure as cardinality generalization failure, can the G0 objective-set controller retain valid value estimation across variable seen cardinalities when the validated Phase-1 critic-support procedure is generalized to active objective tokens?

This gate exists because the first G1 trainer changed critic training from the validated Phase-1 procedure (64-step representative support + ridge head refresh) to online 32-step SGD. That change is now treated as a confound.

## Frozen actor

Use the exact G0 objective-set model. Actor parameters and log_std are frozen exactly.

No actor update is allowed.

## Folds

G1-2 critic support cardinalities: {3,4}; m=2 remains absent.
G1-3 critic support cardinalities: {2,4}; m=3 remains absent.

No held-out cardinality may be used in support fitting or validation for C0.

## Support

For every allowed active set:
- center + all heavy endpoints;
- 64-step deterministic policy rollout;
- split into early [0,32) and late [32,64) phases;
- exact active-objective truncated returns.

The critic body is frozen.

For each active objective token e_i, collect feature/return pairs only when i is active.
Fit the token-query value basis with ridge lambda=1.0 using all allowed support.

No inactive objective target is fabricated or zero-filled.

## Validation

Use fresh matched reset seeds not used in fitting.
Evaluate every training-seen active set and the m=4 anchor.

Fresh H32 critic gate:

    mean EV > 0
    negative EV fraction <= 0.25

Report per-cardinality and per-objective EV/bias.

## Actor invariance

Because actor is frozen:

    max action drift = 0
    max pre-tanh drift = 0

within numerical tolerance 1e-6.

## Decision

PASS in both folds:
- generalized token-query critic substrate is viable;
- authorize G1-C1 actor re-enable using this support/ridge critic procedure.

FAIL:
- stop G1 training work and localize critic representation before any new actor experiment.

This gate does not evaluate held-out cardinality generalization and cannot authorize G2.
