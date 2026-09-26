# V3-G1-C0 — Fixed-Actor Token-Query Critic Isolation Contract

Status: **PREDECLARED — FROZEN BEFORE EXECUTION**
Date: 2026-09-25

## Question

Does the generalized token-query critic remain valid on seen m=4 when actor parameters are frozen exactly at the G0 representation-equivalent controller?

This gate isolates critic training/support from actor drift.

## Frozen actor

Source:

    runs/objective_set_g0_structural_parity-2026-09-25/objective_set_g0_init.pt

All actor/family/log_std parameters are frozen and must be bitwise unchanged after training.

## Folds

G1-2-C0:
    allowed training cardinalities = {3,4}
    forbidden                     = 2

G1-3-C0:
    allowed training cardinalities = {2,4}
    forbidden                     = 3

Use seeds 73101, 73102, 73103.

## Critic training

Reuse the exact G1 active-set sampler, rollout horizon, reward normalization, active-token query semantics and critic optimizer settings:

    updates          = 30
    horizon          = 32
    num_envs         = 8
    gamma            = .99
    lambda           = .95
    critic lr        = 1e-3
    grad clip        = 1.0

Only critic_body and value_basis parameters may update.

Inactive objectives are absent:
- no critic query;
- no return target;
- no loss contribution.

## Leakage rule

Held-out cardinality is forbidden from rollout generation, critic targets, support, and checkpoint selection.

## Evaluation

At u30 evaluate fresh m=4 H32 critic validity using the same matched endpoint/center suites as Phase 1.

PASS:

    mean H32 EV > 0
    negative EV fraction <= .25

Also require actor parameter max absolute drift = 0 exactly.

## Interpretation

If critic fails across the fixed-actor runs:
    token-query online critic training/support is independently insufficient.

If critic passes while joint G1 failed:
    critic failure is induced by joint actor/state-distribution drift, and the next repair must generalize the Phase-1 critic-support/refresh substrate rather than alter objective-set architecture.

No held-out semantic evaluation is authorized in C0.
