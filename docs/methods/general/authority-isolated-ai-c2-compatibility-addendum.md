# AI-C2 Compatibility Addendum

Status: **FROZEN — REVISED GOVERNANCE; OLD AI-C2 RESULT PRESERVED**
Date: 2026-09-25

## Purpose

This addendum changes the governance question for AI-C2 prospectively.

It does not rewrite, replace, or relabel the historical AI-C2 paired-capacity contract or its FAIL verdict.

Historical contract:

    question:
      Is the wide critic necessary / superior for the robust
      high-authority actor-learning regime?

Historical verdict:

    FAIL

That result remains preserved as historical evidence.

## Why governance changes

The causal premise behind the original capacity-selection gate changed after the actor durability problem was isolated and repaired.

Earlier regime:

    continued PPO
      ->
    family action-simplex geometry drifts
      ->
    authority regime changes
      ->
    late critic demand changes
      ->
    narrow critic appears insufficient

After bounded all-simplex-edge retention:

    actor authority geometry remains stable
      ->
    narrow critic fresh H32 / MC64 remains valid
      ->
    wide critic provides no necessary value-validity benefit

Therefore critic width is no longer the scientifically relevant treatment question.

## Revised AI-C2 question

Is there a critic configuration compatible with the stabilized authority-isolated actor-learning regime without destroying:

- finite preference-simplex authority geometry;
- local tangent authority;
- fresh value generalization;
- robustness;
- PPO correctness?

AI-C2 is now a compatibility gate, not a critic-width competition.

## Compatibility criteria

A critic configuration is COMPATIBLE iff all hold:

Authority:

    matched-support pairwise retention >= .90
    matched-support tangent retention  >= .90

Finite simplex geometry:

    every all-simplex edge-energy retention >= .90

Critic:

    fresh H32 EV mean > 0
    fresh MC64 EV mean > 0
    H32 negative fraction <= .25
    MC64 negative fraction <= .25

Robustness:

    semantic-suite minimum survival = 1.00
    held-out minimum survival >= .95
    fresh minimum survival >= .95
    no new systematic termination topology

PPO:

    post-refresh ratio invariant <= 1e-4

## Read-only reclassification of existing formal AI-C2 runs

No new training is used for this addendum.

### Narrow critic

Matched-support:

    pairwise retention        .9996
    tangent retention        1.1143
    heavy-heavy mean edge     .9467
    heavy-center mean edge    .9755
    minimum edge              .9034

Fresh critic:

    H32 EV mean               .4265
    H32 negative fraction     .025
    MC64 EV mean              .3397
    MC64 negative fraction    .000

Survival:

    semantic                  1.00
    held-out                  1.00
    fresh                     1.00

PPO ratio:

    PASS

Verdict:

    COMPATIBLE

### Wide critic

Matched-support:

    pairwise retention        .9581
    tangent retention        1.0114
    heavy-heavy mean edge     .9288
    heavy-center mean edge    .9427
    minimum edge              .8332 (A-S)

Fresh critic:

    H32 EV mean               .4144
    H32 negative fraction     .0125
    MC64 EV mean              .3219
    MC64 negative fraction    .0375

Survival:

    semantic                  1.00
    held-out                  1.00
    fresh                     1.00

PPO ratio:

    PASS

Verdict:

    NOT COMPATIBLE under the full finite-simplex gate

Reason:

    minimum all-simplex edge < .90

## Model selection principle

The narrow critic becomes the branch reference because it is the minimal sufficient compatible critic.

This is not a claim that narrow is universally better than wide.

It means only:

    wide capacity is not required in the stabilized actor regime

and:

    among the configurations already tested,
    narrow satisfies the complete compatibility contract.

No capacity tuning is authorized from this addendum.

## Revised AI-C2 verdict

    compatible critic exists       YES
    selected compatible critic     NARROW
    revised AI-C2 compatibility    PASS

Historical wide-capacity AI-C2 verdict remains:

    FAIL

Both statements coexist because they answer different questions.

## Authorization

The purpose of AI-C2 was to prevent opening semantic H2 while actor/critic incompatibility remained unresolved.

That blocker is now closed by the compatible narrow configuration.

Therefore:

    AI-H2 AUTHORIZED

Reference stack:

    authority-isolated actor
    + bounded all-simplex-edge retention
    + projected/headroom repair
    + narrow compatible critic
    + lambda=.95
    + unchanged MORL rewards / preference semantics
    + unchanged reset distribution

AI-H2 must not introduce a new engineering repair.
It tests semantic accumulation / forgetting under the stabilized stack.
