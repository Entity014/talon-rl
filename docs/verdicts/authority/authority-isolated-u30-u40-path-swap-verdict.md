# U30/U40 Common-vs-Family Path-Swap Robustness Verdict

Status: **FROZEN — DISTRIBUTED PATH CONTRIBUTION; FULL U40 REGRESSION REQUIRES COMMON/FAMILY INTERACTION IN MOST CASES**
Date: 2026-09-25

## Question

Which actor component owns the fresh robustness regression that appears after continuing the durability-valid u30 actor to u40?

Actor decomposition:

COMMON path:
    actor_body
    actor_mean

FAMILY path:
    all family_* parameters
    including base realization, preference hypernetwork, and generated bases

## Policies

Reference:
    full u30

Endpoint:
    full u40

Hybrids:

    H_common:
      common = u40
      family = u30

    H_family:
      common = u30
      family = u40

The audit is frozen/read-only.

All 20 fresh preference x seed cases are evaluated for every policy.

## Important center-preference interpretation

Unlike the previous preference-delta interpolation audit, swapping family_* parameters changes the full family realization block, including its center operating point.

Therefore C can change under a family-path swap.

This audit localizes implementation-path ownership, not only preference-delta ownership.

## Authority geometry

NARROW:

    policy      pairwise   tangent   min edge

    u30          1.034      1.020      .931
    u40          1.552      1.544     1.118
    H_common     1.213      1.222     1.022
    H_family     1.454      1.459     1.075

WIDE:

    u30          1.034      1.020      .931
    u40          1.586      1.609     1.144
    H_common     1.241      1.271     1.036
    H_family     1.465      1.435     1.106

All hybrids retain valid simplex geometry.

Therefore hybrid rescue is not obtained by collapsing preference authority below the validated lower floor.

## Fresh robustness

NARROW:

    policy      failed lanes

    u30             0
    u40             3
    H_common        1
    H_family        2

WIDE:

    u30             0
    u40             6
    H_common        1
    H_family        1

Neither path alone reproduces the full u40 failure topology.

## NARROW failure ownership

Full u40 failures:

    T / 9700000 / lane2
    T / 9700226 / lane6
    A / 9701000 / lane2

H_common:

    T / 9700000 / lane2

H_family:

    T / 9700000 / lane2
    A / 9701000 / lane2

Interpretation:

    T / 9700000
      can be triggered by either changed common or changed family path
      -> distributed susceptibility

    A / 9701000
      transfers with u40 family path
      -> family-path-dominant

    T / 9700226
      neither hybrid fails
      -> common x family interaction required

No hybrid creates a new failure outside the full-u40 failure set.

## WIDE failure ownership

Full u40 failures:

    T / 9700000 / lane2
    T / 9700226 / lane6
    A / 9701000 / lane2
    A / 9701226 / lane5
    S / 9703226 / lane3
    C / 9704339 / lane0

H_common:

    T / 9700000 / lane2

H_family:

    A / 9701000 / lane2

Thus:

    T / 9700000
      -> common-path-dominant

    A / 9701000
      -> family-path-dominant

    T / 9700226
    A / 9701226
    S / 9703226
    C / 9704339
      -> neither path alone reproduces failure
      -> interaction candidates

Again no hybrid creates a new failure outside the full-u40 topology.

## Center case

WIDE C / 9704339 / lane0:

    u30       survives
    u40       fails at t=53
    H_common  survives
    H_family  survives

This is particularly informative.

The C regression is not owned by the changed common path alone or by the changed family realization alone.

It requires their u40 combination.

Therefore at least one robustness mechanism is genuinely an interaction effect between state-only and family realization changes.

## Operating-regime checks

Examples:

WIDE T / 9700000 / lane2:

    u30:
      survives
      max pre-tanh abs 37.43
      saturation fraction .754

    u40:
      fails at t=60
      max pre-tanh abs 47.91
      saturation fraction .723

    H_common:
      fails at t=60
      max pre-tanh abs 48.52
      saturation fraction .755

    H_family:
      survives
      max pre-tanh abs 36.70
      saturation fraction .754

This case follows common-path ownership.

WIDE A / 9701000 / lane2:

    u30       survives
    u40       fails
    H_common  survives
    H_family  fails

This case follows family-path ownership.

WIDE C / 9704339 / lane0:

    u30:
      survives
      saturation .884

    u40:
      fails t=53
      saturation .724

    H_common:
      survives
      saturation .781

    H_family:
      survives
      saturation .750

Thus saturation level alone does not explain the regression.

## Causal conclusion

Fresh u40 robustness regression is heterogeneous but structurally organized:

1. common-path-dominant cases exist;
2. family-path-dominant cases exist;
3. interaction-only cases are the majority in WIDE;
4. no evidence supports single-path ownership of the whole regression;
5. no evidence supports global authority magnitude or saturation as the sole cause.

The WIDE endpoint is especially clear:

    6 full-u40 failures
    only 1 reproduced by H_common
    only 1 reproduced by H_family
    4 require the combined u40 actor

Therefore:

    fresh robustness regression
      != common drift only
      != family drift only
      ~= path-specific failures + common/family interaction

## Decision

    common-only training repair      NOT AUTHORIZED
    family-only training repair      NOT AUTHORIZED
    global upper authority cap       NOT AUTHORIZED
    new saturation penalty           NOT AUTHORIZED
    AI-H2                            REMAINS BLOCKED

The validated lower simplex floor remains retained.

## Next justified gate

Do not train yet.

The next clean question is whether the interaction-only failures arise from incompatibility between the u40 common-state representation and u40 family realization at the same visited states.

A minimal read-only next audit should target only the interaction cases:

    narrow:
      T / 9700226

    wide:
      T / 9700226
      A / 9701226
      S / 9703226
      C / 9704339

Use matched pre-contact states and evaluate the 2x2 actor-function matrix:

    common30 + family30
    common40 + family30
    common30 + family40
    common40 + family40

at the same states.

Measure:
- pre-tanh action decomposition;
- coordinate-wise common contribution;
- coordinate-wise family contribution;
- interaction residual caused by nonlinear family realization;
- action/state divergence in the 5-10 steps before the full-u40 split.

The purpose is not to identify another global penalty.
It is to test whether continued PPO created a compatibility mismatch between the two actor pathways in specific late-state regimes.

Only if a recurrent cross-case interaction signature appears should a training-side compatibility constraint be considered.
