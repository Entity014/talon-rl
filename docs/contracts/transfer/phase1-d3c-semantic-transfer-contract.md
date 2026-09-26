# Phase-1 D3-C — Zero-Adaptation MORL Semantic Transfer Contract

Status: **PREDECLARED — AUTHORIZED AFTER D3-B PASS**
Date: 2026-09-26

## Question

Given exact D3-A interface equivalence and viable D3-B nominal dynamics, do the preference-conditioned T/A/O/S semantics of the frozen Phase-1 controller transfer to MuJoCo without adaptation?

Primary reset arm:

    canonical D2 joint-state initialization

Native MuJoCo home is descriptive only for D3-C; it is not used to select or tune the controller.

## Frozen semantic definitions

The MuJoCo evaluator reimplements the exact Phase-1 weighted reward terms:

Tracking:

    1.5 * exp(-||cmd_xy - v_xy||^2 / 0.25)
  + 0.75 * exp(-(cmd_yaw - wz)^2 / 0.25)

Angular stability:

    -0.05 * (wx^2 + wy^2)

Orientation stability:

    -2.5 * (gx^2 + gy^2)

Smoothness:

    -0.01 * ||a_t - a_(t-1)||^2

Frozen normalization divisors:

    T  1.7194554805755615
    A  0.15590913593769073
    O  0.01563369482755661
    S  0.08311229199171066

No semantic formula or normalization may change after results are observed.

## Matched suites

Four fixed nonzero command suites:

    suite 0  forward      [0.5, 0.0,  0.0]
    suite 1  turn_left    [0.3, 0.0,  0.3]
    suite 2  turn_right   [0.3, 0.0, -0.3]
    suite 3  lateral      [0.0, 0.25, 0.0]

Within each suite, all preferences use the identical reset state and command.

## Preferences

Endpoints + center:

    T  [0.7,0.1,0.1,0.1]
    A  [0.1,0.7,0.1,0.1]
    O  [0.1,0.1,0.7,0.1]
    S  [0.1,0.1,0.1,0.7]
    C  [0.25,0.25,0.25,0.25]

Pairwise continua:

    T-A
    T-O
    T-S
    A-O
    A-S
    O-S

with alpha:

    0, .25, .5, .75, 1

Each rollout:

    64 policy steps
    20 ms policy period
    zero adaptation

## Physical semantic proxies

Use the exact H2a proxy conventions:

    T  |vx-cmd_x| + |wz-cmd_yaw|    lower is better
    A  ||omega_xy||                   lower is better
    O  body tilt degrees              lower is better
    S  ||a_t-a_(t-1)||                lower is better

Additional vy tracking may be reported descriptively but is not substituted for the frozen T proxy.

## Endpoint gate

For T/A/O:

    objective heavy-vs-center correct on >= .75 suites
    physical heavy-vs-center correct on >= .75 suites

S:

    report valid / inconsistent / engineering-confounded
    not a hard D3-C acceptance requirement

## Preference-family gate

    center within heavy envelope      >= .75
    continuum monotonicity            >= .65
    continuum endpoint-between        >= .65

## Robustness interpretation

Survival is reported separately from semantic direction.

If a semantic comparison is dominated by collapse/termination, classify it engineering-confounded rather than semantic failure.

## Decision

D3-C PASS if:
- T/A/O endpoint semantics pass;
- center compromise passes;
- continuum monotonicity and endpoint-between pass;
- no new broad engineering collapse prevents interpretation.

S is characterized separately.

D3-C PASS authorizes D4 guarded hardware bring-up.

D3-C FAIL freezes the semantic-transfer limitation. It does not authorize policy tuning under D3.
