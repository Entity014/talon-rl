# Authority-Isolated Over-Expansion Causal Audit Verdict

Status: **FROZEN — GLOBAL UPPER AUTHORITY CAP NOT AUTHORIZED**
Date: 2026-09-25

## Question

Are the fresh u40 robustness regressions caused by preference-action simplex expansion beyond a robust operating envelope?

If yes, a two-sided authority band would be justified:

    gamma_min d_ij^20 <= d_ij^theta <= gamma_max d_ij^20

If no, the current lower simplex floor should not be converted into a global upper cap.

## Policies

Robust reference:
    edge-retained u30

Expanded endpoints:
    formal AI-C2 narrow u40
    formal AI-C2 wide u40

## Intervention

The first action-space interpolation attempt was discarded because direct residual interpolation could produce actions outside the valid actuator range.

That result is not used for the verdict.

The authoritative audit uses physically valid pre-tanh preference-response interpolation.

For each current state:

    z_lambda(s,w)
      = z_40(s,C)
        + Delta z_30(s,w)
        + lambda [Delta z_40(s,w)-Delta z_30(s,w)]

with:

    lambda in {1.0, .75, .5, .25, 0.0}

and:

    a_lambda = tanh(z_lambda)

Thus:
- the u40 center/common response is preserved;
- only preference-conditioned family response is interpolated;
- every action remains in the valid action range.

All 20 fresh preference x seed cases are evaluated at every lambda, providing both failing-lane and survivor controls.

## Authority response

NARROW:

    lambda   pairwise   tangent   min edge
    1.00      1.569      1.632      1.160
     .75      1.527      1.548      1.140
     .50      1.510      1.512      1.149
     .25      1.515      1.522      1.139
    0.00      1.541      1.576      1.151

WIDE:

    lambda   pairwise   tangent   min edge
    1.00      1.624      1.688      1.193
     .75      1.573      1.615      1.173
     .50      1.546      1.587      1.174
     .25      1.544      1.602      1.162
    0.00      1.565      1.661      1.184

The interpolation reduces authority somewhat, but tanh/output operating geometry keeps the simplex substantially expanded relative to u20.

No lambda approaches the lower authority floor.

## Fresh robustness response

NARROW failed lanes:

    lambda   failed lanes
    1.00         3
     .75         3
     .50         3
     .25         2
    0.00         2

WIDE:

    lambda   failed lanes
    1.00         5
     .75         4
     .50         7
     .25         4
    0.00         6

There is no monotonic relationship:

    authority scale down
      !=
    robustness monotonically up

The best lambda differs by arm and does not eliminate failure.

## Failure transitions

NARROW baseline failures:

    T / 9700000
    T / 9700226
    A / 9701000

At lambda=.25:
- both T failures are rescued;
- A/9701000 persists;
- a new S/9703226 failure appears.

At lambda=0:
- both T failures remain rescued;
- A/9701000 persists;
- S/9703226 remains newly failed.

Thus authority reduction trades failure regimes rather than globally restoring robustness.

WIDE baseline failures in this audit:

    T / 9700000
    A / 9701000
    A / 9701226
    S / 9703226
    C / 9704339

Examples:
- lambda=.75 rescues T/9700000 but leaves A/A/S/C failures;
- lambda=.50 adds T/9700226 and A/9701113 failures;
- lambda=.25 rescues A/9701000 and S/9703226 but adds T/9700226;
- lambda=0 rescues S/9703226 but adds A/9701113 and retains other failures.

Again the intervention exchanges failure topology instead of producing a common robust band.

## Invariant center failure

WIDE:

    C / seed 9704339 / lane0

fails at exactly:

    t=53

for every lambda.

This is a strong internal negative control.

For C:

    Delta z(s,C) = 0

so the interpolation leaves the C action policy unchanged by construction.

Therefore this failure cannot be caused by preference-simplex over-expansion.

At least one formal AI-C2 fresh failure is outside the authority-expansion mechanism.

## Run-to-run variability

The original formal endpoint audit reported 6 WIDE fresh failed lanes.

The physically valid lambda=1 reproduction in this audit produced 5.

The discrepancy is a borderline near-horizon case and indicates small simulation/runtime variability around some marginal failures.

Therefore one-lane differences are not treated as decisive causal evidence.

The central verdict relies on:
- non-monotonic response;
- failure swapping;
- cross-arm inconsistency;
- the invariant C negative control.

## Operating-regime interpretation

The physically valid pre-tanh intervention does not produce the dramatic rescue seen in the discarded unbounded action-space interpolation.

Thus the earlier apparent rescue was partly an artifact of leaving the valid action manifold.

With the corrected intervention:
- actions remain bounded;
- authority remains high;
- failure topology changes non-monotonically.

There is no clean scalar authority threshold separating safe and unsafe behavior.

## Decision

Supported:

    lower simplex floor remains validated
    authority durability remains solved
    authority expansion correlates with some fresh failures
    some T failures are causally sensitive to preference-response scale

Not supported:

    global over-expansion as the sole fresh-robustness cause
    monotonic robust authority envelope
    single global upper authority cap
    two-sided edge band as the next training intervention

Therefore:

    upper authority cap         NOT AUTHORIZED
    formal AI-C2                remains FAIL
    AI-H2                       remains BLOCKED

## Next implication

Do not reopen broad mechanism mining.

The current evidence says the u40 fresh failures are heterogeneous:

1. preference-response-sensitive failures
   - e.g. some T cases can be rescued by reducing family response;

2. preference-response-insensitive failures
   - e.g. WIDE C/9704339 is invariant under the intervention;

3. tradeoff cases
   - reducing response rescues one lane while creating another.

The next useful step should therefore not be a global authority regularizer.

A clean next branch would test whether the fresh failures are a consequence of continuing all actor parameters after u30 rather than of authority scale itself:

    freeze/retain the validated family simplex mechanism
    isolate common/state-path continuation drift versus family-path continuation drift
    on the exact fresh cases

This should be a read-only hybrid/path-swap audit before any new training intervention.
