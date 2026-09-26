# Phase 3 T1a — Custom Matched-State Cross-Engine Audit Verdict

Status: **FROZEN — SOURCE ANCHOR INVALID / NOT CAUSALLY DECISIVE**
Date: 2026-09-26

## Purpose

T1a attempted to isolate physics by forcing both engines to the same manually chosen root pose, canonical joint pose, zero velocity, fixed command, and clean observation.

## Key result

Initial preference response is exact across engines:

    T-vs-center action-vector max error at t=0 = 0.0
    for all four commands

Thus the deployment policy function and preference input are not the source of the discrepancy.

After closed-loop evolution:

    mean MuJoCo/Isaac T-vs-center action-separation ratio = 1.420
    mean cross-engine action-response cosine H8          = 0.473
    mean cross-engine action-response cosine H64         = 0.318

This is strong evidence that physics-induced state visitation rapidly changes the effective preference-conditioned response.

However the source anchor failed:

    Isaac T semantic correctness under this custom clean matched protocol = 2 / 4

The canonical source result is T-valid, so the custom matched initialization/command protocol changes the source behavioral regime itself.

## Interpretation

T1a demonstrates:

    same initial state + same policy + same preference
        -> exact initial action response

but:

    different physics
        -> rapid trajectory/state divergence
        -> different later preference-conditioned action response

This supports state-visitation sensitivity.

It does NOT establish that this mechanism explains the frozen D3 T reversal, because the custom source protocol no longer reproduces the validated Isaac T semantic regime.

## Decision

    T1a causal anchor        INVALID
    T2 adaptation            NOT AUTHORIZED

Next:

    T1b source-distribution anchor validation

Use canonical H2a reset seeds/state-command distribution and first verify that a deterministic-clean Isaac replay preserves T semantics before replaying those states into MuJoCo.
