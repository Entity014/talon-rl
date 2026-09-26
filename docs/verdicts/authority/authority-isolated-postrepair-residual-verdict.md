# Authority-Isolated Post-Repair Residual Verdict

Status: **FROZEN — RESIDUAL SPLITS INTO TWO MECHANISMS**
Date: 2026-09-25

## Best training-side candidate so far

Projected PPO/tail conflict removal + 5% active tail descent, u10.

Results:
- pairwise authority retention: 1.448
- tangent Jacobian retention: 1.634
- fixed-probe tail-fraction delta: -0.0049
- training termination fraction: 0 for updates 4..10
- semantic suites2/3 failed lanes: 5
- minimum survival: 0.75
- frozen u75 baseline failed lanes: 9

This is a substantial partial repair but does not pass the min-survival >= .95 gate.

## Remaining failures

    suite2 / O : lane 2, lane 7
    suite3 / O : lane 0
    suite3 / S : lane 0
    suite3 / C : lane 0

All terminate by base_contact.

## Mechanism split

### A. suite3 / O / lane0: invariant residual

u75 and repaired-u10 are nearly identical in both action and state trajectory and both fail at step 14.

Representative pre-contact values are effectively unchanged.

This failure is therefore not explained by the repaired tail/headroom mechanism and should not be targeted by stronger global tail shaping.

### B. suite2 / O: front-left support geometry

The repaired actor changes the front-left action geometry more clearly.

For the new regression suite2/O/lane7:
- frozen u75 survives
- repaired-u10 fails at step 15
- pre-contact roll-rate is substantially larger under repaired-u10
- largest action differences are FL_hip and FL_calf

Counterfactual donor rescue from u75 evaluated on the repaired actor's current states:

    repaired alone                FAIL @ 15
    all u75 coordinates          SURVIVE
    u75 FL_hip only              SURVIVE
    u75 FL_calf only             FAIL @ 15
    u75 RR_hip only              FAIL @ 15
    u75 FL_hip + FL_calf         SURVIVE
    u75 front-left leg           SURVIVE
    u75 front legs               SURVIVE

Thus FL_hip alone is sufficient to rescue the new suite2/O/lane7 regression.

## Interpretation

The headroom repair solved part of the global late-saturation problem while preserving/growing preference authority, but the residual is no longer a single mechanism.

1. suite2/O contains a coordinate-specific FL-hip support-geometry regression.
2. suite3/O/lane0 is almost invariant to the repair and appears to be a separate state/dynamics robustness limit.
3. suite3/S and suite3/C are partially shifted by the repair and fail one step later, but remain on the same lane0 contact boundary.

Therefore further global headroom tuning is not justified.

## Decision

- projected + active tail descent remains the best partial repair
- do not open AI-C2/H2
- do not increase kappa globally
- do not hard-code FL_hip as the final method
- do not treat all five residual failures as one saturation problem

The next branch should separately diagnose:
- generic actuator-aware support preservation for O-heavy regimes, using FL_hip only as a causal probe;
- the invariant suite3/O/lane0 state/dynamics failure, which likely requires a different robustness mechanism.
