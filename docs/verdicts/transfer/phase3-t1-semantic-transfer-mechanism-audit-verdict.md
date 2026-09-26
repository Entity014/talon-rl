# Phase 3 T1 — Semantic Transfer Mechanism Audit Verdict

Status: **FROZEN — MECHANISM IDENTIFIED ENOUGH TO AUTHORIZE ONE MINIMAL ACTUATOR-CALIBRATION TEST**
Date: 2026-09-26

## Frozen baseline

D3 remains unchanged:

    D3-A interface equivalence      PASS
    D3-B nominal dynamics           PASS
    D3-C semantic transfer          FAIL

No D3 artifact or verdict is modified by T1.

## T1a — custom matched-state audit

A manually matched clean state produced exact initial T-vs-center policy response across engines:

    max t=0 T-vs-center action-vector error = 0.0

but later action response diverged strongly:

    mean cross-engine response cosine H8   0.473
    mean cross-engine response cosine H64  0.318

However this custom source protocol preserved T semantics in only 2/4 Isaac commands.

Therefore:

    T1a source anchor INVALID

T1a supports closed-loop state-visitation sensitivity but is not causally decisive for D3.

## T1b — source-distribution anchor

Canonical H2a reset seeds/state-command distribution was retained, while observation corruption alone was disabled for deterministic parity.

Isaac clean source anchor:

    T objective direction     4 / 4
    T physical direction      3 / 4
    VERDICT                   PASS

Exact 32 source physical initial states and commands were replayed in MuJoCo.

MuJoCo source-state replay:

    T objective direction     2 / 4
    T physical direction      0 / 4
    VERDICT                   FAIL

Therefore the semantic degradation survives exact source-state replay and is attributable to the target physics/model transition rather than a custom reset choice.

## T1c — same-state, same-action one-step transition

The exact T/C actions produced in Isaac were applied to the exact same source physical states in MuJoCo.

After 20 ms:

    MuJoCo T better than center      3 / 4 suites
    one-step semantic sign flip      1 / 4 suites

Target-vs-source next-state mismatch under identical action:

    mean T linear-velocity RMSE      0.0838 m/s
    mean T angular-velocity RMSE     0.4006 rad/s
    mean T joint-velocity RMSE       2.9132 rad/s

Thus the target dynamics differ substantially at one step, especially in joint/angular velocity, but T semantics are not systematically reversed at that one step.

## T1d — temporal accumulation

MuJoCo source-state replay shows the T-vs-center physical relation becoming wrong at approximately:

    suite 0   H=4
    suite 1   H=4
    suite 2   H=8
    suite 3   H=4

At 50 Hz this corresponds to roughly:

    80–160 ms

T-vs-center action separation grows as trajectories diverge, reaching large closed-loop separations by H64.

Therefore the dominant proximate mechanism is:

> rapid multi-step closed-loop state-visitation divergence under mismatched target dynamics, rather than an incorrect initial preference response or a universal one-step semantic inversion.

## T1e — actuator-model audit

Source Isaac A1:

    actuator model        DC motor
    stiffness             25
    damping               0.5
    actuator friction     0
    effort limit          33.5 Nm
    saturation effort     33.5 Nm
    velocity limit        21 rad/s

Frozen MuJoCo baseline:

    position kp           100
    joint damping         1 for abduction / 2 otherwise
    joint frictionloss    0.2
    force limit           33.5 Nm

The force ceiling matches, but stiffness/damping/friction differ substantially.

This mismatch is consistent with:
- the large one-step joint-velocity RMSE;
- the rapid 80–160 ms state-visitation divergence;
- the absence of an immediate policy-semantic error at t=0.

This does not prove actuator mismatch is the sole cause.

It is sufficient to authorize one minimal causal intervention.

## T1 decision

    preference function error             ruled out as initial cause
    custom reset artifact                 ruled out
    universal one-step semantic flip      not supported
    multi-step state-visitation drift     strongly supported
    actuator/joint dynamics mismatch      strongest authorized upstream candidate

Next:

    T2 — actuator/joint-dynamics calibration only

No policy fine-tuning is authorized.
