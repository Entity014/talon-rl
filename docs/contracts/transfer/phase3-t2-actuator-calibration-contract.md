# Phase 3 T2 — Minimal Actuator/Joint-Dynamics Calibration Contract

Status: **PREDECLARED — ONE AUTHORIZED TRANSFER ADAPTATION CANDIDATE**
Date: 2026-09-26

## Purpose

Test whether the source-vs-target actuator/joint-dynamics mismatch identified by T1 is a causal contributor to semantic transfer failure.

This is not policy training.

## Frozen baseline

Reference:

    D3 zero-adaptation MuJoCo baseline

Must remain unchanged and reproducible.

## Sole intervention

Create a separate MuJoCo model variant with:

    position actuator kp     100 -> 25
    joint damping            1/2 -> 0.5 for all 12 joints
    joint frictionloss       0.2 -> 0

Keep unchanged:

    policy artifact
    observation adapter
    action scale 0.25
    force range +/-33.5 Nm
    robot masses/inertias
    joint limits
    contact/friction parameters
    terrain
    MuJoCo version
    dt / solver / iterations
    reset protocol
    reward definitions
    preference definitions

Do not modify the frozen baseline XML files in place.

The calibrated model must be a new named artifact.

The Isaac velocity limit of 21 rad/s is NOT added in T2 because doing so would introduce a second actuator intervention. It may be characterized descriptively.

## Causal ladder

T2-A:
    rerun same-state same-action 20 ms transition audit

Primary mechanistic expectation:
    joint-velocity RMSE should decrease from the frozen T1 value 2.9132 rad/s.

T2-B:
    rerun D3-B nominal viability

Require:
    survival >= .75
    finite states/actions
    no control contract violation

T2-C:
    rerun full D3-C semantic protocol

Required for T2 success:

    T objective correctness      >= .75
    T physical correctness       >= .75

    A objective correctness      >= .75
    A physical correctness       >= .75

    O objective correctness      >= .75
    O physical correctness       >= .75

    S                           report honestly

    center compromise            >= .75
    continuum monotonicity       >= .65
    continuum endpoint-between   >= .65

No semantic threshold may be changed after results are seen.

## Decision

If one-step dynamics mismatch decreases but semantic transfer still fails:
    actuator mismatch contributes to dynamics error but is not sufficient for semantic repair.

If T/A/O and preference-family gates recover:
    actuator calibration is a successful minimal transfer adaptation and D4 may be reconsidered only after a fresh deployment/safety validation of the adapted simulator assumptions.

If dynamics or semantics regress:
    freeze T2 failure; do not tune kp/damping/friction by search.

No parameter sweep is authorized inside T2.
