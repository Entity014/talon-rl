# Phase 3 T4 — Passive Joint Dynamics Correction Contract

Status: **PREDECLARED — AUTHORIZED CANDIDATE**
Date: 2026-09-26

## Motivation

The read-only plant audit identified a direct policy-free mismatch:

Source PhysX passive joint properties:

    damping       0
    friction      0
    armature      0

Frozen MuJoCo baseline:

    damping       1 / 2
    frictionloss  0.2
    armature      0.01

A matched free-space torque pulse showed strong target-side dissipation absent in the source.

## Frozen base

T4 starts from the T3 model:

    explicit source DCMotor torque law
    transparent unit-gain torque actuators
    frozen D3 masses/inertias
    frozen D3 contact/solver settings

## Sole intervention

For all 12 MuJoCo leg joints:

    damping       -> 0
    frictionloss  -> 0
    armature      -> 0

Keep unchanged:

    explicit DCMotor law
    kp 25 / kd 0.5 inside controller
    torque-speed clipping
    effort limit 33.5 Nm
    velocity limit 21 rad/s
    masses/inertias
    COM
    joint limits
    contact geometry
    friction
    solref/solimp
    timestep
    solver
    policy
    observation adapter
    action scale
    reset protocols
    semantic definitions

No parameter sweep is authorized.

## T4-A — policy-free plant probe

Repeat the exact +5 Nm / 10 ms gravity-off FL_thigh pulse.

Required directional evidence:

    target qdot peak increases toward source
    target post-pulse decay decreases toward source
    qdot trajectory RMSE decreases relative to frozen MuJoCo baseline

This is mechanistic validation only.

## T4-B — same-state same-action one-step parity

Repeat the frozen T1c source-state/action audit using explicit DCMotor T4 plant.

Report:

    linear velocity RMSE
    angular velocity RMSE
    joint velocity RMSE
    joint position RMSE

No hard requirement that every metric beat T2/T3.

## T4-C — nominal viability

Repeat exact D3-B protocol:

    canonical reset
    native reset
    5 commands
    5 preferences
    64 steps

Canonical requirements:

    finite / control contract          1.0
    immediate-10-step survival         >= .75
    64-step survival                   >= .75

If canonical viability fails:

    T4-D semantic transfer NOT REACHED
    T4 FAIL

## T4-D — semantic transfer

Only if T4-C passes.

Use exact frozen D3-C thresholds:

    T objective / physical       >= .75 / .75
    A objective / physical       >= .75 / .75
    O objective / physical       >= .75 / .75
    S                            report
    center compromise            >= .75
    continuum monotonicity       >= .65
    continuum endpoint-between   >= .65

## Interpretation

If T4-A fixes free-space pulse but T4-C fails:
    passive mismatch is real but insufficient for closed-loop transfer.

If T4-C passes but T4-D fails:
    passive correction restores plant viability but not MORL semantics.

If T4-D passes:
    passive-joint correction is a successful minimal transfer adaptation candidate.

No inertial/contact correction may be added inside T4.
