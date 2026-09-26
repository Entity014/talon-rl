# Phase 3 T3 — Explicit DCMotor Torque-Law Emulation Contract

Status: **PREDECLARED — AUTHORIZED CANDIDATE**
Date: 2026-09-26

## Motivation

T2 showed that copying nominal PD parameters into a MuJoCo position actuator substantially improved one-step transition parity but destroyed closed-loop viability.

Therefore T3 tests actuator-law equivalence rather than gain equivalence.

## Frozen references

    D3 zero-adaptation baseline              FROZEN
    T1 mechanism audit                      FROZEN
    T2 direct PD calibration                FROZEN FAIL

No prior artifact or verdict may be rewritten.

## Sole intervention

Replace MuJoCo position-servo actuation with explicit joint torque actuation implementing the Isaac source DCMotor law.

Policy-side target remains:

    q_target = q_default + 0.25 * action

Raw PD torque:

    tau_raw = 25 * (q_target - q) - 0.5 * qdot

Source DC motor limits:

    effort_limit       = 33.5 Nm
    saturation_effort  = 33.5 Nm
    velocity_limit     = 21 rad/s

Torque-speed clipping must match IsaacLab DCMotor._clip_effort exactly:

    vel_at_effort_lim = velocity_limit * (1 + effort_limit / saturation_effort)

    qdot_clip = clip(qdot, -vel_at_effort_lim, +vel_at_effort_lim)

    tau_top    = saturation_effort * ( 1 - qdot_clip / velocity_limit )
    tau_bottom = saturation_effort * (-1 - qdot_clip / velocity_limit )

    tau_max = min(tau_top, effort_limit)
    tau_min = max(tau_bottom, -effort_limit)

    tau = clip(tau_raw, tau_min, tau_max)

MuJoCo must receive tau directly as joint torque.

## Model rules

Create a new MuJoCo model artifact.

Remove position actuator dynamics from the T3 model.

Keep unchanged:

    body masses/inertias
    joint limits
    joint damping/friction from frozen D3 baseline
    contact parameters
    floor friction
    timestep / solver / iterations
    policy
    observation adapter
    action scale
    reward definitions
    preference definitions
    reset protocols

Important:

T3 changes the actuator law only.

It does NOT also import T2 damping/friction edits.

## T3-A — one-step source-transition parity

Use the exact frozen T1c source states and exact source policy actions.

Compare against frozen D3 baseline and T2:

Primary metrics:

    linear velocity RMSE
    angular velocity RMSE
    joint velocity RMSE
    joint position RMSE

Expected direction:

    T3 should improve one-step parity relative to D3 baseline.

T3 need not beat every T2 local metric to proceed if the explicit law is correct and closed-loop behavior is viable.

## T3-B — nominal viability

Use the exact D3-B protocol:

    canonical reset arm
    native MuJoCo reset arm
    5 commands
    5 preferences
    64 policy steps
    50 Hz policy rate

Required:

    finite state/action                1.0
    control contract                   1.0
    immediate-10-step survival        >= .75 canonical
    64-step survival                  >= .75 canonical
    action saturation                 report

If canonical viability fails:

    T3-C NOT REACHED
    T3 FAIL

## T3-C — semantic transfer

Only if T3-B passes.

Use the exact frozen D3-C semantic protocol.

Required:

    T objective correctness      >= .75
    T physical correctness       >= .75

    A objective correctness      >= .75
    A physical correctness       >= .75

    O objective correctness      >= .75
    O physical correctness       >= .75

    S                            report honestly

    center compromise            >= .75
    continuum monotonicity       >= .65
    continuum endpoint-between   >= .65

## Decision interpretation

T3-A improves + T3-B fails:
    explicit actuator law alone is insufficient for closed-loop transfer.

T3-B passes + T3-C fails:
    actuator equivalence restores viability but is insufficient for semantic transfer.

T3-C passes:
    explicit DCMotor emulation is a successful minimal transfer adaptation candidate.

No actuator-parameter sweep is authorized inside T3.
