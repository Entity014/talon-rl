# Phase 3 T2 — Direct Actuator-Parameter Calibration Verdict

Status: **FROZEN — FAIL**
Date: 2026-09-26

## Intervention

Only the predeclared MuJoCo actuator/joint parameters were changed in a separate model artifact:

    position kp       100 -> 25
    joint damping     1/2 -> 0.5
    frictionloss      0.2 -> 0

Unchanged:

    policy
    action scale
    force range
    contacts/friction
    masses/inertias
    timestep/solver
    reward semantics
    preference semantics

Baseline XML files remained unchanged.

## T2-A — matched one-step dynamics

The intervention substantially reduced source-target transition mismatch.

Frozen T1 baseline:

    mean T linear-velocity RMSE     0.0838 m/s
    mean T angular-velocity RMSE    0.4006 rad/s
    mean T joint-velocity RMSE      2.9132 rad/s

T2 calibrated position actuator:

    mean T linear-velocity RMSE     0.0134 m/s
    mean T angular-velocity RMSE    0.1917 rad/s
    mean T joint-velocity RMSE      1.1632 rad/s

Thus direct parameter calibration moved the one-step target transition materially toward the source transition.

## T2-B — nominal viability

Despite better one-step transition parity, closed-loop locomotion collapsed immediately.

Canonical-state arm:

    survival64             0 / 25
    immediate-10-step OK   0 / 25
    finite states          25 / 25
    ctrl contract          25 / 25

Native-home arm:

    survival64             0 / 25
    immediate-10-step OK   0 / 25

Typical failure occurred around:

    6–9 policy steps

Action saturation remained finite and below the frozen .95 aggregate gate, so the failure is not a numerical or interface fault.

## T2-C

    NOT REACHED

Per the predeclared ladder, semantic evaluation is not interpretable after nominal viability fails.

## Interpretation

Direct numerical transplantation of stiffness/damping/friction is not actuator equivalence.

The source Isaac A1 uses an explicit DCMotor model:

    tau_des = 25*(q_target-q) + 0.5*(0-qdot)

followed by a velocity-dependent DC motor torque-speed saturation law with:

    effort_limit       33.5 Nm
    saturation_effort   33.5 Nm
    velocity_limit      21 rad/s

The frozen MuJoCo baseline uses a position actuator with a different actuation law.

Therefore T2 shows:

> matching nominal PD parameters can improve one-step state-transition parity while simultaneously destroying closed-loop viability when the underlying actuator laws differ.

## Decision

    T2 direct parameter calibration     FAIL
    T2-C semantic validation            NOT REACHED
    D4 hardware                         BLOCKED

No gain sweep is authorized.

A future candidate, if opened, must test source-equivalent explicit DCMotor torque-law emulation as a new predeclared intervention rather than tuning position-servo gains.

## Artifacts

    docs/contracts/transfer/phase3-t2-actuator-calibration-contract.md
    scripts/rl/phase3_t2_generate_actuator_model.py
    scripts/rl/phase3_t2a_one_step_compare.py
    scripts/rl/phase3_t2b_nominal_dynamics.py

    runs/phase3_t2_actuator_calibration/
    runs/phase3_t2a_one_step/
    runs/phase3_t2b_nominal_dynamics/
