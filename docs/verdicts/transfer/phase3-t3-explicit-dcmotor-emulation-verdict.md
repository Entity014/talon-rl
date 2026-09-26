# Phase 3 T3 — Explicit DCMotor Torque-Law Emulation Verdict

Status: **FROZEN — FAIL AT T3-B / T3-C NOT REACHED**
Date: 2026-09-26

## Candidate

T3 replaced the MuJoCo position-servo actuator with a transparent unit-gain torque actuator and implemented the Isaac source DCMotor law explicitly at every MuJoCo physics substep.

Policy target:

    q_target = q_default + 0.25 * action

Raw torque:

    tau_raw = 25 * (q_target - q) - 0.5 * qdot

Torque-speed clipping:

    effort_limit       33.5 Nm
    saturation_effort  33.5 Nm
    velocity_limit     21 rad/s

The torque command was recomputed every 2 ms MuJoCo physics step, matching the source design principle that the actuator model is recomputed every physics substep.

Frozen D3 joint damping/friction/contact/inertia values were retained.

## Actuator transparency

T3 MuJoCo actuator model:

    motor gear                    1.0
    actuator gain                 1.0
    actuator bias                 0
    ctrl limiting                 disabled

Validation:

    max |actuator_force - ctrl|   0.0

Therefore runtime DCMotor torque reaches the joint without a hidden position-servo law.

## T3-A — one-step source-transition parity

Frozen D3 baseline:

    linear velocity RMSE      0.0838 m/s
    angular velocity RMSE     0.4006 rad/s
    joint velocity RMSE       2.9132 rad/s

T2 direct-PD calibration:

    linear velocity RMSE      0.0134 m/s
    angular velocity RMSE     0.1917 rad/s
    joint velocity RMSE       1.1632 rad/s

T3 explicit DCMotor:

    linear velocity RMSE      0.0357 m/s
    angular velocity RMSE     0.4090 rad/s
    joint velocity RMSE       3.1181 rad/s

Interpretation:

- linear-velocity parity improves materially relative to D3;
- angular-velocity parity is approximately unchanged/slightly worse;
- joint-velocity parity is worse than D3 and much worse than T2.

Thus source actuator-law emulation does not by itself reproduce the source one-step plant transition.

This implies remaining source-target differences outside the actuator law, such as plant/contact/inertia/solver interactions, are materially involved.

## T3-B — nominal closed-loop viability

Canonical-state arm:

    rollouts                         25
    immediate-10-step survival       25 / 25 = 1.00
    64-step survival                 10 / 25 = 0.40
    finite-state fraction            1.00
    control-contract fraction        1.00
    action saturation fraction       0.9009

Native-MuJoCo arm:

    rollouts                         25
    immediate-10-step survival       25 / 25 = 1.00
    64-step survival                 25 / 25 = 1.00
    finite-state fraction            1.00
    control-contract fraction        1.00
    action saturation fraction       0.9349

Predeclared canonical survival requirement:

    >= 0.75

Observed:

    0.40

Therefore T3-B FAILS.

## Failure-mode characterization

Canonical failures:

    total failures               15 / 25
    stand                         5
    lateral                       5
    turn-left                     2
    turn-right                    2
    forward                       1

Preference distribution among failures:

    O-heavy                       5
    center                        4
    T-heavy                       2
    A-heavy                       2
    S-heavy                       2

Failure timing:

    approximately 17–25 policy steps

No failure was caused by:
- non-finite state;
- control-contract violation;
- trunk-floor contact;
- excessive tilt.

All 15 failures crossed the predeclared trunk-height floor:

    0.18 m

Observed minimum heights at failure were approximately:

    0.17897–0.17999 m

Maximum tilt among these failures was only approximately:

    5.8 deg

This makes the failure close to the strict height boundary, but the threshold was preregistered and is not changed after observing results.

## Comparison with T2

T2 direct parameter calibration:

    canonical survival64       0 / 25
    native survival64          0 / 25
    typical failure            6–9 policy steps

T3 explicit DCMotor:

    canonical survival64      10 / 25
    native survival64        25 / 25
    immediate survival       50 / 50

Thus explicit DCMotor emulation materially improves closed-loop viability relative to T2.

However it does not satisfy the primary canonical transfer gate.

## T3-C

    NOT REACHED

Per the predeclared contract, semantic transfer is not evaluated after canonical nominal viability fails.

## Scientific interpretation

T3 establishes two useful points.

First:

> actuator-law fidelity matters substantially for closed-loop viability.

Replacing nominal-gain matching with the correct DCMotor structure recovers the native target-simulator regime and avoids the immediate collapse seen in T2.

Second:

> actuator-law equivalence is still insufficient for source-like closed-loop equivalence.

Even with the source DCMotor law, source-target joint/angular transition parity remains poor and the canonical initialization crosses the strict viability floor in 60% of rollouts.

Therefore the remaining transfer gap cannot be attributed solely to the actuator controller law.

## Decision

    T3-A one-step parity                 MIXED
    T3-B canonical nominal viability     FAIL
    T3-C semantic transfer               NOT REACHED
    T3 overall                           FAIL
    D4 hardware                          BLOCKED

No gain sweep, height-threshold change, or semantic evaluation is authorized inside T3.

Any next candidate must be opened as a new predeclared intervention and preserve D3/T2/T3 as frozen baselines.

## Artifacts

    docs/contracts/transfer/phase3-t3-explicit-dcmotor-emulation-contract.md
    scripts/rl/phase3_t3_generate_dcmotor_model.py
    scripts/rl/phase3_t3a_one_step_compare.py
    scripts/rl/phase3_t3b_nominal_dynamics.py

    talon_rl/assets/data/Robots/unitree_a1/mujoco/a1_t3_dcmotor.xml
    talon_rl/assets/data/Robots/unitree_a1/mujoco/scene_t3_dcmotor.xml

    runs/phase3_t3_dcmotor/
    runs/phase3_t3a_one_step/
    runs/phase3_t3b_nominal_dynamics/
