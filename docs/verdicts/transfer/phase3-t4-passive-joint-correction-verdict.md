# Phase 3 T4 — Passive Joint Dynamics Correction Verdict

Status: **FROZEN — FAIL AT T4-C / T4-D NOT REACHED**
Date: 2026-09-26

## Candidate

T4 started from T3 explicit DCMotor torque-law emulation and changed only target passive joint properties:

    damping       1 / 2 -> 0
    frictionloss  0.2   -> 0
    armature      0.01  -> 0

All masses, inertias, COMs, contacts, solver settings, timestep, policy and semantic definitions remained frozen.

## T4-A — policy-free torque pulse

Probe:

    gravity off
    FL_thigh_joint
    +5 Nm
    10 ms
    direct torque

Frozen MuJoCo baseline:

    qdot trajectory RMSE vs Isaac     3.5155 rad/s
    peak qdot                         1.1088 rad/s
    qdot at 50 ms                     0.0366 rad/s

T4 passive-corrected target:

    qdot trajectory RMSE vs Isaac     0.0133 rad/s
    peak qdot                         3.8298 rad/s
    qdot at 50 ms                     3.7326 rad/s

Isaac reference:

    peak qdot                         3.8160 rad/s
    qdot at 50 ms                     3.7191 rad/s

Therefore T4-A strongly validates that the passive-joint mismatch was real and dominant for the free-space joint response.

## T4-B — same-state same-action 20 ms parity

Mean T-state RMSE:

    linear velocity      0.0132 m/s
    angular velocity     0.0468 rad/s
    joint velocity       0.3233 rad/s

These values are substantially smaller than T3:

    angular velocity     ~0.409 rad/s
    joint velocity       ~3.118 rad/s

T-heavy is better than center after one target transition in:

    4 / 4 MuJoCo suites

Thus local source-target transition parity is strongly improved.

## T4-C — nominal closed-loop viability

Canonical arm:

    survival64              0 / 25
    immediate-10-step OK    0 / 25
    finite states           25 / 25
    control contract        25 / 25
    action saturation       0.8118

Native arm:

    survival64              0 / 25
    immediate-10-step OK    1 / 25
    finite states           25 / 25
    control contract        25 / 25
    action saturation       0.7877

Failure timing:

    canonical roughly 7–9 policy steps
    native roughly 6–10 policy steps

Failure mode:

    trunk-height floor crossing

Canonical minimum heights at failure:

    approximately 0.166–0.178 m

Native minimum heights at failure:

    approximately 0.169–0.179 m

No failure was caused by:
- non-finite state;
- control-contract violation;
- trunk-floor contact;
- excessive tilt.

Maximum observed tilt remained below approximately 9 degrees.

## T4-D

    NOT REACHED

Per the predeclared contract, semantic transfer is not evaluated after nominal viability fails.

## Scientific interpretation

T4 establishes an important separation.

First:

> the source-target passive-joint mismatch is real and can dominate local free-space transition error.

Correcting damping/friction/armature nearly reproduces the source torque-pulse response and materially improves one-step source-state parity.

Second:

> local plant parity is not sufficient for closed-loop viability.

Once the target passive joints are made source-like, the frozen controller rapidly lowers the robot through the height floor under both reset arms.

Therefore the residual transfer problem is not explained by passive-joint dynamics alone.

Remaining known structural mismatches include:
- total/trunk mass distribution;
- body inertial representation;
- contact/solver/compliance;
- timestep/integration differences.

## Decision

    T4-A plant probe                 PASS strongly
    T4-B one-step parity             PASS / strongly improved
    T4-C nominal viability           FAIL
    T4-D semantic transfer           NOT REACHED
    T4 overall                       FAIL
    D4 hardware                      BLOCKED

No partial damping, armature, friction or gain sweep is authorized.

Any next intervention requires a new read-only residual plant audit first.

## Artifacts

    docs/contracts/transfer/phase3-t4-passive-joint-correction-contract.md
    scripts/rl/phase3_t4_generate_passive_corrected_model.py
    scripts/rl/phase3_t4a_joint_pulse.py
    scripts/rl/phase3_t4b_one_step_compare.py
    scripts/rl/phase3_t4c_nominal_dynamics.py

    talon_rl/assets/data/Robots/unitree_a1/mujoco/a1_t4_passive_corrected.xml
    talon_rl/assets/data/Robots/unitree_a1/mujoco/scene_t4_passive_corrected.xml

    runs/phase3_t4_passive_correction/
    runs/phase3_t4b_one_step/
    runs/phase3_t4c_nominal_dynamics/
