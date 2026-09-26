# Phase 3 — Post-T4 Residual Plant / Closed-Loop Audit Verdict

Status: **FROZEN — NO SINGLE T5 PLANT CORRECTION AUTHORIZED**
Date: 2026-09-26

## Context

Frozen results:

    D3 zero-adaptation semantic transfer      FAIL
    T1 dynamics-driven divergence             SUPPORTED
    T2 nominal PD matching                    FAIL
    T3 explicit DCMotor emulation             materially helpful, insufficient
    T4 passive-joint correction               local parity restored, closed-loop viability FAIL

The purpose of this residual audit is to decide whether one additional plant parameter group is sufficiently dominant to justify a T5 intervention.

## Corrected inertial probe

A first MuJoCo base-force probe was discarded because its initial joint pose was not matched to the Isaac canonical pose.

That invalid result is not used.

Corrected matched probe:

    gravity off
    canonical joint pose
    +20 N trunk force
    10 ms

Trunk/root COM vx at 10 ms:

    Isaac          0.01547 m/s
    MuJoCo T4      0.01375 m/s

This is a material but moderate difference, not the initially observed 4x discrepancy.

Static plant mass remains different:

    Isaac total mass      13.741 kg
    MuJoCo total mass     12.453 kg
    delta                 -1.288 kg

The mass gap is real but does not produce a dominant impulse-response mismatch comparable to the passive-joint error found before T4.

## Fixed canonical stance probe

Purpose:

Determine whether the T4 target plant/contact model collapses even without MORL policy feedback.

Protocol:

    canonical root/joint state
    zero policy action
    q_target = canonical default pose
    source-equivalent explicit DCMotor law
    64 policy periods / 1.28 s
    no preference-conditioned action changes

Isaac:

    minimum trunk/root height     0.2201 m
    final height                  0.2745 m
    max tilt                      8.37 deg

MuJoCo T4:

    minimum height                0.2116 m
    final height                  0.2673 m
    max tilt                      8.04 deg

Both remain above the frozen 0.18 m viability floor.

The fixed-target target-simulator response is therefore close enough that the T4 closed-loop collapse cannot be attributed to an unconditional stance/contact failure.

## Relation to T4 closed-loop failure

T4 with the frozen MORL policy:

    canonical survival64      0 / 25
    native survival64         0 / 25
    failure timing            roughly 6–10 policy steps
    failure mode              height floor crossing

T4 with fixed zero action:

    no height-floor failure over 64 policy periods

Therefore:

> the residual failure requires closed-loop policy feedback acting on small source-target state differences.

This is consistent with the earlier D2 finding that very small execution differences can accumulate into different long-horizon behavior.

## Remaining plant differences

Known residual differences still include:

- total/trunk mass distribution;
- detailed inertial representation;
- separate-foot versus merged-calf representation;
- contact compliance / solver formulation;
- PhysX 5 ms versus MuJoCo 2 ms integration;
- friction model differences.

However current evidence does not isolate any one of these as the dominant remaining cause.

Contact onset in the earlier drop probe was nearly matched:

    Isaac       0.245 s
    MuJoCo      0.248 s

while post-contact evolution differed, but that probe is jointly sensitive to inertial, passive and solver/contact effects.

## Decision

The evidence no longer supports a clean single-parameter plant correction.

Therefore:

    inertial correction candidate       NOT AUTHORIZED
    contact/solver correction candidate NOT AUTHORIZED
    mixed parameter fitting             NOT AUTHORIZED

No T5 plant-tuning candidate is opened.

The strongest residual interpretation is:

> after correcting the largest actuator/passive mismatches, relatively small distributed plant differences are amplified by the frozen policy's closed-loop state visitation, producing viability and semantic divergence.

Any next research phase would need to address transfer robustness / closed-loop sensitivity at the controller-training level or use a separately preregistered multi-parameter system-identification framework.

That would be a new research branch, not continuation of T4 plant tuning.

## Artifacts

    docs/verdicts/transfer/phase3-t4-passive-joint-correction-verdict.md
    scripts/rl/phase3_residual_isaac_base_impulse.py
    scripts/rl/phase3_residual_mujoco_base_impulse.py
    scripts/rl/phase3_residual_isaac_fixed_stance.py
    scripts/rl/phase3_residual_mujoco_fixed_stance.py

    runs/phase3_residual_plant_audit/
