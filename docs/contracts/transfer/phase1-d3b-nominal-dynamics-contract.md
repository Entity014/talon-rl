# Phase-1 D3-B — Zero-Adaptation Nominal Dynamics Transfer Contract

Status: **PREDECLARED — AUTHORIZED AFTER D3-A PASS**
Date: 2026-09-26

## Question

With the exact frozen D2 deployment controller and exact D3-A interface adapter, does nominal locomotion remain viable in MuJoCo without any policy or simulator rescue tuning?

## Frozen policy/runtime

    artifacts/phase1_canonical_actor_state.pt
    Phase1StandaloneActor / Phase1EagerStateRuntime
    PyTorch 2.7.0+cu128
    CUDA 12.8
    50 Hz policy rate

MuJoCo:

    3.3.7
    dt = 0.002 s
    10 physics steps per 20 ms policy action
    XML hashes frozen by D3-A

No policy, reward, action scale, actuator gain, friction, contact, timestep, solver, or model parameter may change.

## Reset arms

Arm A — canonical-state initialization:
- MuJoCo root pose from native home keyframe;
- root/joint velocities zero;
- 12 joint positions overwritten with canonical D2 default pose;
- previous action zero.

Arm B — native-MuJoCo initialization:
- exact MuJoCo home keyframe;
- root/joint velocities zero;
- previous action zero.

These arms answer different questions and are reported separately.

## Fixed command set

    stand       [0.0, 0.0,  0.0]
    forward     [0.5, 0.0,  0.0]
    turn_left   [0.3, 0.0,  0.3]
    turn_right  [0.3, 0.0, -0.3]
    lateral     [0.0, 0.25, 0.0]

## Preference set

    T-heavy [0.7,0.1,0.1,0.1]
    A-heavy [0.1,0.7,0.1,0.1]
    O-heavy [0.1,0.1,0.7,0.1]
    S-heavy [0.1,0.1,0.1,0.7]
    center  [0.25,0.25,0.25,0.25]

D3-B uses preferences only as a nominal viability sweep. Semantic ordering is not interpreted until D3-C.

## Rollout

Each arm × command × preference:

    64 policy steps
    20 ms per policy step
    10 MuJoCo physics steps per policy step

Total first-pass rollouts:

    2 × 5 × 5 = 50

## Survival definition

A rollout is alive while all are true:
- all qpos/qvel/actions are finite;
- trunk height >= 0.18 m;
- trunk tilt <= 75 degrees;
- no trunk collision geom contacts the world/floor.

Immediate collapse:

    survival failure within first 10 policy steps

## D3-B primary gates

Per contract:

    all states/actions finite                         required
    no action-target/ctrlrange contract violation    required
    no immediate collapse in first 10 policy steps   required for >=75% rollouts
    64-step survival                                  >=0.75 aggregate
    scalar action saturation |a|>=0.98                <0.95 aggregate fraction

Report reset arms independently and combined.

The canonical-state arm is the primary dynamics-engine isolation arm.
The native-home arm characterizes additional reset/model mismatch.

## Stop rule

If the canonical-state arm fails the aggregate 0.75 survival gate:
- freeze zero-adaptation dynamics-transfer failure;
- D3-C is NOT REACHED.

If canonical-state passes but native-home fails:
- D3-B passes for engine transfer with a documented reset-state mismatch.

If both pass:
- D3-C MORL semantic transfer is authorized.

No failure authorizes simulator or controller tuning before the zero-adaptation verdict is frozen.
