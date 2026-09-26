# Phase-1 D3-A — MuJoCo Interface Equivalence Verdict

Status: **FROZEN — PASS**
Date: 2026-09-26

## Executive verdict

The Phase-1 MuJoCo adapter is structurally and live-runtime consistent with the frozen D2 deployment interface.

    D3-A1 structural/interface math   PASS
    D3-A2 live runtime mapping        PASS

Therefore D3-B nominal zero-adaptation dynamics transfer is authorized.

## Frozen secondary runtime

    Python          3.12.3
    MuJoCo          3.3.7
    NumPy           2.1.3
    MuJoCo dt       0.002 s
    integrator      Euler
    solver          Newton
    iterations      100
    line-search     50

Model hashes:

    scene.xml
    415eb95e2982c5a90c873f10d45e3efee94186ef6050e2c9aaea40135ff569ad

    a1.xml
    492978cb7700b0064bcefbc2b93dba651a828cf9212df91d56e951666890807c
## D3-A1 structural result

Joint vocabulary:

    exact 12/12 match

MuJoCo order is leg-grouped.
Canonical Phase-1 order is joint-type grouped.

All permutations are derived by joint name.

Synthetic physical-state observation equivalence:

    max |Delta obs| = 0.0

Canonical action target round-trip:

    max error = 0.0

MuJoCo native home differs from canonical D2 default pose and remains frozen as a transfer mismatch.

## D3-A2 live runtime semantics

### Quaternion / projected gravity

Known identity, roll90, pitch90, yaw90 and mixed orientations were tested against MuJoCo compiled body rotation.

    max error = 1.11e-16
    PASS

### Free-joint velocity semantics

Finite-difference integration established:

    qvel[:3]    world-frame linear velocity of free-joint/root origin
    qvel[3:6]   body-frame angular velocity

Errors:

    translation finite-difference    1.05e-10
    body angular finite-difference   9.65e-10

The legacy direct spatial-velocity assumption is therefore superseded.
## Root COM velocity correction

Canonical Isaac observation uses:

    root_com_lin_vel_b

not root-link/free-joint-origin velocity.

The correct MuJoCo mapping is:

    v_com_b
      = R^T v_free_world
        + omega_b x r_com_b

MuJoCo trunk inertial offset:

    [0.0, 0.0041, -0.0005] m

Finite-difference MuJoCo COM position versus adapter formula:

    max error = 1.44e-10
    PASS

This correction is required for canonical 48-D observation equivalence.

## Joint extraction

qpos/qvel addresses are obtained from compiled joint IDs, not fixed slice assumptions.

Controlled joint-state injection:

    q max error     0.0
    qdot max error  0.0

PASS

## Full live 48-D observation

Using one controlled nontrivial live MuJoCo state with:
- non-identity root quaternion;
- nonzero root linear/angular velocity;
- nonzero q/qdot;
- command;
- previous action;

the canonical adapter produced:

    max |Delta obs| = 0.0

PASS
## Actuator and timing semantics

MuJoCo compiled actuator order is resolved by actuator->joint name.

Canonical target:

    q_target = q_default_D2 + 0.25 * a

Target permutation round-trip:

    max error = 0.0

Position actuator force model at the controlled state:

    max error = 0.0

MuJoCo timestep:

    2 ms

Frozen D2 policy period:

    20 ms

Action hold:

    10 MuJoCo physics steps per policy action

Observed ctrl drift across ten substeps:

    0.0

PASS

## Final decision

    structural interface       PASS
    live state semantics       PASS
    COM velocity semantics     PASS
    joint mapping              PASS
    action mapping             PASS
    50 Hz hold contract        PASS

D3-A:

    PASS

D3-B:

    AUTHORIZED

No policy, reward, actuator, friction, contact, or XML parameter was modified to obtain this result.
## D3-B reset characterization

D3-B will report two zero-adaptation reset arms separately.

### Arm 1 — canonical-state initialization

Set the MuJoCo runtime to the canonical D2 initial/root and joint state as closely as the frozen interface specifies.

Purpose:

> isolate physics-engine / actuator / contact transfer while minimizing initial joint-state mismatch.

No XML parameter is changed.

### Arm 2 — native MuJoCo initialization

Use the frozen MuJoCo home keyframe without replacing its joint pose with D2 defaults.

Purpose:

> measure full model + reset + physics-engine transfer mismatch.

The two arms answer different questions and must not be pooled into one pass/fail statistic before being reported separately.

Neither arm may tune the controller or simulator.

## Source artifacts

    docs/contracts/transfer/phase1-d3-zero-adaptation-sim2sim-contract.md
    talon_rl/phase1_mujoco_adapter.py
    scripts/rl/phase1_d3a_structural_interface.py
    scripts/rl/phase1_d3_runtime_provenance.py
    scripts/rl/phase1_d3a_freejoint_semantics.py
    scripts/rl/phase1_d3a_live_mapping.py
    runs/phase1_d3a_structural_interface/
    runs/phase1_d3_runtime_provenance/
    runs/phase1_d3a_live_mapping/
