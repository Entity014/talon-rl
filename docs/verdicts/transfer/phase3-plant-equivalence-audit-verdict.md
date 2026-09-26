# Phase 3 — Read-Only Plant / Contact Equivalence Audit Verdict

Status: **FROZEN — PASSIVE-JOINT MISMATCH DOMINANT IDENTIFIABLE CANDIDATE**
Date: 2026-09-26

## Context

Frozen prior results:

    D3 zero-adaptation semantic transfer       FAIL
    T1 dynamics-driven divergence              SUPPORTED
    T2 direct PD calibration                   FAIL
    T3 explicit DCMotor emulation              materially helpful, insufficient

This audit used no policy tuning and no plant parameter fitting.

## Static inertial comparison

Canonical local Isaac USD plant:

    total mass          13.741 kg
    trunk mass           6.001 kg

Frozen MuJoCo baseline:

    total mass          12.453 kg
    trunk mass           4.713 kg

Difference:

    total / trunk gap   -1.288 kg

The canonical training environment additionally randomizes trunk mass at startup by:

    add [-1, +3] kg

around the local USD value.

Therefore the frozen MuJoCo trunk is below the nominal source plant and slightly below the source randomization lower edge.

Other body-level differences include representation changes in which Isaac carries separate 0.06 kg foot rigid bodies while MuJoCo merges that mass into the calf representation.

No inertial correction is authorized from static comparison alone.

## Passive joint properties

Actual source PhysX DOF properties were queried from the live physics view.

PhysX passive DOF:

    stiffness      0
    damping        0
    friction       0
    armature       0

The apparent values:

    stiffness 25
    damping    0.5

in ArticulationData are explicit DCMotor-controller parameters, not passive PhysX joint properties.

Frozen MuJoCo baseline:

    hip damping              1.0
    thigh/calf damping       2.0
    armature                 0.01
    frictionloss             0.2

Thus the target simulator adds passive resistance and reflected inertia that do not exist in the source PhysX plant.

## Policy-free matched torque pulse

Probe:

    gravity off
    canonical pose
    FL_thigh_joint
    +5 Nm
    10 ms pulse
    direct joint torque
    no policy
    no DCMotor controller

Results:

    peak qdot
        Isaac       3.816 rad/s
        MuJoCo      1.109 rad/s

    qdot at 10 ms
        Isaac       3.686 rad/s
        MuJoCo      1.109 rad/s

    qdot at 50 ms
        Isaac       3.719 rad/s
        MuJoCo      0.0366 rad/s

    base angular-velocity peak
        Isaac       0.257 rad/s
        MuJoCo      0.055 rad/s

The source joint nearly free-coasts after the pulse while the target joint dissipates almost all motion within roughly 40 ms.

This is direct policy-free evidence of a large passive/effective-joint-dynamics mismatch.

## Contact/drop probe

Zero-torque canonical-pose drop from root z=0.60 m:

    first contact
        Isaac       0.245 s
        MuJoCo      0.248 s

Contact onset is closely aligned.

Post-contact response differs strongly:

    Isaac minimum COM z        ~0.056 m
    MuJoCo minimum COM z       ~0.140 m

    Isaac post-contact vz max   +0.256 m/s
    MuJoCo post-contact vz max  remained negative

However this probe contains both contact/solver effects and passive-joint effects.

Therefore it establishes a post-contact regime mismatch but does not isolate contact as the dominant cause.

## Contact/solver structural differences

Isaac:

    PhysX dt                         0.005 s
    solver position iterations      4
    solver velocity iterations      0
    robot material static friction  0.8
    robot material dynamic friction 0.6
    ground static/dynamic friction  1.0 / 1.0
    restitution                     0

MuJoCo:

    dt                              0.002 s
    Newton solver                   100 iterations
    floor sliding friction          1.0
    foot sliding friction           0.8
    MuJoCo solref/solimp compliance representation

These are real structural differences but are not isolated by the current drop probe.

## Decision

Evidence ranking:

    passive joint dynamics      STRONG / DIRECT / POLICY-FREE
    inertial distribution       REAL / STATICALLY CONFIRMED
    contact/solver dynamics     REAL / DYNAMICALLY CONFOUNDED

The strongest single mismatch with direct causal evidence is the target passive-joint dynamics.

Therefore one T4 candidate is authorized:

> remove target passive joint damping, frictionloss and armature while retaining T3 explicit DCMotor actuation.

T4 must NOT change:
- trunk/body mass;
- body inertia;
- COM;
- contact parameters;
- floor friction;
- solver;
- timestep;
- policy;
- action scale.

This preserves the remaining mismatches so the passive-joint hypothesis can be tested causally.

## Artifacts

    docs/contracts/transfer/phase3-t4-readonly-plant-equivalence-audit-contract.md
    scripts/rl/phase3_plant_audit_isaac_snapshot.py
    scripts/rl/phase3_plant_audit_isaac_nominal_snapshot.py
    scripts/rl/phase3_plant_audit_mujoco_snapshot.py
    scripts/rl/phase3_plant_probe_isaac_physx_dof_props.py
    scripts/rl/phase3_plant_probe_isaac_joint_pulse.py
    scripts/rl/phase3_plant_probe_mujoco_joint_pulse.py
    scripts/rl/phase3_plant_probe_isaac_drop.py
    scripts/rl/phase3_plant_probe_mujoco_drop.py

    runs/phase3_plant_equivalence_audit/
