# Phase-1 D2 — Deployment Equivalence Validation Contract

Status: **PREDECLARED — DEPLOYMENT IMPLEMENTATION ONLY**
Date: 2026-09-26

## Question

Can the exact frozen canonical Phase-1 controller be exported and executed through a deployment boundary without changing its policy function, observation semantics, action semantics, timing assumptions, or safety behavior?

Canonical checkpoint:

    runs/authority_isolated_simplex_edge_retain-2026-09-25/model_30.pt

No policy weight, architecture, reward, semantic threshold, or checkpoint selection may change in D2.

## D2 interpretation

D2 tests deployment equivalence, not learning quality.

If D2 fails because of:
- feature ordering;
- frame convention;
- action mapping;
- serialization;
- runtime shape/finiteness checks;
- stale-data handling;
- timing;
- safety wrapper implementation;

the deployment implementation may be corrected and retested.

If the exact deployment-equivalent controller retains an already-known behavioral limitation, report it and do not tune the policy.
## D2-A — observation contract

Canonical policy observation width:

    48

Exact concatenated term order:

    0: base_lin_vel          [3]
    1: base_ang_vel          [3]
    2: projected_gravity     [3]
    3: velocity_commands     [3]
    4: joint_pos             [12]
    5: joint_vel             [12]
    6: actions               [12]

Semantic definitions:

- base_lin_vel: robot/base-frame linear velocity;
- base_ang_vel: robot/base-frame angular velocity;
- projected_gravity: gravity direction projected into base frame;
- velocity_commands: [v_x, v_y, yaw_rate] command generated for base_velocity;
- joint_pos: joint position relative to default pose;
- joint_vel: joint velocity relative to default velocity;
- actions: previous normalized policy action.

Canonical IsaacLab observation corruption is enabled in the source training/evaluation config.

Therefore D2 separates two questions:

1. tensor-level policy parity:
   deployment receives the exact frozen 48-D observation tensor already produced by the source manager;

2. observation reconstruction parity:
   deployment sensor/state adapter independently reconstructs those 48 values term by term.

No extra learned observation normalization exists in the canonical Phase-1 model.
## D2-B — action contract

Policy action width:

    12

Exact joint order:

    0  FL_hip_joint
    1  FR_hip_joint
    2  RL_hip_joint
    3  RR_hip_joint
    4  FL_thigh_joint
    5  FR_thigh_joint
    6  RL_thigh_joint
    7  RR_thigh_joint
    8  FL_calf_joint
    9  FR_calf_joint
    10 RL_calf_joint
    11 RR_calf_joint

Policy output:

    a = tanh(pre_tanh)
    a in [-1, 1]

IsaacLab joint-position command mapping:

    q_target = q_default + 0.25 * a

Default offsets, in action order:

    [ 0.1, -0.1,  0.1, -0.1,
      0.8,  0.8,  1.0,  1.0,
     -1.5, -1.5, -1.5, -1.5 ]

No downstream action remapping may change signs or joint order.

Any hardware-side joint limits/current/torque safety clamps are external safety constraints and must be logged when active.
## D2-C — runtime / serialization parity

Build a deployment actor from the exact frozen canonical checkpoint.

Interface:

    action = policy(obs_48, preference_4)

Preference order:

    [T, A, O, S]

Preference requirements:

    finite
    each weight >= 0
    sum(weights) = 1 within 1e-6

Hard parity bank:
- frozen probe states already used by Phase-1 authority audits;
- T/A/O/S heavy endpoints;
- center;
- sampled simplex interior preferences.

Compare source PyTorch model against exported deployment artifact.

Primary numerical gates:

    max |Delta pre_tanh| <= 1e-6
    max |Delta action|   <= 1e-6

If CPU-vs-GPU numerical kernels prevent 1e-6, record the observed error and require:

    max |Delta action| <= 1e-5

No tolerance may be loosened after examining semantic outcomes.

Additional runtime gates:
- deterministic repeat call returns identical output;
- wrong observation width rejected;
- wrong preference width rejected;
- non-finite observation rejected;
- non-finite preference rejected;
- invalid simplex preference rejected;
- non-finite policy output rejected.
## D2-D — timing and stale-data contract

Canonical source control period:

    sim dt       = 0.005 s
    decimation   = 4
    policy dt    = 0.020 s
    policy rate  = 50 Hz

Deployment target rate:

    50 Hz

Measure:
- median inference latency;
- p95;
- p99;
- max;
- deadline miss count.

Initial gate:

    p99 inference latency < 10 ms
    zero inference calls > 20 ms in offline parity benchmark

This leaves explicit headroom for sensing, state estimation, transport, and actuator command generation.

Stale-data behavior:

    age <= 20 ms       normal inference
    20 < age <= 40 ms  hold previous safe action + warning
    age > 40 ms        safe-stop request

D2 may implement and validate this behavior, but hardware-specific brake/disable semantics belong to D4.
## D2-E — safety contract

Deployment boundary must reject or safe-stop on:
- NaN/Inf observation;
- invalid preference simplex;
- missing observation packet;
- stale observation beyond limit;
- action NaN/Inf;
- action outside normalized [-1,1] contract before actuator mapping.

Simulation safety signals:
- base-contact/fall termination;
- command timeout;
- explicit estop latch;
- startup state with no valid command;
- preference bounds.

Estop must be latched until explicit reset.

Startup behavior:

    no valid observation/command -> no policy actuation request

Recovery from fall is not part of D2 unless already represented by the frozen policy; D2 must not invent an autonomous recovery policy.
## D2-F — deployment-wrapper closed-loop simulation

After A-E pass, execute the canonical IsaacLab environment through the deployment wrapper rather than direct model calls.

Use the same frozen matched suites as canonical H2a.

Evaluate:
- T/A/O/S endpoint semantics;
- center compromise;
- continuum;
- survival;
- preference authority.

Primary requirement:

    deployment-wrapper metrics reproduce direct-call canonical metrics
    within numerical / rollout tolerance.

Hard qualitative no-regression:
- T remains PASS;
- A remains PASS for canonical seed;
- O remains PASS;
- S remains reported exactly, not rescued or hidden;
- no new broad termination topology appears.

D2-F does not select a new checkpoint.

## D2 completion rule

D2 passes only if:
1. observation contract is explicit and tested;
2. action/joint mapping is explicit and tested;
3. serialization/runtime action parity passes;
4. timing/stale-data guards pass;
5. safety boundary tests pass;
6. closed-loop wrapper simulation shows no deployment-induced behavioral regression.

Then D3 sim-to-sim is authorized.

## Source artifacts

Canonical freeze:

    docs/protocols/freeze/phase1-d0-canonical-freeze-manifest.md

D1 verdict:

    docs/verdicts/transfer/phase1-d1-multiseed-final-characterization-verdict.md

Runtime contract probe:

    scripts/rl/phase1_d2_contract_probe.py
    runs/phase1_d2_contract_probe/runtime_contract.json
