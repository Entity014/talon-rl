# Phase-1 D2 — Deployment Equivalence Validation Verdict

Status: **FROZEN — PASS ON CUDA EAGER-STATE RUNTIME; CPU AND JIT PATHS NOT AUTHORIZED**
Date: 2026-09-26

## Executive verdict

The exact frozen Phase-1 canonical controller can be executed outside the training actor stack with exact policy-function and closed-loop equivalence, provided deployment uses the validated actor-only eager-state runtime on CUDA.

Official deployment artifact:

    artifacts/phase1_canonical_actor_state.pt

Artifact SHA256:

    ea47e3b8b8d87f0b5536383b88a62da22457bfcf9ccc1e55c4099bb24a397f63

Canonical source checkpoint remains:

    runs/authority_isolated_simplex_edge_retain-2026-09-25/model_30.pt

No policy weights or method parameters were changed in D2.

## D2-A — observation contract

Canonical policy input:

    48 dimensions

Exact order:

    base_lin_vel          3
    base_ang_vel          3
    projected_gravity     3
    velocity_commands     3
    joint_pos_rel        12
    joint_vel_rel        12
    previous action      12

Clean manual reconstruction was compared against the IsaacLab ObservationManager over 32 closed-loop steps.

Result:

    max absolute error = 0.0
    VERDICT           = PASS

The source training/evaluation environment uses observation corruption. The deterministic deployment reconstruction contract is the clean underlying observation definition; noise/corruption is not required as a deployment transform.
## D2-B — action contract

Policy output:

    12-D normalized action in [-1, 1]

Joint order:

    FL_hip_joint
    FR_hip_joint
    RL_hip_joint
    RR_hip_joint
    FL_thigh_joint
    FR_thigh_joint
    RL_thigh_joint
    RR_thigh_joint
    FL_calf_joint
    FR_calf_joint
    RL_calf_joint
    RR_calf_joint

Target mapping:

    q_target = q_default + 0.25 * action

Default offsets:

    [ 0.1, -0.1,  0.1, -0.1,
      0.8,  0.8,  1.0,  1.0,
     -1.5, -1.5, -1.5, -1.5 ]

Manual mapping versus the IsaacLab action term:

    max absolute error = 0.0
    VERDICT           = PASS

Control period:

    physics dt   5 ms
    decimation   4
    policy dt    20 ms
    policy rate  50 Hz
## D2-C — deployment actor equivalence

A standalone actor architecture was separated from the training actor-critic and populated only from the frozen canonical actor state.

Frozen-probe parity:

    CPU max action error    0.0
    CUDA max action error   0.0

Therefore the actor-state artifact and standalone deployment actor preserve the deterministic policy function exactly on both tested devices.

    VERDICT = PASS

### Compiled TorchScript finding

TorchScript/script and traced artifacts passed pointwise frozen-probe checks, including sub-micro or exact errors depending on device/probe.

However in matched closed-loop continuum evaluation, tiny compiled-graph numerical deviations accumulated enough to alter the aggregate continuum gate.

Observed examples:

    JIT CPU online max action error     6.90e-6
    continuum endpoint-between          0.6493  < 0.65

    JIT/trace CUDA online max error      4.69e-7
    continuum endpoint-between          0.6215  < 0.65

The threshold was not relaxed.

Therefore compiled JIT artifacts are not authorized as the canonical deployment representation for this controller.

This is a deployment numerical-sensitivity finding, not a policy-learning failure.
## D2-D — timing and stale-data behavior

Official CUDA eager-state runtime:

    median latency     0.322 ms
    p95                0.800 ms
    p99                1.127 ms
    max                1.477 ms
    calls >20 ms       0 / 1000

Predeclared gate:

    p99 < 10 ms
    zero calls >20 ms

CUDA:

    PASS

CPU eager-state runtime:

    median             0.939 ms
    p95                7.064 ms
    p99               10.975 ms
    max               14.127 ms
    calls >20 ms       0 / 1000

CPU violates the frozen p99 <10 ms gate.

CPU:

    NOT AUTHORIZED

No timing threshold was changed after measurement.

Stale-data contract on the official runtime:

    <=20 ms            normal inference
    20-40 ms           hold previous safe action
    >40 ms             zero action + latched estop

30 ms hold test:

    PASS

41 ms safe-stop/latch test:

    PASS
## D2-E — deployment guards

Official eager-state runtime rejects:

    wrong observation width       PASS
    wrong preference width        PASS
    NaN observation               PASS
    NaN preference                PASS
    negative preference weight    PASS
    invalid simplex sum           PASS

Explicit estop:

    zero-action latch             PASS
    remains latched               PASS
    explicit reset restores run   PASS

Deterministic repeated inference:

    bitwise identical             PASS

No hardware motor-disable semantics are claimed here. D2 validates the software policy boundary; physical estop/drive-disable integration remains D4 scope.

## D2-F — deployment-wrapper closed-loop equivalence

The standalone actor-state runtime was placed in the canonical IsaacLab closed loop and evaluated on the exact frozen H2a endpoint and continuum suites.

Online source-vs-deployment action parity:

    max absolute error = 0.0

Endpoint semantic result:

    T   PASS
    A   PASS
    O   PASS
    S   FAIL / canonical limitation preserved

All endpoint correctness fractions and survival values match the canonical direct-call report exactly.

Preference-family result:

    continuum monotonicity        0.7473958333
    continuum endpoint-between    0.6666666667
    center compromise             0.875
    minimum survival              0.875

Canonical deltas:

    endpoint metrics              0
    continuum monotonicity        0
    continuum between             0
    center compromise             0

Therefore:

    D2-F VERDICT = PASS
## Final D2 decision

    D2-A observation equivalence       PASS
    D2-B action equivalence            PASS
    D2-C actor-state equivalence       PASS
    D2-D CUDA timing/stale behavior    PASS
    D2-E safety/runtime guards         PASS
    D2-F closed-loop equivalence       PASS

Official Phase-1 deployment boundary:

    actor-only eager-state artifact
    + Phase1StandaloneActor
    + Phase1EagerStateRuntime
    + CUDA execution
    + 50 Hz policy loop

Rejected for canonical deployment:

    TorchScript scripted actor
    TorchScript traced actor
    CPU runtime under current p99 gate

Overall:

    D2 DEPLOYMENT EQUIVALENCE          PASS
    D3 SIM-TO-SIM                      AUTHORIZED

## Scientific interpretation

D2 demonstrates that the frozen canonical policy can leave the training actor-critic stack without changing its deterministic controller function or validated closed-loop behavior.

It also reveals that this controller is unusually sensitive to tiny numerical perturbations over long closed-loop horizons: compiled graph differences below 1e-6 at the action level were sufficient to change the continuum qualification.

Accordingly, serialization/runtime equivalence for this controller must be established behaviorally as well as pointwise.

This finding does not reopen policy training or method selection.
## Source artifacts

Contract:

    docs/contracts/transfer/phase1-d2-deployment-equivalence-contract.md

Deployment module:

    talon_rl/phase1_deployment.py

Official actor-state artifact:

    artifacts/phase1_canonical_actor_state.pt

Runtime contract probe:

    scripts/rl/phase1_d2_contract_probe.py
    runs/phase1_d2_contract_probe/runtime_contract.json

Observation/action parity:

    scripts/rl/phase1_d2_obs_action_parity.py
    runs/phase1_d2_obs_action_parity/obs_action_parity.json

Official actor-state parity:

    scripts/rl/phase1_d2_eager_state_parity.py
    runs/phase1_d2_eager_state_parity/eager_state_parity.json

Official runtime guard/timing:

    scripts/rl/phase1_d2_eager_runtime_guard.py
    runs/phase1_d2_eager_runtime_guard/runtime_guard.json

Closed-loop equivalence:

    scripts/rl/phase1_d2_closed_loop.py
    runs/phase1_d2_closed_loop/

Rejected compiled-path diagnostics:

    scripts/rl/phase1_d2_runtime_parity.py
    scripts/rl/phase1_d2_trace_parity.py
    artifacts/phase1_canonical_actor.pt
    artifacts/phase1_canonical_actor_trace.pt
