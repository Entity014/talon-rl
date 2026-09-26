# Phase-1 D3-A — Live MuJoCo Interface Equivalence Verdict

Status: **FROZEN — PASS**
Date: 2026-09-26

## Runtime provenance

Secondary engine:

    Python       3.12.3
    MuJoCo       3.3.7
    NumPy        2.1.3

Model:

    talon_rl/assets/data/Robots/unitree_a1/mujoco/scene.xml
    scene SHA256  415eb95e2982c5a90c873f10d45e3efee94186ef6050e2c9aaea40135ff569ad

    talon_rl/assets/data/Robots/unitree_a1/mujoco/a1.xml
    model SHA256  492978cb7700b0064bcefbc2b93dba651a828cf9212df91d56e951666890807c

Physics:

    timestep      0.002 s
    integrator    Euler
    solver        Newton
    iterations    100
    gravity       [0,0,-9.81]

Actuation:

    12 position actuators
    kp            100
    force limits  +/-33.5 Nm

## A1 structural interface

    joint vocabulary exact              PASS
    actuator vocabulary exact           PASS
    joint-name permutation              PASS
    synthetic 48-D reconstruction       0.0 error
    action target roundtrip             0.0 error

## A2 live-runtime semantics

MuJoCo free-joint conventions were verified live.

Canonical observation mapping:

    root COM linear velocity in base frame
      = R^T v_free_origin_world
        + omega_body x trunk_COM_offset_body

    root angular velocity in base frame
      = free-joint rotational qvel

    projected gravity
      = R^T [0,0,-1]

Trunk COM offset in body frame:

    [0, 0.0041, -0.0005]
Live controlled-state results:

    COM velocity vs engine body COM       0.0 max error
    angular velocity vs engine            0.0 max error
    full 48-D observation                 0.0 max error

    action -> actuator ctrl roundtrip      0.0 max error
    full canonical action cube             within all ctrl ranges

    position actuator:
      zero position error force            0.0
      +0.05 rad target error force         +5.0
      kp=100 semantic error                0.0

    physics step                           2 ms
    policy hold                            10 steps
    policy hold duration                   20 ms exact

Therefore the engine sends and consumes state/action semantics exactly as required by the canonical Phase-1 interface once the explicit adapter is applied.

## Important velocity finding

The canonical IsaacLab observation uses root COM linear velocity, not merely free-joint-origin velocity.

Using mj_objectVelocity(local=1) directly would not reproduce the canonical quantity.

The validated mapping explicitly accounts for the trunk COM offset.

## D3-A final decision

    D3-A1 structural/interface math       PASS
    D3-A2 live runtime mapping            PASS
    D3-A overall                          PASS

    D3-B nominal dynamics                 AUTHORIZED
    D3-C semantic transfer                BLOCKED pending D3-B

No policy, action scale, actuator gain, contact parameter, reward, or model parameter was adapted to obtain this result.

## Source artifacts

    docs/contracts/transfer/phase1-d3-zero-adaptation-sim2sim-contract.md
    talon_rl/phase1_mujoco_adapter.py
    scripts/rl/phase1_d3_runtime_provenance.py
    scripts/rl/phase1_d3a_structural_interface.py
    scripts/rl/phase1_d3a_freejoint_basis_probe.py
    scripts/rl/phase1_d3a_canonical_velocity_probe.py
    scripts/rl/phase1_d3a_live_interface_final.py

    runs/phase1_d3_runtime_provenance/
    runs/phase1_d3a_structural_interface/
    runs/phase1_d3a_live_runtime/
