# Phase-1 D3 — Zero-Adaptation Sim-to-Sim Transfer Contract

Status: **PREDECLARED — D3-A INTERFACE WORK AUTHORIZED; D3-B/C BLOCKED UNTIL MUJOCO RUNTIME PROVENANCE IS FROZEN**
Date: 2026-09-26

## Scientific question

How much of the frozen canonical Phase-1 controller behavior transfers from the source IsaacLab simulator to an independently implemented MuJoCo A1 simulator when the policy, reward definitions, preference semantics, and deployment boundary are held fixed?

D3 is the first true transfer gate.

It is not:
- policy optimization;
- domain adaptation;
- system-identification-driven retuning;
- reward redesign;
- actuator retuning to rescue behavior.

First-pass D3 is zero adaptation.

## Frozen policy side

Every D3 experiment must use exactly:

    artifacts/phase1_canonical_actor_state.pt
    Phase1StandaloneActor
    Phase1EagerStateRuntime
    CUDA
    50 Hz policy rate

No training checkpoint object or TorchScript policy is allowed in D3 evidence.

## D3 decomposition

    D3-A  interface equivalence
    D3-B  nominal dynamics transfer
    D3-C  MORL semantic transfer

Failure classification:

    interface mismatch
        -> integration failure

    interface exact + locomotion collapse
        -> dynamics-transfer failure

    locomotion survives + T/A/O ordering disappears
        -> preference-semantic transfer failure

    T/A/O preserved + S changes
        -> characterize S transfer separately
## Secondary simulator freeze

Target engine:

    MuJoCo

Target model:

    talon_rl/assets/data/Robots/unitree_a1/mujoco/scene.xml
    includes a1.xml

Existing XML facts frozen before first rollout:

    position actuator kp          100
    actuator force range          [-33.5, 33.5] Nm
    model joint damping           class-dependent, base default 2
    collision friction            base default 0.6
    foot friction                 0.8 0.02 0.01

MuJoCo home pose:

    hip-abduction    0
    thigh            0.9
    calf            -1.8

This is intentionally not equal to the canonical Isaac default pose:

    [ 0.1, -0.1,  0.1, -0.1,
      0.8,  0.8,  1.0,  1.0,
     -1.5, -1.5, -1.5, -1.5 ]

The mismatch is a D3 dynamics/model fact, not something to fix before the first transfer measurement.

Before D3-B begins, freeze:
- exact Python interpreter;
- MuJoCo package version;
- model XML SHA256;
- MuJoCo timestep;
- integrator/solver options;
- actuator ctrl/force limits;
- contact/friction parameters;
- action hold/substep convention.

Current runtime status:

    pyproject requires mujoco>=3.0
    IsaacLab Python environment has no mujoco package
    uv --offline cannot resolve mujoco from cache

Therefore D3-B and D3-C are currently BLOCKED on secondary-runtime provenance, not on controller logic.
## D3-A — interface equivalence

Canonical observation vector:

    48-D

Exact order:

    base_lin_vel_b       3
    base_ang_vel_b       3
    projected_gravity_b  3
    velocity_command     3
    joint_pos_rel       12
    joint_vel_rel       12
    previous_action     12

Canonical joint order:

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

D3-A must prove:

1. MuJoCo joint state is reordered by joint name, not hardcoded index assumption.
2. Joint position observation is relative to the D2 canonical default offsets, not MuJoCo's home keyframe.
3. Joint velocity is represented in canonical joint order and units rad/s.
4. Base linear/angular velocities are expressed in the canonical base frame.
5. Projected gravity is computed in the canonical base frame.
6. Command remains [vx, vy, yaw_rate].
7. Previous action remains the prior normalized policy output in canonical joint order.
8. Policy action remains normalized [-1,1].
9. MuJoCo position target is:

       q_target_canonical = q_default_D2 + 0.25 * action

   followed only by canonical-name -> MuJoCo-actuator-order permutation.

Hard D3-A unit gate:

    same synthetic physical state
      -> Isaac canonical builder
      -> MuJoCo canonical builder

    max |Delta obs| <= 1e-7

and:

    same normalized action
      -> canonical target builder
      -> MuJoCo reordered target

    inverse-reordered target
      == q_default_D2 + 0.25*a

    max error <= 1e-7
## D3-B — nominal dynamics transfer

Prerequisites:

    D3-A PASS
    secondary runtime provenance frozen

No adaptation.

Use matched command/preference/reset families, but do not require bit-identical trajectories across physics engines.

Primary questions:
- does the robot remain numerically stable?
- does it avoid immediate base collapse?
- does it sustain locomotion for the evaluation horizon?
- are velocity, height, tilt, action and contact trajectories physically bounded?

Predeclared first-pass requirements:

    all states/actions finite                         required
    no actuator-target contract violation            required
    no immediate collapse in first 10 policy steps   required
    survival over 64-step matched horizon            >= 0.75 aggregate
    median trunk height                              physically positive and bounded
    no persistent action saturation > 95% samples    required

These are transfer viability gates, not semantic claims.

Failure does not authorize actuator tuning during the first D3 pass.

## D3-C — MORL semantic transfer

Only run if D3-B establishes nominal locomotion viability.

Use the same preference family:

    T-heavy
    A-heavy
    O-heavy
    S-heavy
    center
    all six pairwise continua
    sampled simplex interior

Requirements:

T/A/O:
- objective endpoint direction must remain correct on >= 0.75 matched suites;
- physical proxy direction must remain correct on >= 0.75 matched suites.

S:
- report valid / inconsistent / engineering-confounded;
- S is not a hard transfer acceptance requirement.

Preference-family structure:

    center compromise                  >= 0.75
    continuum monotonicity aggregate   >= 0.65
    endpoint-between aggregate         >= 0.65

Robustness:
- report survival separately;
- do not require exact Isaac numerical equality.

## Adaptation firewall

Before the frozen D3 result is recorded, forbidden:
- policy retraining/fine-tuning;
- reward changes;
- preference changes;
- observation feature changes;
- action-scale changes;
- actuator gain tuning for policy rescue;
- friction/contact tuning for policy rescue;
- adding filters not present in the frozen interface.

After D3 is frozen, any adaptation must be a new explicitly named phase and compared against this zero-adaptation baseline.

## Stop rules

If D3-A fails:
    stop; repair interface only.

If D3-A passes and D3-B fails:
    freeze dynamics-transfer failure.
    D3-C is NOT REACHED.

If D3-B passes and D3-C fails:
    freeze semantic-transfer limitation.

If D3-C passes:
    authorize D4 guarded hardware bring-up.

No D3 outcome reopens Phase-1 training.
