# Phase 4 — Read-Only Controller Transfer Robustness Audit Contract

Status: **PREDECLARED — READ ONLY**
Date: 2026-09-26

## Motivation

The plant-calibration branch is closed.

Frozen findings:

    actuator-law mismatch              materially addressed
    passive-joint mismatch            materially addressed
    free-space local qdot response     near source
    fixed canonical stance viability  preserved
    single dominant residual plant parameter not identified

Yet the frozen controller still diverges in closed loop under small distributed source-target differences.

Primary research question:

> How sensitive is the frozen controller to small distributed model/state errors, and is there evidence of high closed-loop amplification that explains transfer failure?

## Non-goals

This phase does NOT:
- train or fine-tune the policy;
- change rewards;
- change simulator parameters;
- change semantic thresholds;
- add robustness regularizers;
- perform domain randomization;
- select a new controller.

Any later intervention requires a separate predeclared branch.

## Frozen controller

    artifacts/phase1_canonical_actor_state.pt
    Phase1EagerStateRuntime
    CUDA
    50 Hz

Primary source states come from the validated H2a / T1b source distribution.

## Audit ladder

### R0 — same-state actor sensitivity

For fixed source-valid observations:

    J_s = da / ds

Estimate with deterministic centered finite differences.

Report:
- spectral norm / top singular value approximation;
- per-observation-block sensitivity;
- action perturbation norm per unit state perturbation;
- T-heavy and center separately.

Observation blocks:

    base linear velocity
    base angular velocity
    projected gravity
    command
    joint position
    joint velocity
    previous action

Preference input remains frozen within each probe.

### R1 — one-step closed-loop amplification

Use matched source state and action.

Inject small generic state perturbations before policy evaluation, then propagate one source/target transition.

Measure:

    ||delta s_t||
    ||delta a_t||
    ||delta s_(t+1)||

Derived gains:

    G_pi = ||delta a_t|| / ||delta s_t||
    G_dyn = ||delta s_(t+1)|| / ||delta a_t||
    G_cl = ||delta s_(t+1)|| / ||delta s_t||

Compare:
- source-valid Isaac transition;
- T4 target transition.

### R2 — finite-horizon divergence ladder

From the same initial physical state:

    H = 1, 2, 4, 8, 16, 32, 64

Track:

    state divergence
    action divergence
    joint-state divergence
    height divergence
    tracking semantic divergence
    cumulative objective-margin divergence

Estimate an early closed-loop divergence rate from the pre-saturation region.

### R3 — sensitivity-to-semantics link

Compare states/trajectories that remain source-valid with target trajectories that later fail.

Ask whether large local/finite-horizon gain predicts:

    height-floor crossing
    T semantic reversal
    A semantic degradation
    continuum breakdown

This is diagnostic correlation only, not a predictor-mining phase.

## Perturbation rule

Use small symmetric perturbations with scales tied to the existing observation normalization / physically meaningful units.

No perturbation scale may be adjusted after seeing semantic outcomes.

Default normalized perturbation amplitudes:

    base linear velocity     0.01
    base angular velocity    0.01
    projected gravity        0.005
    joint position           0.005 rad
    joint velocity           0.05 rad/s
    previous action          0.01

Command and preference are held fixed for the primary audit.

## Decision logic

If source-valid and transfer-failing regimes separate clearly by closed-loop gain / divergence rate:

    authorize one controller-robustness intervention branch.

Candidate families may include:
- domain randomization around the frozen residual envelope;
- Jacobian / Lipschitz regularization;
- closed-loop robustness regularization;
- robust training across a plant ensemble.

The specific intervention must be chosen from the observed mechanism, not from preference.

If actor Jacobian is high but does not predict closed-loop failure:

    do not authorize Jacobian regularization.

If local gains are modest but finite-horizon amplification is high:

    treat the failure as dynamics-feedback accumulation rather than purely actor-local sensitivity.

If no stable separation appears:

    freeze distributed closed-loop sensitivity as a limitation.
    No robustness regularizer is authorized.

## Governance

Plant-calibration results remain frozen.

No simulator parameter fitting is reopened inside Phase 4.

D4 hardware remains blocked until a future branch passes a fresh semantic-transfer gate.
