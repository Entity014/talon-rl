# Phase 4 — Controller Transfer Robustness Audit Verdict

Status: **FROZEN — FEEDBACK AMPLIFICATION CONFIRMED; NO SINGLE ROBUSTNESS REGULARIZER AUTHORIZED**
Date: 2026-09-26

## Context

Frozen prior results:

    plant calibration branch                 CLOSED
    D3 zero-adaptation semantic transfer     FAIL
    T1 dynamics-driven divergence            SUPPORTED
    T2 nominal PD calibration                FAIL
    T3 explicit DCMotor emulation            materially helpful, insufficient
    T4 passive-joint correction              local parity strongly improved, closed-loop FAIL
    post-T4 single-parameter plant tuning    NOT AUTHORIZED

Phase 4 asked:

> How sensitive is the frozen controller to small distributed source-target errors, and does a compact sensitivity mechanism explain semantic transfer failure?

No training, regularization, domain randomization, policy update or simulator fitting was performed.

## R0 — actor-local state sensitivity

Deterministic centered finite-difference Jacobians were evaluated on 32 canonical H2a source states.

Mean action Jacobian spectral norm:

    T-heavy        10.07
    center          9.05

T-heavy exceeds center on only:

    56.25% of matched states

Largest mean block sensitivities:

    T-heavy:
        joint position          7.07
        base linear velocity    5.57
        projected gravity       2.91

    center:
        joint position          6.37
        base linear velocity    5.03
        projected gravity       2.62

Interpretation:

The actor is locally sensitive, especially to joint position and base linear velocity.

However T-heavy is not uniformly more sensitive than center and actor-local sensitivity alone does not identify the transfer-failing regime.

No Jacobian regularization is authorized from R0.

## R1 — one-step closed-loop amplification

Blockwise symmetric observation perturbations were propagated from matched physical source states.

Actor gain G_pi is identical across engines, as expected from the same frozen policy.

Source PhysX one-step closed-loop gain is generally higher than the T4 target gain.

Representative mean G_cl values:

    T-heavy                     Isaac      MuJoCo T4
    ------------------------------------------------
    base linear velocity        518.98       377.48
    base angular velocity       337.59       170.20
    projected gravity           624.75       203.12
    joint position              341.08       201.95
    joint velocity               32.93        22.81
    previous action             152.54        55.91

Center shows the same qualitative pattern.

Interpretation:

> the target simulator is not locally more explosive than the source under one-step perturbation gain.

Therefore a simple “target local closed-loop gain is too large” hypothesis is rejected.

## R2 — finite-horizon source-target divergence

A corrected matched protocol was used:

- same source physical state;
- same actual command;
- same preference;
- previous-action state reset identically;
- target traces aligned post-step with source traces.

Earlier intermediate R2 reports with stale previous-action state or pre/post-step misalignment are invalid and are not used in this verdict.

At H1:

    action divergence            approximately 0
    T height difference          ~0.23 mm
    C height difference          ~0.22 mm
    T tracking difference        ~0.009
    C tracking difference        ~0.010

Thus policy response begins equivalent and the first discrepancy is produced by residual plant dynamics.

Action divergence then accumulates:

                         T-heavy       center
    H1                   ~0            ~0
    H2                    0.047         0.061
    H4                    0.084         0.131
    H8                    0.180         0.274
    H16                   0.580         0.655
    H32                   0.846         0.760
    H64                   0.958         0.863

Instantaneous normalized state divergence grows strongly over the early horizon.

H8 / H1 instantaneous state-divergence ratio:

    T-heavy        14.81
    center         13.18

Early log-divergence slope:

    T-heavy        ~0.312 per policy step
    center         ~0.291 per policy step

Semantic source-target agreement is initially preserved and begins to separate at multi-step horizon.

At H1:

    source T tracking direction       4/4
    target T tracking direction       4/4

At H8:

    source T tracking direction       3/4
    target T tracking direction       2/4

This directly supports:

> small residual plant differences create state error first; repeated policy feedback then creates increasing action divergence and trajectory-level semantic separation.

## R3 — sensitivity-to-semantic-failure link

The decisive question was whether a compact early-divergence metric predicts which source-valid lanes lose T semantics after transfer.

Across all 32 matched T-vs-center lane pairs:

    early action divergence vs tracking degradation
        Spearman rho   0.027
        p               0.881

    early action divergence vs objective degradation
        Spearman rho   0.130
        p               0.478

Among lanes that are source-semantic-valid on both physical and objective criteria:

    source-valid lanes                 12
    target semantics retained           8
    target semantics lost               4

Mean early H2-H8 action divergence:

    retained                           0.335
    lost                               0.500

But the distributions overlap strongly:

    Mann-Whitney p                     0.683

One failed lane has very low early divergence while another has very high divergence.

Therefore no stable threshold or monotonic relation exists between the tested early action-divergence scalar and semantic transfer loss.

## Final interpretation

Phase 4 confirms a system-level mechanism:

    residual plant mismatch
        -> small first-step state error
        -> policy receives shifted state
        -> action trajectories diverge
        -> repeated feedback amplifies trajectory differences
        -> semantic behavior can separate

However the audit does NOT support the stronger claim:

    one compact actor Jacobian / local gain / early action-divergence scalar
        -> predicts semantic failure

The transfer limitation is therefore best characterized as:

> distributed multi-step closed-loop sensitivity rather than a single local-sensitivity defect.

This is consistent with the broader thesis chain:

    local action parity
        != closed-loop equivalence

    local plant parity
        != closed-loop viability

    closed-loop divergence
        != a single compact semantic-failure predictor

## Decision

    actor-local sensitivity mechanism         INSUFFICIENT
    target one-step gain excess              NOT SUPPORTED
    finite-horizon feedback amplification    STRONGLY SUPPORTED
    compact gain -> semantic failure link     NOT SUPPORTED

Therefore:

    Jacobian regularization                  NOT AUTHORIZED
    Lipschitz regularization                 NOT AUTHORIZED
    sensitivity-metric-specific penalty      NOT AUTHORIZED
    ad-hoc domain randomization tuning       NOT AUTHORIZED

No controller-training intervention is opened from this audit.

A future robustness branch would need a broader preregistered plant-ensemble training formulation rather than optimization against the audited scalar metrics.

D4 hardware remains blocked.

## Artifacts

    docs/contracts/transfer/phase4-controller-transfer-robustness-audit-contract.md

    scripts/rl/phase4_r0_capture_source_obs.py
    scripts/rl/phase4_r0_actor_sensitivity.py
    scripts/rl/phase4_r1_isaac_closed_loop_gain.py
    scripts/rl/phase4_r1_mujoco_closed_loop_gain.py
    scripts/rl/phase4_r2_isaac_source_trajectories.py
    scripts/rl/phase4_r2_mujoco_target_trajectories.py
    scripts/rl/phase4_r2_compare_divergence.py

    runs/phase4_controller_robustness/

## Invalid intermediate analyses

The following intermediate R2 states were diagnosed and discarded before verdict:

1. source center rollout with stale ActionManager previous-action state;
2. target trace logged pre-step while source trace was logged post-step.

Neither invalid intermediate result contributes to this verdict.
