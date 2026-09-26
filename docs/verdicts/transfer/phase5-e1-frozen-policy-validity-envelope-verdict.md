# Phase 5 P5-E1 — Frozen-Policy Validity Envelope Verdict

Status: **FROZEN — CHARACTERIZATION COMPLETE / P5-E2 AUTHORIZED SUBJECT TO INITIALIZATION MANIFEST**
Date: 2026-09-26

## Decision

P5-E1 establishes a clear pre-training semantic-robustness gap.

The frozen canonical Phase-1 controller remains valid on its canonical source-domain anchor, while deterministic plant variations largely preserve endpoint locomotion viability but substantially reduce preference-semantic validity.

    P5-E0  ensemble freeze             PASS / FROZEN
    P5-E1  baseline envelope           COMPLETE / FROZEN
    P5-E2  training screen             AUTHORIZED AFTER E2 INIT MANIFEST
    P5-E3  held-out validation         BLOCKED
    P5-E4  MuJoCo                     BLOCKED
    D4     hardware                    BLOCKED

Final machine-readable E1 report:

    runs/phase5_e1_validity_envelope/e1_final_report.json
## Source-domain anchor

The exact frozen H2a source-domain report is retained as the no-regression anchor.

This is distinct from a deterministic local plant tuple because canonical Phase-1 source evaluation includes the original source-domain environment/randomization contract.

Source-domain result:

    T semantics                  PASS
    A semantics                  PASS
    O semantics                  PASS
    S semantics                  FAIL / known limitation

    endpoint survival            1.00
    continuum monotonicity       0.7474
    endpoint-between             0.6667
    center compromise            0.875

    critic H32 EV mean           0.0471
    critic negative fraction     0.2063
    critic                       PASS

This preserves the frozen Phase-1 claim.
## Correction to first E1 aggregate

The first deterministic-bank run labeled the exact tuple:

    mass delta       0
    passive blend    0
    contact blend    0

as "nominal source".

That label is not equivalent to the canonical source-domain anchor because it removes the original source-domain plant randomization.

No plant data were invalidated.

The tuple is retained and renamed:

    P0_source_nominal_tuple

The source-domain anchor is the frozen exact H2a evaluation.

This correction changes no policy, plant parameter, metric, threshold, or training decision.
## Deterministic nominal tuple

P0_source_nominal_tuple:

    endpoint survival            1.00

    T semantics                  PASS
    A semantics                  FAIL
    O semantics                  PASS
    S semantics                  FAIL

    center compromise            PASS
    continuum monotonicity       PASS
    endpoint-between             FAIL

    critic validity              FAIL

This result itself is informative:

> the local nominal plant point is not equivalent to the broader source-domain distribution on which the controller was validated.

## In-ensemble baseline

16 deterministic Sobol plants:

    endpoint survival gate       16 / 16 = 1.000

Semantic pass fraction:

    T                            6 / 16  = 0.375
    A                           10 / 16  = 0.625
    O                           15 / 16  = 0.9375
    S                            1 / 16  = 0.0625
Preference-family structure:

    center pass                 15 / 16 = 0.9375
    continuum monotonic pass   16 / 16 = 1.000
    endpoint-between pass       2 / 16 = 0.125

Authority retention relative to deterministic nominal tuple:

    pairwise >= .90            14 / 16 = 0.875
    tangent >= .90              9 / 16 = 0.5625

Critic validity:

    0 / 16

Thus the main baseline failure is not broad endpoint locomotion collapse.

The controller usually remains operational while semantic ordering, differential authority, endpoint-between structure, and critic validity become plant-sensitive.

## Held-out baseline

12 frozen held-out plants:

    endpoint survival gate      11 / 12 = 0.9167

Semantic pass fraction:

    T                            7 / 12 = 0.5833
    A                            5 / 12 = 0.4167
    O                           10 / 12 = 0.8333
    S                            0 / 12 = 0.0000

Preference-family structure:

    center pass                 11 / 12 = 0.9167
    continuum monotonic pass   12 / 12 = 1.000
    endpoint-between pass       1 / 12 = 0.0833
Authority retention:

    pairwise >= .90             9 / 12 = 0.750
    tangent >= .90              6 / 12 = 0.500

Critic validity:

    0 / 12

Held-out plants are not uniformly worse than in-ensemble plants on every objective.

They are worse on A, O, S pass fractions, while T happens to be higher.

Therefore E1 does not support a one-dimensional "held-out is harder" interpretation.

It establishes heterogeneous plant-dependent semantic validity.

## Engineering envelope

Endpoint locomotion is broadly preserved:

    in-ensemble endpoint survival     1.000
    held-out endpoint survival        0.9167

The lower global-protocol survival values are driven partly by the already-known continuum reset failure pattern and must not be interpreted as broad locomotion collapse.

No non-finite or control-contract pathology dominated E1.

Action saturation remains high and is reported descriptively, not used to modify the ensemble.
## Plant sensitivity

Plant-latent correlations are descriptive only.

Notable associations include:

    passive blend vs pairwise authority      rho ~ -0.637
    passive blend vs O objective margin      rho ~ +0.821
    passive blend vs O physical margin       rho ~ -0.722

    mass vs A physical margin                rho ~ -0.542
    mass vs O objective margin               rho ~ -0.528
    mass vs O physical margin                rho ~ +0.644

    contact blend vs S physical margin       rho ~ +0.461

These relationships are not used to alter ranges, weighting, or sampling.

The heterogeneous signs reinforce the decision not to convert E1 into parameter-specific tuning.

## Critic interpretation

The frozen source critic passes on the canonical source-domain anchor but fails the frozen H32 gate on all deterministic plant-bank evaluations.

This is a baseline generalization limitation.

It does not authorize a critic-specific repair.

Under the Phase-5 contract, E2 may update the same actor/critic machinery through the sole treatment:

    plant distribution during training

A successful E2 treatment must recover critic validity while preserving source-domain semantics and authority.
## E1 interpretation

Observed regime:

    endpoint locomotion mostly preserved
    +
    semantic validity degrades across plant variation

Therefore E1 matches the intended Phase-5 treatment question:

> Can exposure to the frozen bounded plant ensemble improve semantic robustness, rather than merely increase survival?

This is closest to preregistered Case 2.

Important qualification:

- O semantics are already comparatively robust.
- T and A show substantial headroom.
- S remains a characterized limitation and is not a treatment success target.
- critic validity has substantial off-source headroom.
- continuum monotonicity is robust, but endpoint-between structure is fragile.

## Authorization boundary

P5-E2 is scientifically authorized, but training may not start until a dedicated E2 initialization manifest freezes:

- source training state;
- treatment/control seed;
- training budget;
- treatment plant sampler;
- nominal control plant distribution;
- checkpoint policy;
- output isolation;
- source-domain no-regression evaluation;
- prohibition on E1/E3/E4 semantic checkpoint selection.

No E2 result may modify the P5-E0 ensemble.

P5-E3 and P5-E4 remain blocked.
