# Phase 5 P5-E1 — Frozen-Policy Validity Envelope Contract

Status: **PREDECLARED — CHARACTERIZATION ONLY**
Date: 2026-09-26

## Question

Before any Phase-5 training, how wide is the frozen canonical Phase-1 controller's semantic and engineering validity envelope over the frozen P5-E0 plant family?

## Policy

Frozen checkpoint:

    runs/authority_isolated_simplex_edge_retain-2026-09-25/model_30.pt

No training, fine-tuning, checkpoint selection, reward change, objective weighting change, plant-range change, or ensemble modification is allowed.

## Plant bank

Evaluate:

    nominal source                         1 plant
    deterministic in-ensemble bank       16 plants
    frozen E0 held-out bank              12 plants

The 16 in-ensemble plants use the frozen Sobol sampler after the 128 E0 sanity points:

    sampler       scrambled Sobol
    seed          26092650
    fast-forward  128
    draw          16
Held-out plants are copied exactly from:

    artifacts/phase5_e0_plant_ensemble_manifest.json

Every plant tuple is applied deterministically after environment reset.

## Semantic protocol

Reuse the exact H2a semantic protocol:

    4 matched suites
    8 lanes per plant
    64 policy steps

Endpoints:

    T A O S C

Continuum:

    T-A T-O T-S A-O A-S O-S
    alpha = 0,.25,.5,.75,1

Endpoint semantic gates remain:

    objective correctness >= .75
    physical correctness  >= .75
    endpoint survival     >= .95

S is characterized honestly and is not a required Phase-5 success objective.
## Engineering characterization

For every plant report:

    endpoint / global survival
    minimum trunk height
    maximum tilt
    normalized-action saturation fraction
    non-finite fraction
    control-contract violation fraction

Action saturation is descriptive:

    coordinate saturation = fraction(|a_j| >= .95)

No E1 checkpoint or ensemble decision may use this metric.

## Authority characterization

On center-rollout observations, query T/A/O/S/C on the same states.

Report:

    mean pairwise action distance
    minimum pairwise edge distance
    heavy-heavy mean distance
    heavy-center mean distance
    simplex tangent finite-difference norm

Plant authority retention is descriptive relative to nominal source.

Reference durability threshold for reporting only:

    authority retention >= .90
## Critic characterization

Use the frozen H2a H32 current-policy validation:

    mean EV
    negative-EV fraction
    mean absolute bias

Reference critic gate:

    EV mean > 0
    negative fraction <= .25

## Aggregation

Nominal source:

    individual report

In-ensemble:

    mean
    median
    p10
    worst case
    per-objective semantic pass fraction
    survival pass fraction
    authority-retention fraction
    critic-valid fraction

Held-out:

    all 12 plants individually
    aggregate pass fractions
## Plant sensitivity

For descriptive characterization only, report rank correlations of frozen plant latents:

    mass_delta_kg
    passive_blend
    contact_blend

against:

    survival
    T/A/O/S objective margins
    T/A/O/S physical margins
    continuum monotonicity
    center compromise
    authority metrics

These correlations cannot alter E0 support, choose objective weighting, or select E2 training samples.

## Interpretation

Case 1:
    semantics mostly valid across plants
    -> low E2 headroom.

Case 2:
    survival remains high while semantics degrade
    -> direct support for the Phase-5 semantic-robustness treatment question.

Case 3:
    locomotion broadly collapses
    -> ensemble is too harsh for a clean semantic-learning interpretation.

Case 4:
    held-out plants are materially worse than in-ensemble bank
    -> pre-training generalization gap established.

## Governance

E1 is characterization only.

P5-E2 remains BLOCKED until E1 results are frozen.

No objective-specific rescue, especially S-specific treatment, is authorized.
