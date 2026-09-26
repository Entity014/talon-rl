# Phase 5 P5-E2 — Training Screen Evaluation Contract

Status: **PREDECLARED BEFORE U30 SEMANTIC EVALUATION**
Date: 2026-09-26

## Fixed checkpoints

Treatment:
    runs/phase5_e2_ensemble_seed75001/model_30.pt

Control:
    runs/phase5_e2_control_seed75001/model_30.pt

Both are fixed u30 checkpoints. No best-checkpoint selection is permitted.

## Source-domain hard gates

For the ensemble treatment checkpoint:

    T endpoint semantics                  PASS
    O endpoint semantics                  PASS
    A                                    report under D1 seed-sensitive interpretation
    S                                    report only
    endpoint survival                     >= .95
    fixed-probe pairwise authority        >= .90 retention from u20
    fixed-probe tangent authority         >= .90 retention from u20
    critic H32 EV mean                    > 0
    critic H32 negative-EV fraction       <= .25

Matched control is evaluated identically and reported.

## In-ensemble hard gates

Use exactly the 16 P5-E1 Sobol plants I01-I16.

No E3 held-out plant may be evaluated in E2.

For treatment:

    endpoint survival gate pass fraction  >= .90
    T semantic plant-pass fraction        >= .75
    O semantic plant-pass fraction        >= .75

Matched-control comparison:

    treatment T plant-pass fraction >= control T plant-pass fraction
    treatment O plant-pass fraction >= control O plant-pass fraction

A and S are characterized identically and cannot become treatment-specific targets.

## Descriptive, non-gating outputs

Report:

    center compromise
    continuum monotonicity
    endpoint-between
    action saturation
    plant sensitivity
    pairwise/tangent authority on in-ensemble states
    critic validity on in-ensemble states

These may explain the outcome but cannot alter the E2 verdict.

## Stop rule

If source T/O, source authority, or source critic fails:

    E2 FAIL
    E3 BLOCKED

If source passes but in-ensemble T/O or survival hard gates fail:

    E2 FAIL
    E3 BLOCKED

If treatment passes all absolute gates but is below matched control on T or O:

    E2 does not establish ensemble-training benefit
    E3 BLOCKED

Only a full E2 PASS authorizes E3 held-out evaluation.
