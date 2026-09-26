# Trajectory Information Attribution Audit Contract

Status: **PREDECLARED — MEASUREMENT ONLY, NO TRAINING / NO REWARD REDESIGN**

Date: 2026-09-24

## Motivation

The previous finite-policy audit rejected trajectory-level sufficiency of the current scalar objective returns:
- local objective semantics remain valid;
- actual normalized objective return can improve while semantic competence deteriorates.

This audit asks a narrower question:

> Which trajectory information seen by the semantic endpoint is discarded by the current total/mean objective return?

The goal is attribution, not feature mining and not reward redesign.

## Policy paths

Use exactly the four policy paths from the trajectory-sufficiency audit:

1. V2-B lambda=.95: u10, u25, u50, u75
2. V2-B lambda=1.00: u10, u25, u50, u75
3. paired weighted-PPO control: 0, 5, 10, 15, 20, 25
4. paired max-min floor: 0, 5, 10, 15, 20, 25

Use the exact matched endpoint suites and seeds from the frozen semantic evaluator:
- 4 suites
- seeds 840001..840004
- 64 steps
- T/A/O/S-heavy and center preferences
- same V2-B evaluator environment and objective normalization

No optimizer step is allowed.

## Per-step signals

For each axis i and suite, record across all 64 steps:

1. normalized objective signal under i-heavy preference:
   r_i^H(t)
2. normalized objective signal under center preference:
   r_i^C(t)
3. physical proxy under i-heavy preference:
   p_i^H(t)
4. physical proxy under center preference:
   p_i^C(t)

Define higher-is-better relational signals:

    a_obj_i(t)  = r_i^H(t) - r_i^C(t)
    a_phys_i(t) = p_i^C(t) - p_i^H(t)

The semantic endpoint uses the same direction convention.

## Frozen descriptor families

### A. Aggregate baseline
- total/mean heavy objective return
- mean objective advantage
- mean physical advantage

### B. Temporal structure
For objective advantage and physical advantage separately:
- early mean: steps 0..20
- middle mean: steps 21..42
- late mean: steps 43..63
- late-minus-early change
- least-squares time slope

### C. Tail / dispersion
For heavy physical proxy:
- standard deviation
- p90
- p95
- maximum excursion

For heavy objective signal:
- standard deviation
- p10 (lower tail)

For relational advantage signals:
- standard deviation
- worst 16-step rolling mean

### D. Relational persistence
For objective and physical advantage separately:
- fraction of steps with positive advantage
- fraction of steps with negative advantage
- sign-flip count
- longest consecutive negative-advantage run
- final 16-step mean advantage
- minimum cumulative-prefix mean advantage

Zero tolerance for sign is fixed at 1e-10.

No externally chosen physical threshold is introduced; all persistence measures are relative to the matched center trajectory.

## Attribution targets

For every consecutive policy transition and axis, compute descriptor changes and compare against:

- Delta semantic objective margin
- Delta semantic physical margin
- PASS -> FAIL event
- FAIL -> PASS event

Primary continuous target is the corresponding semantic margin:
- objective-derived descriptors -> Delta M_obj
- physical/relational-physical descriptors -> Delta M_phys

## Analysis rule

The audit is family-first, not best-single-feature-first.

For every descriptor report:
- Spearman correlation with target change
- Pearson correlation with target change
- sign agreement where descriptor orientation is higher-is-better

For each descriptor family report:
- median absolute Spearman across its descriptors
- maximum absolute Spearman (diagnostic)
- number of descriptors with |Spearman| >= 0.50

A descriptor is considered a **candidate missing trajectory statistic** only if:
1. |Spearman| >= 0.50 overall;
2. its sign/meaning is interpretable a priori from the frozen definition;
3. the same qualitative relation is present in at least 3 of 4 policy paths OR at least 3 of 4 axes where applicable;
4. it improves materially over the scalar-return baseline Spearman from the previous audit.

A family is considered **informative** if:
- at least two descriptors in that family satisfy |Spearman| >= 0.50, and
- the family median absolute Spearman exceeds the aggregate-baseline family median by >= 0.10.

## PASS -> FAIL attribution

For every PASS -> FAIL transition, report before/after values of:
- total objective return
- worst-16 objective advantage
- worst-16 physical advantage
- positive-advantage fraction
- longest disadvantage run
- late-minus-early advantage

No classifier is fit because the event count is small and fitting would encourage overinterpretation.

## Decision

Possible outcomes:

### TEMPORAL / TAIL / RELATIONAL FAMILY INFORMATIVE
Evidence supports that total return discards a specific class of trajectory information. Reward redesign remains a separate future branch and is not automatically authorized.

### NO SIMPLE TRAJECTORY FAMILY DOMINATES
Simple hand-designed trajectory statistics do not explain semantic transitions reliably. The next hypothesis should move toward richer sequence/state-regime representation rather than manually adding one scalar term.

### AGGREGATE BASELINE SUFFICIENT AFTER ALL
If richer families do not improve over total-return alignment and the baseline unexpectedly dominates, revisit the prior sufficiency diagnosis for implementation consistency.

No training method, reward term, threshold, or optimizer change is authorized inside this audit.
