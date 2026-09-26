# Semantic-Gate Margin / Robustness Audit Contract

Status: **PREDECLARED — READ-ONLY EVALUATOR DIAGNOSTIC, NO TRAINING**

Date: 2026-09-24

## Question

Are held-out semantic PASS -> FAIL events mostly:

A. near-boundary decision flips caused by a thresholded evaluator, or
B. genuine competence collapses with substantial margin loss that remain failures under small threshold/reset perturbations?

No training method, reward, architecture, optimizer, semantic threshold, or PASS definition is changed by this audit.

## Frozen corpus

Use only the strict held-out CONTROL corpus already measured in:

`runs/semantic_gate_factorization_audit-2026-09-24/semantic_gate_factorization_report.json`

This contains:
- 3 independent training seeds
- 25 unique checkpoints
- 24 finite transitions
- 96 axis-transition samples
- 14 PASS -> FAIL events
- 4 matched reset suites per checkpoint

Scale calibration uses only the earlier discovery corpus:

`runs/trajectory_information_attribution_audit-2026-09-24/trajectory_information_attribution_report.json`

No held-out labels are used to fit any scale or threshold.

## Frozen normalization

For each axis i, using discovery checkpoints only:

    s_obj_i  = median(|mean objective heavy-center margin|) + 1e-8
    s_phys_i = median(|mean physical heavy-center margin|) + 1e-8

Held-out suite margins are normalized as:

    z_obj_{i,k}  = m_obj_{i,k}  / s_obj_i
    z_phys_{i,k} = m_phys_{i,k} / s_phys_i

Higher is better in both channels.

## Exact continuous distance to the frozen PASS boundary

The semantic gate passes a channel when at least 3 of 4 suites have positive margin.

For four suite margins sorted ascending:

    z_(1) <= z_(2) <= z_(3) <= z_(4)

at least 3/4 are positive iff:

    z_(2) > 0

Therefore define channel gate margins:

    G_obj  = second-smallest(z_obj over 4 suites)
    G_phys = second-smallest(z_phys over 4 suites)

and semantic gate margin:

    G_sem = min(G_obj, G_phys)

Under numerical tolerance 1e-10:

    G_sem > 0  <=> frozen semantic PASS

This identity must be verified for every checkpoint-axis state before further analysis.

## PASS -> FAIL event quantities

For every frozen PASS -> FAIL event compute:

    G_source > 0
    G_target <= 0

Boundary clearance:

    C = min(G_source, -G_target)

Crossing magnitude:

    Delta G = G_target - G_source

A large positive source margin and large negative target margin indicate a genuine crossing rather than a threshold graze.

## Small threshold sweep

Without changing the frozen evaluator, perform a diagnostic sensitivity sweep on normalized suite sign threshold:

    tau in {-0.25, 0.00, +0.25}

At each tau, a channel is correct in a suite iff:

    z > tau

and checkpoint PASS still requires >=3/4 objective suites and >=3/4 physical suites.

For each original PASS -> FAIL event report whether the flip persists at:
- lenient threshold tau=-0.25
- original threshold tau=0
- stricter threshold tau=+0.25

Define threshold-sweep robust flip:

    source PASS and target FAIL at all three tau values

No other tau values are inspected.

## Exact paired reset bootstrap

The four suites are treated as matched reset contexts.

Enumerate all ordered bootstrap resamples of four suite indices with replacement:

    4^4 = 256 paired resamples

For each resample, use the same suite-index multiset for source and target, then evaluate the original tau=0 semantic PASS rule.

Report:
- source PASS probability
- target PASS probability
- paired PASS->FAIL probability

Because all 256 resamples are enumerated, there is no Monte Carlo seed or sampling noise.

## Predeclared event classes

### B — ROBUST GENUINE COLLAPSE
An original PASS -> FAIL event is class B only if all hold:

1. boundary clearance C >= 0.25 normalized units;
2. threshold-sweep robust flip = true;
3. paired bootstrap PASS->FAIL probability >= 0.75.

### A — NEAR-BOUNDARY / EVALUATOR-SENSITIVE FLIP
An event is class A if any hold:

1. boundary clearance C < 0.10; or
2. paired bootstrap PASS->FAIL probability < 0.50; or
3. the original flip disappears under either tau=-0.25 or tau=+0.25.

### MIXED
All events not satisfying A or B are class MIXED.

The thresholds 0.10 / 0.25 / 0.50 / 0.75 are frozen before reading event results.

## Aggregate gate

### FORGETTING ROBUST
Conclude that semantic forgetting is robust to evaluator discretization if:

1. >=60% of PASS -> FAIL events are class B;
2. <=25% are class A;
3. median boundary clearance >=0.25;
4. median paired bootstrap PASS->FAIL probability >=0.75;
5. class-B events are present in all 3 training seeds.

### EVALUATOR SENSITIVITY MATERIAL
Conclude that evaluator discretization materially contributes if:

1. >=50% of PASS -> FAIL events are class A; or
2. median boundary clearance <0.10; or
3. median paired bootstrap PASS->FAIL probability <0.50.

### MIXED / INCONCLUSIVE
Otherwise.

## Secondary descriptive checks

Report separately:
- source G_sem distribution;
- target G_sem distribution;
- objective-limiting vs physical-limiting channel counts;
- semantic-score drop versus Delta G correlation;
- event class by axis and training seed;
- whether the two residual counterexamples from prior audits are class A, B, or MIXED.

These do not alter the primary verdict.

## Decision scope

This audit may justify revising how thesis results describe semantic forgetting robustness.

It does NOT authorize:
- changing the semantic threshold;
- changing PASS definition;
- reward redesign;
- ranking/context losses;
- architecture or optimizer changes;
- any new training run.
