# Semantic-Gate Factorization Audit Contract

Status: **PREDECLARED — READ-ONLY, NO TRAINING / NO REWARD REDESIGN**

Date: 2026-09-24

## Motivation

Current evidence establishes:
- absolute heavy-policy return is insufficient;
- heavy-versus-center relation is informative and held-out validated;
- direct scalar relational augmentation is still not semantically sufficient.

Counterexamples exist where:

    Delta J_H > 0
    Delta (J_H - J_C) > 0
    but semantic PASS -> FAIL

Therefore the next question is whether semantic competence is a distributional / conjunctive property across contexts rather than another scalar trajectory summary.

## Frozen data

Use only the strict held-out CONTROL paths from the three independent training seeds already evaluated in:

`runs/relational_persistence_heldout-2026-09-24/heldout_report.json`

The underlying checkpoint trajectories are re-evaluated with the same frozen endpoint protocol:
- 3 training seeds: 980001, 981001, 982001
- shared base + u1..u8 per seed
- 4 matched suites / reset seeds 840001..840004
- 64 steps
- T/A/O/S-heavy and center preferences
- unchanged V2-B objective normalization and physical proxies

No optimizer step or reward change is allowed.

## Semantic-gate factors

For every checkpoint, axis, and suite compute:

### Suite-level directional correctness
- objective sign correctness: heavy objective margin > 0
- physical sign correctness: heavy physical margin > 0 under higher-is-better sign convention
- joint correctness: both objective and physical correct in the same suite

### Magnitude
- objective heavy-center mean margin
- physical heavy-center mean margin
- absolute margin magnitude

### Cross-suite heterogeneity
Across the 4 suites:
- mean margin
- standard deviation
- minimum margin
- maximum margin
- fraction positive
- sign-disagreement count
- range

### Objective-physical agreement
Across suites:
- fraction where objective and physical signs agree
- fraction where objective says correct but physical says incorrect
- fraction where physical says correct but objective says incorrect

### Phase factorization
For each suite and axis, using frozen thirds:
- early (0..20)
- middle (21..42)
- late (43..63)

For objective and physical heavy-center advantage separately:
- phase sign correctness
- number of phases positive
- early/mid/late margins
- phase sign-flip count
- late failure indicator while total mean remains positive

### Worst-context factors
- worst suite objective margin
- worst suite physical margin
- worst phase-within-suite objective margin
- worst phase-within-suite physical margin
- count of negative suite-phase cells out of 12

## Transition decomposition

For every consecutive policy transition and axis classify PASS -> FAIL cause using frozen priority-independent flags:

1. **mean_relation_degradation**
   - mean objective or physical relation decreases.

2. **suite_sign_loss**
   - objective or physical correctness fraction decreases.

3. **worst_suite_collapse**
   - minimum suite margin crosses from >=0 to <0 or becomes more negative by >1e-10.

4. **phase_specific_collapse**
   - count of negative suite-phase cells increases.

5. **objective_physical_disagreement_growth**
   - objective-vs-physical sign disagreement fraction increases.

6. **heterogeneity_growth**
   - suite margin standard deviation increases while mean relation is non-decreasing.

Flags may co-occur; no single-label assignment is imposed.

## Primary questions

Across all PASS -> FAIL events report:
- fraction with mean relation deterioration;
- fraction with suite correctness loss;
- fraction with worst-suite collapse;
- fraction with increased negative suite-phase cells;
- fraction with objective-physical disagreement growth;
- fraction where mean relation is non-decreasing but competence still fails;
- fraction where both objective and physical mean relations are non-decreasing but some suite/context sign flips negative.

## Distributional hypothesis gate

Evidence supports **context-distributional competence** if all hold:

1. At least 75% of PASS -> FAIL events show either worst-suite collapse or increased negative suite-phase cells.
2. At least 25% of PASS -> FAIL events occur while at least one mean relation is non-decreasing, demonstrating that mean relation alone is insufficient.
3. At least 20% of PASS -> FAIL events occur while both objective and physical mean relations are non-decreasing OR while one is non-decreasing and the failure is explained by cross-suite/phase heterogeneity growth.
4. Worst-context or suite-correctness factors identify a larger fraction of PASS -> FAIL events than heavy-only J deterioration.
5. Pattern direction is observed in all 3 training seeds.

If these hold, the next hypothesis is that competence is distributional/conjunctive across matched contexts and phases.

## Interpretation

### DISTRIBUTIONAL HYPOTHESIS SUPPORTED
Semantic competence is better characterized as simultaneous correctness across contexts/phases than by scalar mean return or scalar relational margin.

This would justify a later diagnostic of chance-constrained / minimum-context / CVaR-like objectives, but does not authorize training in this audit.

### MOSTLY MEAN-DRIVEN
If PASS -> FAIL is explained mainly by mean relational deterioration and context factors add little, the relational mean remains the primary missing statistic.

### INCONCLUSIVE
If no factor class clearly dominates or event count is too small, reward redesign remains blocked.

No distributional objective, CVaR term, threshold, architecture, PPO rule, or optimizer change is authorized here.
