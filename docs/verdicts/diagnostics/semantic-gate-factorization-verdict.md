# Semantic-Gate Factorization Audit Verdict

Status: **FROZEN — INCONCLUSIVE FOR DOMINANT DISTRIBUTIONAL MECHANISM; CONTEXT-LEVEL COUNTEREXAMPLES VALIDATED**

Date: 2026-09-24

## Question

Is semantic competence primarily a distributional / conjunctive property across resets and trajectory phases, rather than a scalar relational property?

The audit factorizes the frozen endpoint gate itself over the strict held-out control paths from three independent training seeds.

## Evidence base

Exact held-out protocol:
- training seeds 980001, 981001, 982001
- 25 unique checkpoints
- 24 finite policy transitions
- 96 axis-transition samples
- 4 matched reset suites per checkpoint
- 64-step trajectories
- early / middle / late phase decomposition
- unchanged V2-B objective and physical semantic definitions

No optimizer step, reward change, or threshold change was made.

## PASS -> FAIL event set

There are **14 held-out PASS -> FAIL events**:
- seed 980001: 4
- seed 981001: 6
- seed 982001: 4

The failure pattern is present in all three independent training seeds.

## Factor frequencies

Among the 14 PASS -> FAIL events:

| Factor | Fraction |
|---|---:|
| suite correctness loss | **100%** |
| context sign flip to negative | **100%** |
| mean relational deterioration | **85.7%** |
| worst-suite collapse | **78.6%** |
| increased negative suite-phase cells | **78.6%** |
| worst-suite OR phase collapse | **85.7%** |
| objective-physical disagreement growth | 42.9% |
| heterogeneity growth with non-decreasing mean | 7.1% |
| at least one mean relation non-decreasing | **14.3%** |
| both means non-decreasing, or one non-decreasing plus heterogeneity growth | **14.3%** |

For comparison, heavy-only objective return deteriorates in only **64.3%** of PASS -> FAIL events.

Thus suite/phase context factors are more sensitive to forgetting than heavy-only return, but mean relational deterioration is also present in most events.

## Predeclared distributional gate

Passed:
- worst-suite or phase collapse in >=75% of PASS -> FAIL events: **PASS (85.7%)**
- context factors identify more forgetting events than heavy-only J deterioration: **PASS**
- pattern direction present in all three seeds: **PASS**

Failed:
- >=25% of PASS -> FAIL events with at least one non-decreasing mean relation: **FAIL (14.3%)**
- >=20% of events with both means non-decreasing, or one non-decreasing plus heterogeneity growth: **FAIL (14.3%)**

Formal result:

> **INCONCLUSIVE**

The audit does not support the stronger claim that context-distributional collapse is the dominant mechanism beyond mean relational deterioration.

## What is nevertheless established

### 1. Semantic forgetting is context-visible

Every PASS -> FAIL transition contains at least one matched context whose directional correctness flips negative.

In 12/14 events, either the worst suite deteriorates or the number of negative suite-phase cells increases.

Therefore the semantic gate is genuinely sensitive to cross-context structure, not merely the heavy-only return.

### 2. Mean relation explains most, but not all, forgetting

Mean objective and/or physical relational margin deteriorates in 12/14 PASS -> FAIL events.

This is consistent with the preceding held-out result that mean heavy-versus-center relation is the strongest currently validated scalar indicator.

### 3. Two decisive counterexamples show that mean relation is not sufficient

#### Seed 980001, u3 -> u4, Smoothness

Both mean relations improve:

    Delta objective mean relation = +4.87e-5
    Delta physical mean relation  = +2.30e-4

Yet semantic competence changes:

    PASS -> FAIL

The primary factorization signal is increased objective-physical sign disagreement across suites.

Thus a better average relation does not guarantee that the objective and physical semantic criteria remain jointly correct across contexts.

#### Seed 981001, u4 -> u5, Tracking

Again both mean relations improve:

    Delta objective mean relation = +4.98e-5
    Delta physical mean relation  = +1.07e-3

Yet:

    PASS -> FAIL

Here the worst-suite margin deteriorates and cross-suite heterogeneity grows despite the improving means.

This is direct evidence that a small subset of semantic failures is distributional in a way the mean relation cannot represent.

## Interpretation

The strongest supported hierarchy is now:

    absolute heavy return
        -> insufficient

    mean heavy-versus-center relation
        -> strongly informative / held-out validated
        -> explains most observed forgetting events

    context / suite / phase structure
        -> detects nearly all forgetting events
        -> provides real counterexamples beyond the mean
        -> but is not yet established as the dominant mechanism

Therefore semantic competence should not yet be redefined wholesale as a CVaR/min-context objective.

The data support a more cautious statement:

> semantic competence is primarily associated with matched relational behavior, while a minority of failures additionally require cross-context conjunctive information that scalar mean relations discard.

## Decision

**Distributional-objective training is NOT authorized.**

No chance constraint, CVaR term, minimum-context objective, phase penalty, or reset-wise minimum is introduced.

The factorization branch is closed as a diagnostic result.

If further research is pursued, the next useful step is not another broad architecture or reward sweep. It would be a narrowly designed test of whether preserving **suite-wise directional correctness** adds causal benefit beyond preserving the mean relation, with explicit protection against overfitting to the four evaluation suites.

## Thesis wording

Recommended wording:

> Semantic-gate factorization showed that most competence losses were accompanied by deterioration of the mean heavy-versus-center relation, but context-level structure remained relevant. Worst-suite or phase-specific collapse occurred in 85.7% of held-out PASS-to-FAIL events, and every forgetting event included at least one context-level sign loss. Importantly, two failures occurred even while both objective and physical mean relational margins improved, demonstrating that mean relational competence is not sufficient. However, such counterexamples represented only 14.3% of forgetting events, so the evidence did not establish distributional context failure as the dominant mechanism.

Primary artifacts:
- `docs/contracts/diagnostics/semantic-gate-factorization-contract.md`
- `scripts/rl/semantic_gate_factorization_audit.py`
- `runs/semantic_gate_factorization_audit-2026-09-24/semantic_gate_factorization_report.json`
