# Relational-Objective Design Audit Verdict

Status: **FROZEN — RELATION VALID, DIRECT SCALAR CANDIDATE NOT AUTHORIZED**

Date: 2026-09-24

## Question

Can the held-out validated heavy-versus-center relation be injected into the objective as

    J_cand = J_H + eta (J_H - J_C)

without producing false improvement through center degradation or semantic deterioration?

This audit is offline only. No optimizer step or reward change occurs.

## Frozen data

The audit reuses only the strict held-out CONTROL set:
- 3 independent training seeds
- 24 finite policy transitions
- 96 axis-transition samples

No new rollout is collected.

## Baseline relation remains strongly validated

Heavy-only objective return:

    Spearman(Delta J_H, Delta objective correctness) = 0.409

Pure heavy-versus-center relation:

    Spearman(Delta R, Delta objective correctness) = 0.732

Difference:

    +0.323

Thus the relational signal remains robustly present in exactly the sample used for this objective-design audit.

## Frozen scalar candidates

The predeclared eta values were:

    0.25, 0.50, 1.00

### eta = 0.25

- objective-correctness Spearman = **0.500**
- semantic-score Spearman = **0.516**
- objective semantic false-gain fraction = **18.5%**
- center-degradation false-gain fraction = **0.0%**
- PASS->FAIL false approval = **28.6%**
- same after heavy-performance guardrail = **28.6%**

Fails:
- required >= +0.10 Spearman gain over J_H
- required <=20% guardrailed PASS->FAIL false approval

### eta = 0.50

- objective-correctness Spearman = **0.559**
- semantic-score Spearman = **0.578**
- objective semantic false-gain fraction = **18.2%**
- center-degradation false-gain fraction = **3.6%**
- candidate-positive transitions retained by heavy guardrail = **96.4%**
- PASS->FAIL false approval = **28.6%**
- same after heavy-performance guardrail = **28.6%**

This is the cleanest scalar candidate with respect to center degradation, but it still approves too many semantic forgetting events.

### eta = 1.00

- objective-correctness Spearman = **0.633**
- semantic-score Spearman = **0.658**
- objective semantic false-gain fraction = **9.4%**
- center-degradation false-gain fraction = **11.3%**
- candidate-positive transitions retained by heavy guardrail = **88.7%**
- PASS->FAIL false approval = **14.3%**
- same after heavy-performance guardrail = **14.3%**

This has the best semantic alignment of the three candidates, but violates the predeclared <=10% center-degradation false-gain criterion.

## Why the heavy-only guardrail is insufficient

The audit used a strict guardrail:

    Delta J_H >= 0

with no epsilon tuning.

However, several PASS->FAIL events still receive positive candidate score even though heavy performance is non-degrading.

Examples at eta=.5:

### seed 980001, u3 -> u4, Smoothness

    Delta J_H   = +1.185e-4
    Delta J_C   = +6.98e-5
    Delta R     = +4.87e-5
    Delta J_cand= +1.428e-4

Yet:

    objective correctness decreases by 0.25
    semantic score decreases by 0.25
    PASS -> FAIL

Both heavy return and relational margin improve, yet semantic competence is lost.

### seed 981001, u4 -> u5, Tracking

    Delta J_H   = +1.338e-4
    Delta J_C   = +8.40e-5
    Delta R     = +4.98e-5
    Delta J_cand= +1.587e-4

Yet:

    semantic score decreases by 0.25
    PASS -> FAIL

Again, neither center degradation nor heavy-return degradation explains the semantic failure.

Other PASS->FAIL false approvals do involve relational deterioration, confirming that both pathology types occur.

## Interpretation

The offline audit separates two issues.

### 1. Center-degradation pathology is real but not dominant at moderate eta

At eta=.5, only 3.6% of candidate-positive transitions are false gains caused by non-improving heavy performance.

Therefore the concern that a relational term may simply make the center policy worse is valid, but a strict heavy-performance guardrail controls most of that pathology.

### 2. A more important mismatch remains

Even when both:

    Delta J_H > 0
    Delta (J_H - J_C) > 0

semantic competence can still deteriorate.

Therefore the held-out validated relational signal is **descriptively informative**, but direct scalarization of absolute and relational returns is not yet a sufficient causal training objective.

This mirrors the earlier diagnosis:

    informative surrogate != sufficient semantic objective

## Formal decision

Predeclared result:

> **RELATION VALID BUT CANDIDATE PATHOLOGICAL**

No eta passes all authorization criteria.

Therefore:
- no relational reward training pilot is authorized;
- no eta selection is authorized;
- no temporal term is added;
- no PPO / architecture / lambda change is made.

## Updated scientific state

    Absolute heavy-only return
        -> insufficient

    Heavy-versus-center relation
        -> held-out validated as information

    J_H + eta (J_H - J_C)
        -> improves alignment
        -> but still admits semantic false approvals
        -> and high eta begins to admit center-degradation false gains

    Heavy-performance guardrail alone
        -> controls center pathology
        -> does not guarantee semantic preservation

Thus the next formulation, if research continues, should not be another unconstrained scalar weighted sum.

A narrower next diagnostic would test **conjunctive / constrained objective logic**, for example requiring separate non-degradation of both absolute heavy performance and relational competence rather than allowing one scalar term to compensate for another. Such a branch would require a new predeclared contract and is not authorized by this verdict itself.

## Thesis wording

Recommended wording:

> Offline objective-design analysis confirmed that adding the held-out validated heavy-versus-center relation to the absolute objective improved semantic alignment, but no tested scalar weighting satisfied all safety and semantic criteria. Moderate weighting largely controlled center-degradation pathology, yet some semantic PASS-to-FAIL transitions occurred even when both the heavy-policy return and the relational margin improved. Thus relational information is informative but is not, by itself, a sufficient scalar training objective.

Primary artifacts:
- `docs/contracts/diagnostics/relational-objective-design-audit-contract.md`
- `scripts/rl/relational_objective_design_audit.py`
- `runs/relational_objective_design_audit-2026-09-24/relational_objective_design_report.json`
