# Context-Incremental Retention Audit Contract

Status: **PREDECLARED — OFFLINE, READ-ONLY, NO TRAINING**

Date: 2026-09-24

## Question

When semantic competence is currently PASS and the mean heavy-versus-center relation is known, does additional context/phase sign robustness predict whether competence will survive the next policy update?

This is deliberately a **prospective retention** question.

## Why this formulation

The frozen semantic PASS gate is itself defined from suite-level objective and physical correctness fractions. Using those same fractions to predict PASS at the same checkpoint would be tautological.

Therefore this audit uses only:
- features measured at the **source checkpoint**;
- target = whether the **next checkpoint** retains PASS;
- context features that are not directly used by the frozen gate: suite×phase sign structure and worst suite-phase margins.

No new rollout is required. Reuse:

`runs/semantic_gate_factorization_audit-2026-09-24/semantic_gate_factorization_report.json`

## Sample

Take every axis-transition in the strict held-out 3-seed CONTROL set for which:

    PASS(source) = True

Binary target:

    y = 1  if PASS(target) = True   (retained)
        0  if PASS(target) = False  (forgotten)

No FAIL-origin transition is used in the primary retention analysis.

## Frozen baseline features

Mean relational state at the source checkpoint:

1. objective mean heavy-minus-center margin
2. physical mean center-minus-heavy margin

These are the strongest validated scalar indicators from prior audits.

## Frozen context features

Source-checkpoint context structure, none of which is part of the PASS rule directly:

1. objective positive suite×phase fraction = 1 - negative objective cells / 12
2. physical positive suite×phase fraction = 1 - negative physical cells / 12
3. worst objective suite-phase margin
4. worst physical suite-phase margin
5. objective suite-margin standard deviation
6. physical suite-margin standard deviation
7. objective-physical sign-agreement fraction across suites

No feature selection or threshold tuning is allowed after seeing results.

## Models

Use two fixed logistic models with L2 regularization:

BASE:

    retention ~ objective_mean_relation + physical_mean_relation

BASE+CTX:

    retention ~ objective_mean_relation + physical_mean_relation
              + all 7 frozen context features

All features are standardized using training-fold mean/std only.

Regularization strength is fixed at C=1.0.

## Evaluation

Use leave-one-training-seed-out cross-validation:
- train on 2 seeds
- test on the third
- repeat for all 3 seeds

Primary metrics pooled over held-out predictions:
- ROC AUC
- average precision
- Brier score
- log loss

Also report per-seed metrics when defined.

## Nonparametric companion check

To avoid relying only on logistic functional form:

1. compute a scalar baseline robustness score:

       B = z(obj_mean_relation) + z(phys_mean_relation)

2. divide PASS-origin transitions into lower vs upper half of B within each training seed;
3. within each half compare retention rate between source checkpoints with:

       both objective and physical positive suite×phase fraction >= 0.75

   versus those below that criterion.

This 0.75 threshold is fixed a priori to mirror the semantic gate's 3-of-4 suite correctness level, but applied to 12 suite×phase cells rather than the gate itself.

## Incremental-value gate

Context-wise structure is considered to have **incremental prospective retention value** only if all hold:

1. BASE+CTX pooled ROC AUC exceeds BASE by >= 0.10.
2. BASE+CTX pooled average precision exceeds BASE by >= 0.10.
3. BASE+CTX Brier score is lower than BASE by >= 0.02.
4. BASE+CTX log loss is lower than BASE.
5. Improvement direction (AUC or, when undefined, Brier) is non-worse on all 3 held-out seeds.
6. In the nonparametric companion check, context-robust checkpoints have >=0.15 higher retention rate in at least one baseline-relation stratum and are not lower in the other stratum.

If the PASS-origin sample has fewer than 20 transitions or fewer than 8 forgetting events, the audit is automatically marked **UNDERPOWERED / INCONCLUSIVE** regardless of model metrics.

## Decision

### INCREMENTAL CONTEXT VALUE SUPPORTED
Phase/context robustness predicts future retention beyond mean relation. This authorizes a separate minimal intervention design audit, not training.

### NO INCREMENTAL VALUE
Mean relation explains prospective retention as well as the frozen context features. Distributional branch closes.

### UNDERPOWERED / INCONCLUSIVE
Insufficient PASS-origin/forgetting events or mixed metrics. No training authorized.

No reward term, suite penalty, CVaR/min-context objective, optimizer, architecture, lambda, or semantic threshold changes are authorized in this audit.
