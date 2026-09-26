# Context-Incremental Retention Audit Verdict

Status: **FROZEN — UNDERPOWERED / INCONCLUSIVE; NO INCREMENTAL CONTEXT VALUE ESTABLISHED**

Date: 2026-09-24

## Question

When an axis is currently semantically competent, does source-checkpoint suite×phase robustness predict whether that competence will survive the next update beyond what is already explained by the mean heavy-versus-center relation?

## Causal-clean formulation

To avoid tautology, the audit does not use same-checkpoint suite correctness to predict same-checkpoint PASS.

Instead:
- sample only transitions whose source checkpoint is PASS;
- features are measured only at the source checkpoint;
- target is future PASS retention at the next checkpoint;
- context features are suite×phase quantities not directly used by the frozen PASS rule.

## Sample

Strict held-out CONTROL set across three independent training seeds:
- PASS-origin transitions: **18**
- retained PASS: **4**
- forgotten PASS: **14**

The predeclared minimum sample requirement was:
- at least 20 PASS-origin transitions;
- at least 8 forgetting events.

Forgetting count is sufficient, but PASS-origin count is not.

Therefore the formal result is automatically:

> **UNDERPOWERED / INCONCLUSIVE**

## Prospective model comparison

Baseline logistic model:

    retention ~ objective mean relation + physical mean relation

Context model:

    baseline
    + objective/physical suite×phase positive fraction
    + worst suite-phase margins
    + suite-margin standard deviations
    + objective-physical sign agreement

Leave-one-training-seed-out predictions were pooled.

| Metric | Mean-relation BASE | BASE + context | Change |
|---|---:|---:|---:|
| ROC AUC | **0.411** | 0.339 | -0.071 |
| Average precision | **0.229** | 0.208 | -0.021 |
| Brier score | **0.251** | 0.276 | worsened by 0.026 |
| Log loss | **0.930** | 1.022 | worsened by 0.092 |

Thus the frozen context feature set does not show a positive prospective generalization trend in this small held-out sample.

## Per-seed behavior

Seed 980001:
- 4 PASS-origin transitions
- all 4 forget
- AUC is undefined because there is no retained positive class
- context model Brier is worse than baseline

Seed 981001:
- 9 PASS-origin transitions
- 3 retain / 6 forget
- baseline AUC = 0.556
- context AUC = 0.500
- Brier improves slightly under context features

Seed 982001:
- 5 PASS-origin transitions
- 1 retain / 4 forget
- baseline AUC = 0.500
- context AUC = 0.250
- context Brier is worse

The improvement direction is therefore not consistent across held-out training seeds.

## Nonparametric companion check

The predeclared source-context robustness condition required both objective and physical suite×phase positive fractions >=0.75.

No PASS-origin source checkpoint satisfies this condition:
- lower mean-relation stratum: 0 robust / 8 total
- upper mean-relation stratum: 0 robust / 10 total

Therefore the stratified retention comparison is not estimable under the frozen threshold.

The threshold is not changed post hoc.

## Interpretation

This result does not invalidate the earlier semantic-gate factorization evidence.

That earlier audit established that:
- context sign loss accompanies all observed PASS->FAIL events;
- worst-suite or phase collapse accompanies 85.7%;
- two counterexamples fail despite improving objective and physical mean relations.

However, those are **concurrent failure descriptions**.

The current prospective audit asks a stronger question:

> can context structure measured before the next update predict future retention beyond mean relation?

Within the available held-out sample, the answer is not established.

In fact, the frozen context feature set generalizes worse than the mean-relation baseline in pooled leave-one-seed-out metrics.

Therefore the evidence supports:

    context structure is descriptively real
    but prospective incremental causal value is not demonstrated

## Decision

No suite-wise context auxiliary, sign penalty, chance constraint, CVaR objective, or minimum-context intervention is authorized.

The distributional/context branch remains closed for method design under the current evidence.

The strongest validated description remains:

> mean heavy-versus-center relation is the best scalar indicator currently established; context-level failures provide genuine counterexamples to its sufficiency, but the tested source-context features do not yet provide reliable prospective retention information beyond that mean relation.

## Thesis wording

Recommended wording:

> Although context-level sign loss accompanied semantic forgetting and exposed counterexamples to mean relational sufficiency, a prospective held-out retention audit did not establish incremental predictive value from source-checkpoint suite-by-phase robustness beyond the mean heavy-versus-center relation. The available PASS-origin sample was underpowered (18 transitions), and the frozen context-augmented model did not improve leave-one-seed-out discrimination or calibration. Context structure therefore remains a valid descriptive factor but not a validated causal intervention target.

Primary artifacts:
- `docs/contracts/diagnostics/context-incremental-retention-contract.md`
- `scripts/rl/context_incremental_retention_audit.py`
- `runs/context_incremental_retention_audit-2026-09-24/context_incremental_retention_report.json`
