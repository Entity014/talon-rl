# Prospective Relational-Slope Validation Verdict

Status: **FROZEN — SLOPE DIAGNOSTIC ONLY; TEMPORAL TARGET NOT AUTHORIZED**

Date: 2026-09-24

## Question

Does within-rollout heavy-versus-center relational slope measured at a currently competent source checkpoint prospectively predict robust semantic collapse at the next checkpoint, beyond source mean relation?

## Independent validation corpus

Three new control-only training seeds were generated after the v17 hypothesis was frozen:

    983001
    984001
    985001

All use the frozen final-reference controller/training contract:
- V2-B single-site FiLM actor
- Foundation V2
- GAE lambda=.95
- unchanged weighted MORL PPO update
- 8 finite updates
- no retention, architecture, ranking, context, or temporal intervention

Measurement used:
- 25 unique checkpoints (shared base + 24 new checkpoints)
- 4 axes
- heavy and center preferences
- 4 matched reset suites
- 64 rollout steps

Raw-cache integrity:

    expected checkpoints = 25
    cache files          = 25
    missing              = 0
    extra                = 0
    invalid              = 0

## Prospective target

Only transitions whose source checkpoint is semantic PASS were eligible.

A positive target is a next-checkpoint **robust semantic collapse** using the frozen v16 operational rule:
- target FAIL;
- clearance >=.25;
- PASS->FAIL persists at normalized tau {-0.25,0,+0.25};
- exact paired-bootstrap flip probability >=.50.

Observed:

    eligible source-PASS transitions = 22
    robust-collapse positives         = 10
    retained/non-robust outcomes      = 12

Therefore the predeclared power gate passes.

## Primary prospective result

Frozen risk scores use no fitted coefficients.

| Source predictor | ROC AUC for next robust collapse |
|---|---:|
| mean heavy-center relation | 0.542 |
| full-rollout relational slope (64 steps) | 0.625 |
| **early-half relational slope (32 steps)** | **0.483** |
| source continuous gate margin | 0.367 |

Primary early-half slope is effectively non-discriminative and does not outperform source mean relation.

## Channel-specific secondary AUC

    objective early-half slope : 0.492
    physical early-half slope  : 0.467
    objective full slope       : 0.700
    physical full slope        : 0.625

The strongest secondary score is objective full-rollout slope at AUC 0.700, still below the frozen 0.75 validation threshold and not eligible to replace the predeclared primary metric post hoc.

## Seed consistency

### seed 983001

    eligible = 6
    robust collapse = 2
    median Slope32, collapse     = -0.0053
    median Slope32, non-collapse = +0.0844

Direction supports the hypothesis.

### seed 984001

    eligible = 9
    robust collapse = 4
    median Slope32, collapse     = +0.0603
    median Slope32, non-collapse = -0.0884

Direction is opposite to the hypothesis.

### seed 985001

    eligible = 7
    robust collapse = 4
    median Slope32, collapse     = +0.0581
    median Slope32, non-collapse = +0.1652

Direction supports the hypothesis only weakly.

Thus prospective slope separation is not seed-consistent.

## Frozen zero-slope risk split

At source checkpoint:

    Slope32 < 0
        n = 8
        robust-collapse rate = 37.5%

    Slope32 >= 0
        n = 14
        robust-collapse rate = 50.0%

Difference:

    -12.5 percentage points

This is opposite to the predeclared expectation that an already negative early slope should identify higher future-collapse risk.

Among retained/non-robust outcomes, 41.7% are nevertheless classified as high risk by `Slope32 < 0`.

## Primary gate

| Criterion | Result |
|---|---|
| >=8 eligible and >=3 robust collapses | **PASS: 22 / 10** |
| AUC(Slope32) >= .75 | **FAIL: .483** |
| Slope32 beats mean relation by >=.10 | **FAIL** |
| AUC(Slope64) >= .75 | **FAIL: .625** |
| collapse has more-negative median Slope32 in all comparable seeds | **FAIL** |
| negative-slope risk rate advantage >=.25 | **FAIL: -0.125** |
| mean relation not equivalent within .05 AUC | **FAIL** |

Formal verdict:

> **SLOPE DIAGNOSTIC ONLY**

## Does the v17 retrospective signature reproduce?

Partially, yes.

When comparing source to target after the update:

### full-rollout objective slope deterioration

    robust collapse      9 / 10
    non-collapse         8 / 12

    robust median Delta = -1.16e-5
    control median Delta = -3.15e-6

### full-rollout physical slope deterioration

    robust collapse      9 / 10
    non-collapse         9 / 12

    robust median Delta = -1.21e-3
    control median Delta = -1.69e-4

### early-half slope deterioration

    objective: robust 7/10 vs control 5/12
    physical:  robust 7/10 vs control 5/12

Thus robust collapse again tends to be accompanied by a stronger negative change in relational slope, especially in magnitude. However, slope deterioration also occurs frequently in non-collapse transitions.

This sharpens the interpretation:

> relational-slope deterioration is primarily a **concurrent / post-update signature of collapse**, not a reliable precursor already present at the source checkpoint.

## Axis distribution limitation

Eligible / positive counts:

    Tracking     1 / 0
    Angular      8 / 3
    Orientation  4 / 2
    Smoothness   9 / 5

The new corpus is therefore informative overall but sparse for Tracking. This does not rescue the failed pooled prospective gate; it is a limitation on axis-specific claims.

## Scientific conclusion

v17 identified temporal relational erosion after restricting analysis to robust collapse. The independent-seed prospective test now shows that this erosion should **not** be promoted to a causal training lever based on current evidence.

A source policy can have a positive or non-negative early relational slope and still collapse after the next update, while some retained policies already have negative source slopes.

Therefore:

    robust collapse is associated with slope deterioration

but

    source slope does not reliably predict robust collapse

and

    temporal-relational slope is not yet a validated causal target.

## Decision

No slope loss, persistence loss, temporal relation reward, or other slope-target training branch is authorized.

The temporal-target branch is closed under the present evidence.

The result should be retained as a mechanistic diagnostic:

> **closed-loop temporal erosion describes what robust collapse looks like after an update, but the source trajectory does not contain a sufficiently reliable slope precursor to justify optimizing that quantity directly.**

## Thesis wording

> Although robust semantic collapse was accompanied by pronounced within-rollout relational-slope deterioration, an independent three-seed prospective validation did not show that source-checkpoint slope predicted future collapse. Among 22 competent source transitions with 10 subsequent robust collapses, early-half slope achieved ROC AUC 0.483 and full-rollout slope 0.625, compared with 0.542 for source mean relation. The direction of early-slope separation also failed to reproduce across all seeds. Relational-slope erosion should therefore be interpreted as a collapse-associated trajectory signature rather than a validated causal precursor or training target.
