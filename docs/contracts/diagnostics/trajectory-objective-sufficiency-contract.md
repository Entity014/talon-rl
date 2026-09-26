# Trajectory-Level Objective Sufficiency Audit Contract

Status: **PREDECLARED — READ-ONLY, NO TRAINING**

Date: 2026-09-24

## Diagnosis being tested

Closed:
- local objective semantics
- per-objective PPO gradient availability
- first-order gradient conflict repair
- private capacity / specialization
- update magnitude / plasticity
- critic / PPO foundation

Open:
- trajectory-level semantic sufficiency of the existing T/A/O/S training objectives.

The question is not whether reward signs are locally correct. The question is:

> when the closed-loop return of training objective i improves after a finite policy change, does the semantic competence criterion for axis i reliably improve as well?

## Existing evidence reused

No new rollout or optimizer step is allowed in this gate.

Use matched-reset semantic reports from four existing policy paths evaluated with the same 64-step endpoint protocol:

1. V2-B, lambda=.95:
   - u10, u25, u50, u75

2. V2-B, lambda=1.00:
   - u10, u25, u50, u75

3. Paired restart weighted-PPO control from u50:
   - steps 0, 5, 10, 15, 20, 25

4. Paired restart max-min floor arm from u50:
   - steps 0, 5, 10, 15, 20, 25

This yields 16 consecutive policy transitions and 64 axis-transition samples.

## Quantities per policy checkpoint and axis i

Training-return quantity:

    J_i = mean normalized objective-i return
          under the i-heavy preference across the four matched suites.

Semantic objective margin:

    M_obj_i = mean [ objective_i(i-heavy) - objective_i(center) ]

Higher is better.

Semantic physical margin:

    M_phys_i = mean [ physical_i(center) - physical_i(i-heavy) ]

The sign is chosen so higher is better for all axes.

Binary semantic competence:

    PASS_i = frozen endpoint PASS criterion

with the unchanged 0.75 objective-correct, 0.75 physical-correct, and 0.95 survival thresholds.

For each consecutive pair theta_a -> theta_b compute:

    Delta J_i
    Delta M_obj_i
    Delta M_phys_i
    PASS transition

## Primary tests

Across all 64 axis-transition samples and separately per axis report:

- Pearson correlation: Delta J vs Delta M_obj
- Spearman rank correlation: Delta J vs Delta M_obj
- Pearson correlation: Delta J vs Delta M_phys
- Spearman rank correlation: Delta J vs Delta M_phys
- sign agreement fraction for Delta J and Delta M_obj
- sign agreement fraction for Delta J and Delta M_phys
- among Delta J > 0 cases, fraction with Delta M_obj < 0
- among Delta J > 0 cases, fraction with Delta M_phys < 0
- among PASS -> FAIL events, fraction for which Delta J >= 0
- among FAIL -> PASS events, fraction for which Delta J <= 0

Near-zero changes use tolerance 1e-10 and are excluded from sign-agreement denominators.

## Predeclared sufficiency criterion

Trajectory-level surrogate sufficiency is supported only if **all** hold:

1. overall Spearman(Delta J, Delta M_obj) >= 0.50;
2. overall Spearman(Delta J, Delta M_phys) >= 0.50;
3. objective-margin sign agreement >= 0.65;
4. physical-margin sign agreement >= 0.65;
5. among Delta J > 0 cases, semantic-objective deterioration fraction <= 0.20;
6. among Delta J > 0 cases, semantic-physical deterioration fraction <= 0.20;
7. among PASS -> FAIL events, no more than 20% occur with Delta J >= 0.

If any criterion fails, the existing training objective is classified as **trajectory-level semantically insufficient under the audited policy changes**.

This is a scope-bounded empirical diagnosis, not a claim that the reward/objective is globally invalid.

## Interpretation rule

SUFFICIENCY PASS:
- keep the existing trajectory objective semantics closed;
- remaining explanation shifts toward higher-order/path-distribution effects not captured by this finite-difference audit.

SUFFICIENCY FAIL:
- update thesis diagnosis to:
  - local objective semantics: CLOSED
  - trajectory-level semantic sufficiency: NOT VALIDATED / empirically mismatched
- do not propose a new training method inside this audit.
- next work, if any, must first identify which semantic trajectory statistic is missing from the current training surrogate.
