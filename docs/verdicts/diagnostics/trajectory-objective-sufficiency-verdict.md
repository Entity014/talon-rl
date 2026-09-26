# Trajectory-Level Objective Sufficiency Audit Verdict

Status: **FROZEN — TRAJECTORY-LEVEL SEMANTIC SUFFICIENCY FAILS IN THE AUDITED POLICY CHANGES**

Date: 2026-09-24

## Diagnosis update

The following remain closed / validated:
- local objective semantics
- per-objective PPO gradient availability
- first-order conflict repairability
- private policy capacity / specialization
- update magnitude / plasticity
- critic / PPO foundation

The following is **not** validated:

> trajectory-level semantic sufficiency of the existing T/A/O/S training objectives.

The audit asks whether finite improvements in the actual normalized training-objective return reliably correspond to improvements in the matched closed-loop semantic endpoint criteria.

## Evidence base

No new rollout and no optimizer step were used.

The audit reuses matched-reset 64-step endpoint reports from four existing policy paths:

1. V2-B, lambda=.95: u10 -> u25 -> u50 -> u75
2. V2-B, lambda=1.00: u10 -> u25 -> u50 -> u75
3. paired weighted-PPO restart: step 0 -> 5 -> 10 -> 15 -> 20 -> 25
4. paired max-min floor restart: step 0 -> 5 -> 10 -> 15 -> 20 -> 25

Total:
- 16 consecutive finite policy transitions
- 4 semantic axes per transition
- **64 axis-transition samples**

For each axis i:

    J_i = normalized objective-i return under i-heavy preference
    M_obj_i = objective_i(i-heavy) - objective_i(center)
    M_phys_i = physical_i(center) - physical_i(i-heavy)

All quantities use higher-is-better sign convention.

## Overall alignment

Across all 64 samples:

| Metric | Result | Predeclared sufficiency requirement |
|---|---:|---:|
| Pearson(Delta J, Delta M_obj) | **0.573** | diagnostic only |
| Spearman(Delta J, Delta M_obj) | **0.251** | >= 0.50 |
| Pearson(Delta J, Delta M_phys) | **0.499** | diagnostic only |
| Spearman(Delta J, Delta M_phys) | **0.230** | >= 0.50 |
| objective-margin sign agreement | **0.578** | >= 0.65 |
| physical-margin sign agreement | **0.609** | >= 0.65 |
| Delta J > 0 but semantic objective worsens | **42.3%** | <= 20% |
| Delta J > 0 but physical semantic worsens | **38.5%** | <= 20% |
| PASS -> FAIL with nonnegative Delta J | **33.3%** | <= 20% |

Every predeclared sufficiency criterion fails.

The moderate Pearson correlations do not rescue the result: rank ordering is weak, sign agreement is below threshold, and positive-return changes frequently coincide with semantic deterioration.

## Per-axis result

### Tracking
- Spearman DeltaJ / objective-margin = **0.044**
- Spearman DeltaJ / physical-margin = **0.159**
- objective sign agreement = **0.563**
- physical sign agreement = **0.750**

### Angular
- Spearman DeltaJ / objective-margin = **0.035**
- Spearman DeltaJ / physical-margin = **-0.068**
- objective sign agreement = **0.500**
- physical sign agreement = **0.500**
- among DeltaJ > 0 cases, semantic objective worsens **57.1%**
- among DeltaJ > 0 cases, physical semantic worsens **57.1%**

### Orientation
Orientation shows the strongest correlation of the four axes, but still fails the frozen sufficiency gate:
- Spearman DeltaJ / objective-margin = **0.447**
- Spearman DeltaJ / physical-margin = **0.415**
- objective sign agreement = **0.625**
- physical sign agreement = **0.625**
- among DeltaJ > 0 cases, both semantic objective and physical semantic worsen **40%**
- 5 PASS -> FAIL events total; 40% occur with nonnegative DeltaJ

### Smoothness
- Spearman DeltaJ / objective-margin = **0.418**
- Spearman DeltaJ / physical-margin = **0.156**
- objective sign agreement = **0.625**
- physical sign agreement = **0.563**

No axis independently provides strong evidence that its training return is a sufficient finite-policy-change surrogate for its semantic endpoint behavior.

## Decisive mismatch examples

### Orientation, lambda=.95, u10 -> u25

Training objective improves:

    Delta J_O = +0.003153

but semantic behavior deteriorates:

    Delta M_obj_O  = -0.000647
    Delta M_phys_O = -0.09621
    O: PASS -> FAIL

This is direct evidence that increasing the closed-loop normalized Orientation training return is not sufficient to preserve the frozen Orientation semantic criterion.

### Orientation, lambda=1.00, u10 -> u25

Again:

    Delta J_O = +0.003146
    Delta M_obj_O  = -0.000277
    Delta M_phys_O = -0.02751
    O: PASS -> FAIL

The same mismatch occurs under a different GAE-lambda training path.

### Angular, paired weighted-PPO restart, step 5 -> 10

    Delta J_A = +0.000270
    Delta M_obj_A = -1.04e-5
    A: PASS -> FAIL

The physical margin improves slightly in this transition, showing that even the two semantic views can disagree near the threshold; the binary semantic competence nevertheless disappears while the training return increases.

## Interpretation

The audit changes the thesis diagnosis materially.

It is no longer accurate to state that reward/objective semantics are completely closed.

The supported statement is:

> **Local objective semantics are validated, but trajectory-level semantic sufficiency is not.**

The existing objectives provide useful local learning signals, and their gradients can be made jointly first-order compatible. However, finite policy changes that improve an objective's own normalized rollout return do not reliably improve — or even preserve — the matched semantic behavior used to define competence.

The causal chain is therefore:

    local objective semantics                     VALIDATED
             ↓
    per-objective PPO gradient                    VALIDATED
             ↓
    joint first-order compatibility               ACHIEVABLE
             ↓
    finite policy update                          VALIDATED
             ↓
    new closed-loop trajectory distribution
             ↓
    semantic endpoint                             NOT GUARANTEED

This explains why max-min gradient balancing can succeed at every update while semantic competence still disappears.

## Decision

**TRAJECTORY-LEVEL SEMANTIC SUFFICIENCY FAILS IN THE AUDITED POLICY CHANGES.**

No new training method is authorized by this diagnostic alone.

The next scientific question, if pursued, should identify which trajectory-level statistic or temporal structure is missing from the current objective surrogate before changing PPO, architecture, replay, or gradient-combination rules again.

## Thesis wording

Recommended scope-bounded statement:

> The normalized objective returns were locally meaningful and yielded valid per-objective policy gradients, but finite improvements in those returns were not reliably aligned with improvements in the matched closed-loop semantic competence criteria. Across 64 axis-transition samples, rank correlations between objective-return changes and semantic-margin changes were weak, and positive objective-return changes frequently coincided with semantic deterioration. Therefore local objective validity did not imply trajectory-level semantic sufficiency.

Primary artifacts:
- `docs/contracts/diagnostics/trajectory-objective-sufficiency-contract.md`
- `scripts/rl/trajectory_objective_sufficiency_audit.py`
- `runs/trajectory_objective_sufficiency_audit-2026-09-24/trajectory_objective_sufficiency_report.json`
