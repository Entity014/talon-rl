# Robust Semantic Collapse Mechanism Audit Verdict

Status: **FROZEN — TEMPORAL RELATIONAL DEGRADATION SIGNATURE IDENTIFIED; CONTROL-SPECIFIC EVIDENCE LIMITED**

Date: 2026-09-24

## Scope

Primary causal corpus is the v16 robust semantic collapse subset only:

    8 / 14 PASS -> FAIL events

Boundary-sensitive flips are excluded from the primary mechanism analysis.

No simulator rerun or training change was performed.

## 1. Robust-only descriptor re-ranking

Among the 8 robust collapses:

### Mean / scalar quantities

    heavy-only J deteriorates                 4 / 8
    objective mean relation deteriorates      7 / 8
    physical mean relation deteriorates       7 / 8

Thus absolute heavy return is again weak, while mean relation remains strongly associated with robust collapse.

### Temporal relational quantities

The strongest robust-only pattern is late-trajectory degradation:

    objective late relation deteriorates      8 / 8
    physical late relation deteriorates       8 / 8

    objective final16 deteriorates             8 / 8
    physical final16 deteriorates              8 / 8

    objective late-minus-early deteriorates    8 / 8
    physical late-minus-early deteriorates     7 / 8

    objective relational slope deteriorates    8 / 8
    physical relational slope deteriorates     7 / 8

This is stronger than heavy-only return, ordering deterioration, or context-collapse flags.

### Ordering / context

    joint interior ordering deteriorates       5 / 8
    joint reset-consistent ordering deteriorates 5 / 8
    worst-suite collapse                       6 / 8
    phase-specific collapse                    6 / 8

Therefore the robust-collapse subset does not revive the preference-ordering hypothesis.

## 2. Matched retained controls

Only 4 PASS->PASS retained transitions exist in the three-seed held-out corpus.

Matching used source continuous semantic margin with same-axis priority and reuse only after the unique control pool was exhausted.

Because this pool is small and some robust events require reused or cross-axis controls, the comparison is descriptive only.

### Mean relation is not collapse-specific

    objective mean relation deterioration
        robust collapse:   87.5%
        matched retained:  87.5%

    physical mean relation deterioration
        robust collapse:   87.5%
        matched retained:  87.5%

Thus deterioration of the mean heavy-versus-center relation alone does **not** discriminate robust collapse from retained competence.

### Late/final relation is universal but only partially specific

    objective late deterioration
        robust:   100%
        retained: 62.5%

    physical late deterioration
        robust:   100%
        retained: 87.5%

    objective final16 deterioration
        robust:   100%
        retained: 62.5%

    physical final16 deterioration
        robust:   100%
        retained: 62.5%

These are strong collapse signatures, but they also occur frequently in retained transitions.

### Relational slope is the cleanest descriptive discriminator

    objective slope deterioration
        robust:   100%
        retained: 37.5%

    physical slope deterioration
        robust:   87.5%
        retained: 37.5%

Median delta also changes sign between groups:

    objective slope
        robust median   = -1.26e-5
        retained median = +6.38e-6

    physical slope
        robust median   = -1.37e-3
        retained median = +2.25e-4

Under the frozen descriptive specificity rule, these are the only two descriptors that clearly separate robust-collapse from matched retained transitions.

This should be interpreted as a **temporal relational degradation signature**, not yet as a validated causal training target.

## 3. Earliest quarter divergence

Fixed quarters:

    Q1 = 0..15
    Q2 = 16..31
    Q3 = 32..47
    Q4 = 48..63

### Objective relation

Deterioration counts:

    Q1: 6 / 8
    Q2: 3 / 8
    Q3: 5 / 8
    Q4: 8 / 8

### Physical relation

    Q1: 6 / 8
    Q2: 5 / 8
    Q3: 4 / 8
    Q4: 8 / 8

Thus the common pattern is not simply "collapse begins late."

For 6/8 events, degradation is already visible in Q1 in both channels.

However, by Q4:

    objective deterioration = 8 / 8
    physical deterioration  = 8 / 8

The effect therefore **amplifies / persists toward the end of rollout**, rather than emerging only at the end.

### Temporal lead

    simultaneous objective + physical onset : 7 / 8
    physical-first                           : 1 / 8
    objective-first                          : 0 / 8

This argues against a simple one-channel causal sequence in which reward-side or physical-side failure consistently occurs first.

## 4. Objective-limited versus physical-limited robust collapse

The robust corpus splits evenly:

    target objective-limited : 4
    target physical-limited  : 4

### Objective-limited group

This group shows broad deterioration across both objective and physical relational descriptors.

Notably:
- mean objective and physical relation: 4/4 deteriorate
- late objective and physical relation: 4/4
- final16 objective and physical relation: 4/4
- objective and physical slope: 4/4

### Physical-limited group

This group remains more heterogeneous:
- mean objective relation: 3/4
- mean physical relation: 3/4
- late objective and physical relation: 4/4
- final16 objective and physical relation: 4/4
- objective slope: 4/4
- physical slope: 3/4
- worst-suite collapse: 4/4
- objective-physical disagreement growth: 4/4

Thus there is some evidence for two submodes:

1. **broad coupled relational decay**, prominent in objective-limited collapses;
2. **context/disagreement-heavy physical-limited collapse**, where worst-suite and objective-physical disagreement are especially common.

But n=4 per group is too small for a strong causal claim.

## Main conclusion

The robust-only analysis changes the diagnosis in an important way.

The most reliable pattern is not simply:

> mean heavy-versus-center relation decreases.

That decrease also occurs frequently in retained transitions.

Instead, robust collapse is characterized more specifically by:

> **progressive temporal degradation of the heavy-versus-center behavioral relation across the rollout, with relational slope deterioration providing the clearest descriptive separation from retained competence.**

This degradation usually begins early and becomes universal by the final quarter.

However, because the retained control pool contains only four transitions and required reuse for some matches, this remains a diagnostic signature rather than a validated causal mechanism.

## Decision

No new training method is authorized.

Specifically, this result does not yet authorize:
- slope/persistence reward terms;
- temporal ranking losses;
- context penalties;
- architecture or optimizer changes.

The appropriate next scientific statement is:

> Robust semantic collapse is associated with temporally worsening preference-conditioned relational behavior rather than with absolute return loss, mean relational loss alone, or preference-ordering deformation. The strongest current discriminator is the within-rollout relational slope, but independent retained-control evidence is still limited.

## Thesis wording

> Restricting analysis to robust semantic collapses revealed a stronger temporal pattern than was visible in pooled PASS-to-FAIL events. Objective and physical late-window relations deteriorated in all eight robust collapses, and the final 16-step relation deteriorated in all cases. Mean relational degradation alone was not specific, occurring equally often in matched retained transitions, whereas relational slope deterioration occurred in 100% of robust collapses for the objective channel and 87.5% for the physical channel, compared with 37.5% in matched retained controls. The degradation typically began early but became universal by the final quarter, suggesting progressive closed-loop erosion of preference-conditioned behavior rather than a purely endpoint or ordering failure.

Primary artifacts:
- `docs/contracts/diagnostics/robust-collapse-mechanism-audit-contract.md`
- `scripts/rl/robust_collapse_mechanism_audit.py`
- `runs/robust_collapse_mechanism_audit-2026-09-24/robust_collapse_mechanism_report.json`
