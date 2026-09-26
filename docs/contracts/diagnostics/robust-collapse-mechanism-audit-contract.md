# Robust Semantic Collapse Mechanism Audit Contract

Status: **PREDECLARED — READ-ONLY, NO METHOD DESIGN / NO TRAINING**

Date: 2026-09-24

## Primary corpus

Use only the v16 operational **robust semantic collapse** corpus:

    clearance >= 0.25
    threshold-sweep robust flip = true
    paired-bootstrap PASS->FAIL probability >= 0.50

Expected n = 8 events.

Boundary-sensitive flips and the single mixed event are excluded from primary causal inference and reported only as secondary context.

## Sources

Read-only reuse of frozen artifacts:
- semantic-collapse corpus v16
- relational-persistence held-out report
- semantic-gate factorization report
- preference-to-behavior ordering report
- exact five-point raw trajectory cache (25/25 checkpoints)
- semantic-gate robustness report

No simulator rerun is authorized.

## Analysis 1 — Robust-only descriptor comparison

For each robust collapse, evaluate change in higher-is-better descriptors:

### Scalar / relational
- J_heavy
- objective mean heavy-center relation
- physical mean heavy-center relation

### Temporal / persistence
- objective late relation
- physical late relation
- objective late-minus-early
- physical late-minus-early
- objective slope
- physical slope
- objective final16 relation
- physical final16 relation
- objective worst16 relation
- physical worst16 relation
- objective positive-fraction
- physical positive-fraction

### Ordering
- joint primary interior ordering accuracy
- joint reset-consistent pair fraction

### Context
- worst-suite collapse flag
- phase-specific collapse flag
- objective-physical disagreement-growth flag
- heterogeneity-growth flag

A continuous descriptor is called "deteriorated" iff its transition delta < -1e-10. Context flags use their frozen boolean definitions.

Report deterioration count / 8 and median delta. Do not choose a new descriptor after seeing results; all frozen descriptors above are reported.

## Analysis 2 — Retained-transition controls

Control pool:

    source PASS AND target PASS

from the same three held-out control paths.

Matching priority for each robust event:
1. same semantic axis, nearest absolute source G_sem;
2. if no same-axis retained control exists, any retained control with nearest source G_sem.

Controls are used without replacement when possible. If the pool is too small, reuse is allowed only after all unique candidates are exhausted and must be reported.

Because the retained pool is expected to be small, this analysis is descriptive only.

For each descriptor report:
- robust-collapse deterioration fraction;
- matched-retained deterioration fraction;
- difference in fractions;
- robust median delta versus retained median delta.

A descriptor is considered a **specific collapse candidate** only descriptively if:

    robust deterioration >= 75%
    AND matched-retained deterioration <= 50%
    AND difference >= 25 percentage points.

This does not authorize a training target.

## Analysis 3 — Earliest quarter divergence

Using cached time series, reconstruct matched heavy-versus-center relation for each robust event at source and target over fixed quarters:

    Q1 = steps  0..15
    Q2 = steps 16..31
    Q3 = steps 32..47
    Q4 = steps 48..63

For each axis:
- objective relation = heavy objective_i - center objective_i
- physical relation = center physical_i - heavy physical_i

where heavy is q=.70 and center is q=.25 in the frozen five-point ladder.

For each event/channel compute target-source delta in Q1..Q4.

Earliest divergence quarter = first quarter whose delta is negative by >1e-10.

Report:
- earliest-divergence quarter distribution;
- fraction with Q1 deterioration;
- fraction with deterioration appearing only after Q1;
- quarter-wise deterioration counts and median deltas.

No quarter is selected post hoc.

## Analysis 4 — Objective-side versus physical-side collapse

Use frozen continuous gate margins from v15/v16.

Classify each robust event by target limiting channel:
- objective-limited
- physical-limited

For each group report descriptor deterioration patterns separately.

Also classify temporal lead:
- objective-first: earliest objective quarter < earliest physical quarter
- physical-first: earliest physical quarter < earliest objective quarter
- simultaneous: equal earliest quarter
- one-channel-only: only one channel deteriorates

## Interpretation

### COMMON ROBUST-COLLAPSE PRECURSOR
A descriptor/pattern may be called a common precursor only if:
- it deteriorates in >=7/8 robust events; and
- it is not equally common in matched retained controls; and
- its deterioration is not solely a consequence of the target binary gate definition.

### MULTIPLE FAILURE MODES
If objective-limited and physical-limited robust collapses show materially different descriptor/lead patterns, report multiple robust-collapse modes rather than forcing one mechanism.

### NO COMMON PRECURSOR
If no descriptor reaches the common-precursor criterion, preserve the negative result. Do not open a method branch.

## Decision scope

This audit is diagnostic only.
It does not authorize reward redesign, ranking/context loss, architecture changes, retention mechanisms, or new training.
