# Preference-to-Behavior Ordering Audit Verdict

Status: **FROZEN — ORDERING NOT SUPPORTED; RANKING INTERVENTION NOT AUTHORIZED**

Date: 2026-09-24

## Question

Is semantic winner rotation primarily a deformation of the continuous preference-to-behavior map,

    w -> behavior,

such that interior preference ordering degrades beyond what is already explained by the heavy-versus-center mean relation?

## Exact measurement corpus

The predeclared full held-out corpus was completed exactly:
- V2-B + Foundation V2 + GAE lambda=.95
- 25 unique checkpoints
- 3 independent weighted-PPO control seeds
- 4 semantic axes
- 5 preference levels per axis
- 4 matched reset suites
- 64 rollout steps

Preference ladder per axis:

    q = w_i in {0.10, 0.25, 0.40, 0.55, 0.70}
    w_j = (1-q)/3, j != i

The direct heavy-versus-center pair (0.70, 0.25) was excluded from the primary ordering metric to prevent the audit from reducing to the already validated endpoint relation.

## Raw-cache integrity

Each checkpoint was measured in a fresh Isaac process to avoid long-lived simulator mutex instability. This changed execution only, not the scientific protocol.

Raw cache integrity after completion:

    expected checkpoint hashes = 25
    cache files                = 25
    missing                    = 0
    extra                      = 0
    invalid metadata/shape     = 0

Each cache contains objective and physical time series with shape:

    [4 axes, 4 reset suites, 5 preferences, 64 steps, 4 signals]

and frozen preference/reset metadata.

## Primary verdict

> **ORDERING NOT SUPPORTED**

The predeclared ordering hypothesis does not pass.

### PASS -> FAIL ordering deterioration

There are 14 held-out PASS -> FAIL events.

Primary joint interior ordering deterioration, defined as a decrease in either:
- joint objective+physical primary ordering accuracy; or
- joint reset-consistent pair fraction,

occurs in:

    10 / 14 = 71.4%

Required:

    >= 75%

**FAIL**

### Comparison with strongest mean-relation control

Mean objective heavy-versus-center relation deteriorates in:

    12 / 14 = 85.7%

Mean physical heavy-versus-center relation deteriorates in:

    12 / 14 = 85.7%

Thus the strongest scalar relation control has sensitivity:

    85.7%

Primary ordering sensitivity:

    71.4%

The contract required ordering sensitivity to exceed the strongest relation control by >=0.10.

Observed ordering is instead **14.3 percentage points lower**.

**FAIL**

### Ordering beyond non-degrading mean relation

The key incremental criterion asked whether ordering deteriorates in PASS -> FAIL events even when **both** mean objective and physical relations are non-decreasing.

Observed:

    0 / 14 = 0%

Required:

    >= 25%

**FAIL**

Therefore interior ordering does not explain an additional class of forgetting events beyond the mean relation in this corpus.

### Ordering-change correlation with semantic change

Primary joint interior ordering change versus semantic-score change:

    Spearman rho = 0.299

Required:

    |rho| >= 0.50

**FAIL**

Per-seed Spearman:

    seed 980001: 0.113
    seed 981001: 0.432
    seed 982001: 0.359

All signs are positive, so the predeclared direction-consistency criterion passes, but effect magnitude is too weak.

### FAIL -> PASS recovery

Ordering improves in:

    12 / 16 = 75.0%

Required:

    >= 60%

**PASS**

Thus ordering is behaviorally related to competence recovery, but not strongly enough to identify it as the missing mechanism.

## Frozen gate summary

| Criterion | Result |
|---|---|
| PASS->FAIL ordering deterioration >=75% | **FAIL: 71.4%** |
| Beats strongest mean-relation control by >=0.10 | **FAIL** |
| >=25% PASS->FAIL with both mean relations non-decreasing but ordering deteriorating | **FAIL: 0%** |
| `|rho(Delta ordering, Delta semantic)| >=0.50` | **FAIL: 0.299** |
| Positive direction across 3/3 seeds | **PASS** |
| FAIL->PASS ordering recovery >=60% | **PASS: 75.0%** |

Overall:

> **ORDERING HYPOTHESIS NOT SUPPORTED**

## Secondary ordering channels

Some post-measurement descriptive channels correlate more strongly with semantic change:

    objective rank-rho change: rho = 0.573
    physical rank-rho change:  rho = 0.583

However, these are not the predeclared primary joint ordering metric and may not be selected post hoc to rescue the hypothesis.

Primary joint pairwise ordering remains:

    rho = 0.299

Therefore no ranking intervention is authorized from these secondary correlations.

## Global shape observation

Across all checkpoint-axis states in the three held-out paths:

    joint primary ordering accuracy
    minimum = 0.139
    median  = 0.389
    maximum = 0.694

No checkpoint-axis state reaches the prospective robustness threshold:

    joint ordering >= 0.75
    0 / 108 states

Among the 18 PASS-origin transitions:

    high-order source checkpoints >=.75 = 0
    low-order source checkpoints       = 18
    retention rate in low-order group  = 22.2%

This means semantic endpoint competence can exist even when the full five-point objective+physical preference curve is not globally monotonic under the strict joint ordering definition.

Thus global monotonic ordering is not a necessary property of the current semantic PASS criterion.

## Decisive residual counterexamples

The two previously identified failures where both mean objective and physical relations improve are especially informative.

### Seed 980001, u3 -> u4, Smoothness

    Delta objective mean relation > 0
    Delta physical mean relation  > 0
    Delta joint primary ordering  = +0.194

Yet:

    semantic score decreases by 0.25
    PASS -> FAIL

The preference-map ordering improves rather than degrades.

### Seed 981001, u4 -> u5, Tracking

    Delta objective mean relation > 0
    Delta physical mean relation  > 0
    Delta joint primary ordering  = +0.083

Yet:

    semantic score decreases by 0.25
    PASS -> FAIL

Again, ordering improves while semantic competence is lost.

These directly reject the stronger explanation that the residual semantic failures are caused by preference-map ordering collapse.

## Interpretation

The evidence hierarchy is now:

    absolute heavy return
        -> insufficient

    heavy-versus-center mean relation
        -> strongest validated scalar signal
        -> explains most forgetting events

    full preference-to-behavior interior ordering
        -> related to competence changes
        -> recovers during many FAIL->PASS events
        -> but does not outperform mean relation
        -> does not explain the residual mean-relation counterexamples

Therefore winner rotation should **not** be reinterpreted as primarily a monotonic preference-map deformation under the tested continuum definition.

The learned map can be non-monotonic or folded at intermediate preferences while still satisfying the endpoint semantic gate, and the specific residual semantic failures can occur while ordering improves.

## Decision

No pairwise ranking loss, monotonicity regularizer, ranking margin, or ordering intervention is authorized.

The ranking branch is **CLOSED** under the frozen contract.

No method branch should be opened from secondary ordering metrics without a new independent hypothesis and evidence base.

## Thesis wording

Recommended wording:

> A full five-point preference-to-behavior ordering audit did not support the hypothesis that semantic winner rotation was primarily caused by deformation of the continuous preference map. Interior joint ordering deteriorated in 71.4% of held-out PASS-to-FAIL events, less often than the mean heavy-versus-center relation (85.7%), and ordering changes correlated only moderately with semantic-score changes (Spearman 0.299). Moreover, the two residual failures that occurred despite improving objective and physical mean relations also showed improving joint ordering. Thus preference ordering is behaviorally associated with competence but does not provide incremental explanatory power sufficient to justify a ranking-based intervention.

Primary artifacts:
- `docs/contracts/preference_control/preference-behavior-ordering-audit-contract.md`
- `scripts/rl/preference_behavior_ordering_audit.py`
- `scripts/rl/preference_behavior_ordering_offline_verdict.py`
- `runs/preference_behavior_ordering_audit-2026-09-24/preference_behavior_ordering_report.json`
- `runs/preference_behavior_ordering_audit-2026-09-24/raw_cache/`
