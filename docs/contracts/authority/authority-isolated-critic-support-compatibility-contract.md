# Authority-Isolated Critic/Support Compatibility Audit Contract

Status: **PREDECLARED — READ-ONLY / FROZEN ACTOR**
Date: 2026-09-25

## Question
Why does the validated Foundation V2 critic remain valid through AI-H1 u50 but fail at u75?

Distinguish:
A. support/refresh incompatibility with the stronger authority-isolated actor distribution;
B. critic-body representation insufficiency;
C. localized phase/preference coverage failure.

No actor parameter may change.

## Frozen checkpoints
Compare exactly:
- AI-H1 u50: critic-valid reference
- AI-H1 u75: critic-failed target

Actor and critic body are frozen for all analyses.

## Evaluation distribution
Fresh deterministic current-policy rollouts:
- T/A/O/S/C
- reset suites 840001..840004
- 64 steps
- early H32 and late H32
- same normalized four-objective return targets and gamma=.99
## Support regimes
For each checkpoint collect fresh current-policy support independently from evaluation seeds.

Freeze exactly four head-fit regimes, all ridge lambda=1:

1. CURRENT_HEAD
   Saved critic head; no refit.

2. SELECTED12
   3 diverse early + 3 diverse late units from anchor pool
   plus 3 diverse early + 3 diverse late units from adaptive/current-policy pool.

3. EXPANDED
   All collected anchor/current-policy support units used by the selected pool.

4. DENSE_CURRENT
   Fresh current-policy support over all five anchor preferences,
   8 independent reset seeds per preference,
   64 steps each; use all early+late units.

No support-size or ridge-lambda sweep beyond these four regimes.

## Body-sufficiency probe
Using DENSE_CURRENT features from the frozen critic body:
- fit ridge head on 75% of dense support reset seeds;
- evaluate on held-out 25% dense-support reset seeds;
- also evaluate on the independent semantic reset suites.

This is a frozen-body linear-probe test only.

No critic-body optimization is allowed.
## Metrics
Report for u50 and u75, per regime:
- early/late aggregate EV;
- early/late negative-EV fraction;
- mean absolute bias;
- EV by objective head;
- EV by preference;
- EV by reset suite.

Also report support-to-evaluation feature geometry:
- standardized nearest-support distance;
- Mahalanobis-style PCA distance in critic feature space;
- support/evaluation feature mean-shift norm;
- early vs late separately.

## Frozen decision rules

### SUPPORT COMPATIBILITY PROBLEM
At u75, same frozen critic body with EXPANDED or DENSE_CURRENT head refit restores:
- early EV > 0;
- late EV > 0;
- early negative fraction <= .25;
- late negative fraction <= .25;
- combined negative fraction <= .25;

and the dense held-out-body probe also satisfies these bounds.

### CRITIC BODY INSUFFICIENT
DENSE_CURRENT head refit and held-out frozen-body probe both fail late EV >0 or late negative fraction <=.25.

### SUPPORT IMBALANCE / LOCALIZED COVERAGE
Aggregate u75 can be restored but one preference/head/suite remains strongly negative, or feature-distance diagnostics show a specific phase/preference coverage gap.

### INCONCLUSIVE
No regime gives a consistent distinction.

No actor treatment, critic architecture, ridge threshold, or semantic gate may be changed after observing results.
## Decision scope
- SUPPORT COMPATIBILITY PROBLEM -> authorize a separate architecture-neutral support-rule amendment and AI-H1 revalidation.
- CRITIC BODY INSUFFICIENT -> do not patch support cadence; any critic-capacity change is a new method branch.
- LOCALIZED COVERAGE -> only a separately predeclared coverage repair may proceed.
- No AI-H2 semantic evaluation is authorized by this audit alone.
