# B1-R0.2 — History / Privileged-Information Value Audit

Status: **DIAGNOSTIC DRAFT — READ-ONLY / NO RL UPDATE**

R0/R0.1 pair matching was inconclusive because near-observation pairs were too sparse. R0.2 replaces pair matching with an information-value test over event-labeled trajectories from every P2 and S1 seed.

## Fixed event and horizon

Use the unchanged event definition: `tilt > 15° OR height < 0.15 OR base contact OR done`. Predict whether an event occurs within a fixed future horizon `H=10` steps. Do not alter thresholds to increase positives.

## Feature groups

1. **Current actor observation:** `o_t` with command dimensions retained for the prediction audit but reported separately.
2. **History stack:** `[o_{t-3}, o_{t-2}, o_{t-1}, o_t]`, masked at episode boundaries.
3. **Privileged/full-state cues:** current simulator fields available to the critic/evaluator (tilt, height, actual velocity, contact, root/angular state where available), explicitly excluding future labels.
4. **Actor plus privileged cues:** diagnostic upper-bound feature set only; never a training policy input.

## Analysis

Collect fixed-condition no-update trajectories from P2 and S1 final checkpoints for seeds 0, 1, and 2. Split by trajectory/seed, never random rows from the same trajectory. Fit the same lightweight offline predictor and report held-out AUROC, average precision, Brier score, and future-tilt/height regression error for each feature group.

Report bootstrap confidence intervals across trajectories and positive-event counts. The audit is invalid if a feature contains future information or crosses a reset boundary.

## Decision rule

- history materially outperforms current observation across seeds → recurrent PPO/LSTM candidate;
- privileged/full-state cues materially outperform actor observation while history does not → SRM-style reconstruction candidate;
- both improve → select the smaller incremental mechanism first; do not stack them;
- neither improves → close memory/representation hypotheses and revisit state/action formulation.

No RL parameters, reward, std schedule, monitor, or training budget are changed by R0.2.
