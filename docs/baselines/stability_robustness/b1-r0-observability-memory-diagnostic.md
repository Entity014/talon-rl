# B1-R0 — Observability / Memory Diagnostic

Status: **DIAGNOSTIC DRAFT — READ-ONLY / NO UPDATE**

B1-P1, B1-P2, and B1-S1 are closed as failed or partial interventions. The next question is not another optimizer or smoothness coefficient:

> Does the feed-forward policy observe enough instantaneous information to choose a stable gait, or does action selection depend on recent history/latent state?

## Scope and non-goals

R0 does not train, update parameters, alter reward/std/env semantics, or authorize B0.2. It uses existing P2/S1 checkpoints and newly collected short no-update rollouts only. No recurrent model is introduced in R0.

## Diagnostic A — near-observation collision

Collect time-major tuples `(o_t, history_t, a_t, done_t, seed, checkpoint)` from the same deterministic monitor/rollout contract. Find pairs with small instantaneous observation distance but different recent histories. Compare deterministic actor means/actions:

- `||o_i - o_j||`;
- history distance over the preceding `K` observations/actions;
- `||mu(o_i) - mu(o_j)||` and action distance;
- whether the pair occurs near a fall, high tilt, or recovery transition.

The collision search must exclude pairs across reset boundaries and report pair counts, distance thresholds, and state/episode provenance.

## Diagnostic B — history-conditioned predictability

For each current observation bucket, measure whether the next deterministic action and short-horizon behavior are better predicted from `(o_t)` alone or from a bounded history window. This is an offline diagnostic only: use fixed regressors/statistics, not a trained policy. Report conditional action variance and outcome variance by observation bucket.

Evidence for a memory problem requires repeated same/near-same observations with materially different action or short-horizon outcomes conditioned on history. A simple high-dimensional observation distance without behavioral divergence is not sufficient.

## Diagnostic C — representation coverage

Audit the policy observation layout and identify information unavailable at one timestep but present in simulator fields or existing rollout metadata: velocity/phase cues, contact transitions, and recent action context. Separate:

- information already present but potentially hard for the MLP to use;
- information genuinely absent from policy observations;
- privileged information that must not leak into the baseline monitor.

## Decision rule

- Strong history dependence with adequate instantaneous fields → candidate **P2 + recurrent PPO/LSTM**.
- Missing reconstructable latent/phase information with weak direct history evidence → candidate **P2 + SRM-style reconstruction**.
- Neither signal → stop escalating architecture and revisit task observability/representation assumptions before training.

## Candidate screen

Only after R0 evidence, run short one-seed screens for P2+LSTM and P2+SRM, one variable at a time. A candidate with no signal is stopped. A candidate with a predeclared signal is frozen and confirmed on three seeds using the unchanged update-500 deterministic gate.

Until R0 is complete, B0.2, MOPPO, KL coefficient sweeps, and ACAPS coefficient sweeps remain closed.
