# S Trajectory-Level Temporal Outcome Decomposition Contract
Status: **PREDECLARED — READ-ONLY, FIXED U30, NO TRAINING**
Date: 2026-09-25

## Question
At the validated u30 checkpoint, how does Smoothness semantic effect accumulate over time in matched S-heavy versus center trajectories?

This gate does not fit a predictor, search features, add a loss, or modify training.

## Frozen protocol
Checkpoint: `runs/authority_isolated_simplex_edge_retain-2026-09-25/model_30.pt`
Preferences: C=[.25,.25,.25,.25], S=[.10,.10,.10,.70]
Suites: seeds 840001..840004, 8 lanes each, 64 steps.
No optimizer update is allowed.

## Per-step outputs
For every matched lane record:
- exact `Delta r_S(t)`, with `r_S=-0.01||a_t-a_(t-1)||^2`;
- cumulative `Delta J_S(0:t)`;
- `Delta action_rate(t)`;
- `||a_S(t)-a_C(t)||_2`;
- `||s_S(t)-s_C(t)||_2` using policy observations;
- termination and contact timing for both branches.

Temporal checkpoints are H=4,8,16,32,64.
## Frozen classification
For each lane set `tau=max(1e-6,0.10*max_t |cumulative Delta J_S|)`.
Checkpoint values above +tau are positive, below -tau negative, otherwise neutral.

A — early-wrong -> stays wrong:
- H64 negative;
- H8 negative;
- H16/H32/H64 never positive.

B — early-right -> later reversal:
- H64 negative;
- at least one of H4/H8/H16 positive.

C — delayed benefit:
- H64 positive;
- H4 and H8 not positive;
- at least one of H16/H32/H64 positive.

D — oscillatory / basin-sensitive / residual:
- every remaining pattern.
D deliberately stays residual; no subclasses may be invented after seeing results.

## Primary decision
The decisive population is final-wrong lanes (`Delta J_S(0:64)<0`).
Report overall, wrong-only, and per-suite class counts.
A recurring mechanism authorizes exactly one repair candidate only if:
1. A or B contains >=60% of final-wrong lanes; and
2. that same class occurs in at least 3 of 4 suites containing a wrong lane.

Interpretation:
- B majority: late closed-loop reversal after initially correct accumulation.
- A majority: early few-step semantic failure that persists.
- neither: stop S mechanism mining and freeze the limitation.

Class C is descriptive only.

## Secondary descriptive outputs
Report median traces for final-correct vs final-wrong lanes:
- cumulative Delta J_S;
- Delta action-rate;
- action separation;
- state separation;
- first meaningful cumulative-sign time;
- contact/termination timing.

## Non-goals
Do not fit a feature predictor, tune thresholds after results, add a semantic loss,
modify PPO/actor/critic, change the fixed objective set, or open objective generalization.

## Gate consequence
Recurring A/B mechanism -> authorize one mechanism-matched S repair candidate.
Otherwise -> stop S mechanism mining; H2a remains incomplete and H2b unauthorized.
