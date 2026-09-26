# V2-B State-Fixed Phase-Local Credit Audit

Status: FROZEN — ONE-STEP LOCAL FD COMPARATOR INVALID; NO MONOTONIC STATE-FIXED LATE CREDIT COLLAPSE

Date: 2026-09-23

## Question

At matched O-heavy states from t8/t20/t36/t52, does semantic credit itself become intrinsically wrong in late-state regimes, or is the previously observed late trajectory drift primarily visitation/sequence dependent?

## Coordinate contract

The one-step Orientation action derivative was estimated in applied-action coordinates and mapped with the exact tanh Jacobian into pre-tanh action-mean coordinates before comparison with MC and GAE score directions. This avoids the coordinate error diagnosed previously in C40.

## One-step local derivative stability

A central finite-difference sweep was run with h = 0.04, 0.02, 0.01, 0.005.

The local derivative is not numerically stable enough to serve as a causal comparator:

| Phase | cos local(h=.02), local(h=.01) | cos local(h=.01), local(h=.005) | local norm behavior |
|---|---:|---:|---|
| t8 | 0.547 | 0.472 | ~1.23 -> 1.18 -> 13.18 |
| t20 | 0.743 | 0.792 | ~0.74 -> 1.05 -> 1.02 |
| t36 | 0.455 | 0.145 | ~0.51 -> 7.07 -> 5.80 |
| t52 | 0.358 | 0.015 | ~1.17 -> 1.25 -> 2.45 |

The h=.04 direction is also frequently inconsistent with h=.02. Therefore claims based on local-vs-MC or local-vs-GAE cosine are rejected for this audit. The one-step physical action effect is too weak/noisy under the simulator/reward discretization to identify a reliable tangent with this finite-difference method.

## State-fixed MC vs GAE credit

At the same frozen phase states, MC-vs-GAE Orientation score cosine was:

- t8: 0.518 ± 0.316
- t20: 0.577 ± 0.164
- t36: 0.544 ± 0.067
- t52: 0.439 ± 0.356

This is imperfect alignment, but there is no clean monotonic phase collapse analogous to the trajectory-level profile that fell from ~0.7-0.8 early to negative late.

## Interpretation

The previous trajectory audit remains valid as a statement about realized closed-loop rollouts: as O-heavy and center trajectories diverged, time-local GAE-vs-MC score alignment degraded strongly and became negative late.

However, the state-fixed audit does not reproduce a universal intrinsic late-state collapse. Therefore the strongest current interpretation is:

> The GAE/MC drift is primarily associated with the evolving closed-loop visitation/trajectory distribution and sequence context, rather than a simple phase label or an intrinsic property shared by all late states.

This is consistent with a trajectory-distribution / temporal-realization problem: the estimator and long-horizon objective become differently weighted as the policy visits diverging states over time.

## Causal status

- Conditioning/action authority: established.
- Early Orientation semantic realization: established.
- Cross-objective override: rejected as primary.
- O objective semantic sign error: rejected.
- One-step local action-gradient comparator: INVALID / numerically unstable.
- Universal late-state local GAE failure: NOT SUPPORTED.
- Closed-loop visitation-dependent GAE/MC drift: SUPPORTED.
- Dynamics/trajectory coupling: remains involved.

## Decision

V2-C remains OFF.

The next useful diagnostic should not add architecture and should not rely on one-step action finite differences. It should isolate visitation dependence directly, for example by evaluating MC/GAE credit on the same frozen state set under early-vs-late continuation distributions or by replaying common states with controlled continuation horizons. The purpose is to determine whether horizon/continuation distribution, rather than local state geometry, creates the semantic drift.