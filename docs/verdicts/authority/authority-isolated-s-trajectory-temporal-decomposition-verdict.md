# S Trajectory-Level Temporal Outcome Decomposition Verdict
Status: **FROZEN — NO SINGLE RECURRING TEMPORAL FAILURE CLASS; S MECHANISM MINING STOPPED**
Date: 2026-09-25

## Question
At fixed validated u30, does the S-heavy versus center semantic failure follow a recurring temporal accumulation pattern strong enough to justify one mechanism-matched repair?

Checkpoint:

    runs/authority_isolated_simplex_edge_retain-2026-09-25/model_30.pt

Protocol:
- read-only;
- 4 matched reset suites;
- 8 lanes per suite;
- 64 steps;
- exact Smoothness reward `-0.01 ||a_t-a_(t-1)||^2`;
- full per-step S-heavy versus center traces;
- no optimizer update;
- no feature fitting.

The temporal classification and 60% + cross-suite authorization rule were frozen before the run.
## Result
There are 32 matched lane pairs:

    final S-correct    13
    final S-wrong      19

Overall temporal classes:

    A early-wrong -> stays wrong          9
    B early-right -> later reversal       2
    C delayed benefit                    10
    D oscillatory / basin / residual     11

Among the 19 final-wrong lanes:

    A                                  9 / 19 = 47.37%
    B                                  2 / 19 = 10.53%
    D                                  8 / 19 = 42.11%

Class A appears in all four suites containing wrong lanes, but its fraction is below the predeclared 60% threshold.

Class B appears in only two suites and is rare overall.

Therefore neither A nor B satisfies the frozen recurring-mechanism rule.
## Per-suite wrong-lane decomposition
Suite 0: 7 wrong -> A=4, B=1, D=2
Suite 1: 2 wrong -> A=1, B=0, D=1
Suite 2: 4 wrong -> A=2, B=0, D=2
Suite 3: 6 wrong -> A=2, B=1, D=3

The failure population is heterogeneous even within suites.

## Interpretation
The data do **not** support the strong hypothesis that S is usually locally/early correct and later reverses. Only 2/19 wrong lanes are class B.

There is meaningful evidence for persistent early failure in some lanes: class A is the largest single wrong-lane class and appears in every suite. But it is not dominant enough to define a general mechanism because 8/19 wrong lanes fall into residual/oscillatory class D.

Thus the fixed-u30 S inconsistency is better described as heterogeneous trajectory-level closed-loop behavior than as one compact temporal mechanism.

This is consistent with the earlier hierarchy:
- one-step local alignment: rejected as main explanation;
- H16 closed-loop sensitivity: real but not transferable;
- phase-local H16 target: rejected;
- trajectory temporal decomposition: no dominant repairable class.
## Decision

    recurring A mechanism                    NOT ESTABLISHED
    recurring B reversal mechanism           NOT ESTABLISHED
    one S repair candidate                   NOT AUTHORIZED
    further S mechanism mining               STOP
    H2a                                      REMAINS FAIL / INCOMPLETE
    H2b durability                           NOT AUTHORIZED
    objective generalization                 REMAINS PHASE 2 / FUTURE WORK

Under the predeclared stop rule, the correct next thesis action is to freeze the S inconsistency as a limitation rather than invent another loss or architecture change from heterogeneous traces.

The supported thesis-level statement is:

> At the validated u30 operating point, Smoothness preference semantics were not consistently realized. The failure was not explained by reward/proxy mismatch, lack of preference authority, a transferable local gradient defect, a phase-local short-horizon sensitivity, or a single dominant temporal accumulation pattern. The remaining inconsistency is therefore treated as heterogeneous closed-loop trajectory-level behavior rather than as evidence for a compact repair target.

Primary artifacts:
- `docs/contracts/authority/authority-isolated-s-trajectory-temporal-decomposition-contract.md`
- `scripts/rl/authority_isolated_s_trajectory_temporal_decomposition.py`
- `runs/authority_isolated_s_trajectory_temporal_decomposition-2026-09-25/trajectory_temporal_decomposition.json`
- `runs/authority_isolated_s_trajectory_temporal_decomposition-2026-09-25/PROVENANCE_MANIFEST.json`
