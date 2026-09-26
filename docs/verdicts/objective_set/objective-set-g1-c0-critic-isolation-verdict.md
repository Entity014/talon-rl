# V3-G1-C0 Fixed-Actor Token-Query Critic Isolation Verdict

Status: **FROZEN — CRITIC SUBSTRATE FAILS INDEPENDENTLY OF ACTOR DRIFT**
Date: 2026-09-25

## Question

Does the generalized token-query critic remain valid on seen m=4 when actor parameters are frozen exactly at the G0 controller?

## Result

All six preregistered runs preserved actor parameters exactly:

    max actor parameter drift = 0

Allowed training cardinalities were respected:
- G1-2-C0 used only m=3,4
- G1-3-C0 used only m=2,4

No held-out cardinality entered critic training.

Fresh m=4 critic results:

    G1-2 / 73101    EV -7.240   neg frac .800
    G1-2 / 73102    EV -2.580   neg frac .769
    G1-2 / 73103    EV -12.651  neg frac .956

    G1-3 / 73101    EV -1.141   neg frac .519
    G1-3 / 73102    EV -3.978   neg frac .838
    G1-3 / 73103    EV -8.348   neg frac .956

Frozen critic gate:

    mean EV > 0
    negative fraction <= .25

Passes:

    0 / 6
## Interpretation

The G1 critic failure does not require actor parameter drift.

Therefore the direct online Adam training of the shared token-query critic on mixed active-set cardinalities is independently insufficient under the current support/target scheme.

This sharply narrows the G1 blocker:

    objective-set representation          VALID
    permutation invariance               VALID
    actor preference authority           RETAINED
    actor architecture                   NOT CURRENT BLOCKER
    token-query critic API               STRUCTURALLY VALID
    token-query critic training/support  FAIL

The earlier joint G1 semantic regression is therefore confounded by a critic that is already invalid even under a frozen actor.

This does not prove that fixing the critic alone will restore T/A/O semantics, but critic validity must be restored before cardinality generalization can be interpreted.

## Decision

    G1 cardinality claim             NOT INTERPRETABLE
    G2                              BLOCKED
    actor architecture changes      NOT AUTHORIZED
    objective-specific patches      NOT AUTHORIZED
    critic-support repair           AUTHORIZED

The next justified branch is to generalize the Phase-1 critic compatibility machinery itself:
- reset-diverse representative support;
- active-token target collection;
- shared token-query critic fit/refresh;
- no inactive objective queries;
- no held-out cardinality leakage.

The actor must remain frozen while that repair is validated first.

## Primary artifacts

- docs/contracts/objective_set/objective-set-g1-c0-critic-isolation-contract.md
- scripts/rl/objective_set_g1_c0_critic_isolation.py
- runs/objective_set_g1_c0_critic_isolation-2026-09-25/critic_isolation.json
- runs/objective_set_g1_c0_critic_isolation-2026-09-25/PROVENANCE_MANIFEST.json
