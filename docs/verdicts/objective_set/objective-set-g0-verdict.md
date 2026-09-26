# Objective-Set Generalized MORL — G0 Verdict

Status: **FROZEN — G0 REPRESENTATION EQUIVALENCE PASS; G1 AUTHORIZED**
Date: 2026-09-25

## Purpose

G0 tested whether the validated Phase-1 authority-isolated controller could be migrated from a fixed-index 4D preference interface to a permutation-invariant objective-set interface without changing its function or semantic behavior.

Source Phase-1 checkpoint:

    runs/authority_isolated_simplex_edge_retain-2026-09-25/model_30.pt

Objective-set initialization:

    runs/objective_set_g0_structural_parity-2026-09-25/objective_set_g0_init.pt

No training update was performed.

## G0-A structural parity

Compatibility tokens are the canonical one-hot T/A/O/S identities. Weighted set aggregation therefore satisfies exactly:

    z_O = sum_i w_i e_i = w

The migrated actor consumes z_O through the same authority-isolated family path.

The critic public interface is changed from a fixed four-vector output to shared token queries:

    V_i = V(s,e_i,z_O)

with the legacy four value rows loaded as token-conditioned value bases.
Structural results on the frozen Phase-1 probe bank:

    pre-tanh max abs error       0
    action max abs error         0
    log-prob max abs error       0
    critic max abs error         8.34e-7

Permutation test over all 24 orderings:

    pre-tanh max drift           0
    action max drift             0
    queried value max drift      0

Frozen tolerance:

    1e-6

Decision:

    G0-A PASS

Unit tests:

    tests/test_objective_set_actor_critic.py
    4 / 4 PASS

The same model instance also accepts m=2,3,4 token sets without changing parameter shapes.

## G0-B behavioral parity

The full frozen H2a matched-reset protocol was replayed using the objective-set model.

Endpoint semantics:

    T   PASS   objective=1.00 physical=1.00 survival=1.00
    A   PASS   objective=1.00 physical=1.00 survival=1.00
    O   PASS   objective=1.00 physical=1.00 survival=1.00
    S   FAIL   objective=0.50 physical=0.50 survival=1.00
Mean endpoint effects remain exactly equal to Phase 1:

    T objective delta       +0.000664894
    T physical delta        -0.0315489

    A objective delta       +0.000776573
    A physical delta        -0.0353405

    O objective delta       +0.00891025
    O physical delta        -0.744545 deg

    S objective delta       -0.000125681
    S physical delta        +0.0218054

Continuum / compromise:

    monotonicity            0.747396
    endpoint-between        0.666667
    center compromise       0.875

Critic:

    H32 EV mean             0.0471441
    negative fraction       0.20625
    mean abs bias           0.0479726

Difference versus frozen Phase-1 H2a:

    all endpoint metrics    exactly 0
    continuum metrics       exactly 0
    critic EV drift         -8.02e-9

The known continuum reset survival minimum remains 0.875, exactly as in the Phase-1 H2a run. Endpoint survival remains 1.00 and the migration introduces no new robustness regression.

Decision:

    G0-B PASS
## Scientific interpretation

The fixed-index representation was not required to reproduce the validated Phase-1 behavior.

A permutation-invariant objective-set representation can reproduce the same controller exactly when initialized through the compatibility token basis.

Therefore the Phase-2 question is now cleanly separated:

    representation migration      SOLVED
    variable-cardinality learning OPEN
    unseen-combination learning   NOT YET TESTED

The result does not improve or repair Smoothness. S remains the same frozen Phase-1 limitation.

This is desirable for G0 because G0 tests equivalence, not method improvement.

## Authorization

    Phase 1 frozen                  YES
    G0 structural parity            PASS
    G0 behavioral parity            PASS
    hard permutation gate           PASS
    critic token-query migration    PASS
    G1 variable-cardinality         AUTHORIZED
    G2 compositional holdout        BLOCKED UNTIL G1
    G3 unseen objective identity    STRETCH / NOT AUTHORIZED

## Primary artifacts

- docs/contracts/objective_set/objective-set-generalized-morl-phase2-contract.md
- talon_rl/objective_set_actor_critic.py
- tests/test_objective_set_actor_critic.py
- scripts/rl/objective_set_g0_structural_parity.py
- scripts/rl/objective_set_g0_behavioral_parity.py
- runs/objective_set_g0_structural_parity-2026-09-25/g0_structural_parity.json
- runs/objective_set_g0_behavioral_parity-2026-09-25/g0_behavioral_parity.json
