# Formal AI-C2 After Simplex-Edge Durability Repair

Status: PREDECLARED
Date: 2026-09-25

## Starting actor

Use the durability-valid endpoint:

    runs/authority_isolated_simplex_edge_retain-2026-09-25/model_30.pt

Actor/log_std are copied exactly into both critic arms.

Actor optimizer/support/RNG continuation state comes from:

    runs/authority_isolated_simplex_edge_retain-2026-09-25/resume_state.pt

## Sole paired contrast

NARROW:
    critic body 48 -> 128 -> 128 -> 128 -> 4

WIDE:
    critic body 48 -> 256 -> 256 -> 128 -> 4

Both critics are re-equilibrated on the same frozen edge-u30 actor support before actor learning.

## Frozen actor-side contract

Both arms use identically:
- projected PPO/headroom repair
- kappa=.05
- GAE lambda=.95
- continuous preference schedule
- original reset distribution
- same support/head-refresh schedule
- simplex-edge durability retention
- gamma=.90
- rho=.25
- beta0=2.497041993384243
- frozen u20 edge reference support
- same update budget and seed schedule

No Jacobian loss and no semantic rehearsal.

## Continuation

    global u30 -> u40
    10 actor updates

## Primary gates

Authority:
    matched-support pairwise retention >= .90
    matched-support tangent retention >= .90

Edge geometry:
    minimum overall simplex edge retention >= .90

Critic:
    fresh H32 EV mean > 0
    fresh MC64 EV mean > 0
    H32 negative fraction <= .25
    MC64 negative fraction <= .25

Robustness:
    semantic-suite min survival = 1.00
    held-out min survival >= .95
    fresh min survival >= .95

PPO:
    ratio invariant <= 1e-4

## Interpretation

If WIDE passes all gates:
    AI-C2 PASS and AI-H2 is authorized.

If NARROW fails while WIDE passes:
    critic-capacity compatibility/necessity is supported in this repaired regime.

If both pass:
    wide critic is compatible but strict capacity necessity is not supported.

If WIDE fails:
    AI-H2 remains blocked.
