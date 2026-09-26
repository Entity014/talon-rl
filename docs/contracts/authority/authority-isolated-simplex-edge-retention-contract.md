# Authority-Isolated Simplex Edge-Margin Durability Contract

Status: PREDECLARED
Date: 2026-09-25

## Question

Can bounded asymmetric preservation of the full preference-conditioned action simplex prevent post-u20 authority contraction during continued PPO learning?

## Frozen start

Exact robust u20 continuation state:

    runs/authority_isolated_coverage2_control-2026-09-25/resume_state.pt

Frozen:
- authority-isolated actor
- wide critic
- actor optimizer state
- lambda=.95
- projected PPO/tail repair
- kappa=.05
- reset/training distribution
- preference schedule
- critic support/head-refresh contract
- u20 reference state support
- update budget u20 -> u30
- rho=.25
- beta0=2.497041993384243

## Sole treatment

CONTROL:
    current projected/headroom continuation

TREATMENT:
    same
    + asymmetric all-simplex-edge retention

Preferences:

    T / A / O / S / C

All 10 unordered edges are primary:

    T-A
    T-O
    T-S
    T-C
    A-O
    A-S
    A-C
    O-S
    O-C
    S-C

## Frozen reference support

Reuse exactly:

    runs/authority_isolated_u20_authority_support-2026-09-25/u20_authority_support.npz

No new states are added.

Per update:
- balanced 120-state sample
- 8 states from each origin x phase cell

## Edge quantity

For each state s and preference pair i,j:

    d_ij^theta(s)
      = || pi_theta(s,w_i) - pi_theta(s,w_j) ||_2

Frozen u20 edge:

    d_ij^20(s)

Asymmetric floor loss:

    L_edge
      = E [ max(0, gamma d_ij^20(s) - d_ij^theta(s))^2 ]

## Gamma

Predeclared single value:

    gamma = 0.90

No gamma sweep is authorized in this branch.

Interpretation:
- edge >= 90% of frozen u20 edge: no penalty
- edge < 90%: retention gradient activates
- authority growth is not penalized

## Gradient budget

Reuse validated bounded rule:

    alpha_t =
      min(beta0,
          rho ||g_base|| / (||g_edge|| + eps))

with:

    rho   = 0.25
    beta0 = 2.497041993384243

No rho/beta tuning.

## Primary authority gate

On frozen u20 matched support:

    pairwise authority retention >= .90
    tangent authority retention  >= .90

## Edge geometry gate

Report separately:

1. heavy-heavy mean edge-energy retention
2. heavy-center mean edge-energy retention
3. every individual edge
4. minimum edge-energy retention

Predeclared edge floor:

    every important edge-energy retention >= .90

A mean >= .90 is not sufficient if one edge collapses materially.

## Mechanism gate

Required:
- treatment improves edge-floor deficit vs control
- no systematic heavy-heavy compression remains
- family parameter geometry remains non-degenerate
- treatment materially improves pairwise authority vs control

## No-regression

    semantic-suite minimum survival = 1.00
    held-out reset minimum survival >= .95
    fresh H32 / MC64 critic validity PASS
    PPO ratio invariant <= 1e-4
    no new termination topology
    projected/headroom repair remains active

## Decision logic

PASS:
- pairwise >= .90
- tangent >= .90
- minimum edge-energy retention >= .90
- no-regression gates pass

PAIRWISE PASS / TANGENT FAIL:
- finite simplex geometry preservation is sufficient for pairwise authority
- local differential authority remains a separate blocker
- only then may a tangent/Jacobian branch be considered

FAIL:
- do not add Jacobian loss automatically
- first classify whether failure is edge-specific, phase-specific, or the edge-floor formulation itself is insufficient

AI-C2 remains blocked until this durability gate passes.
AI-H2 remains blocked until formal AI-C2 passes.
