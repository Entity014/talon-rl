# Authority-Isolated Functional Delta-a Durability Retention Contract

Status: PREDECLARED
Date: 2026-09-25

## Question

Can a bounded functional retention constraint preserve the experimentally confirmed u20 preference authority during continued PPO learning, without changing MORL semantics or damaging robustness, critic validity, PPO correctness, or headroom repair?

## Start

Exact robust u20 continuation state:

    runs/authority_isolated_coverage2_control-2026-09-25/resume_state.pt

Both arms inherit identical:
- actor
- wide critic
- actor optimizer state
- adaptive critic support
- anchor support
- RNG state
- lambda=.95
- projected PPO/tail repair
- kappa=.05
- training/reset distribution
- preference schedule
- update budget

## Arms

CONTROL:
    existing u20 -> u30 continuation

TREATMENT:
    same
    + bounded functional Delta-a retention

## Frozen reference support

Reference states come from the frozen robust u20 actor, not the historical generic fixed probe.

Seeds:
    840003
    840004
    850101
    850202
    850303

Origins:
    T / A / O / S / C

Phases:
    initial t0-3
    early   t4-31
    late    t32-63

Stored support:
    128 states per origin x phase cell
    1,920 states total

Per training update:
    sample 8 states from every cell
    = 120 states
    evaluate all four heavy preferences against center
    = 480 matched Delta-a pairs

## Functional target

For heavy preference w_i:

    Delta a_theta(s,w_i)
      = pi_theta(s,w_i) - pi_theta(s,w_C)

Frozen target:

    Delta a_20(s,w_i)

Retention loss:

    L_retain
      = E ||Delta a_theta(s,w_i)
             - Delta a_20(s,w_i)||^2

No semantic returns, PASS events, parameter norms, coefficient magnitudes, generated-parameter norms, or endpoint scores enter this loss.

## Gradient budget

Validated bounded auxiliary-gradient rule is frozen:

    rho   = 0.25
    beta0 = 2.497041993384243

Current base actor update is the existing projected PPO + active tail-descent direction.

For its gradient g_base and retention gradient g_retain:

    alpha_t =
      min(beta0,
          rho * ||g_base|| / (||g_retain|| + eps))

Final pre-clipping actor gradient:

    g_total = g_base + alpha_t g_retain

The retention term is not allowed to exceed 25% of the existing actor-update gradient norm unless beta0 is tighter.

## Screen

    u20 -> u30
    10 actor updates

## Primary authority gate

Measured on matched representative u20 state support:

    pairwise action authority retention >= .90
    simplex-tangent action Jacobian retention >= .90

Historical fixed probe is diagnostic only and is not the primary support.

## Mechanism gate

Required:
- matched-state Delta-a error remains bounded relative to control;
- functional specific-response RMS retention >= .90;
- family parameter / coefficient geometry remains non-degenerate;
- treatment must materially outperform matched control on functional authority.

## No-regression gates

    semantic-suite minimum survival = 1.00
    held-out reset minimum survival >= .95
    fresh H32 EV > 0
    fresh MC64 EV > 0
    negative-EV fractions <= .25
    PPO ratio invariant <= 1e-4
    no new termination topology
    tail/headroom repair remains active

## Interpretation

This is not the previous semantic-retention rehearsal method.

The causal role is narrower:

    prevent experimentally localized family-path functional authority contraction after u20

Passing this gate authorizes a formal AI-C2 rerun.

AI-H2 remains blocked until formal AI-C2 passes.
