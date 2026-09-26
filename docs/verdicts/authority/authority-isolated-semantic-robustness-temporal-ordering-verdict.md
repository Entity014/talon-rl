# Semantic–Robustness Temporal Ordering Verdict

Status: **FROZEN — SEMANTIC-FIRST SUPPORTED, WITH RESET-REGIME CAVEAT**
Date: 2026-09-25

## Question

At the first H2 collapse transition u35 -> u40, does semantic relation degradation precede, follow, or co-emerge with closed-loop stability degradation?

No training was performed.

## Important negative-control correction

An initial absolute semantic-onset detector was rejected because it triggered transient sign reversals even at u35, where semantic endpoint competence still passed.

Specifically:

    u35 joint semantic onset present in 16 / 16 primary axis-reset cases

Therefore absolute sign-crossing was not treated as collapse-specific evidence.

The audit was refined to use paired deterioration relative to u35:

    Delta relation(t)
      = relation_u40(t) - relation_u35(t)

The collapse-specific semantic onset is the first persistent window where both:
- objective heavy-vs-center relation deteriorates;
- physical heavy-vs-center relation deteriorates.

This removes ordinary u35 oscillation from the temporal-ordering question.

## Primary semantic suites

Seeds:

    840001..840004

Axes:

    T / A / O / S

Total primary axis-reset cases:

    16

Classification:

    semantic-first    15
    co-emergent        1
    robustness-first   0

Median lead from semantic deterioration to matched stability divergence:

    24 steps

Range:

    0 .. 39 steps

Thus on the frozen semantic suites, collapse-specific preference-to-behavior deterioration usually appears well before the policy trajectory departs materially from its u35 stability reference.

All primary u40 semantic-suite trajectories still have:

    survival = 1.00

Therefore semantic failure does not require local base_contact or episode termination on the semantic evaluation trajectories themselves.

## Primary onset examples

    seed 840001 / A:
      semantic deterioration t6
      stability divergence   t15

    seed 840002 / T:
      semantic deterioration t4
      stability divergence   t35

    seed 840003 / O:
      semantic deterioration t8
      stability divergence   t29

    seed 840004 / T:
      semantic deterioration t4
      stability divergence   t43

Only:

    seed 840001 / T

is classified co-emergent:

    semantic deterioration t15
    stability divergence   t15

No primary case is robustness-first under the frozen differential definition.

## Fresh reset transfer

Fresh preference-specific reset audit:

    T / 9700000
    A / 9701000
    O / 9702000
    S / 9703000

Results:

    T   semantic-first
    A   robustness-first
    O   semantic-first
    S   semantic-first

Thus fresh reset generalization is not perfectly uniform, but semantic-first remains the dominant pattern.

## Decisive failure lane

The known first fresh failure:

    T / seed 9700000

shows:

    semantic deterioration onset    t11
    stability divergence            t27
    first base_contact              t57
    survival                        .875

Ordering:

    semantic deterioration
      -> ~16 steps
    stability divergence
      -> ~30 steps
    base_contact

This is the strongest direct evidence against:

    robustness loss first
      -> semantic score collapses only as a consequence

for the known u40 failing lane.

## Interpretation

The current evidence supports:

    preference-to-behavior semantic degradation
      occurs before
    closed-loop stability degradation

more strongly than the reverse direction.

Combined with prior H2 findings:

    authority geometry       preserved / stronger
    critic EV                valid
    semantic competence      collapses
    robustness               later regresses

the causal picture is now:

    PPO continuation
      ->
    preference-conditioned behavioral mapping degrades
      while action authority remains strong
      ->
    trajectory-level stability margins subsequently erode
      in susceptible reset regimes
      ->
    base_contact may occur later

This is compatible with:

    semantic mapping degradation
      -> poorer closed-loop policy choice
      -> robustness loss

## What is NOT established

This audit does not prove that semantic degradation is the sole cause of robustness failure.

Reasons:
- stability divergence is defined relative to matched u35 trajectory dispersion, not a physical hard safety boundary;
- fresh reset ordering is not unanimous;
- A/9701000 is robustness-first;
- different reset regimes can expose different closed-loop dynamics.

Therefore do not claim:

    semantic collapse deterministically causes every robustness failure

The supported statement is narrower:

    robustness-first is not the dominant explanation of the u35->u40 semantic collapse;
    semantic degradation usually precedes measurable stability deterioration,
    including in the first known failing fresh lane.

## Decision

Predeclared categories:

    SEMANTIC-FIRST SUPPORTED            YES
    ROBUSTNESS-FIRST SUPPORTED          NO
    COMMON-REGIME / CO-EMERGENT         not dominant
    DECOUPLED ACROSS RESET REGIMES      partial caveat only

Formal verdict:

    SEMANTIC-FIRST SUPPORTED
    with reset-regime heterogeneity

## Consequence for next branch

Do not return to:
- authority retention;
- critic width;
- headroom mining;
- external robustness mining;
- joint-specific safety patching.

The next unresolved mechanism is upstream of robustness:

    what changes in the preference-to-behavior semantic mapping
    between u35 and u40 while action-space authority remains strong?

Any next intervention should first target or diagnose semantic mapping quality, not treat robustness loss as the initiating cause.

A training intervention is still not automatically authorized by this audit.
