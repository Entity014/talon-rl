# Authority-Isolated Authority Durability Diagnostic Verdict

Status: **FROZEN — FUNCTIONAL FAMILY-PATH DURABILITY FAILURE CONFIRMED; BOUNDED FUNCTIONAL AUTHORITY RETENTION AUTHORIZED**
Date: 2026-09-25

## Question

Does post-u20 authority decay correspond to loss of causal preference response relative to the frozen robust u20 reference while the underlying preference-family geometry remains intact?

## Reference checkpoints

u20 robust reference:

    runs/authority_isolated_coverage2_control-2026-09-25/model_20.pt

u30 exact continuation:

    runs/authority_isolated_ai_c2_continuity-2026-09-25/model_30.pt

## Endpoint fixed-probe finding

On the historical fixed probe-state set:

    pairwise action retention              0.934
    action tangent-Jacobian retention      1.015

Meanwhile family-side representation metrics do not collapse:

    coefficient pairwise retention         1.009
    generated-parameter pairwise retention 1.032
    parameter tangent retention            1.034
    centered parameter specific energy     1.034

Ranks:

    centered parameter rank      3 -> 3
    centered functional rank     4 -> 4

Thus the family does not lose parameter-space dimensionality or preference-conditioned generated-parameter separation.

This also shows that the apparent continuation authority failure is state-regime dependent rather than a uniform global collapse.

## Parameter drift

Relative parameter L2 drift u20 -> u30:

    shared/common actor path       0.0136
    family all                     0.1170
    family hypernetwork            0.0227
    family generated bases         0.1373
    family base network            0.1064
    log_std                        0.1204

The largest deterministic actor drift is inside the family realization parameters, not the shared state-only actor trunk.

## Matched-state audit

To remove state-distribution confounding, states were collected from the frozen u20 actor on:
- suite2 seed 840003
- suite3 seed 840004
- held-out suite4 seed 850101
- held-out suite5 seed 850202
- held-out suite6 seed 850303
- all T/A/O/S/C rollout origins
- early and late trajectory phases

The same exact states were then evaluated by u20, u30, and hybrid actors.

### Aggregate matched states

Initial-state support:

    u30 pairwise retention                 0.830
    u30 tangent retention                  0.803
    u30 functional-specific RMS retention  0.850

Early trajectory support:

    u30 pairwise retention                 0.864
    u30 tangent retention                  0.900
    u30 functional-specific RMS retention  0.852

Late trajectory support:

    u30 pairwise retention                 0.847
    u30 tangent retention                  0.871
    u30 functional-specific RMS retention  0.855

Therefore the post-u20 authority loss is real on the state distribution actually visited by the robust policy, even though it is weak or absent on the older fixed probe set.

## Causal hybrid swap

Two functional hybrids were constructed:

    H_common:
      common/shared actor from u30
      family path from u20

    H_family:
      common/shared actor from u20
      family path from u30

### Aggregate early states

    H_common pairwise retention    0.971
    H_family pairwise retention    0.851

### Aggregate late states

    H_common pairwise retention    0.977
    H_family pairwise retention    0.817

### Initial states

    H_common pairwise retention    1.024
    H_family pairwise retention    0.844

Therefore:
- replacing only the shared/common path with u30 largely preserves u20 authority;
- replacing only the family path with u30 reproduces most of the authority loss.

This localizes the causal durability failure to the family path.

## Cross-preference recurrence

The same pattern recurs across rollout-origin preferences.

Examples, pairwise retention on late states:

    T-origin:
      u30       0.841
      H_common  0.976
      H_family  0.772

    A-origin:
      u30       0.791
      H_common  0.965
      H_family  0.769

    O-origin:
      u30       0.757
      H_common  0.963
      H_family  0.787

    S-origin:
      u30       0.939
      H_common  1.003
      H_family  0.870

    C-origin:
      u30       0.961
      H_common  1.009
      H_family  0.872

The effect is strongest on T/A/O late-state regimes and weaker on S/C, but it is not a single-preference or single-reset artifact.

## Interpretation

The failure is not:

    family rank collapse
    coefficient collapse
    generated-parameter magnitude collapse
    shared state-trunk overwrite
    critic capacity failure
    PPO ratio failure

Instead:

    family parameter geometry remains expressive
        ->
    continued PPO changes family realization parameters
        ->
    on states actually visited by the robust policy,
    preference-conditioned action response contracts
        ->
    pairwise separation / tangent authority falls

This is a functional durability problem.

## Authorization

A bounded functional-authority retention treatment is now justified.

The retention target should be defined against the frozen u20 reference on matched representative states.

Preferred target:

    Delta a_theta(s,w)
      = pi_theta(s,w) - pi_theta(s,w_C)

retain:

    Delta a_theta(s,w)
      ~= Delta a_u20(s,w)

or equivalently preserve the local simplex-tangent response:

    J_w^theta(s)
      ~= J_w^u20(s)

The target must operate on functional preference response, not:
- raw family parameter norm;
- coefficient magnitude;
- generated-parameter norm;
- global action magnitude;
- semantic rollout outcomes.

## Gradient-budget rule

Reuse the already validated bounded rehearsal budget:

    alpha_t
      = min(beta_0,
            rho * ||g_mixed|| / (||g_retain|| + eps))

with the previously validated bounded budget, rho approximately 0.25, unless a fresh predeclared calibration shows otherwise.

The purpose is not to create new authority.
The purpose is to prevent continued PPO from erasing the functional authority already present at u20.

## Next experiment

Paired continuation from exact robust u20:

CONTROL:
    current projected/headroom PPO continuation

TREATMENT:
    same
    + bounded functional-authority retention to frozen u20

Primary gate:

    pairwise authority retention >= 0.90
    tangent authority retention >= 0.90

No-regression:

    frozen semantic survival = 1.00
    held-out survival >= 0.95
    fresh H32/MC64 critic validity PASS
    PPO ratio invariant PASS
    no new termination topology

If treatment passes:
    rerun formal AI-C2
    then authorize AI-H2.

## Decision

    authority-durability diagnostic     PASS
    family-path causal localization     PASS
    bounded functional retention        AUTHORIZED
    AI-H2                               STILL BLOCKED
