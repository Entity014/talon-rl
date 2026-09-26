# Authority-Isolated Direction vs Amplitude Authority Audit Verdict

Status: **FROZEN — HEAVY-vs-CENTER FLOOR IS NOT SUFFICIENT; SIMPLEX EDGE CONTRACTION IS THE NEXT MECHANISTIC TARGET**
Date: 2026-09-25

## Question

Is post-u20 authority loss primarily explained by contraction of heavy-vs-center response magnitude / projected reference authority, such that an asymmetric projected-authority floor is justified as the next retention treatment?

## Inputs

Frozen u20 reference:

    runs/authority_isolated_coverage2_control-2026-09-25/model_20.pt

Endpoints:

    control u30
    bounded raw-Delta-a retention u30

Reference support:

    1,920 robust-u20 visited states
    balanced across T/A/O/S/C origins
    initial / early / late phases
    frozen + held-out reset suites

## Per-state decomposition

For each heavy preference:

    r_mag
      = ||Delta a_theta|| / (||Delta a_20|| + eps)

    c_dir
      = cos(Delta a_theta, Delta a_20)

    m_parallel
      = Delta a_theta^T Delta a_20
        / (||Delta a_20||^2 + eps)

The previous raw-Delta-a treatment clearly improves direction:

median cosine, CONTROL -> RETAIN

    T   .738 -> .906
    A   .300 -> .481
    O   .277 -> .515
    S   .467 -> .667

However magnitude remains contracted on important axes:

mean Delta-a norm ratio, CONTROL -> RETAIN

    T   .852 -> .852
    A  1.071 -> .862
    O   .855 -> .776
    S  1.223 -> 1.153

Thus raw MSE can reduce Euclidean error by improving direction while still allowing response-scale contraction.

## Reference-energy-weighted audit

To avoid instability from states where the frozen reference Delta-a is nearly zero, state contributions were weighted by ||Delta a_20||^2.

Weighted projected reference authority m_parallel:

    axis    CONTROL    RETAIN
    T        .512       .680
    A        .406       .479
    O        .404       .467
    S        .342       .448

The gamma=.9 projected-floor loss also decreases under RETAIN on every axis.

Therefore the raw-Delta-a treatment already improves projected reference authority in aggregate.

Yet the primary authority metrics worsen:

    pairwise retention:
      control  .906
      retain   .828

    tangent retention:
      control  .862
      retain   .783

This is decisive:

    improved heavy-vs-center projected authority
      does not imply
    preserved preference authority geometry

Therefore a heavy-vs-center projected-authority floor alone is not justified as the next training treatment.

## Cell-level correlation

Across matched origin x phase cells:

For RETAIN:

    m_parallel vs pairwise retention:
      Pearson   .642
      Spearman  .489

This is stronger than cosine alone, so projected authority is relevant.

However its improvement is not sufficient to preserve the gate.

Magnitude ratio is more related to tangent retention:

    magnitude ratio vs tangent retention:
      Pearson   .500
      Spearman  .729

This reinforces that pairwise and tangent authority are distinct geometric quantities.

## Simplex-edge audit

All 10 pairwise preference edges among T/A/O/S/C were measured on the same frozen-u20 states.

Energy-norm edge retention:

CONTROL:

    T-A  .855
    T-O  .884
    T-S  .791
    T-C  .817
    A-O  .913
    A-S  .821
    A-C  .930
    O-S  .942
    O-C  .896
    S-C  .987

RETAIN:

    T-A  .846
    T-O  .858
    T-S  .801
    T-C  .836
    A-O  .837
    A-S  .728
    A-C  .801
    O-S  .876
    O-C  .835
    S-C  .956

Aggregate:

    heavy-heavy mean edge retention:
      control  .868
      retain   .824

    heavy-center mean edge retention:
      control  .907
      retain   .857

The strongest treatment regression is not reducible to one radial heavy-center response.

Examples:

    A-S:
      control .821
      retain  .728

    A-O:
      control .913
      retain  .837

    O-S:
      control .942
      retain  .876

Therefore the bounded raw-Delta-a treatment improves alignment to each heavy-vs-center reference direction while compressing the multi-preference action simplex.

## Causal interpretation

The current evidence supports:

    family realization path drift
        ->
    preference-conditioned action simplex contracts / deforms
        ->
    heavy-heavy and heavy-center margins shrink
        ->
    pairwise authority declines
        ->
    local simplex tangent authority also declines

The failure is not adequately described by:

    direction error alone
    radial magnitude alone
    projected heavy-vs-center authority alone

The authority quantity is multi-directional.

## Decision

    direction-vs-amplitude audit               COMPLETE
    projected authority is relevant           SUPPORTED
    projected heavy-vs-center floor alone      NOT AUTHORIZED
    norm floor alone                           NOT AUTHORIZED
    raw Delta-a MSE                            REJECTED
    bounded gradient budget                    RETAIN
    matched u20 support                        RETAIN
    AI-C2                                      BLOCKED
    AI-H2                                      BLOCKED

## Next justified target

The next retained quantity should correspond directly to the multi-preference authority geometry that the gate measures.

Preferred diagnostic/training candidate:

    simplex edge-margin preservation

For preference pairs i,j:

    d_ij^theta(s)
      = ||pi_theta(s,w_i) - pi_theta(s,w_j)||

Use an asymmetric floor relative to u20:

    L_edge
      = E_{s,i<j}
        [ max(0,
              gamma * d_ij^20(s)
              - d_ij^theta(s))^2 ]

This does not penalize authority growth.

It directly preserves:
- heavy-heavy separation;
- heavy-center separation;
- the action-simplex geometry whose contraction is observed.

A separate local tangent term should not be added in the same first treatment.

First test whether edge-margin preservation alone recovers:
- pairwise authority >= .90;
- tangent authority >= .90 through induced local geometry preservation.

Only if pairwise passes but tangent still fails should tangent-Jacobian retention be considered separately.

## Stop conditions

Do not:
- increase rho;
- increase beta0;
- enlarge the replay set;
- combine edge and Jacobian losses immediately;
- open AI-H2.

Change only the retained functional quantity in the next treatment.
