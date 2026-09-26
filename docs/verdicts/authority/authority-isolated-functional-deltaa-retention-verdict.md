# Authority-Isolated Functional Delta-a Durability Retention Verdict

Status: **FROZEN — TARGET LOSS IMPROVES, AUTHORITY DURABILITY FAILS; RAW DELTA-A MSE IS INSUFFICIENT**
Date: 2026-09-25

## Question

Can bounded functional Delta-a retention against the robust u20 reference prevent post-u20 authority contraction during continued PPO learning?

## Frozen setup

Both arms start from the exact robust u20 continuation state and keep:
- authority-isolated actor
- wide critic
- actor optimizer state
- lambda=.95
- projected PPO/tail repair
- kappa=.05
- reset/training distribution
- preference schedule
- critic support/head-refresh contract
- 10 continuation updates

CONTROL:
    existing continuation

TREATMENT:
    same
    + bounded functional Delta-a retention

## Reference support

Frozen u20 visited-state support:
- suite2 / suite3
- held-out suite4 / suite5 / suite6
- T/A/O/S/C rollout origins
- initial / early / late phases
- 128 states per origin x phase
- 1,920 states total

Per update treatment sample:
- 8 states per origin x phase cell
- 120 states total
- all T/A/O/S heavy-vs-center responses
- 480 Delta-a pairs

## Retention target

    Delta a_theta(s,w)
      = pi_theta(s,w) - pi_theta(s,w_C)

Loss:

    L_retain
      = E ||Delta a_theta - Delta a_u20||^2

## Gradient budget

Frozen validated budget:

    rho   = 0.25
    beta0 = 2.497041993384243

Observed weighted retention/base-gradient ratio:
- always <= .25
- mean = .1292
- several updates beta-cap limited
- one update reached the .25 norm budget exactly

PPO ratio remained exact.
Training-rollout termination fraction remained 0.

Thus optimization hijack and immediate robustness collapse are not explanations.

## Control reproduction

CONTROL reproduces the previously observed u20 -> u30 drift:

    fixed-probe pairwise retention     0.71465
    fixed-probe tangent retention      0.76466
    train survival                     1.00
    PPO ratio error                    0

This validates the paired implementation.

## Treatment endpoint

Historical fixed probe:

    pairwise retention:
      control  0.71465
      retain   0.72920

    tangent retention:
      control  0.76466
      retain   0.77738

The treatment produces only a small improvement and remains far below the .90 gate.

## Primary matched-support audit

On frozen u20 visited-state support:

CONTROL:

    Delta-a MSE vs u20                 0.05055
    pairwise retention                 0.90605
    tangent retention                  0.86152
    functional-specific RMS retention  0.90517
    parameter rank                     3
    functional rank                    4
    authority gate                     FAIL

TREATMENT:

    Delta-a MSE vs u20                 0.03589
    pairwise retention                 0.82758
    tangent retention                  0.78264
    functional-specific RMS retention  0.85360
    parameter rank                     3
    functional rank                    4
    authority gate                     FAIL

Therefore the retention loss successfully improves its own target while worsening the primary authority metrics.

## Phase breakdown

Pairwise retention:

    initial:
      control  .875
      retain   .830

    early:
      control  .899
      retain   .840

    late:
      control  .904
      retain   .804

Tangent retention:

    initial:
      control  .803
      retain   .800

    early:
      control  .988
      retain   .925

    late:
      control  1.050
      retain   .943

The treatment does not merely miss one phase. The pairwise regression recurs across initial, early, and late support.

## Why lower MSE does not imply stronger authority

Heavy-vs-center Delta-a decomposition shows that treatment improves directional alignment to the u20 response but does not preserve response amplitude sufficiently.

Median cosine to frozen u20 Delta-a:

    T:
      control  .737
      retain   .906

    A:
      control  .299
      retain   .481

    O:
      control  .276
      retain   .515

    S:
      control  .467
      retain   .667

Thus treatment improves reference-direction fidelity on every heavy axis.

However mean Delta-a norm ratio relative to u20 is:

    T:
      control  .852
      retain   .852

    A:
      control  1.071
      retain   .862

    O:
      control  .855
      retain   .776

    S:
      control  1.223
      retain   1.153

The most important regression is O, whose authority amplitude contracts further under treatment. A also moves from slight overshoot to substantial contraction.

Therefore raw squared-error retention conflates:
- directional fidelity,
- response amplitude,
- and coordinate/state weighting.

It can reduce Euclidean error by rotating responses toward the reference while still allowing authority magnitude/margins to shrink.

## Causal interpretation

The previous authority-durability diagnostic remains valid:

    family-path functional response is the post-u20 blocker.

But the first retention formulation is insufficient:

    raw Delta-a MSE
      !=
    authority durability

This is another proxy-vs-mechanism separation:
- causal localization was correct;
- the first functional proxy does not enforce the authority quantity required by the gate.

## Decision

    bounded gradient machinery                PASS / RETAIN
    matched u20 support construction          PASS / RETAIN
    raw Delta-a MSE retention                 FAIL
    authority durability gate                 FAIL
    formal AI-C2 rerun                        NOT AUTHORIZED
    AI-H2                                     BLOCKED

Because the primary authority gate fails, expensive fresh critic and semantic endpoint gates are not promoted for this treatment.

## Next justified question

Do not increase rho or beta0 and do not enlarge the replay set by default.

The next diagnostic should separate the two components explicitly:

    direction retention
    versus
    authority-amplitude / margin retention

Candidate future targets, only after a predeclared fixed-policy audit:
- normalized Delta-a direction + separate norm/margin floor;
- pairwise heavy-vs-center authority-margin preservation;
- tangent-Jacobian norm preservation on matched states.

The target should preserve the experimentally required action-response magnitude rather than only minimize raw Euclidean response error.
