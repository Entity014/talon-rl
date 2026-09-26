# H2a — Semantic Validity at the Validated U30 Operating Point

Status: **FROZEN — H2a FAIL; T/A/O VALID, S SEMANTICS NOT RELIABLY VALID; H2b NOT AUTHORIZED**
Date: 2026-09-25

## Question

At a checkpoint where engineering prerequisites are already valid, does the continuous-preference MORL controller realize the intended behavior semantics?

This is **not** a forgetting test.

H2a asks only for semantic validity at the frozen validated operating point.

## Checkpoint

    runs/authority_isolated_simplex_edge_retain-2026-09-25/model_30.pt

This checkpoint was selected before H2a because it passed the engineering prerequisite stack:

- authority ownership;
- authority durability;
- all-simplex-edge geometry;
- wide-critic validity;
- frozen semantic-suite robustness;
- held-out robustness;
- PPO ratio contract.

No training update is performed during H2a.

## Protocol

Endpoint preferences:

    T = [.7,.1,.1,.1]
    A = [.1,.7,.1,.1]
    O = [.1,.1,.7,.1]
    S = [.1,.1,.1,.7]
    C = [.25,.25,.25,.25]

Endpoint evaluation:
- 4 matched reset suites;
- same reset seed for T/A/O/S/C within each suite;
- 64 steps;
- 8 lanes.

Semantic endpoint criterion:
- objective-direction correctness >= .75;
- physical-proxy direction correctness >= .75;
- endpoint survival >= .95.

Physical proxies:
- T: tracking error;
- A: xy angular velocity;
- O: body tilt;
- S: action rate.

Continuum:
- all six heavy-heavy simplex edges:
  - T-A
  - T-O
  - T-S
  - A-O
  - A-S
  - O-S
- alpha = 0,.25,.5,.75,1;
- matched reset within each path/suite;
- aggregate monotonicity threshold .65;
- aggregate between-endpoint threshold .65.

Center compromise:
- center objective value must lie inside the envelope spanned by the four heavy endpoints in >= .75 of objective x suite cells.

## Endpoint semantics

### T-heavy

    objective correctness       1.00
    physical correctness        1.00
    min survival                1.00

Mean versus center:

    objective delta             +0.000665
    tracking-error delta        -0.03155

Decision:

    PASS

### A-heavy

    objective correctness       1.00
    physical correctness        1.00
    min survival                1.00

Mean versus center:

    objective delta             +0.000777
    xy-angular-velocity delta   -0.03534

Decision:

    PASS

### O-heavy

    objective correctness       1.00
    physical correctness        1.00
    min survival                1.00

Mean versus center:

    objective delta             +0.008910
    tilt delta                  -0.74455 deg

Decision:

    PASS

### S-heavy

    objective correctness       0.50
    physical correctness        0.50
    min survival                1.00

Mean versus center:

    objective delta             -0.000126
    action-rate delta           +0.02181

Decision:

    FAIL

The S-heavy policy is not reliably smoother than center.

Per-suite S-heavy versus center:

Suite 0:
    objective delta             -0.0000898   wrong
    action-rate delta           -0.000664    correct but negligible
    survival                    1.00

Suite 1:
    objective delta             +0.000261    correct
    action-rate delta           -0.03819     correct
    survival                    1.00

Suite 2:
    objective delta             +0.0000185   correct
    action-rate delta           +0.00819     wrong
    survival                    1.00

Suite 3:
    objective delta             -0.000692    wrong
    action-rate delta           +0.11788     wrong
    survival                    1.00

Therefore the S failure is not attributable to endpoint robustness.

The controller survives all S endpoint evaluations, yet the S preference does not consistently produce the intended lower-action-rate behavior.

## Center compromise

    between-heavy-envelope fraction    .875
    threshold                          .750

Decision:

    PASS

The center is generally an interior compromise rather than an out-of-family extreme.

## Continuum

Across all six heavy-heavy paths:

    monotonicity fraction               .7474
    threshold                           .65

    between-endpoint fraction           .6667
    threshold                           .65

Decision:

    aggregate continuum structure PASS

However S-related continuum structure is visibly weaker and more variable than the aggregate score suggests, especially on T-S and A-S in some suites.

Therefore aggregate continuum PASS does not override the endpoint S failure.

## Robustness during H2a

All T/A/O/S/C endpoint evaluations:

    min survival = 1.00

One continuum rollout fails the global no-regression criterion:

    path      T-S
    suite     3
    alpha     0.0
    policy    exact T endpoint
    survival  .875

This is a robustness failure on a continuum-protocol reset seed, not an intermediate-preference semantic failure.

Therefore:

- H2a's global survival criterion is FAIL;
- the S endpoint semantic failure remains unconfounded because S endpoint survival is 1.00 in all four endpoint suites.

## Critic during H2a

    H32 EV mean              .04714
    negative fraction        .20625
    mean absolute bias       .04797

Predeclared critic condition:

    H32 EV mean > 0
    negative fraction <= .25

Decision:

    PASS

The semantic endpoint failure is not accompanied by critic collapse under this gate.

## Collateral

    center tracking error                1.0379
    maximum heavy/center tracking ratio  1.0111
    allowed maximum                      2.0

Decision:

    PASS

No broad tracking collapse explains the S result.

## H2a criteria

    T semantic endpoint             PASS
    A semantic endpoint             PASS
    O semantic endpoint             PASS
    S semantic endpoint             FAIL

    center compromise               PASS
    continuum monotonicity          PASS
    continuum between-endpoint      PASS
    critic validity                 PASS
    tracking collateral             PASS
    all-evaluation survival         FAIL

Overall:

    H2a SEMANTIC VALIDITY FAIL

## Causal interpretation

This is the first semantic test performed at a checkpoint where the major engineering prerequisites were already validated.

The result therefore separates two issues:

Engineering validity:
    largely solved at u30.

Semantic validity:
    incomplete.

Specifically:

    preference authority exists;
    preference simplex geometry exists;
    critic can represent the regime;
    controller is robust on the endpoint suites;

but:

    S-heavy does not reliably map to S behavior.

This means authority durability was a real blocker, but it was **not a complete explanation** of the original MORL semantic problem.

The architecture now has functional preference authority, yet one objective semantics remains weak/inconsistent.

## Forgetting interpretation

Do not call this result forgetting.

A single checkpoint cannot establish forgetting.

Correct terminology:

    H2a semantic validity at u30: FAIL

H2b semantic durability:
    NOT YET TESTED

Because H2a fails, H2b is not authorized yet under the current roadmap.

There is no reason to study semantic durability of a mapping that is not fully valid at its starting checkpoint.

## Checkpoint selection

U30 was not selected because H2a looked favorable.

It was selected before semantic evaluation by engineering eligibility:

    authority PASS
    simplex-edge PASS
    critic PASS
    robustness PASS
    held-out robustness PASS
    PPO contract PASS

This preserves the distinction between checkpoint selection and semantic outcome.

A generic methodology should define:

    eligible checkpoint
      iff engineering validation contract passes

and only then apply semantic evaluation.

The semantic evaluator must not participate in selecting the checkpoint used to test semantic validity.

## Decision

    mechanism-specific u40 mining       CLOSED
    H2a at validated u30                COMPLETE
    H2a                                FAIL
    H2b durability                     NOT AUTHORIZED
    u40 stabilization                  NOT PRIORITY
    S-semantic diagnosis               NEXT CORE QUESTION

## Next justified branch

Do not return to u40 late-training robustness.

The immediate research question is now:

    Why does S-heavy preference fail to reliably reduce action-rate / improve the S objective
    despite valid preference authority and robust endpoint behavior?

The next audit should remain read-only and semantic-specific.

Suggested first gate:

1. verify reward/metric semantic alignment for S:
   - exact normalized S objective decomposition;
   - action-rate reward term;
   - any other S constituent terms;
   - whether evaluation physical proxy matches the trained S objective;

2. compare S-heavy versus center on matched states/trajectories:
   - action rate;
   - action acceleration / coordinate contributions;
   - reward decomposition;
   - whether S authority is expressed in dimensions unrelated to the measured smoothness proxy;

3. verify that S semantics are not being masked by reward scale or conflicting reward components.

Only after establishing whether S is a reward-definition problem, evaluation-proxy problem, or policy-semantic problem should a new training intervention be considered.
