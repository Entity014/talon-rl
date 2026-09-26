# S Short-Horizon Closed-Loop Preference-Return Sensitivity Verdict

Status: **FROZEN — TEMPORAL CLOSED-LOOP CREDIT PARTIALLY SUPPORTED; LOCAL ALIGNMENT REMAINS REJECTED; NO TRAINING REGULARIZER AUTHORIZED**
Date: 2026-09-25

## Question

Does S semantic sign reversal emerge only after repeated closed-loop policy/dynamics evolution, rather than from one-step local preference semantics?

Define the simplex tangent:

    d_S = w_S - w_C
        = [-.15,-.15,-.15,+.45]

For horizon H:

    D_S^H(s)
      = d/d epsilon
        J_S^H(pi(w_C + epsilon d_S) | s)
        at epsilon=0

where J_S^H is cumulative exact weighted S reward:

    r_S(t) = -.01 sum_j (a_tj-a_{t-1,j})^2

and repeated policy response, simulator dynamics, and previous-action dependence are all included.

## Source-state contract

Primary source:

    exact reset state t=0

For every suite/lane:
- same reset seed;
- same command;
- same simulator initialization;
- same previous-action history;
- plus/minus perturbations differ only in preference weight.

This avoids simulator-state teleportation and manager-history confounds.

32 matched lane samples:

    4 suites x 8 lanes

Trajectory-level semantic labels are the existing exact full-horizon S-heavy-vs-center reward signs.

## Finite difference

Primary initial screen:

    epsilon = .05

Numerical control:

    epsilon = .02

Horizon ladder:

    H = 1,4,8,16,32

Because long closed-loop rollouts may amplify arbitrarily small preference changes, an epsilon-convergence follow-up was preconditioned on any apparent long-horizon signal.

Convergence ladder:

    epsilon = .005,.01,.02,.05

## Initial horizon ladder

### H=1

    AUC correct vs wrong       .522
    Spearman vs final Delta S  .015
    sign agreement             .500

Result:

    NULL

This reproduces the exact local semantic-gradient audit conclusion.

### H=4

    AUC                       .498
    Spearman                 -.041
    sign agreement            .563

Result:

    NULL

### H=8

    AUC                       .692
    Spearman                  .263
    p                         .146
    sign agreement            .594

A weak separation begins to appear.

### H=16

At epsilon=.05:

    AUC                       .660
    Spearman                  .220
    sign agreement            .719

Signal is stronger in sign but not yet clean.

### H=32

At epsilon=.05:

    AUC                       .789
    Spearman                  .457
    p                         .0086
    sign agreement            .750

This initially appears strongly predictive.

However H32 fails the required epsilon-stability test.

## Epsilon convergence

### H=8

The result is extremely stable across epsilon.

At epsilon=.005:

    AUC                       .696
    Spearman                  .265
    p                         .143
    sign agreement            .594

Epsilon=.005 vs .01/.02/.05:

    Pearson                   > .99999
    sign agreement            1.00
    median relative difference ~.004 or less

Interpretation:

    H8 is a valid local closed-loop sensitivity estimate,
    but its trajectory-semantic discrimination is weak.

### H=16

At epsilon=.005:

    AUC                       .765
    Spearman                  .429
    p                         .0143
    sign agreement            .750

Correct lanes:

    mean D_S^16              +.138

Wrong lanes:

    mean D_S^16              -.027

This is the first horizon where a small-epsilon closed-loop sensitivity shows statistically meaningful relation to full-horizon semantic outcome.

Convergence:

epsilon .005 vs .01:

    Pearson                   .845
    sign agreement            .969
    median relative difference .053

epsilon .005 vs .02:

    Pearson                   .737
    sign agreement            .906
    median relative difference .056

epsilon .005 vs .05:

    Pearson                   .422
    sign agreement            .906
    median relative difference .259

Interpretation:

    a real small-epsilon signal exists,
    but the system is already becoming nonlinear with larger perturbations.

### H=32

At epsilon=.005:

    AUC                       .737
    Spearman                  .277
    p                         .125
    sign agreement            .688

At epsilon=.01:

    AUC                       .850
    Spearman                  .531
    p                         .0018
    sign agreement            .750

At epsilon=.02/.05 the statistics change again.

Convergence against epsilon=.005:

    epsilon .01:
      Pearson                 .652
      median relative diff    .434

    epsilon .02:
      Pearson                 .316
      median relative diff    .559

    epsilon .05:
      Pearson                -.009
      median relative diff    .835

Interpretation:

    H32 is not a stable local derivative.

The strong H32 screen at epsilon=.05 cannot be interpreted as D_S^32 at epsilon->0.

Instead it reflects finite preference perturbations entering different closed-loop trajectory basins.

## Cross-suite transfer

Using the natural sign rule:

    D_S^H > 0 => predict semantic-correct

at epsilon=.005.

H8 balanced accuracy by suite:

    suite0   .357
    suite1   .583
    suite2   .625
    suite3   .667

mean:

    .558

H16:

    suite0   .429
    suite1   .417
    suite2   .875
    suite3  1.000

mean:

    .680

Thus H16 improves aggregate discrimination, but the signal does not transfer uniformly across suites.

It is highly informative in suites 2/3 and weak in suites 0/1.

H32 sign performance appears stronger on average:

    mean balanced accuracy .753

but H32 is not epsilon-convergent and therefore is not admissible as a local sensitivity mechanism.

## Horizon consistency

At epsilon=.005:

    H8 vs H16:
      Pearson .178
      sign agreement .781

    H16 vs H32:
      Pearson .488
      sign agreement .625

    H8 vs H32:
      Pearson -.031
      sign agreement .594

The semantic signal changes substantially with horizon.

This is further evidence that the effect is genuinely temporal/closed-loop, not a scaled version of the one-step local tangent.

## Causal interpretation

Rejected:

    one-step local semantic-gradient mismatch as the main cause

Supported:

    semantic outcome acquires meaningful preference-return structure only after repeated closed-loop evolution

Specifically:

    H1        null
    H4        null
    H8        weak but numerically stable signal
    H16       meaningful small-epsilon signal
    H32       stronger finite-perturbation outcome relation,
              but local derivative no longer stable

Therefore the evidence supports a temporal closed-loop credit mechanism.

However it does **not** support a simple scalar short-horizon derivative as a universal semantic predictor.

The H16 signal is not consistent across all suites.

## Decision

    local alignment regularizer              NOT AUTHORIZED
    one-step semantic repair                 REJECTED
    temporal closed-loop credit mechanism    SUPPORTED
    H16 sensitivity as universal target      NOT YET SUPPORTED
    H32 local derivative interpretation      REJECTED
    H2a                                      REMAINS FAIL
    H2b                                      NOT AUTHORIZED

## Important distinction

The data support:

    S semantic reversal is a multi-step phenomenon.

They do not yet support:

    maximize D_S^16 everywhere.

That would overinterpret a predictor that performs well only in part of the state/reset distribution.

## Next justified question

Do not return to handcrafted state features and do not train an alignment loss.

The next read-only gate should localize **when** the H16 closed-loop semantic signal becomes informative along the nominal center trajectory.

Use matched source states from several nominal phases, for example:

    source t = 0,4,8,12,16

and evaluate a fixed small-epsilon H=16 sensitivity from each source state, restoring the full simulator state and previous-action history.

Question:

    Is there a recurrent phase where D_S^16 becomes discriminative before the eventual semantic outcome?

If a consistent phase-local signal transfers across suites, this would support temporal credit/localization.

If phase-local H16 also varies idiosyncratically across suites, stop short-horizon mechanism mining and conclude that S semantics emerge from longer, trajectory-level closed-loop structure without a compact local repair target.
