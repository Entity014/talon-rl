# S Sign-Context Audit Verdict

Status: **FROZEN — NO SIMPLE TRANSFERABLE PRE-RESPONSE CONTEXT PREDICTOR; CONTEXT-CONDITIONED PENALTY NOT AUTHORIZED**
Date: 2026-09-25

## Question

Can source-state or early control-context variables predict whether changing the preference from center to S-heavy will improve or worsen the exact S objective?

Define the matched-pair semantic effect:

    Delta S = S_reward(pi(w_S)) - S_reward(pi(w_C))

Correct:

    Delta S > 0

Wrong/reversed:

    Delta S < 0

The causal unit is a matched S-heavy / center trajectory pair from the same reset and lane.

## Dataset

Validated u30 checkpoint:

    runs/authority_isolated_simplex_edge_retain-2026-09-25/model_30.pt

Four matched reset suites:

    seed 840001
    seed 840002
    seed 840003
    seed 840004

Eight lanes per suite:

    32 matched trajectory pairs

Labels are computed from the full-horizon exact weighted S reward term:

    action_rate_l2, weight -0.01

No evaluator proxy is used for the label.

## Label distribution

Correct semantic sign by suite:

    suite 0: 1 / 8
    suite 1: 6 / 8
    suite 2: 4 / 8
    suite 3: 2 / 8

Overall:

    13 / 32 correct

This is important because sign reversal occurs within suites, not only between suites.

Therefore suite identity alone is not a sufficient explanation.

## Predictor windows

Two leakage-controlled windows were tested.

### Strict source state

Features taken at reset before the first policy action.

Families:
- command / target velocity and yaw;
- projected gravity / orientation;
- base linear velocity;
- base angular velocity;
- joint velocity;
- previous action state;
- contact fraction.

### Early control-context

Features averaged over center-policy steps t0-t3 only.

The S-heavy trajectory is not used for predictor features.

This permits limited locomotion-phase/context information while avoiding features created by the S treatment itself.

## Validation

Leave-one-suite-out:

    train on 3 suites = 24 lane pairs
    test on held-out suite = 8 lane pairs

Each suite is held out once.

A simple L2-regularized logistic regression is fitted after train-fold-only standardization.

Feature-family models are evaluated separately to avoid high-dimensional feature fishing.

## Univariate source-state signal

Strongest strict source-state correlations with continuous Delta S:

    projected gravity x       rho = -0.290
    command x                 rho = -0.152
    command y                 rho = +0.084
    command speed             rho = +0.065
    command yaw               rho = +0.060

No source feature has a strong association.

The largest absolute Spearman correlation is only .29.

## Univariate early control-context signal

Strongest t0-t3 center-context correlations:

    base linear y             rho = +0.342
    leg velocity LR asymmetry rho = -0.198
    previous action norm      rho = -0.182
    base linear z             rho = +0.161
    command x                 rho = -0.158

Again no feature provides a strong, clean threshold.

The strongest feature, base linear y, is only borderline as a descriptive association and is not sufficient for a method.

## Cross-suite prediction

### Strict source state

Mean leave-one-suite-out balanced accuracy:

    command              .283
    orientation          .548
    linear velocity      .500
    joint velocity       .500
    action context       .500
    contact              .406
    all features         .375

No source-state family gives reliable held-out prediction.

Orientation is the nominal best at .548, essentially weak.

### Early center context

Mean leave-one-suite-out balanced accuracy:

    command              .336
    orientation          .254
    linear velocity      .551
    joint velocity       .589
    action context       .469
    contact              .406
    all features         .454

The nominal best is early joint-velocity context:

    mean balanced accuracy = .589

But fold results are unstable:

    held-out suite 0   .86
    held-out suite 1   .33
    held-out suite 2   .50
    held-out suite 3   .67

Pooled AUC:

    .389

Therefore this is not a transferable predictor.

The all-feature model also fails:

    mean balanced accuracy = .454
    pooled AUC             = .401

Adding more handcrafted context does not solve the problem.

## Contact/support hypothesis

Contact fraction by itself:

    balanced accuracy = .406

Therefore coarse support/contact phase does not predict semantic sign.

More elaborate handcrafted contact-phase mining is not justified from the current evidence.

## Interpretation

Supported:

    S semantic sign reversal is real.
    Sign reversal occurs within the same reset suite.
    Some weak descriptive associations with early dynamics exist.

Not supported:

    command regime as a transferable predictor;
    initial orientation as a transferable predictor;
    base velocity as a transferable predictor;
    joint-velocity context as a transferable predictor;
    coarse support/contact phase as a transferable predictor;
    a simple combined context classifier.

The best early-context family does not generalize reliably across held-out suites.

Therefore the current evidence does not support a simple rule of the form:

    if context X, S-heavy is semantically correct;
    else S-heavy reverses.

## Decision

    context-conditioned semantic penalty      NOT AUTHORIZED
    handcrafted context threshold             NOT AUTHORIZED
    additional feature-family mining          STOP
    H2a                                       FAIL
    H2b                                       NOT AUTHORIZED

## Consequence

The S semantic failure is now constrained more tightly:

    reward definition mismatch         rejected
    evaluator proxy mismatch           rejected
    lack of preference authority       rejected
    simple pre-response context rule   rejected

Remaining interpretation:

    PPO has learned an S-conditioned action mapping whose local semantic effect is not globally consistent across the visited state distribution.

The inconsistency is not captured by a low-dimensional handcrafted context variable tested here.

## Next research-level question

Do not continue handcrafted feature mining.

The next useful question should move from:

    "which context feature predicts sign?"

to:

    "why does optimization of the local S objective fail to produce a globally sign-consistent preference-conditioned response?"

A clean next diagnostic should operate directly on the learned policy/objective differential rather than on manually selected state features.

Candidate read-only gate:

    local S semantic gradient alignment audit

At matched states, compare:

1. policy preference tangent in the S direction:

       da/dw_S

2. local gradient of the exact S reward/objective with respect to action:

       d r_S / da

3. their directional alignment:

       g_S(s)
         = (d r_S / da)^T (da/dw_S)

Interpretation:

    g_S > 0
      increasing S weight locally improves S reward

    g_S < 0
      increasing S weight locally moves action in the wrong semantic direction

Then ask whether negative local semantic-gradient alignment recurs in the regimes/lanes that show negative trajectory-level Delta S.

This moves the analysis from descriptive context prediction to the actual differential semantic mechanism.

Only if that local differential misalignment is confirmed should a training-side semantic alignment method be considered.
