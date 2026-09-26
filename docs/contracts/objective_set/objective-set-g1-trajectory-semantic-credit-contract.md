# V3-G1 — Trajectory-Level Active-Set Semantic Credit Contract

Status: **PREDECLARED — READ-ONLY / NO TRAINING COMMIT**
Date: 2026-09-25

## Question

When an active-set scalarized update opposes the gradient of an individual objective — especially Tracking — how does that disagreement propagate into closed-loop trajectory-level semantic behavior on training-seen support?

This gate does not test held-out-cardinality generalization.

## Frozen policy point

Use the exact G0 objective-set checkpoint:

    runs/objective_set_g0_structural_parity-2026-09-25/objective_set_g0_init.pt

Audit independently:

    G1-2 training support: m in {3,4}
    G1-3 training support: m in {2,4}

Held-out cardinalities are forbidden.

## Shared on-policy source batch

For each fold:
- fit token-query critic with the validated fixed-policy support/ridge procedure on allowed cardinalities only;
- collect one H64 stochastic on-policy batch for every allowed cardinality;
- same states, pre-tanh actions, old log-probabilities, active objective sets, weights, rewards, and done masks feed all gradient arms;
- use normalized_objective_vector exactly as in G1;
- no post-hoc batch selection.

## Virtual update directions

Primary directions:

    g_T^MC
    g_mixed^MC
    g_mixed^critic

where:
- g_T^MC uses only Tracking's active-token MC64 PPO term;
- g_mixed^MC uses active-set scalarization m * sum_i w_i A_i^MC;
- g_mixed^critic uses the same scalarization with token-GAE.

Secondary sacrifice-matrix directions:

    g_A^MC
    g_O^MC
    g_S^MC

when the objective is present in the sampled allowed-cardinality support.

All directions are policy-gradient-only. Tail/retention gradients are excluded.

## Norm-matched finite virtual update

For every direction g:

    delta_theta = -eta * g / (||g|| + eps)

with shared:

    eta = 1e-4 * ||theta_actor||

No optimizer state is used.
No update is committed or saved as a training checkpoint.

## Closed-loop matched evaluation

For every virtual direction and the base model, run matched deterministic rollouts from identical reset seeds.

Evaluate only active sets belonging to training-seen cardinalities.

Primary Tracking evaluation:
- for every seen active set containing T;
- compare T-heavy against center within that same set.

Horizon ladder:

    H = 8, 16, 32, 64

At every horizon compute:

    Delta J_T(H)
        = cumulative normalized T return margin
          [T-heavy minus center]

    Delta semantic_T(H)
        = virtual T normalized margin
          minus base T normalized margin

    Delta physical_T(H)
        = virtual physical tracking margin
          minus base physical tracking margin

where physical tracking margin is:

    tracking_error(center) - tracking_error(T-heavy)

so larger is better.

Also report survival and body-stability descriptors:
- termination fraction;
- angular-velocity XY;
- body tilt.

These are descriptive unless survival drops below 0.95.

## General per-objective sacrifice matrix

For each semantic objective i in {T,A,O,S} and each MC update direction j in {T,A,O,S,mixed}:

    M_ij(H) = semantic_margin_i^virtual(H) - semantic_margin_i^base(H)

Evaluate cells only when objective i is active in the evaluated seen set and gradient j exists.

Aggregate separately by fold and horizon.

The primary sacrifice matrix uses normalized objective margin.
A parallel physical-proxy matrix is also reported.

## Return-direction diagnostic

For every per-objective MC direction g_j, report the corresponding objective-return change:

    Delta J_j(H)

This prevents interpreting a semantically bad update as a gradient bug when it actually improves the optimized return while worsening the semantic proxy.

## Primary decision tree

### A — multi-objective interaction is proximate cause

Supported if, in both folds and at H32 or H64:
- g_T^MC improves/preserves T return and T semantic/physical margin;
- g_mixed^MC worsens T semantic or physical margin beyond deadband;
- the T sacrifice under mixed MC is larger than under g_T^MC.

Then active-set scalarization is a proximate cause of T sacrifice on seen support.

### B — objective-return / semantic mismatch dominates

Supported if, in both folds and at H32 or H64:
- g_T^MC improves T normalized return;
- but g_T^MC worsens T semantic physical margin beyond deadband;
- mixed MC is not required for the failure.

Then objective-return credit itself is not semantically sufficient.

### C — critic error + interaction both contribute

Supported if:
- mixed critic is consistently worse than mixed MC on T semantic margin;
- but mixed MC still worsens T in at least one fold/horizon;
- critic-vs-MC separation is therefore real but does not eliminate interaction.

### D — no compact gradient-level explanation

Supported if:
- A/B/C do not transfer across both folds;
- or sign/classification changes materially across H8/H16/H32/H64;
- or sacrifice matrices are heterogeneous with no stable T-specific pattern.

Then stop actor-gradient mechanism mining.

## Deadbands

For each base semantic/physical margin:

    tau = max(1e-6, 0.05 * |base margin|)

A virtual effect is:
- improve if Delta > +tau;
- degrade if Delta < -tau;
- neutral otherwise.

## Stop rule

If outcome is D:
- no further actor-gradient explanation branch;
- freeze conclusion that the G1 training objective is not semantically stable on seen support under the present formulation;
- do not retrain G1;
- do not open G2.

If A/B/C is stable across folds:
- authorize exactly one mechanism-matched follow-up diagnostic or repair branch.

## Non-goals

Do not:
- use held-out cardinalities;
- change reward definitions;
- add objective-specific losses;
- change architecture;
- commit virtual updates;
- tune eta after observing results;
- call MC semantic ground truth;
- reopen G1 training before this gate closes.
