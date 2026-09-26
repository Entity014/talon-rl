# V3-G1 — Critic-Derived vs MC-Derived Actor Update Audit

Status: **PREDECLARED — READ-ONLY / NO TRAINING COMMIT**
Date: 2026-09-25

## Question

Does the early V3-G1 actor semantic drift arise primarily from inaccurate token-query critic credit, or from the active-set PPO objective/update geometry itself?

This gate compares two actor update directions on identical on-policy data:

    g_critic = grad L_PPO(A_token-GAE)
    g_MC     = grad L_PPO(A_MC64)

Everything except the source of credit is held fixed.

## Frozen policy point

Primary audit point is the exact G0 objective-set checkpoint:

    runs/objective_set_g0_structural_parity-2026-09-25/objective_set_g0_init.pt

This is the last point where:
- T/A/O semantics are valid;
- preference authority is valid;
- permutation invariance is exact;
- the actor has not yet undergone G1 semantic drift.

The audit is run independently for G1-2 and G1-3 training supports using seed 73101.

No held-out cardinality is used.

## Shared on-policy data

For each fold and each allowed training cardinality:
- collect one H64 stochastic on-policy batch;
- use exactly the same states, pre-tanh actions, old log-probabilities, active objective tokens, active weights, rewards, and done flags for both arms;
- reward normalization is the frozen normalized_objective_vector used by Phase 1/G1;
- scalarization is exactly:

    m * sum_i w_i A_i

where i ranges only over active objectives.

Gradients from the fold's allowed cardinalities are averaged with equal cardinality weight, matching the uniform-cardinality G1 sampler.

## Credit arm A — token-query GAE

Before collecting the audit batch, fit the token-query value basis using the validated current-policy 64-step support/ridge procedure on allowed cardinalities only.

Use:

    gamma  = 0.99
    lambda = 0.95

and active-token GAE exactly as in G1.

No held-out token/cardinality target is queried.

## Credit arm B — critic-independent MC64

Do not use token-query critic values in constructing the credit target.

For every active objective:

    G_t^MC = sum_{k=t}^{63} gamma^(k-t) r_k

with termination masking identical to the environment rollout.

Set:

    A_t^MC = G_t^MC

No learned value baseline, token-query prediction, bootstrapping value, or critic-derived correction is used.

MC64 is a critic-independent control arm, not semantic ground truth.

## Actor-gradient scope

Compare PPO policy gradients only.

Tail control and edge-retention gradients are excluded from the primary credit comparison because they are identical auxiliary machinery and would obscure the source-of-credit contrast.

They may be reported separately but are not part of g_critic or g_MC.

Actor parameters included:
- actor_body;
- actor_mean;
- family_hyper;
- family bases;
- log_std.

## 1. Credit agreement

Report:

    cosine(g_critic, g_MC)
    ||g_critic||
    ||g_MC||
    ||g_critic-g_MC|| / (||g_MC||+eps)

Also compute per-objective gradients.

For objective k:
- include only transitions whose active set contains k;
- use only that objective's PPO term;
- no mixed-objective scalarization.

Report per-objective:
- critic-vs-MC gradient cosine;
- critic and MC gradient norms;
- cosine of each per-objective gradient with the corresponding mixed gradient.

This separates:
- critic credit disagreement;
- multi-objective gradient interaction.

## 2. Functional effect

On the frozen fixed probe-state bank, compute a norm-matched virtual update for each arm.

Let theta be actor parameters and g one audit gradient.

Use descent direction:

    delta_theta = -eta * g / (||g|| + eps)

with the same parameter-space step norm for both arms:

    eta = 1e-4 * ||theta||

No optimizer state is used.

For each virtual model report:
- RMS and max |Delta pre-tanh|;
- RMS and max |Delta action|;
- cosine between critic-arm and MC-arm functional deltas.

Evaluate these at m=4 center and T/A/O/S-heavy preferences.

No virtual update is saved as a training checkpoint.

## 3. Semantic direction

Replay the frozen matched m=4 T/A/O anchor suites after each virtual update.

Primary semantic margin for objective k:

    M_k = mean_suite[
        normalized_return_k(k-heavy) - normalized_return_k(center)
    ]

and physical semantic margin:

    P_k = mean_suite[
        physical_k(center) - physical_k(k-heavy)
    ]

so larger is better for both.

For each arm report:

    Delta M_k = M_k(virtual) - M_k(base)
    Delta P_k = P_k(virtual) - P_k(base)

for T/A/O.

Also report S descriptively.

A semantic direction is counted as preserving/improving objective k when neither normalized nor physical margin decreases by more than a frozen numerical deadband:

    deadband = max(1e-6, 0.05 * |base margin|)

and at least one of the two margins improves beyond its deadband.

Because the virtual step is tiny, raw deltas and signs are primary evidence; endpoint PASS/FAIL is secondary.

## Decision rules

### Critic-derived credit is the proximate blocker

Supported only if:
1. mixed gradient cosine is low or negative:
       cos(g_critic,g_MC) < 0.5
2. MC virtual update preserves/improves at least 2 of T/A/O;
3. critic virtual update degrades at least 2 of T/A/O;
4. this qualitative separation appears in both G1-2 and G1-3 audits.

Then authorize a critic/credit repair branch.

### Active-set PPO/objective geometry is the blocker

Supported if:
1. mixed gradient cosine >= 0.8 in both folds; and
2. both virtual updates degrade at least 2 of T/A/O in both folds.

Then close the critic-only hypothesis and investigate active-set scalarization / multi-objective gradient geometry.

### Critic differs but is not sufficient explanation

If gradients differ materially but both virtual updates degrade semantics, critic error may exist but is not sufficient to explain G1 semantic drift.

### Inconclusive / noisy

If MC semantic direction is inconsistent across folds/objectives or deltas remain within deadband:
- do not authorize G1 retraining;
- escalate to trajectory-level active-set semantic credit audit.

## Non-goals / prohibitions

Do not:
- commit either virtual update;
- change architecture;
- change objective rewards;
- add objective-specific losses;
- use held-out cardinalities;
- select a favorable batch after observing results;
- treat MC return as semantic ground truth;
- open G2.

This gate diagnoses the seen-support actor-learning failure only.
