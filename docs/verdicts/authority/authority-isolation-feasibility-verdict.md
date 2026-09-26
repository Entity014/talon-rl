# Authority-Isolation Feasibility Verdict

Status: **FROZEN — FUNCTION-PRESERVING-ISH TRANSFER FEASIBLE; RL TRAINING NOT YET AUTHORIZED**

Date: 2026-09-25

## Question
Can the frozen V2-B actor function be transferred into a model whose shared actor trunk is state-only and whose only actor-side preference pathway is the generated policy-family block?

## Transfer result
The supervised transfer succeeds on held-out states and unseen interior preferences:

    pre-tanh RMSE                 0.005065
    mean tanh-action L2 error     0.004172
    p95 tanh-action L2 error      0.010975
    max action-coordinate error   0.043023
    preference-separation error   1.28%
    heldout/fit RMSE ratio        1.012

Independent fixed probes also generalize:

    mean action L2 error          0.005297
    preference-separation error   1.95%

## Simplex-Jacobian contract repair
The initial report incorrectly evaluated the full 4-D derivative with respect to w, even though valid preferences satisfy sum(w)=1. The off-simplex normal derivative is therefore behaviorally irrelevant.

After projecting onto an orthonormal basis of the 3-D simplex tangent space:

    held-out tangent Jacobian relative error   8.26%
    held-out tangent Jacobian cosine            0.9966

    fixed-probe tangent Jacobian relative error 10.77%
    fixed-probe tangent Jacobian cosine          0.9942

The same frozen thresholds are used (<=15% relative error, >=.95 cosine). No threshold or fitting budget changed.

## Frozen gate
All ten feasibility criteria pass after the geometry repair.

Formal verdict:

> **FUNCTION-PRESERVING-ISH TRANSFER FEASIBLE**

This does not mean exact parameter/function identity. It means the V2-B actor behavior can be transferred to the authority-isolated architecture with low held-out action error, preserved continuum preference separation, and closely matched local preference derivatives on the valid simplex manifold.

## Scientific interpretation
The direct/embed/FiLM preference pathways are not required to reproduce the trained V2-B actor function. A state-only shared trunk plus a single PF-generated conditional actor path can approximate the same behavior closely on unseen states and unseen preferences.

Therefore architectural authority isolation is feasible without accepting a large initialization-quality confound.

## Decision
No RL training is authorized by this audit itself.

A separately predeclared authority-isolated H0/H1 branch may now be considered, initialized from the transferred student rather than from a random replacement.

The next causal question would be:

> when preference dependence has only one actor-side owner, does family authority remain stable under RL updates instead of drifting back to a competing shared pathway?
