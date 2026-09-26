# S Semantic Alignment Audit Verdict

Status: **FROZEN — REWARD/PROXY MISMATCH REJECTED; S POLICY SEMANTICS ARE CONDITIONAL/INCONSISTENT**
Date: 2026-09-25

## Question

Why does S-heavy fail H2a semantic validity at the validated u30 checkpoint?

Candidate explanations:
1. reward-definition mismatch;
2. evaluation-proxy mismatch;
3. preference authority expressed in behavior dimensions unrelated to smoothness;
4. conditional/inconsistent S semantic mapping.

## Frozen checkpoint

    runs/authority_isolated_simplex_edge_retain-2026-09-25/model_30.pt

No training updates were performed.

## S reward definition

The MORL S objective contains exactly one term:

    control_smoothness = action_rate_l2

Isaac Lab kernel:

    action_rate_l2 = sum_j (a_tj - a_{t-1,j})^2

Reward weight:

    -0.01

Therefore the S objective does not combine torque, joint acceleration, contact smoothness, or other constituents.

Those terms are outside the frozen S objective:
- dof_torques_l2: regularizer
- dof_acc_l2: auxiliary
- feet_air_time: auxiliary/contact term
- lin_vel_z_l2: constraint

Thus multi-term reward-definition masking is rejected.

## Evaluator proxy alignment

H2a physical proxy:

    ||a_t - a_{t-1}||_2

Training kernel:

    sum_j (a_tj-a_{t-1,j})^2

These are not identical, so a proxy mismatch was plausible.

However matched S-heavy versus center evaluation shows the exact training kernel fails with the same 2/4 suite pattern.

S-heavy - center:

Suite 0:
    weighted S reward delta    -0.000373   wrong
    squared action-rate delta  +0.03733    worse
    L2 action-rate delta       -0.000664   slightly better

Suite 1:
    weighted S reward delta    +0.001085   correct
    squared action-rate delta  -0.10848    better
    L2 action-rate delta       -0.03819    better

Suite 2:
    weighted S reward delta    +0.000077   correct
    squared action-rate delta  -0.00770    better
    L2 action-rate delta       +0.00819    slightly worse

Suite 3:
    weighted S reward delta    -0.002878   wrong
    squared action-rate delta  +0.28776    much worse
    L2 action-rate delta       +0.11788    worse

Exact S reward correctness:

    2 / 4 suites

Therefore the H2a S failure is not caused by using an L2-norm evaluator instead of the squared training kernel.

## Other smoothness proxies

Mean S-heavy minus center:

    L2 action rate          +0.02181
    squared action rate     +0.05223
    L1 action rate          +0.04670
    max-coordinate delta    +0.01426
    action jerk L2          +0.02565
    joint-velocity norm     +0.09463
    joint-acceleration norm +3.85798
    torque norm             +0.21780

Correct-fraction versus center:

    weighted S reward       .50
    L2 action rate          .50
    squared action rate     .50
    L1 action rate          .25
    max-coordinate delta    .50
    action jerk             .50
    joint velocity          .25
    joint acceleration      .25
    torque                  .00

There is no evidence that S-heavy is producing a hidden form of globally smoother motion that the action-rate evaluator misses.

Evaluation-proxy mismatch is therefore rejected as the primary explanation.

## Joint-level action-rate redistribution

Mean S-heavy minus center squared action-rate contribution:

    FL_hip      +.00512
    FR_hip      +.00321
    RL_hip      -.01038
    RR_hip      +.00875

    FL_thigh    +.00013
    FR_thigh    +.00586
    RL_thigh    +.02319
    RR_thigh    +.01743

    FL_calf     +.00299
    FR_calf     -.00090
    RL_calf     +.00386
    RR_calf     -.00704

Largest absolute changes:
- RL_thigh
- RR_thigh
- RL_hip
- RR_hip
- RR_calf
- FR_thigh

Thus S preference does alter action-rate geometry, but improvements on some joints are outweighed by larger worsening on others.

This is semantic redistribution, not absence of authority.

## Neighboring simplex points

The S direction is context-dependent.

Suite 1:
- increasing S weight generally improves the exact action-rate objective on T-S and A-S paths;
- S endpoint is among the smoothest points.

Suite 2:
- S endpoint improves T-S and A-S, but O-S is non-monotonic.

Suite 0:
- intermediate 75% S points can outperform the S-heavy endpoint;
- endpoint is not the optimum along several S-adjacent paths.

Suite 3:
- increasing S weight consistently worsens T-S, A-S, and O-S;
- S-heavy is the worst/near-worst point for action rate.

Therefore S semantics are not merely weak in magnitude.
The direction of the semantic effect changes with reset/state regime.

## Interpretation

Rejected:

    S reward contains hidden competing constituents      NO
    H2a action-rate proxy is the main mismatch           NO
    S-heavy produces broad hidden physical smoothness    NO

Supported:

    S preference has real functional authority           YES
    authority redistributes action-rate across joints    YES
    S objective itself improves only conditionally       YES
    S semantic direction is reset/state dependent        YES

The clean interpretation is:

    S policy semantics are conditional/inconsistent.

The same preference weight can improve smoothness in one reset regime and worsen it in another.

This is a genuine policy-semantic mapping problem at u30, not an engineering-prerequisite confound.

## Decision

    reward-definition mismatch      REJECTED
    evaluator-proxy mismatch        REJECTED
    hidden-smoothness explanation   REJECTED
    conditional S semantics         SUPPORTED

H2a remains:

    FAIL

H2b remains:

    NOT AUTHORIZED

## Next justified question

Do not train yet.

The next read-only gate should ask:

    What state/context variables predict the sign of the S semantic response?

Use matched S-heavy vs center trajectories and classify suites/states where:

    Delta S reward > 0
versus
    Delta S reward < 0

Candidate explanatory variables:
- commanded velocity / yaw regime;
- phase/contact pattern;
- base angular velocity / tilt;
- joint-velocity distribution;
- action-rate distribution before S divergence;
- support-leg phase.

The goal is to determine whether S semantics fail because:
1. the S preference-to-action mapping is context-blind where a context-dependent tradeoff is required; or
2. the normalized S reward scaling/credit is insufficient in specific state regimes.

Only after identifying a reproducible context-dependent semantic failure should a training-side semantic repair be designed.
