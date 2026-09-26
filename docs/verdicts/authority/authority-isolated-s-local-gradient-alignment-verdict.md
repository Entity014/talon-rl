# S Local Simplex-Tangent Semantic-Gradient Alignment Verdict

Status: **FROZEN — LOCAL FIRST-ORDER ALIGNMENT IS NOT SUFFICIENT; ALIGNMENT REGULARIZER NOT AUTHORIZED**
Date: 2026-09-25

## Question

Does the local preference tangent toward S align with the local exact S objective gradient in a way that predicts trajectory-level semantic correctness?

Use the simplex-valid center-to-S direction:

    d_S = w_S - w_C
        = [-.15, -.15, -.15, +.45]

At center-policy states:

    J_S(s)
      = d a(s,w_C + eps d_S) / d eps | eps=0

For the exact weighted S reward:

    r_S = -.01 ||a_t-a_{t-1}||^2

with the same previous center action:

    grad_a r_S
      = -.02 (a_t-a_{t-1})

Local semantic alignment:

    G_S(s)
      = grad_a r_S^T J_S(s)

Interpretation:

    G_S > 0
      increasing S weight locally improves the S reward

    G_S < 0
      increasing S weight locally moves action in the locally wrong direction

## Causal-clean protocol

- states come only from the center-policy trajectory;
- S-heavy post-treatment states are never used;
- previous action is the same center-policy previous action used by the reward kernel;
- preference derivative is taken along the simplex-valid center-to-S tangent;
- labels are the previously frozen matched-lane full-horizon exact S reward signs.

Dataset:

    4 suites x 8 matched lanes
    = 32 samples

Labels:

    semantic-correct 13 / 32
    semantic-wrong   19 / 32

## Numerical validation

The JVP/gradient implementation was checked against finite differences.

For eps=.01:

    max absolute error   5.30e-7
    mean absolute error  2.50e-7

For eps=.05:

    max absolute error   2.20e-7
    mean absolute error  1.17e-7

Therefore the directional derivative calculation is numerically correct.

## Correct versus wrong lanes

### t0

    correct mean G   +0.000365
    wrong mean G     +0.000300
    AUC               .522

No separation.

### t0-3

    correct mean G   -0.000541
    wrong mean G     -0.000234
    AUC               .437

Correct lanes are, if anything, more locally negative.

### t0-7

    correct mean G   -0.000888
    wrong mean G     -0.000858
    AUC               .547

Essentially identical.

### t0-11

    correct mean G   -0.000377
    wrong mean G     -0.000400
    AUC               .534

No useful separation.

### t0-15

    correct mean G   -0.000195
    wrong mean G     -0.000296
    AUC               .583

Only weak descriptive separation.

## Negative-state fraction

The fraction of early states with G_S < 0 also does not distinguish the two groups cleanly.

For t0-7:

    correct negative fraction   .673
    wrong negative fraction     .638
    AUC                          .579

Thus semantic-correct trajectories frequently begin with locally negative first-order S alignment.

Local negativity is therefore not sufficient to predict trajectory-level semantic reversal.

## Trajectory-level relation

Spearman correlation between early G summaries and full-horizon exact Delta S reward:

    G_t0             rho  +.015   p=.933
    mean G_t0-3      rho  -.135   p=.460
    mean G_t0-7      rho  +.107   p=.561
    mean G_t0-11     rho  +.055   p=.763
    mean G_t0-15     rho  +.062   p=.736

All correlations are weak and statistically uninformative.

This is the decisive result.

The local differential quantity does not track the trajectory-level semantic outcome.

## Sign-rule classification

A zero-threshold rule:

    predict correct iff G > 0

produces mean leave-one-suite-out balanced accuracy:

    t0       .632
    t0-3     .634
    t0-7     .576
    t0-11    .558
    t0-15    .638

These values might appear nominally above chance at some windows, but they are not supported by:
- AUC;
- continuous Delta S correlation;
- correct-vs-wrong group means;
- temporal consistency.

The apparent threshold accuracy arises from the imbalanced and suite-varying label structure rather than a stable semantic mechanism.

It is not sufficient evidence for an alignment rule.

## Coordinate contributions

Early correct-minus-wrong coordinate contributions vary by window.

Largest t0-3 differences:
- FL_calf
- RL_hip
- FL_hip
- RR_calf
- RL_thigh
- FR_thigh

Largest t0-7 differences:
- RL_hip
- FR_hip
- FL_hip
- FL_calf
- RL_thigh
- FR_calf

No stable coordinate subset provides a consistent local semantic-gradient explanation.

Further coordinate mining is not authorized.

## Key interpretation

The following hypothesis is rejected:

    trajectory-level S semantic reversal
      is primarily caused by
    a locally wrong first-order preference tangent
      at pre-response center states.

Instead:

    many semantically correct trajectories also have G_S < 0 locally;
    many wrong trajectories are not uniquely more negative;
    early G_S does not correlate with the final Delta S return.

Therefore local first-order semantic alignment is not sufficient to explain the H2a S failure.

## Decision

Confirmed:
- simplex-valid preference tangent computation;
- exact local S reward gradient;
- numerical derivative correctness.

Rejected:
- local first-order alignment as a trajectory-sign predictor;
- alignment regularizer as the next semantic repair;
- coordinate-specific alignment penalty.

Therefore:

    local semantic alignment repair   NOT AUTHORIZED
    handcrafted context mining        CLOSED
    H2a                               FAIL
    H2b                               NOT AUTHORIZED

## Updated causal picture

At validated u30:

    S authority exists                      YES
    S reward definition is correct          YES
    evaluation proxy matches mechanism      YES
    simple source/context predictor         NO
    local first-order semantic alignment    NO

The remaining failure is therefore more likely a multi-step closed-loop credit/trajectory effect rather than a one-step local directional error.

## Next justified research question

Do not train yet.

The next useful gate should move from one-step local differential analysis to a short-horizon causal objective response:

    if S preference is perturbed slightly at the same initial state,
    what is the sign of the resulting multi-step S return change?

Conceptually:

    D_S^H(s)
      = d/d eps
        J_S^H(pi(w_C + eps d_S) | s)

for a short matched horizon H.

This quantity includes:
- state transition effects;
- repeated policy response;
- action-rate dependence on the previous action;
- accumulation of the S objective over several steps.

A short-horizon finite-difference or differentiable rollout audit should test whether H-step preference-return sensitivity predicts the observed full-horizon Delta S sign.

If short-horizon sensitivity separates correct/wrong lanes while one-step G_S does not, then the semantic failure is a closed-loop temporal credit problem.

If it also fails, the evidence would argue that the inconsistency emerges only over longer-horizon trajectory/basin effects, and another local semantic regularizer would not be justified.
