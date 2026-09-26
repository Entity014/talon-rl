# V2-B Semantic-Outcome Deterministic Surrogate Gate Verdict

Status: FROZEN — HELD-OUT SURROGATE FIDELITY FAILS; DO NOT OPEN SURROGATE REHEARSAL TRAINING.

Date: 2026-09-24

## Question

Does the existing semantic heavy-vs-center outcome target admit a stable learnable local mapping on the frozen reference distribution once simulator-derivative estimation is removed?

## Frozen contract

- semantic-outcome content unchanged
- same reference seeds / phases / environments
- kappa = 0.5
- rho = 0.25
- same competence references and activation semantics
- architecture / critic / lambda unchanged
- fixed-policy audit only
- no rehearsal training

## Surrogate construction

Two deterministic local surrogate representations were tested.

Minimal representation:
- detached heavy and center rollout context from the first 12 physical/command observation coordinates
- heavy and center mean pre-tanh policy actions as the only differentiable columns
- 48 total features
- outputs are the exact same objective-margin and physical-margin targets used by semantic-outcome rehearsal

Richer temporal representation:
- heavy and center context mean
- heavy and center context standard deviation
- heavy and center start-to-end context delta
- heavy and center mean pre-tanh policy actions as the only differentiable columns
- 96 total features
- same semantic targets

Primary model: ridge regression, lambda = 1.
Capacity check: two-hidden-layer MLP on the same representation.
Validation: leave-one-reference-seed-out, separately for S/O/T at u4 and u7.

## Minimal surrogate fidelity

Held-out explained variance (objective / physical):

u4:
- S: -0.284 / -0.198
- O: -0.238 / -0.220
- T: -0.392 / -1.112

u7:
- S: -0.273 / -0.145
- O: 0.078 / 0.002
- T: -0.006 / -1.111

The corresponding in-sample ridge EV values were substantially higher, approximately 0.4-0.8, showing that the model can fit part of the observed support but the mapping does not transfer reliably to a held-out reference seed.

The nonlinear MLP capacity check did not repair held-out generalization and was usually worse.

## Rich temporal surrogate fidelity

Adding temporal spread and delta information improved in-sample fit but did not improve held-out generalization.

Held-out ridge EV (objective / physical):

u4:
- S: -0.23 / -0.16
- O: -0.50 / -0.46
- T: -0.35 / -0.84

u7:
- S: -0.41 / -0.32
- O: -0.12 / -0.15
- T: -0.24 / -1.47

In-sample EV rose to approximately 0.6-0.8, but held-out EV remained negative. This is the signature of reference-specific fit rather than a stable learnable local mapping.

The MLP on the richer representation was also unstable across held-out seeds; several folds extrapolated catastrophically because the training support was too narrow. Those large negative EV values are treated as evidence of poor support/generalization, not as meaningful quantitative estimates.

## Gradient reproducibility diagnostic

Although fidelity already fails the primary gate, surrogate-derived gradients were also inspected.

For the minimal ridge surrogate, bootstrap-fit pairwise gradient cosine was approximately:
- u4: S 0.34, O 0.57, T 0.54
- u7: S 0.33, O 0.60, T 0.50

This is markedly better than the near-zero score-function and pathwise-FD directions, but it is not sufficient to authorize training because the surrogate is not predicting held-out semantic outcomes faithfully.

A stable gradient of a poorly generalizing surrogate is not evidence of a stable semantic gradient.

## Interpretation

The failure sequence is now:

1. score-function estimator: direction unstable
2. local pathwise finite-difference estimator: direction unstable
3. deterministic surrogate: gradient can become more reproducible, but held-out semantic prediction fails

Therefore the remaining blocker is not merely simulator derivative noise.

The current reference representation does not support a seed-generalizable local mapping from policy/action context to the retained semantic heavy-vs-center outcome margins.

This does not prove that semantic outcomes are intrinsically useless as memory content in every formulation. It does show that the current semantic-outcome rehearsal formulation, with the current reference distribution and representation, does not provide a validated learnable local geometry suitable for gradient rehearsal.

## Causal status

- bounded rehearsal budget: SOLVED / RETAIN
- score-function estimator family: REJECTED under tested budget
- local pathwise FD estimator family: REJECTED under fixed-policy reproducibility gate
- deterministic local surrogate fidelity: REJECTED
- surrogate gradient stability alone: NOT SUFFICIENT
- semantic-outcome content in the current rehearsal formulation: NOT VALIDATED
- Delta-a action-response rehearsal: remains the only validated stable rehearsal baseline in this branch

## Decision

Do not open bounded-budget surrogate rehearsal training.

Do not rerun Delta-a versus semantic-outcome rehearsal under this surrogate.

The semantic-outcome rehearsal branch should be closed in its current formulation unless a materially richer reference representation or different retained semantic object is explicitly introduced as a new method hypothesis.

Such a change would no longer be an estimator-only repair; it would be a new content/representation candidate and must be treated as a new branch rather than a continuation of the current causal comparison.
