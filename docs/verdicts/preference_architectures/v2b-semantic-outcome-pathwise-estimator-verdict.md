# V2-B Semantic-Outcome Pathwise Estimator Verdict

Status: FROZEN — LOCAL PATHWISE / REPARAMETERIZED FINITE-DIFFERENCE ESTIMATION DOES NOT PRODUCE A REPRODUCIBLE SEMANTIC-OUTCOME REHEARSAL-GRADIENT DIRECTION.

Date: 2026-09-24

## Frozen contract
- semantic-outcome content: unchanged
- reference distribution: unchanged
- kappa = 0.5
- rho = 0.25
- activation rule: unchanged
- architecture / critic / lambda: unchanged
- fixed-policy audit only; no training

## Estimator replacement
The Isaac simulation path is not autograd-differentiable end-to-end. Therefore the pathwise family is implemented consistently with C38-C43:
- common-random-number stochastic rollouts,
- two-sided finite differences on the pre-tanh action path,
- shared perturbations for heavy and center branches,
- local chain rule from estimated outcome sensitivity back through the preference-conditioned actor,
- shared log-std perturbation included,
- semantic rehearsal scalar remains the same heavy-vs-center objective/physical margin with the same kappa deficit weighting.

A smoke test exposed and fixed a V2-B API issue: V2-B must use _pre_tanh_dist_with_preference(obs,w), not the inherited _pre_tanh_dist path.

## Fixed-policy reproducibility

### u4
Combined:
- pairwise cosine mean = 0.0148
- pairwise cosine min = -0.0634
- cosine-to-mean mean = 0.4345
- norm CV = 0.669
- SNR = 0.613

Per axis:
- S pairwise cosine = -0.0582, SNR = 0.513
- O pairwise cosine = 0.0276, SNR = 0.612
- T pairwise cosine = -0.0119, SNR = 0.579

### u7
Combined:
- pairwise cosine mean = -0.0156
- pairwise cosine min = -0.1004
- cosine-to-mean mean = 0.4246
- norm CV = 0.570
- SNR = 0.569

Per axis:
- S pairwise cosine = 0.0190, SNR = 0.611
- O pairwise cosine = -0.0200, SNR = 0.578
- T pairwise cosine = -0.0098, SNR = 0.560

## Interpretation
Replacing likelihood-ratio score estimation with a local pathwise finite-difference estimator does not recover a stable normalized rehearsal direction.

The failure therefore cannot be attributed to score-function variance alone. The same semantic-outcome target remains direction-unstable under a substantially different estimator family.

This strengthens the hypothesis that the remaining difficulty is tied to one or both of:
- high local sensitivity / non-smooth geometry of the multi-step semantic-outcome target,
- the semantic-outcome target itself not admitting a stable local policy gradient on the tested reference distribution.

## Decision
- bounded rehearsal budget: RETAIN / SOLVED
- semantic content: still not rejected as a memory object
- score-function family: REJECTED under tested budget
- local pathwise FD family: REJECTED as a reproducible rehearsal-gradient estimator under the same fixed-policy gate
- do not rerun Delta-a vs semantic-outcome training
- next justified estimator-family candidate: deterministic/local outcome surrogate fit on the same reference rollouts, with held-out surrogate fidelity and fixed-policy direction reproducibility required before training

The next surrogate must predict the same semantic heavy-vs-center outcome margins; changing the retained semantic target is not authorized by this result.
