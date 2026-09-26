# V2-B Semantic-Outcome Estimator Repair Verdict

Status: FROZEN — CRN, leave-one-out centering, and 4-replica averaging do not produce a reproducible normalized rehearsal-gradient direction.

Date: 2026-09-24

## Frozen contract
- Semantic-outcome rehearsal content unchanged.
- kappa = 0.5.
- rho = 0.25.
- Same reference distribution and acquisition checkpoints.
- Architecture, critic, lambda, and training path unchanged.
- Fixed-policy audit only; no training.

## Repair ladder
1. Common random numbers (heavy vs center).
2. Leave-one-out per-window baseline centering.
3. Independent replica averaging (2 and 4 replicas).
4. Direction reproducibility gate before training.

## CRN result
CRN reduced gradient-norm variability, but did not improve reproducible direction.
At u4, combined pairwise cosine changed from about 0.024 (raw) to -0.054 (CRN), with SNR 0.520.
At u7, combined pairwise cosine was 0.0068, with SNR 0.565.
Orientation showed the same pattern.
## CRN + centering result
Leave-one-out centering gave only small directional improvement.
At u4, combined pairwise cosine was 0.010 and SNR 0.552; O cosine was 0.013.
At u7, combined pairwise cosine was -0.0095 and SNR 0.561; O cosine was 0.0129.

## Replica averaging result
Using 8 independent centered-CRN replicas:
- u4 average-4: combined SNR 0.991, norm CV 0.515, but cosine between the two independent 4-replica averaged gradients = -0.0158.
- u4 O average-4: SNR 0.994, cosine = -0.0100.
- u7 average-4: combined SNR 0.985, norm CV 0.647, cosine = -0.0376.
- u7 O average-4: SNR 0.992, cosine = -0.0297.

Thus averaging improves magnitude-based SNR while the normalized direction remains non-reproducible. Cosine-to-mean is not sufficient here because with only two averaged groups it can look moderately high even when the two group directions are nearly orthogonal.

A-axis is not separately auditable in this branch because semantic-outcome rehearsal never acquired an A reference memory; there is no active A rehearsal gradient to estimate.

## Decision
- Bounded rehearsal budget: RETAIN.
- Semantic-outcome content: UNRESOLVED / NOT REJECTED.
- CRN: insufficient.
- Leave-one-out centering: insufficient.
- Replica averaging up to 4: insufficient for direction stability.
- Current score-function semantic-outcome rehearsal formulation: PRACTICALLY INADEQUATE under the tested sample budget.
- Do not rerun Delta-a vs semantic-outcome training comparison yet.
- Next gate should change estimator family while keeping semantic content and rho=0.25 frozen.

The next estimator-family candidate should remove the discontinuous/high-variance score-function dependence on stochastic rollout outcomes rather than merely increasing replica count.
