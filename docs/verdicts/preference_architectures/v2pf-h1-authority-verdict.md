# V2-PF H1 Policy-Family Authority Verdict

Status: **FROZEN — H1 FAIL; H2 NOT AUTHORIZED**
Date: 2026-09-25

## Formal verdict
The predeclared H1 gate does **not** pass.

Operational classification:

> **Non-degenerate policy-family geometry learned, but incremental causal preference authority is not retained strongly enough at the final H1 checkpoint.**

This is not the same failure mode as V2-H.

## Foundation validity
All foundation checks pass at update 75:
- early critic EV = 0.497, negative fraction = 0.00;
- late critic EV = 0.354, negative fraction = 0.05;
- combined negative-EV fraction = 0.025;
- max PPO pre-update ratio error = 4.57e-5;
- last-10 termination fraction = 0.025;
- coefficient-output and full-rank basis gradients are both clearly nonzero.

Therefore the H1 result is interpretable as an architecture/mechanism result rather than a critic/PPO/safety failure.

## Parameter-family geometry
Preference coefficients separate strongly:

    mean heavy coefficient distance = 0.0722

Generated full-rank parameter realizations also separate:

    mean heavy parameter distance = 0.2140

After subtracting the common generated parameter component:

    centered parameter s2/s1       = 0.293
    centered effective rank         = 3
    largest centered energy share   = 0.898
    specific-energy fraction        = 0.256

All frozen centered-parameter geometry criteria pass.

The local parameter-manifold Jacobian is also clearly non-degenerate:

    Frobenius norm = 0.503
    s2/s1          = 0.296
    effective rank = 3

Thus V2-PF does learn a multidimensional preference-indexed parameter manifold.

## Functional policy-family geometry
The generated family residual has substantial action authority:

    mean ||Delta a_PF|| = 0.0677

Its preference-centered functional geometry is also non-degenerate:

    centered functional s2/s1      = 0.243
    centered effective rank         = 3
    functional specific fraction    = 0.262
    specific/common RMS ratio       = 0.356

Therefore the family block is neither dead nor purely common.

## The single failed H1 criterion
The frozen contract required masking the family block to reduce either global pairwise preference action separation or the preference Jacobian by at least 5% at update 75.

Observed:

    pairwise action separation
      full       = 0.08418
      PF masked  = 0.08030
      reduction  = 4.61%   < 5%

    preference Jacobian Frobenius norm
      full       = 0.04415
      PF masked  = 0.04339
      reduction  = 1.72%   < 5%

Thus:

    mask_reduces_preference_authority = FAIL

Every other frozen H1 criterion passes.

## Temporal authority behavior
The masking criterion is not merely absent throughout training. It appears transiently and then weakens:

| update | parameter rank | parameter specific fraction | functional rank | functional specific fraction | pairwise authority reduction | Jacobian authority reduction |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 3 | 0.154 | 3 | 0.105 | 2.49% | 0.96% |
| 25 | 3 | 0.306 | 3 | 0.383 | 3.14% | 1.31% |
| 50 | 2 | 0.245 | 3 | 0.225 | **13.03%** | **6.62%** |
| 75 | 3 | 0.256 | 3 | 0.262 | **4.61%** | **1.72%** |

At update 50 the family block temporarily satisfies the 5% masking requirement, but this authority is not retained to the final predeclared H1 checkpoint.

No intermediate checkpoint may rescue the final gate post hoc.

## Comparison with V2-H
V2-H failed because generated parameter deltas remained almost collinear despite a multidimensional coefficient map.

V2-PF fixes that failure:

- substantial full-rank generated block;
- centered parameter rank = 3;
- centered functional rank = 3;
- specific parameter/function energy around 26%.

However, solving parameter-manifold diversity does not produce stable incremental control of the deployed preference-to-action mapping beyond the already-strong V2-B conditioning path.

This is a materially different negative result.

## Decision
H2 semantic accumulation is **not authorized** under the frozen contract.

Do not tune:
- basis count;
- family hidden dimension;
- coefficient network depth;
- learning rate;
- masking threshold.

Do not reinterpret the transient update-50 authority as H1 success.

The V2-PF escalation branch is closed for thesis-critical method selection unless a genuinely new formulation hypothesis is opened explicitly.

## Scientific conclusion
> A substantially larger one-model preference-generated policy block successfully learned multidimensional, preference-specific parameter and functional realizations. Nevertheless, its incremental causal contribution to overall preference authority was not stable: masking the generated block reduced pairwise action separation by only 4.61% and preference-Jacobian magnitude by 1.72% at the final H1 checkpoint, below the predeclared 5% gate. The result therefore rejects the simple hypothesis that insufficient conditional parameter capacity was the remaining bottleneck.

## Thesis wording
> Increasing preference-conditioned parameter-generation capacity from a low-rank action-head delta to a full-rank 144→128→12 generated policy block produced a genuinely multidimensional policy-parameter family (centered effective rank 3; specific parameter and functional energy ≈26%). However, the generated block did not retain sufficient incremental causal authority over the final preference-conditioned action mapping: its removal reduced pairwise preference separation by 4.61% and preference-Jacobian magnitude by 1.72%, below the predeclared 5% H1 criterion. Since all PPO, critic, and survival checks remained valid, the result indicates that simply increasing preference-indexed parameter capacity is insufficient to resolve the observed semantic instability.
