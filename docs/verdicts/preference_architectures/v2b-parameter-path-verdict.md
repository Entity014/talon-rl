# V2-B Paired Parameter-Path Verdict

Status: FROZEN — LATE PATH DIRECTION DIVERGENCE, NOT SIMPLE STEP-LENGTH AMPLIFICATION

Date: 2026-09-24

## Question

Does lambda=1 cause the late Angular collapse simply by taking larger actor parameter steps, or does it drive the shared policy along a different optimization direction?

## Blockwise displacement

Actual checkpoint displacement norms were measured over u10->25, u25->50, and u50->75 for both paired training arms.

During the decisive u50->75 interval:

| block | lambda=.95 displacement | lambda=1.0 displacement |
|---|---:|---:|
| shared body | 0.7212 | 0.7236 |
| direct input | 0.3275 | 0.3156 |
| actor head | 0.1679 | 0.1662 |
| embedding | 0.0370 | 0.0465 |
| FiLM | 0.1298 | 0.1411 |
| log_std | 0.0104 | 0.0089 |

The dominant shared-body/head/direct-path displacement magnitudes are nearly equal. Therefore the lambda=1 semantic collapse is not explained by a globally larger parameter travel distance.

## Path-direction divergence

Cosine between control and treatment displacement vectors decreases across training:

| block | u10->25 | u25->50 | u50->75 |
|---|---:|---:|---:|
| shared body | 0.690 | 0.591 | **0.473** |
| direct input | 0.700 | 0.615 | **0.477** |
| actor head | 0.741 | 0.706 | **0.610** |
| FiLM | 0.783 | 0.804 | **0.645** |
| embedding | 0.890 | 0.946 | **0.725** |
| log_std | 0.890 | 0.859 | **0.468** |

The strongest path divergence occurs in the exact u50->75 interval where lambda=1 Angular semantics change from PASS (.75/.75) to complete failure (.00/.00).

## Interpretation

Static lambda=1 gradients have larger raw norms, but Adam + gradient clipping prevent that from becoming a proportionally larger net parameter displacement.

What accumulates instead is a different direction through shared parameter space.

This is consistent with optimizer-history / sequence effects:
- lambda changes per-update gradient geometry and magnitude
- Adam moment state integrates that history
- policy changes alter subsequent on-policy state/action distributions
- later updates therefore follow a different shared-policy path even while instantaneous A-heavy and mixed-batch Angular gradients remain locally valid

Optimizer-state history is a plausible mechanism but is not directly proven because the existing checkpoints did not serialize Adam moments.

## Causal status

- Simple lambda=1 step-length overshoot: NOT SUPPORTED.
- Late u50->75 path-direction divergence: ESTABLISHED.
- Coincidence of path divergence with Angular collapse: ESTABLISHED.
- Instantaneous Angular gradient failure: REJECTED.
- Adam/history + on-policy feedback as cause of direction divergence: PLAUSIBLE, not yet isolated.

## Decision

Do not justify a lower learning rate solely from the current evidence; displacement magnitude is not larger.

The next causal test, if continued, should isolate optimizer-history from on-policy visitation. The cleanest experiment is a paired replay that serializes/logs optimizer moments and per-update parameter deltas from the beginning, or a controlled optimizer comparison that preserves lambda=1 gradients while removing momentum/history (e.g. diagnostic SGD or Adam-state reset) before any method claim.