# Authority-Isolated Residual Lane Verdict

Status: **FROZEN — RESIDUAL FAILURE LOCALIZED TO COORDINATE-SPECIFIC SUPPORT MARGIN**
Date: 2026-09-25

## Exact residual lane

    suite 3
    preference C
    lane 0

Aligned smooth-compression traces reproduce:

    control   fail @ 14
    c=2.5     survive
    c=2.0     fail @ 15
    c=1.75    fail @ 15

## Step-aligned trace

The state trajectories remain nearly coincident through the pre-contact phase. Around t=14-15, the dominant dynamic action coordinate is FL_calf, which rapidly changes sign for recovery.

However, FL_calf is not the causal difference between c=2.5 and c=2.0: its critical action is nearly the same in both arms.

The systematic difference is that c=2.0 reduces the magnitude of many saturated coordinates from about +/-0.987 to +/-0.964.

## Coordinate rescue

Starting from failing c=2.0 and replacing coordinates with their c=2.5 values on the same current state:

    all coordinates             survive
    FL_hip only                survive

All other single-coordinate replacements fail at step 15.

Group replacements:
    hips                        survive
    FR leg                      survive
    front legs                  survive
    thighs only                 fail
    calves only                 fail
    rear legs                   fail

Thus FL_hip alone is sufficient to rescue the residual lane.

## Reverse causal test

Starting from surviving c=2.5:

- replacing FL_hip by c=2.0 for a single timestep anywhere from t=0..15 does not cause failure;
- replacing FL_hip by c=2.0 continuously from t=0..15 causes failure at step 15.

This rules against a single instantaneous recovery pulse and supports a sustained support-margin mechanism.

## Phase localization

Continuous FL_hip weakening windows in the c=2.5 survivor:

    weakened window    result
    t=0..3             survive
    t=4..7             FAIL @ 15
    t=8..11            survive
    t=12..15           survive

Longer windows containing t=4..7 also fail:

    t=0..7             FAIL
    t=4..11            FAIL
    t=0..11            FAIL
    t=4..15            FAIL

while t=8..15 survives.

Therefore the decisive phase is t=4..7.

## FL hip magnitude in decisive phase

    t      c=2.5 action   c=2.0 action   c=1.75 action
    4       0.9858          0.9636          0.9411
    5       0.9847          0.9626          0.9404
    6       0.9823          0.9603          0.9384
    7       0.9804          0.9584          0.9367

The state differences during this phase are initially very small, but the sustained ~0.02-0.025 FL-hip action reduction moves the closed-loop trajectory across the later base-contact boundary.

## Interpretation

The residual failure is not explained by:
- aggregate action norm alone,
- global saturation fraction,
- FL-calf responsiveness,
- one critical timestep,
- or insufficient preference authority.

It is best described as a coordinate-specific closed-loop support-margin / contact-bifurcation problem.

The actor is operating close enough to a stability boundary that a small sustained reduction in FL-hip command during the early-mid descent phase determines whether base contact occurs roughly eight steps later.

## Training implication

Do not use a generic action-magnitude penalty.

Do not target zero saturation globally.

A training-side headroom repair should reduce pathological extreme logits while preserving actuator-specific support magnitude where needed. The first candidate should therefore be tail/headroom shaping that is:
- selective in the extreme-logit region,
- compatible with per-coordinate action scale,
- validated specifically on FL-hip support margin,
- and constrained to preserve preference authority.

A per-action or actuator-aware headroom formulation is better supported by this audit than a single global shrinkage coefficient.

AI-C2/H2 remain blocked until such a repair passes robustness and authority revalidation.
