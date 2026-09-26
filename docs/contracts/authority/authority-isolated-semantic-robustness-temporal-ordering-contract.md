# Semantic–Robustness Temporal Ordering Audit

Status: PREDECLARED — READ ONLY
Date: 2026-09-25

## Question

At the first H2 collapse transition u35 -> u40, does semantic relation degradation precede, follow, or co-emerge with closed-loop stability degradation?

No optimizer step or policy/environment modification is allowed.

## Primary checkpoints

    u35 = H2 snapshot 5
    u40 = H2 snapshot 10

Primary semantic transition:

    u35: T/A/O PASS
    u40: T/A/O FAIL

Authority remains valid at both checkpoints.

## Two evidence levels

### Level 1 — within semantic matched-reset trajectories

Use frozen semantic seeds:

    840001..840004

For each axis T/A/O/S, compare heavy preference against center under the exact same reset.

Purpose:

    determine whether semantic relation becomes wrong while those same trajectories
    still remain dynamically stable / survive.

This level is primary for temporal semantic ordering.

### Level 2 — fresh robustness transfer

Use the existing fresh audit seed family:

    preference-specific seeds 9700000 + preference offset + 113*suite

Purpose:

    determine when the same policy transition begins to lose stability margin
    on unseen reset regimes.

The known first u40 fresh failure is:

    T / seed 9700000 / base_contact

This level is not treated as the same trajectory as the semantic suite.

## Per-step semantic signals

For heavy axis i vs matched center:

Objective relation:

    r_obj_i(t) = objective_i_heavy(t) - objective_i_center(t)

Correct direction:

    r_obj_i(t) > 0

Physical relation:

    T: center_tracking_error(t) - heavy_tracking_error(t)
    A: center_ang_vel_xy(t)    - heavy_ang_vel_xy(t)
    O: center_tilt_deg(t)      - heavy_tilt_deg(t)
    S: center_action_rate(t)   - heavy_action_rate(t)

Correct direction:

    r_phys_i(t) > 0

Report both:
- instantaneous per-step relation;
- fixed 5-step trailing mean;
- cumulative relation from t=0.

No new scalar semantic reward is constructed.

## Stability signals

Record per env / per step:
- tilt_deg;
- |base angular velocity xy|;
- |base linear velocity z|;
- base height;
- joint-velocity norm;
- action norm;
- action-rate norm;
- termination flag / reason.

For u35 and u40, report:
- matched trajectory traces;
- heavy and center separately;
- first contact time;
- pre-contact 5 / 10 step windows where applicable.

## Stability-margin reference

No hand-tuned absolute threshold is introduced.

For each seed/preference/env and each stability signal, use the matched u35 trajectory as the reference.

Define normalized deterioration at u40:

    D_k(t)
      = (x40_k(t) - x35_k(t)) / (MAD35_k + eps)

for higher-is-worse signals.

For height, use absolute deviation from that trajectory's u35 median height.

A stability-divergence event is descriptive, not a safety limit:

    at least two independent stability signals exceed
    their u35 matched-reference p95 deviation
    for >=3 consecutive steps,

or base_contact occurs.

Report threshold-free raw traces alongside this event.

## Semantic-loss onset

For an axis/reset, semantic-loss onset is the first step where:
- 5-step objective relation is non-positive AND
- 5-step physical relation is non-positive,
- for >=3 consecutive evaluated windows.

Also report one-channel loss onsets separately.

This definition is frozen before inspecting traces.

## Primary ordering classification

For cases with both onsets:

    semantic-first:
      semantic onset at least 3 steps before stability onset

    robustness-first:
      stability onset at least 3 steps before semantic onset

    co-emergent:
      absolute onset difference < 3 steps

For semantic-loss cases with no stability event in the same 64-step trajectory:

    semantic-without-local-instability

For stability-loss cases with no semantic-loss event:

    instability-without-semantic-loss

## Required reporting

Primary semantic suites:
- count semantic-without-local-instability cases;
- count semantic-first / robustness-first / co-emergent cases;
- per-axis onset distributions;
- whether semantic FAIL endpoint labels can occur with survival 1.00 and no local stability event.

Fresh suites:
- first stability-divergence/contact onset;
- whether time-local semantic relation (heavy vs center) is already degraded before that onset;
- especially T/9700000.

## Decision

### SEMANTIC-FIRST SUPPORTED
Only if semantic-first dominates reproducibly across multiple axes/resets and fresh stability loss follows.

### ROBUSTNESS-FIRST SUPPORTED
Only if stability-first dominates and semantic relation remains intact until after stability divergence.

### COMMON-REGIME / CO-EMERGENT
If onsets cluster within +/-2 steps or ordering changes across matched cases.

### DECOUPLED ACROSS RESET REGIMES
If semantic relations collapse on fully stable semantic trajectories while fresh robustness failures occur only on distinct reset regimes, with no reproducible within-trajectory temporal ordering.

This last result means the u35->u40 checkpoint transition changes both semantic mapping and robustness generalization, but the data do not support either as a direct temporal cause of the other.

No training branch is authorized directly by this audit.
