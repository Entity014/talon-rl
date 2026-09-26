# Authority-Isolated Residual Branches A/B Verdict

Status: **FROZEN — READ-ONLY LOCALIZATION COMPLETE**
Date: 2026-09-25

## Context

The projected + active tail-descent repair reduced semantic-suite failures from 9 to 5 while preserving/growing preference authority. The remaining failures were split into two read-only audit branches.

---

## Branch A — O-heavy support-preservation audit

Question:

Does the FL/front-left support pattern reproduce across O-heavy failures strongly enough to justify a generic actuator-aware training repair?

### O-heavy support matrix

Repaired-u10 baseline:

    suite2/O survival = 0.75
    failures = lane2, lane7

    suite3/O survival = 0.875
    failure = lane0

Using frozen u75 actions as donor on the repaired actor's current states:

### suite2/O

Only FL_hip donor changes survival:

    repaired base                  0.75
    all-u75 donor                 0.875
    FL_hip donor only             0.875

Every other single coordinate remains:

    survival                      0.75

FL_hip rescues lane7 only. Lane2 still fails.

Phase test for FL_hip donor:

    t0-5                          no rescue
    t4-8                          no rescue
    t6-10                         rescues lane7
    t11-15                        no rescue

Thus the new lane7 regression is phase-specific and depends on FL_hip behavior around the middle of the trajectory.

### suite3/O

No donor condition changes lane0:

    repaired base                 fail @14
    all-u75 donor                 fail @14
    every single-coordinate donor fail @14
    every FL_hip phase window     fail @14

### Branch A decision

The FL_hip support pattern does **not** reproduce across the O-heavy failures.

It is therefore not currently justified as a generic actuator-aware training method.

Interpretation:

    suite2/O/lane7
      = localized FL-hip phase-specific regression

    suite2/O/lane2
      = separate residual

    suite3/O/lane0
      = separate mechanism

Do not hard-code FL_hip or add global actuator-aware regularization from this evidence alone.

---

## Branch B — suite3/O/lane0 dynamics audit

Question:

Is suite3/O/lane0 an actor-independent reset/environment corner case, or can a different controller geometry rescue it?

### Whole-policy comparison

    u50 actor            SURVIVES
    u75 actor            FAIL @14
    repaired-u10 actor   FAIL @14

Therefore the failure is **not actor-independent** and is not simply an unavoidable reset/environment corner.

### u50 donor into u75

Single-coordinate counterfactuals:

    FL_hip donor         SURVIVES
    RR_hip donor         SURVIVES

All other single-coordinate donors tested remain fail @14.

Thus small controller-structure changes can move the trajectory into a surviving basin.

### Full-policy interpolation u75 -> u50

    alpha 0.10           FAIL @14
    alpha 0.25           SURVIVES
    alpha 0.50           FAIL @15
    alpha 0.75           FAIL @15
    alpha 1.00           SURVIVES

The response is non-monotonic.

This rules against a simple scalar action-magnitude or distance-to-u50 explanation.

### Full u50 donor by phase window

    t0-5                  FAIL @14
    t4-8                  SURVIVES
    t6-10                 SURVIVES
    t9-13                 SURVIVES
    t11-15                FAIL @14

The decisive intervention window is mid-trajectory, before terminal contact.

### Hip-specific margin audit

FL_hip interpolation:

    alpha .10             FAIL
    alpha .25             FAIL
    alpha .50             FAIL
    alpha .75             FAIL
    alpha 1.00            SURVIVES

FL_hip phase windows:

    t0-5                  FAIL
    t4-8                  FAIL
    t6-10                 SURVIVES
    t9-13                 SURVIVES
    t11-15                FAIL

RR_hip interpolation:

    alpha .10             FAIL
    alpha .25             SURVIVES
    alpha .50             SURVIVES
    alpha .75             FAIL
    alpha 1.00            SURVIVES

RR_hip phase windows:

    t0-5                  FAIL
    t4-8                  SURVIVES
    t6-10                 SURVIVES
    t9-13                 FAIL
    t11-15                FAIL

### State/action divergence

u50 differs most strongly from u75/repaired during the middle phase.

Examples:

At t8:

    u50:
      FL_hip  ~ -0.60
      RR_hip  ~ +0.995
      ang_vel ~ [0.32, 0.31, -0.13]

    u75:
      FL_hip  ~ +0.998
      RR_hip  ~ +1.000
      ang_vel ~ [0.87, -0.29, -0.16]

At t10:

    u50:
      FL_hip  ~ -0.979
      RR_hip  ~ +0.835
      ang_vel ~ [0.25, -1.02, +0.97]

    u75:
      FL_hip  ~ +0.998
      RR_hip  ~ +1.000
      ang_vel ~ [1.56, +0.40, -0.77]

By t14:

    u50 roll-rate       ~0.75
    u75 roll-rate       ~4.40
    repaired roll-rate  ~4.41

The surviving actor enters a qualitatively different mid-phase dynamics basin well before contact.

### Branch B decision

suite3/O/lane0 is:

    NOT an actor-independent reset corner
    NOT a generic headroom problem
    NOT a simple minimum-action-margin problem
    NOT a monotonic distance-to-u50 problem

It is best characterized as:

    a state-dependent, phase-sensitive controller/dynamics basin failure

where hip action geometry during roughly t4..13 determines whether the trajectory enters the surviving or failing basin.

---

## Combined decision

The residuals remain split:

### Localized regression

    suite2/O/lane7

Causally tied to FL_hip behavior around t6..10.
This does not replicate sufficiently to justify a global method.

### Dynamics-basin failure

    suite3/O/lane0

Rescuable by alternate hip geometry, especially FL_hip or RR_hip, during the mid-phase.
This requires a separate controller-robustness mechanism.

### Other residuals

    suite2/O/lane2
    suite3/S/lane0
    suite3/C/lane0

remain to be mapped against these two mechanisms before any new training intervention.

## Current authorization state

    global headroom tuning          STOP
    global kappa increase           STOP
    hard-coded FL_hip repair        STOP
    AI-C2                           BLOCKED
    AI-H2                           BLOCKED

Next work should remain read-only and test whether suite2/O/lane2 and suite3/S,C/lane0 align with:
1. the localized FL-hip phase regression, or
2. the mid-phase hip/dynamics-basin mechanism.
