# Authority-Isolated Class-B Invariant Representation Verdict

Status: **FROZEN — READ-ONLY FEATURE AUDIT COMPLETE; NO SINGLE SCALAR INVARIANT AUTHORIZED**
Date: 2026-09-25

## Scope

No training was performed.

Class-B cases:

    suite3/O/lane0
    suite3/S/lane0
    suite3/C/lane0

The audit compared:
- failing repaired-u10 trajectory
- surviving u50 trajectory
- matched surviving repaired lanes
- rescued repaired trajectory

A common rescue intervention was used first:

    whole-u50 action donor during t6..10

This rescue was validated to survive all three Class-B cases through horizon 24.

Independent single-coordinate rescues were also validated to survive:
- O: u50 FL_hip during t6..10
- S: u50 RR_hip during t6..10
- C: u50 RR_hip during t6..10

## Feature families tested

State:
- height
- projected gravity
- base angular velocity
- base linear velocity
- contact fraction/change
- joint velocity norm

Action/support geometry:
- left-right hip combination
- front-rear hip combination
- diagonal hip combination
- hip spread
- leg action left-right/diagonal asymmetry
- leg velocity left-right/diagonal asymmetry

State-action coupling:
- action/joint-velocity cosine
- hip/joint-velocity cosine
- roll-rate x lateral hip support
- pitch-rate x front-rear hip support
- projected-gravity x hip support
- angular support coupling
- support-velocity coupling

## Strong descriptive candidate under whole-policy rescue

Define:

    hip_lr   = (FL_hip + RL_hip) - (FR_hip + RR_hip)
    hip_diag = (FL_hip + RR_hip) - (FR_hip + RL_hip)

and analogous leg-velocity asymmetry terms:

    vel_lr
    vel_diag

Then:

    support_velocity_coupling =
        hip_lr * vel_lr
        + hip_diag * vel_diag

During t8..10:

                     fail      whole-u50 rescue    matched controls
    O                18.49          0.94              -0.95
    S                20.60          1.14               0.57
    C                17.55          1.28               1.25

This change precedes the clear angular-velocity divergence, which becomes large mainly around t10..12.

Diagonal hip geometry shows the same temporal pattern:

                     fail      whole-u50 rescue    matched controls
    O                 1.99          0.24               0.52
    S                 1.76          0.22               0.33
    C                 1.74          0.27               0.38

Thus the failing trajectories fail to perform the mid-phase hip/support reconfiguration seen in the robust and whole-policy-rescued trajectories.

## Specificity check

The above features are not uniquely Class-B markers.

In Class-A suite2/O/lane7, the failing trajectory is also strongly abnormal in hip diagonal/support-velocity features.

More importantly, the successful localized FL-hip rescue for Class A survives while support_velocity_coupling remains high:

    Class-A fail        ~17.44
    Class-A rescue      ~18.52
    matched controls     ~0.54

Therefore support_velocity_coupling is not a universal survival requirement.

## Independent-rescue necessity test inside Class B

The strongest whole-policy feature candidates were retested using independent single-coordinate rescues.

All three single-coordinate rescue interventions were confirmed to survive to horizon 24.

However, feature normalization is incomplete and donor-dependent.

For support_velocity_coupling at t8..10:

                     fail      single-coordinate rescue   controls
    O                18.49             19.92              -0.95
    S                20.60             12.30               0.57
    C                17.55             15.16               1.25

O survives without reducing this feature at all.

For hip diagonal:

                     fail      single-coordinate rescue   controls
    O                 1.99              1.81               0.52
    S                 1.75              0.74               0.33
    C                 1.74              0.94               0.38

Again, survival does not require full normalization to the matched-control region.

The action-velocity cosine moves consistently toward matched survivors under the single-coordinate rescues:

                     fail      rescue      controls
    O                 0.054      0.015      -0.073
    S                 0.020     -0.091      -0.112
    C                 0.046     -0.080      -0.151

but the effect is modest in O and also appears in non-Class-B rescue cases. It is therefore not yet a validated Class-B-specific basin coordinate.

## Decision

The first-pass invariant audit does **not** justify a scalar soft basin-margin penalty on any tested feature.

Specifically:

- do not penalize support_velocity_coupling directly
- do not impose a fixed hip-diagonal target
- do not regularize a specific hip coordinate
- do not use action-velocity cosine as a training objective yet

The evidence supports a weaker but important conclusion:

    Class-B survival depends on a mid-phase
    multi-dimensional state-action trajectory geometry,
    not a single scalar state/action margin.

Different rescue policies can enter the surviving basin through different local action paths while sharing the same eventual stability outcome.

## Next representation level

If this branch continues, the next read-only step should move from scalar features to a low-dimensional trajectory representation over t6..10 or t6..13.

Recommended candidates:

1. vector of symmetry/support features across multiple consecutive timesteps;
2. local state-action transition vector:
       [projected gravity, angular velocity, leg velocity asymmetry,
        hip/action symmetry, action-velocity alignment];
3. short-horizon delta features:
       f(t+1)-f(t)
   rather than absolute f(t);
4. PCA / linear discriminant or nearest-basin distance fitted only on
   surviving matched trajectories, then evaluated on fail/rescue trajectories.

Any learned/read-only representation must be validated out-of-sample across O/S/C and against Class A/C before authorizing a training objective.

## Authorization state

    scalar Class-B robustness objective      NOT AUTHORIZED
    joint-specific repair                    NOT AUTHORIZED
    global headroom tuning                   STOPPED
    low-dimensional trajectory audit         AUTHORIZED
    AI-C2 / AI-H2                            BLOCKED
