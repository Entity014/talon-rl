# Authority-Isolated Class-B Trajectory Representation Final Verdict

Status: **FROZEN — STOP CLASS-B REPRESENTATION MINING**
Date: 2026-09-25

## Question

Can the recurring suite3 O/S/C basin failures be represented by a transferable short-horizon state-action trajectory space that:

1. separates fail vs survive across preferences;
2. places whole-policy rescues on the survivor side;
3. places independent single-coordinate rescues on the survivor side;
4. does not simply classify Class-A/Class-C rescue trajectories as Class-B failures?

## Trajectory representation

Windows tested:

    t6..10
    t6..13

Per-timestep generic features included:
- projected gravity
- base angular velocity
- base linear velocity
- hip/action symmetry
- leg-action symmetry
- leg/joint-velocity asymmetry
- action-joint-velocity alignment
- contact/support indicators

Each feature was converted to short-horizon statistics:
- mean
- slope
- variance
- range
- start-to-end delta

No preference label, reset-lane identity, or specific joint name was supplied to the classifier.

## Rescue validation

On repaired-u10, all Class-B base trajectories fail:

    O fail @14
    S fail @15
    C fail @15

Whole-u50 donor during t6..10 rescues all three through horizon 24.

Independent single-coordinate rescues during t6..10 also survive through horizon 24:

    O : u50 FL_hip donor
    S : u50 RR_hip donor
    C : u50 RR_hip donor

Therefore rescue placement is a valid causal validation target.

## Scalar-feature precursor result

Whole-policy rescue made several features look highly invariant, especially:
- hip diagonal geometry
- support-velocity coupling

However, independent single-coordinate rescues can survive without normalizing these features.

Example support-velocity coupling during t8..10:

                 fail      whole rescue     single rescue
    O            18.49          0.94            19.92
    S            20.60          1.14            12.30
    C            17.55          1.28            15.16

Thus the scalar candidates are descriptive, not necessary basin coordinates.

## High-dimensional linear gate

A standardized short-horizon trajectory vector with a leave-one-preference-out linear discriminator was tested first.

Both t6..10 and t6..13 failed the cross-preference gate because successful independent rescues were not consistently placed on the survivor side.

This result was treated cautiously because feature dimension was high relative to the number of natural failing trajectories.

## Final low-dimensional gate

A final fixed representation was therefore predeclared:

    PCA dimension = 3

PCA was fitted only on natural training trajectories from two preferences:
- matched surviving repaired lanes
- natural Class-B repaired failure
- robust u50 survivor

No rescue trajectories were used to fit PCA or class centroids.

Classification used nearest survivor/failure centroid in the 3D PCA space.

Leave-one-preference-out folds:

    train O+S -> test C
    train O+C -> test S
    train S+C -> test O

### t6..10

The natural Class-B failures are correctly placed on the failure side.

Robust u50 trajectories are generally placed on the survivor side.

But causal rescue placement fails:

- O whole rescue is still classified fail
- O single-coordinate rescue is classified fail
- S single-coordinate rescue is classified fail
- C single-coordinate rescue is classified fail

Matched surviving lanes are also not perfectly separated.

Negative-control rescue trajectories from Class A and Class C are classified as Class-B failure-like in all PCA folds.

Therefore:

    cross_preference_core = FAIL
    Class-A rescue specificity = FAIL
    Class-C rescue specificity = FAIL

### t6..13

The same final gate also fails.

Extending the temporal window does not recover transferable rescue placement or negative-control specificity.

## Interpretation

There is real recurring dynamical similarity across suite3/O/S/C:

- the failures are phase-sensitive;
- alternate mid-phase hip/support geometry can rescue them;
- angular divergence appears after the causal intervention window;
- robust u50 occupies a different trajectory regime.

However, the tested generic short-horizon representations do not identify a common low-dimensional basin coordinate that is both:

    predictive of failure
    AND
    necessary/consistent across independent successful rescue paths.

The discriminators can learn 'failure-like trajectory geometry', but successful counterfactual rescues may remain on that failure-like side.

That makes these representations descriptive rather than a valid training target.

## Stop-rule decision

The predeclared stop condition is met.

Do not continue:
- scalar feature mining;
- PCA-dimension tuning;
- alternative classifier fishing;
- additional hand-crafted trajectory features;
- joint-specific Class-B penalties;
- preference-specific Class-B penalties.

Class-B mechanism mining is stopped.

## Consequence for the project

The current best causal statements remain:

1. late global saturation/headroom pathology is real and substantially repairable;
2. projected + active tail-descent improves robustness while preserving preference authority;
3. remaining failures include recurring phase-sensitive dynamics-basin behavior;
4. no simple transferable basin representation has been validated for using that behavior as a generic regularization target.

Therefore the next methodological step should move back to a broader robustness formulation rather than patching residual basin geometry.

AI-C2 and AI-H2 remain blocked under the current robustness gate.
