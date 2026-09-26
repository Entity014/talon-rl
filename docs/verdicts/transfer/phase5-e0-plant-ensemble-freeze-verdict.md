# Phase 5 P5-E0 — Plant Ensemble Freeze Verdict

Status: **FROZEN PASS**
Date: 2026-09-26

## Decision

P5-E0 ensemble identifiability and sanity validation PASS.

The plant ensemble is now frozen before any Phase-5 actor training.

    P5-E0  ensemble freeze             PASS
    P5-E1  source no-regression        AUTHORIZED
    P5-E2  ensemble training           BLOCKED
    P5-E3  held-out plants             BLOCKED
    P5-E4  MuJoCo                      BLOCKED
    D4     hardware                    BLOCKED

Machine-readable source of truth:

    artifacts/phase5_e0_plant_ensemble_manifest.json

Manifest SHA256:

    80246a7ed3e2808681ba72373ff9d5d9ed6fbf6d2b982a49438007fb059cbe2a
## Frozen training support

Three evidence-derived latent variables are authorized.

### 1. Trunk mass delta

    distribution    uniform
    range           [-1.5, +3.0] kg

Canonical Phase-1 already saw:

    [-1.0, +3.0] kg

Measured MuJoCo trunk discrepancy is approximately:

    -1.288 kg

Therefore Phase 5 extends only the lower edge enough to include the measured target reference.

Mass randomization recomputes trunk inertia by mass ratio.

### 2. Passive-joint blend

    passive_blend in [0,1]

Mapping:
    hip-abduction damping    = 1.0 * passive_blend
    thigh/calf damping        = 2.0 * passive_blend
    all-joint armature        = 0.01 * passive_blend

The block is coupled rather than independently randomized.

Endpoint 0 is the source PhysX passive plant.

Endpoint 1 is the measured frozen MuJoCo passive template, excluding frictionloss.

T4 policy-free torque-pulse evidence established that this family is causally relevant.

### 3. Contact blend

    contact_blend in [0,1]

Mapping:

    robot static friction     = 0.8
    robot dynamic friction    = 0.6 + 0.2 * contact_blend
    restitution               = 0

This spans the source dynamic-friction value to the MuJoCo foot sliding-friction reference without introducing restitution.
## Explicit exclusions

The first Phase-5 treatment does not randomize:

- trunk COM;
- independent inertia scaling;
- DCMotor law or gains;
- MuJoCo frictionloss mapped into PhysX friction;
- restitution;
- contact compliance / solver parameters;
- simulator timestep / integration;
- foot/calf topology.

These exclusions prevent unsupported cross-engine parameter equivalence assumptions.

## Held-out plants

Twelve exact held-out plant tuples are frozen in the manifest before E2.

They include:

- target-like endpoint;
- low/high mass boundary compositions;
- passive-only and contact-only stress points;
- mixed interior compositions.

E2 must log all sampled plant tuples and must never inject these exact E3 tuples.
## E0 sanity protocol

Sample count:

    128 deterministic scrambled-Sobol plants

Seed:

    26092650

Fixed stance:

    canonical joint target
    zero normalized action
    1.28 s
    no policy training

Predeclared viability gate:

    finite physics             1.00 required
    reset feasible             1.00 required
    fixed-stance viable        >= 0.90
    trunk height               >= 0.18 m
    max tilt                   <= 60 deg
    no trunk-floor contact
## E0 sanity result

    finite fraction             1.000
    reset-feasible fraction      1.000
    fixed-stance viable fraction 1.000
    trunk-contact fraction       0.000

Realized normalized support coverage:

    mass       0.0039 .. 0.9986
    passive    0.0066 .. 0.9922
    contact    0.0057 .. 0.9953

Minimum-height distribution:

    min       0.2057 m
    p10       0.2180 m
    median    0.2398 m

Maximum tilt:

    maximum   10.37 deg

All sanity gates pass.
## Distribution check

Sobol sample means:

    mass delta       0.7500 kg
    passive blend    0.5000
    contact blend    0.5000

Cross-latent correlations are effectively zero:

    max |correlation| < 0.0012

Thus E0 samples cover the frozen joint support without introducing an unintended coupled sampling structure beyond the explicitly coupled parameter mappings.

## Governance

The ensemble range may not be widened after E1/E2 results.

No new plant parameter family may be added without a new contract.

MuJoCo semantics remain forbidden for E2 checkpoint selection.

P5-E1 is characterization only.

No Phase-5 training has been authorized by this verdict.
