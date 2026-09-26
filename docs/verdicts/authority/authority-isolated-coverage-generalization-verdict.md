# Authority-Isolated Coverage / Generalization Verdict

Status: **FROZEN — COVERAGE TREATMENT FAIL; REPAIRED CONTROL-u20 PASSES ROBUSTNESS STACK; AI-C2 AUTHORIZED**
Date: 2026-09-25

## Question

Are the remaining repaired-u10 robustness failures caused by insufficient reset/state/dynamics coverage, such that broadening only the training distribution improves held-out robustness while preserving preference authority?

## Paired causal design

Both arms resumed from the exact same repaired-u10 continuation state:
- identical actor parameters
- identical wide critic
- identical optimizer state
- identical critic-support pools
- identical anchor specs
- identical RNG snapshot
- identical PPO / GAE lambda=.95 / MORL objective / preference schedule
- identical projected PPO-tail conflict repair
- identical active tail descent kappa=.05

Only the training distribution differed.

### CONTROL

Original UnitreeA1FlatEnvCfg distribution.

### COVERAGE

Broadened only:
- root z reset: +/-0.03 m
- root roll/pitch reset: +/-0.12 rad
- root linear/angular reset velocity: +/-0.75 instead of +/-0.5
- joint reset changed to offset reset with:
  - position offset +/-0.40 rad
  - joint velocity +/-1.50 rad/s
- friction/restitution resampled every reset:
  - static friction [0.6,1.4]
  - dynamic friction [0.5,1.2]
  - restitution [0,0.1]

No reward, actor architecture, critic architecture, preference semantics, or evaluation rule changed.

## Implementation audit

The authority/headroom branch uses Isaac Lab UnitreeA1FlatEnvCfg directly, not the repository's broader Talon environment config.

Important baseline coverage facts:
- reset x/y/yaw already broad
- root linear/angular velocity already +/-0.5
- initial roll/pitch pose is exactly zero
- joint reset uses reset_joints_by_scale with default joint velocity zero; therefore velocity scaling produces exactly zero joint-velocity diversity
- friction is fixed at startup
- base mass is sampled at startup only
- 10-15 s interval push does not occur inside the 32/64-step collection horizons (~0.64/1.28 s)

Thus the coverage hypothesis was implementation-grounded rather than speculative.

## Clean rerun / artifact isolation

An earlier output namespace was contaminated by a concurrent process writing the same resume-state files.

That branch was discarded.

The clean causal rerun used isolated namespaces:

    runs/authority_isolated_coverage2_control-2026-09-25
    runs/authority_isolated_coverage2_coverage-2026-09-25

Both clean arms were verified to start at:

    update = 10
    rows   = 10
    snaps  = [0,3,10]

and both completed to update 20.

## Training result

### CONTROL-u20

Relative to repaired-u10:

    pairwise preference separation retention    1.06397
    simplex-tangent Jacobian retention          1.32520
    authority gate                              PASS

    probe tail-loss ratio                       0.44208
    probe tail-fraction delta                  -0.09408

All continuation actor rollouts had termination fraction 0.

PPO post-refresh ratio invariant remained exact within the existing tolerance.

### COVERAGE-u20

Relative to repaired-u10:

    pairwise preference separation retention    0.75385
    simplex-tangent Jacobian retention          0.65726
    authority gate                              FAIL

    probe tail-loss ratio                       0.49803
    probe tail-fraction delta                  -0.15202

Coverage training genuinely exposed harder states:
- several continuation updates had termination fraction 0.125
- some reached termination fraction 0.25

Thus the treatment was not a no-op.

However, it traded robustness exposure for a major collapse in preference authority.

## Frozen semantic and held-out generalization evaluation

Evaluation environment was the original nominal UnitreeA1FlatEnvCfg.

Frozen suites:

    suite2 seed 840003
    suite3 seed 840004

Held-out reset suites:

    suite4 seed 850101
    suite5 seed 850202
    suite6 seed 850303

All T/A/O/S/C preferences, 8 lanes each, horizon 64.

### repaired-u10

    min frozen survival       0.75
    mean frozen survival      0.9375
    frozen failed lanes       5

    min held-out survival     0.75
    mean held-out survival    0.98333
    held-out failed lanes     2

### CONTROL-u20

    min frozen survival       1.00
    mean frozen survival      1.00
    frozen failed lanes       0

    min held-out survival     1.00
    mean held-out survival    1.00
    held-out failed lanes     0

### COVERAGE-u20

    min frozen survival       1.00
    mean frozen survival      1.00
    frozen failed lanes       0

    min held-out survival     1.00
    mean held-out survival    1.00
    held-out failed lanes     0

## Causal conclusion

The survival recovery cannot be attributed to broadened coverage.

The control continuation under the original training distribution reaches the same perfect frozen and held-out survival while preserving/growing preference authority.

The coverage treatment adds no measured survival benefit and materially damages preference authority.

Therefore:

    simple coverage-deficit hypothesis     NOT SUPPORTED
    broadened-coverage treatment           FAIL
    ordinary repaired continuation         PASS

The most defensible interpretation is:

    repaired-u10 was a transient robustness checkpoint

and the projected/headroom repair needed additional ordinary continuation time for the robust policy regime to emerge.

## Fresh wide-critic revalidation on CONTROL-u20

Measurement-only reset-diverse revalidation:
- 4 fresh suites per preference
- T/A/O/S/C
- H64
- H32 truncated returns
- MC64 returns

Aggregate:

    H32 EV mean                  0.32398
    H32 negative fraction        0.150
    H32 mean absolute bias       0.03997

    MC64 EV mean                 0.33524
    MC64 negative fraction       0.100
    MC64 mean absolute bias      0.09369

    minimum survival             1.00

Per-preference negative fractions remain within the predeclared <=.25 bound.

Critic revalidation:

    PASS

## Full robustness stack at CONTROL-u20

    authority-isolated actor                 PASS
    preference authority retention           PASS
    projected/headroom repair                PASS
    PPO ratio invariant                      PASS
    frozen suite survival                    PASS (min 1.00)
    held-out reset survival                  PASS (min 1.00)
    fresh wide-critic H32/MC64               PASS
    no new termination topology              PASS

## Decision

The actor-robustness blocker is now closed at CONTROL-u20.

Do not adopt the broadened coverage treatment tested here.

Do not reopen:
- residual mechanism mining
- Class-B scalar/basin mining
- external-wrench adversarial training
- local policy-output smoothing
- coverage expansion in this exact form

The correct continuation checkpoint for the next semantic gate is:

    runs/authority_isolated_coverage2_control-2026-09-25/model_20.pt

## Authorization

    AI-C2     AUTHORIZED
    AI-H2     remains downstream of AI-C2

Next step:

    CONTROL-u20
      ->
    AI-C2 actor/critic compatibility revalidation under the now-robust policy
      ->
    if PASS, AI-H2 semantic forgetting test
      ->
    multi-seed
      ->
    sim-to-sim
      ->
    real robot
