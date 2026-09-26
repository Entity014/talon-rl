# Simplex-Edge Durability + Formal AI-C2 Verdict

Status: **FROZEN — EDGE DURABILITY REPAIR PASSES; FORMAL AI-C2 CAPACITY TREATMENT NOT SUPPORTED; AI-H2 REMAINS BLOCKED**
Date: 2026-09-25

## Part I — Simplex-edge durability treatment

### Treatment

Starting from exact robust u20:

    control:
      existing projected/headroom continuation

    treatment:
      same
      + bounded asymmetric all-simplex-edge floor

All T/A/O/S/C edges are retained against frozen u20:

    L_edge
      = E max(0, .90 d_ij^20 - d_ij^theta)^2

Frozen:
- rho=.25
- beta0=2.497041993384243
- matched u20 support
- critic
- actor optimizer
- PPO/headroom contract
- reset distribution
- update budget

### Optimization validity

Treatment:
- PPO ratio invariant = exact
- train termination fraction = 0
- edge gradient budget <= .25 of base gradient
- headroom/tail repair remains active

### Historical fixed probe

u20 -> u30 edge treatment:

    pairwise retention    .9015
    tangent retention     .9048

Both cross the .90 gate.

### Primary matched-u20 support

CONTROL:

    pairwise retention          .9270
    tangent retention          1.0104
    functional RMS retention    .9206

    heavy-heavy mean edge       .8676
    heavy-center mean edge      .9073
    minimum edge                .7906  (T-S)

TREATMENT:

    pairwise retention          .9969
    tangent retention          1.0827
    functional RMS retention    .9687

    heavy-heavy mean edge       .9582
    heavy-center mean edge      .9846
    minimum edge                .9150  (O-C)

All 10 overall edge-energy retentions exceed .90.

Therefore:

    authority gate     PASS
    edge geometry gate PASS

Phase-local sub-.90 values remain in a few cells, e.g.:
- O-C late ~.858
- T-O late ~.889
- T-S early ~.898

but these do not produce an overall edge collapse.

### No-regression audit

Edge treatment endpoint:

Frozen semantic suites:

    min survival              1.00
    failed lanes              0
    H32 EV mean               .3132
    MC64 EV mean              .2825

Held-out suites:

    min survival              1.00
    failed lanes              0
    H32 EV mean               .2973
    MC64 EV mean              .2594

Fresh critic suites:

    min survival              1.00
    failed lanes              0
    H32 EV mean               .3378
    H32 negative fraction     .10
    MC64 EV mean              .2849
    MC64 negative fraction    .05

Critic gate:

    PASS

### Durability conclusion

The asymmetric all-edge floor succeeds where:
- raw Delta-a MSE failed;
- heavy-vs-center projected-authority floor was insufficient.

The retained quantity that matches observed authority durability is the multi-preference action-simplex edge geometry.

Simplex-edge durability repair:

    PASS

---

## Part II — Formal AI-C2 rerun under edge retention

### Question

With authority durability repaired, does critic representation capacity remain the causal requirement for stable actor learning?

### Pair

Both arms:
- exact same robust-u20 actor and actor optimizer state
- same all-edge retention
- same rho/beta/gamma
- same PPO/headroom repair
- same reset/preference/update schedule
- same critic support/head refresh
- same critic-only prefit dataset
- actor exact after critic prefit

NARROW:

    critic body 128-128-128

WIDE:

    critic body 256-256-128

### Prefit

Same 15,360 frozen-u20 samples.

Final prefit MSE:

    narrow   .01486
    wide     .01305

Actor error after prefit:

    exactly 0 in both arms

### Actor-learning fixed-probe endpoint

NARROW:

    pairwise retention    .8967
    tangent retention     .8443

WIDE:

    pairwise retention    .9107
    tangent retention     .8634

On this historical probe:
- wide crosses pairwise .90
- neither crosses tangent .90

Historical-probe authority gate:

    both FAIL

### Primary matched-u20 support

NARROW:

    pairwise retention          .9996
    tangent retention          1.1143
    functional RMS retention    .9683

    heavy-heavy mean            .9467
    heavy-center mean           .9755
    minimum edge                .9034 (O-C)

    authority gate              PASS
    all-edge gate               PASS

WIDE:

    pairwise retention          .9581
    tangent retention          1.0114
    functional RMS retention    .9255

    heavy-heavy mean            .9288
    heavy-center mean           .9427
    minimum edge                .8332 (A-S)

    authority gate              PASS
    all-edge gate               FAIL

Additional wide regressions:
- A-S  .833
- A-C  .858
- O-C  .886
- O-S  .894
- A-O  .899

Thus the wide treatment preserves local differential authority but deforms finite preference-simplex geometry more than the narrow control.

### Critic / robustness endpoint

NARROW fresh audit:

    H32 EV mean               .4265
    H32 negative fraction     .025
    MC64 EV mean              .3397
    MC64 negative fraction    .000
    min survival             1.00
    failed lanes                0

WIDE fresh audit:

    H32 EV mean               .4144
    H32 negative fraction     .0125
    MC64 EV mean              .3219
    MC64 negative fraction    .0375
    min survival             1.00
    failed lanes                0

Frozen semantic and held-out survival are 1.00 in both arms.

Critic gate:

    narrow PASS
    wide   PASS

### Causal interpretation

Once functional simplex-edge authority is explicitly preserved:

    narrow critic no longer shows the earlier value-generalization failure.

Therefore the old statement:

    high-authority actor necessarily requires the wide critic body

is not supported in this new regime.

The wide critic remains value-valid, but it is not causally superior:
- value EV is comparable in both arms;
- survival is identical;
- narrow preserves the finite action simplex better;
- wide violates the predeclared worst-edge floor.

This suggests the earlier critic-capacity blocker was partly conditional on the drifting actor regime. Once actor authority geometry is stabilized, the narrow critic can remain valid.

## Formal AI-C2 decision

Under the predeclared treatment logic:

    WIDE treatment must preserve
      authority
      critic validity
      survival
      edge geometry

Result:

    local pairwise/tangent authority   PASS
    critic gate                        PASS
    survival                           PASS
    PPO ratio                          PASS
    all-edge geometry                  FAIL

Therefore:

    FORMAL AI-C2 = FAIL / CAPACITY TREATMENT NOT SUPPORTED

This is not a return to the old critic blocker.

Instead, evidence now says:

    critic capacity is no longer the active causal blocker
    after simplex-edge durability repair.

## AI-H2

AI-H2 remains blocked under the frozen roadmap because formal AI-C2 did not pass.

Do not:
- tune wide critic capacity further;
- enlarge the critic again;
- add Jacobian loss;
- increase rho/beta;
- tune gamma;
- reopen robustness mining.

## Next decision point

The project now has a successful actor-durability method:

    bounded all-simplex-edge retention

and a surprising critic result:

    narrow and wide critics are both valid,
    but wide does not preserve finite simplex geometry as well in the paired rerun.

The next methodological question should be whether AI-C2 should be retired as a capacity-selection gate and replaced by a simpler compatibility requirement:

    choose the critic configuration that passes
    fresh H32/MC64 + survival + authority geometry,
    without requiring wide > narrow.

Under that compatibility criterion, the narrow arm currently satisfies all measured gates.

This governance change should be made explicitly before opening AI-H2, rather than silently selecting the narrow arm after a failed capacity-treatment hypothesis.
