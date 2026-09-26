# Formal AI-C2 Verdict After Simplex-Edge Durability Repair

Status: **FROZEN — AI-C2 FAILS FRESH ROBUSTNESS; AUTHORITY AND CRITIC VALUE VALIDITY PASS; AI-H2 REMAINS BLOCKED**
Date: 2026-09-25

## Starting point

Durability-valid actor:

    runs/authority_isolated_simplex_edge_retain-2026-09-25/model_30.pt

Both AI-C2 arms start from this exact actor and actor optimizer state.

Frozen actor-side contract in both arms:
- projected PPO/headroom repair
- simplex-edge retention
- gamma=.90
- rho=.25
- beta0=2.497041993384243
- lambda=.95
- same preference schedule
- same reset distribution
- same support/head-refresh schedule
- same global continuation u30 -> u40

Only critic representation differs.

NARROW:
    48 -> 128 -> 128 -> 128 -> 4

WIDE:
    48 -> 256 -> 256 -> 128 -> 4

Both critics were equilibrated on the same frozen-u30 actor dataset before actor learning.

## Training correctness

Both arms:

    max PPO ratio error            0
    training termination fraction  0

Actor authority on historical fixed probe:

    NARROW:
      pairwise retention   1.394
      tangent retention    1.421

    WIDE:
      pairwise retention   1.349
      tangent retention    1.444

Thus the post-u20 authority-durability blocker is no longer present.

## Matched-support authority and simplex geometry

Relative to frozen robust-u20 support:

NARROW:

    pairwise retention               1.60745
    tangent retention                1.65577
    functional RMS retention         1.31739
    heavy-heavy mean edge retention  1.43453
    heavy-center mean edge retention 1.55447
    minimum edge                     T-O = 1.13877
    authority gate                   PASS
    edge gate                        PASS

WIDE:

    pairwise retention               1.62143
    tangent retention                1.70681
    functional RMS retention         1.32699
    heavy-heavy mean edge retention  1.43912
    heavy-center mean edge retention 1.53653
    minimum edge                     T-O = 1.16976
    authority gate                   PASS
    edge gate                        PASS

No simplex contraction remains.

The asymmetrical floor permits authority growth, and both endpoints substantially exceed the frozen-u20 reference geometry.

## Semantic-suite endpoint

NARROW:

    H32 EV mean              .35338
    H32 negative fraction    .025
    MC64 EV mean             .27780
    MC64 negative fraction   .05
    min survival             1.00
    failed lanes             0

WIDE:

    H32 EV mean              .35073
    H32 negative fraction    .000
    MC64 EV mean             .31371
    MC64 negative fraction   .05
    min survival             1.00
    failed lanes             0

Both pass semantic-suite value and survival measurements.

## Held-out reset endpoint

NARROW:

    H32 EV mean              .38987
    H32 negative fraction    .0167
    MC64 EV mean             .31558
    MC64 negative fraction   .0167
    min survival             .875
    failed lanes             2

Failures:
- suite5 / A: 1 base_contact
- suite6 / A: 1 base_contact

WIDE:

    H32 EV mean              .35279
    H32 negative fraction    .0333
    MC64 EV mean             .33521
    MC64 negative fraction   .000
    min survival             1.00
    failed lanes             0

Thus WIDE clearly outperforms NARROW on the held-out robustness portion of AI-C2.

## Fresh critic / robustness endpoint

NARROW:

    H32 EV mean              .38671
    H32 negative fraction    .025
    MC64 EV mean             .31681
    MC64 negative fraction   .0375
    min survival             .875
    failed lanes             3

Failures:
- T / seed 9700000: 1 base_contact
- T / seed 9700226: 1 base_contact
- A / seed 9701000: 1 base_contact

WIDE:

    H32 EV mean              .35455
    H32 negative fraction    .0125
    MC64 EV mean             .32751
    MC64 negative fraction   .0125
    min survival             .875
    failed lanes             6

Failures:
- T / seed 9700000: 1 base_contact
- T / seed 9700226: 1 base_contact
- A / seed 9701000: 1 base_contact
- A / seed 9701226: 1 base_contact
- S / seed 9703226: 1 base_contact
- C / seed 9704339: 1 base_contact

Fresh critic representation itself remains valid in both arms:
- positive H32 EV
- positive MC64 EV
- negative fractions well inside .25

The formal fresh robustness gate fails because:

    min survival = .875 < .95

## Causal comparison

Three fresh failures are shared across both critic architectures:

    T seed 9700000
    T seed 9700226
    A seed 9701000

This common topology argues against critic capacity being the sole cause of the fresh robustness regression.

WIDE removes the NARROW held-out A failures, but has three additional fresh failures in A/S/C.

Therefore:
- wide critic compatibility with authority is established;
- wide critic value validity under actor learning is established;
- strict critic-capacity necessity is not established;
- the formal AI-C2 all-gates PASS is not established because fresh robustness fails.

## Important new observation

The simplex-edge mechanism is asymmetric:

    edge below .90 reference -> penalized
    edge above floor          -> unconstrained

At u40 both arms show large authority expansion:

    pairwise ~1.61 x u20
    tangent  ~1.66-1.71 x u20
    minimum edge >1.13 x u20

This occurs concurrently with new fresh base_contact failures.

This is a hypothesis-generating observation only.

Do not conclude:

    authority over-expansion -> robustness failure

without a read-only causal audit.

## Decision

    simplex-edge durability mechanism       PASS
    formal AI-C2 authority gate             PASS
    formal AI-C2 critic-value gate          PASS
    formal AI-C2 semantic-suite survival    PASS
    WIDE held-out survival                  PASS
    formal AI-C2 fresh survival             FAIL

Formal AI-C2:

    FAIL

AI-H2:

    BLOCKED

## Next justified gate

Do not reopen generic robustness mining.

The narrow next question is:

    Does the lower-bound-only simplex-edge constraint permit authority over-expansion beyond the robust operating envelope, and is that expansion causally related to the new fresh base_contact failures?

A read-only audit should compare:
- durability-valid u30
- formal narrow u40
- formal wide u40

on the exact fresh failure seeds and matched surviving seeds.

Measure:
- pairwise simplex scale relative to u20/u30
- tangent scale
- action saturation/headroom
- action magnitude/rate
- state divergence before base_contact
- whether measurement-only interpolation of family/simplex response back toward u30 rescues failures

No training intervention is authorized until this audit resolves whether authority expansion is causal or merely correlated.
