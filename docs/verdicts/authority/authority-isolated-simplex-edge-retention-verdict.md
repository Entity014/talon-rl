# Authority-Isolated Simplex Edge-Margin Retention Verdict

Status: **FROZEN — PASS; FORMAL AI-C2 AUTHORIZED**
Date: 2026-09-25

## Question

Can bounded asymmetric preservation of the full preference-conditioned action simplex make the robust-u20 authority regime durable through continued PPO learning?

## Treatment

Frozen robust u20 start.

CONTROL:
- projected/headroom continuation

TREATMENT:
- identical continuation
- plus asymmetric all-edge simplex floor

All 10 T/A/O/S/C edges are retained against 90% of their frozen u20 reference lengths on the matched u20 support.

Gradient budget remains:

    rho   = .25
    beta0 = 2.497041993384243

No Jacobian loss, no gamma sweep, no replay expansion.

## Optimization contract

During u20 -> u30 treatment:

    PPO ratio error                  0
    training termination fraction    0

Retention gradient:
- inactive at u21 because edges remained above floor;
- activated from u22 onward;
- bounded at <= .25 of the base actor-gradient norm;
- mean weighted retain/base ratio = .225.

Therefore the treatment is active but does not hijack actor optimization.

## Historical fixed probe

Treatment:

    pairwise authority retention     .90151
    tangent authority retention      .90482

Both exceed the .90 gate.

The historical fixed probe is diagnostic only; primary evaluation is matched support.

## Primary matched-support authority

On balanced frozen-u20 visited states:

CONTROL:

    pairwise retention               .92698
    tangent retention               1.01039
    functional RMS retention         .92062

TREATMENT:

    pairwise retention               .99686
    tangent retention               1.08274
    functional RMS retention         .96874

Authority gate:

    PASS

The edge-only constraint preserves local tangent authority as an induced consequence; no explicit Jacobian regularizer is needed.

## Simplex edge geometry

CONTROL:

    heavy-heavy mean edge retention  .86762
    heavy-center mean edge retention .90728
    minimum edge                     T-S = .79061
    edge gate                        FAIL

TREATMENT:

    heavy-heavy mean edge retention  .95821
    heavy-center mean edge retention .98461
    minimum edge                     O-C = .91501
    edge gate                        PASS

Individual treatment edges:

    T-A  1.0513
    T-O   .9287
    T-S   .9234
    T-C   .9329
    A-O   .9510
    A-S   .9475
    A-C  1.0331
    O-S   .9474
    O-C   .9150
    S-C  1.0575

Every overall edge exceeds the predeclared .90 floor.

Some phase-local values fall below .90:
- T-O late .889
- T-S early .898
- O-C late .858

These do not produce an overall edge collapse and do not prevent the predeclared aggregate edge gate from passing.

## Representation geometry

Treatment endpoint:

    family parameter rank    3
    functional rank          4

No representation degeneracy appears.

## Robustness and critic no-regression

Frozen semantic suites:

    H32 EV mean              .31316
    H32 negative fraction    .10
    MC64 EV mean             .28246
    MC64 negative fraction   .10
    min survival             1.00
    failed lanes             0

Held-out reset suites:

    H32 EV mean              .29730
    H32 negative fraction    .1667
    MC64 EV mean             .25939
    MC64 negative fraction   .10
    min survival             1.00
    failed lanes             0

Fresh critic suites:

    H32 EV mean              .33783
    H32 negative fraction    .10
    MC64 EV mean             .28493
    MC64 negative fraction   .05
    min survival             1.00
    failed lanes             0

Fresh critic gate:

    PASS

No new termination topology appears.

## Causal conclusion

The post-u20 durability blocker was not:
- missing family representation;
- critic capacity;
- state coverage;
- raw Delta-a direction alone;
- radial heavy-vs-center authority alone.

The failure was contraction/deformation of the multi-preference action simplex.

An asymmetric floor on the actual simplex edges:
- preserves pairwise authority;
- preserves local tangent authority without explicit Jacobian regularization;
- preserves robustness;
- preserves wide-critic validity;
- leaves authority growth unconstrained.

This is the first actor-side durability treatment in this branch to satisfy all predeclared authority, edge, robustness, critic, and PPO gates.

## Decision

    simplex-edge durability gate      PASS
    pairwise authority gate           PASS
    tangent authority gate            PASS
    edge geometry gate                PASS
    robustness gate                   PASS
    fresh critic gate                 PASS
    PPO ratio gate                    PASS

Formal AI-C2:

    AUTHORIZED

AI-H2:

    remains blocked until formal AI-C2 passes

## Authoritative treatment checkpoint

    runs/authority_isolated_simplex_edge_retain-2026-09-25/model_30.pt

## Next experiment

Formal AI-C2 paired actor-updating critic compatibility, now with the successful simplex-edge durability mechanism frozen into both arms.

CONTROL critic:
    original/narrow critic representation

TREATMENT critic:
    wide critic representation

Same:
- actor initialization from the durability-valid regime;
- simplex-edge retention;
- projected/headroom repair;
- actor optimizer contract;
- lambda=.95;
- preference schedule;
- reset distribution;
- support/freshness;
- training budget.

If wide treatment passes authority + critic + survival while narrow control regresses, critic-capacity compatibility is causally supported.

If both pass, wide critic is compatible but strict necessity is not supported.

If wide fails, AI-H2 remains blocked.
