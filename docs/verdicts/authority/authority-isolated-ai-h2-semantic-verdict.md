# AI-H2 Semantic Accumulation / Forgetting Verdict

Status: **FROZEN — SEMANTIC ACCUMULATION FAILS, BUT FINAL VERDICT IS ROBUSTNESS-CONFOUNDED**
Date: 2026-09-25

## Governance

AI-H2 was opened only after the AI-C2 compatibility addendum established that the narrow critic is a compatible minimal-sufficient critic under stabilized simplex-edge authority.

Historical wide-capacity AI-C2 remains a historical FAIL.

Revised critic-compatibility AI-C2 is PASS for the narrow critic.

No new engineering repair was introduced in H2.

## H2 training path

Start:

    compatible narrow endpoint at global u30

Continue:

    u30 -> u55

Snapshots:

    u30
    u35
    u40
    u45
    u50
    u55

Frozen during H2:
- authority-isolated actor;
- narrow critic;
- all-simplex-edge retention;
- edge gamma=.90;
- rho=.25;
- beta0=2.497041993384243;
- projected PPO/headroom repair;
- kappa=.05;
- lambda=.95;
- rewards / preference semantics;
- reset distribution;
- critic support / refresh contract.

Training validity:

    max PPO ratio error              0
    max training termination         0
    mean edge/base gradient ratio    .223

## Endpoint semantic timeline

PASS axes:

    u30   T A O
    u35   T A O
    u40   none
    u45   none
    u50   A
    u55   A O S

Maximum simultaneous competence:

    3 / 4

Final simultaneous competence:

    3 / 4

No checkpoint reaches 4/4.

## Acquisition / retention

First PASS:

    T   u30
    A   u30
    O   u30
    S   u55

Retained fraction after first PASS:

    T   .333
    A   .667
    O   .500
    S  1.000

Predeclared requirement:

    every acquired axis >= .80

Therefore retention gate fails.

## PASS -> FAIL forgetting

PASS -> FAIL events:

    T   u35 -> u40
    A   u35 -> u40
    O   u35 -> u40

PASS -> FAIL rate:

    .429

Predeclared maximum:

    .20

Therefore forgetting gate fails.

The u35 -> u40 event is not merely winner substitution.
Three simultaneously competent axes collapse together.

Later reacquisition occurs:

    A   u45 -> u50
    O   u50 -> u55
    S   u50 -> u55

Final PASS identity differs from the initial competent set:

    u30/u35: T A O
    u55:     A O S

Thus semantic competence does not accumulate monotonically.

## Final endpoint semantics at u55

    T   FAIL
    A   PASS
    O   PASS
    S   PASS

Tracking collateral remains bounded.

Endpoint survival on the four matched semantic suites is 1.00.

## Final continuum

Frozen continuum paths:

    T-A
    T-O
    T-S

Final results:

    monotonicity fraction      .6615   PASS >= .65
    endpoint-between fraction  .4653   FAIL < .65

Thus the continuous preference-to-behavior map is only partially coherent.

## Authority at u55

Relative to frozen robust u20:

    pairwise authority retention       1.906
    tangent authority retention        1.867
    functional-specific RMS retention  1.525

Edge geometry:

    heavy-heavy mean edge              1.697
    heavy-center mean edge             1.875
    minimum edge                       1.323  (T-O)

Ranks:

    parameter rank                     3
    functional rank                    4

Therefore:

    authority gate    PASS
    all-edge gate     PASS

The semantic forgetting observed in H2 is not explained by renewed authority contraction.

Indeed, action-simplex authority is stronger than the u20 reference at u55.

## Critic validity at u55

Fresh value metrics:

    H32 EV mean               .3083
    H32 negative fraction     .0375
    MC64 EV mean              .2531
    MC64 negative fraction    .0500

Value representation remains valid by EV criteria.

## Robustness confound at u55

However fresh robustness regresses:

    fresh minimum survival    .875
    failed lanes              3

Fresh failures:

    T / seed 9700000
    A / seed 9701000
    S / seed 9703226

All are base_contact failures.

Semantic-suite survival:

    1.00

Held-out suite survival:

    1.00

Continuum also contains 15 evaluation points with survival .875.

Therefore the final H2 no-regression compatibility contract is not fully satisfied.

## Formal H2 gate

Predeclared requirements:

1. final endpoint PASS count = 4/4               FAIL
2. max simultaneous PASS count = 4/4             FAIL
3. retained fraction >= .80 for every axis        FAIL
4. PASS->FAIL rate <= .20                         FAIL
5. final semantic survival >= .95                 PASS on endpoint suites
6. final authority compatibility                  PASS
7. final critic compatibility incl fresh survival FAIL
8. final continuum monotonicity >= .65            PASS
9. final continuum between fraction >= .65        FAIL

## Interpretation

The strongest supported statement is:

    durable preference authority is NOT sufficient,
    by itself, to guarantee semantic accumulation.

Evidence:
- authority remains strong;
- simplex geometry remains non-degenerate;
- pairwise and tangent authority grow;
- yet T/A/O semantic competence collapses at u40;
- later competence is reacquired in a different set.

However, because fresh/continuum robustness regresses by u55, the strict H2 causal conclusion is confounded at the final endpoint.

Therefore do NOT claim:

    semantic forgetting persists under a fully unchanged compatibility envelope

without qualification.

Instead claim:

    semantic forgetting is observed despite preserved authority,
    but the full H2 endpoint is robustness-confounded.

## Decision

    authority-durability mechanism          PASS
    semantic accumulation                  FAIL
    semantic retention                     FAIL
    continuum between/interpolation        FAIL
    final authority compatibility          PASS
    value EV compatibility                 PASS
    final fresh robustness                 FAIL

Formal verdict:

    AI-H2 = FAIL / ROBUSTNESS-CONFOUNDED

## What is closed

Do not reopen:
- authority creation;
- simplex-edge retention target search;
- critic-width search;
- Jacobian retention;
- projected-authority-floor search.

Those questions are answered sufficiently for this branch.

## Next scientific question

The next step should not invent another authority proxy.

The unresolved question is now:

    can semantic forgetting be demonstrated on a time interval
    where authority, critic validity, and robustness all remain simultaneously valid?

The most informative existing interval is u30 -> u40 because:
- u30/u35 have 3 semantic passes;
- u40 has 0 semantic passes;
- training survival remains 1.00;
- authority retention machinery is active.

Before any new training intervention, perform a measurement-only compatibility audit at u35 and u40:
- matched-support pairwise/tangent/all-edge authority;
- frozen semantic/held-out/fresh survival;
- fresh H32/MC64.

If u40 still satisfies compatibility while semantics collapse, H2 can be sharpened to an unconfounded early-window semantic-forgetting result without extending training or mining another mechanism.

If compatibility already fails by u40, semantic forgetting remains coupled to robustness degradation and the next branch must be framed accordingly.


## Early-window compatibility addendum

A measurement-only audit was added at the first semantic-collapse transition:

    u35 -> u40

No training was added.

### u35

Semantic:

    PASS axes    T A O
    PASS count   3

Authority:

    pairwise retention     1.254
    tangent retention      1.411
    minimum edge           1.022

Compatibility:

    semantic survival      1.00
    held-out survival      1.00
    fresh survival         1.00

Fresh critic:

    H32 EV mean            .398
    H32 negative fraction  .025
    MC64 EV mean           .301
    MC64 negative fraction .050

Result:

    compatibility PASS

### u40

Semantic:

    PASS axes    none
    PASS count   0

Authority:

    pairwise retention     1.548
    tangent retention      1.700
    minimum edge           1.209

Thus action authority is stronger, not weaker, at the collapse point.

Semantic / held-out survival:

    1.00 / 1.00

Fresh critic:

    H32 EV mean            .370
    H32 negative fraction  .0125
    MC64 EV mean           .278
    MC64 negative fraction .025

Fresh robustness:

    minimum survival       .875
    failed lanes           1

Failure:

    preference T
    seed 9700000
    base_contact

Result:

    compatibility FAIL due fresh survival only

## Early-window conclusion

The first semantic collapse is not explained by authority contraction:

    u35 -> u40
    authority increases
    T/A/O semantics collapse

However, the same u40 checkpoint also introduces the first fresh robustness miss.

Therefore the desired stronger statement:

    semantic forgetting occurs while
    authority + critic + robustness
    all remain simultaneously valid

is NOT established.

Instead the clean statement is:

    semantic collapse occurs while authority remains fully valid,
    but coincides with onset of fresh robustness regression.

This preserves the formal H2 verdict:

    FAIL / ROBUSTNESS-CONFOUNDED

and prevents over-attributing the semantic collapse either to authority or to robustness without further causal evidence.
