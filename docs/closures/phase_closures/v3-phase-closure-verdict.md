# V3 Phase Closure — Generalized Representation Without Generalized Learning

Status: **FROZEN — V3 METHOD SEARCH CLOSED**
Date: 2026-09-25

## Executive verdict

V3 successfully generalized the objective interface, but did not establish semantically stable learning across variable objective sets.

The defensible summary is:

> **V3 proved generalized representation, but not generalized learning.**

This is a scoped extension result, not the main deployment path.

## What V3 established

### G0 — objective-set representation equivalence

    objective-set interface                 PASS
    permutation invariance                  PASS
    actor pre-tanh parity                   PASS
    action parity                           PASS
    log-prob parity                         PASS
    critic parity                           PASS
    H2a behavioral parity                   PASS

The fixed-index Phase-1 controller can therefore be represented through a permutation-invariant objective-set API without changing its behavior.

### G1 — structural capability

    variable-cardinality execution          PASS
    preference authority                    PASS
    fixed-policy token-query critic capacity PASS
    same architecture across set sizes      PASS

The representation is cardinality-independent at execution time.
## What V3 did not establish

### G1 — learning stability

    seen-support semantic preservation      FAIL
    moving-policy critic validity           FAIL
    clean held-out-cardinality test         NOT REACHED
    compact critic/gradient repair mechanism NOT ESTABLISHED

Extensive read-only audits showed:

- critic-derived and MC-derived actor gradients differ materially;
- critic-only explanations are insufficient;
- mixed-objective gradient interaction exists;
- trajectory-level semantic effects vary strongly by fold and horizon;
- no cross-fold compact mechanism survived the preregistered stop rules.

Therefore it is not scientifically justified to continue V3 mechanism mining or to introduce post-hoc objective-specific corrections.

## Claim boundary

Authorized claim:

> The objective-set representation generalizes the controller interface successfully, but the current learning formulation does not maintain semantic validity when trained across variable objective sets.

Also supported:

> A permutation-invariant, cardinality-independent objective representation is not sufficient for semantically stable learning across changing objective sets.

Do not claim:

    variable-cardinality learning/generalization succeeded
    variable-cardinality generalization failed
    unseen-combination generalization failed
    unseen-objective generalization failed

The last three were not cleanly tested because G1 learning failed on seen support first.
## Relationship to Phase 1

Phase 1 remains the main thesis controller and deployment candidate.

Phase-1 result:

    preference authority                    VALID
    authority durability                    VALID
    critic / robustness substrate           VALID
    T semantic validity                     PASS
    A semantic validity                     PASS
    O semantic validity                     PASS
    S semantic validity                     DOCUMENTED LIMITATION

Phase 1 established:

> Preference authority is not sufficient for semantic correctness.

V3 extends that finding:

> Generalized objective representation is not sufficient for generalized semantic learning.

These findings are complementary.

## Scope decision

The thesis-critical path is now:

    STOP V3 method search
        ↓
    freeze V3 as extension / negative result
        ↓
    return to Phase-1 canonical controller
        ↓
    multi-seed final characterization
        ↓
    deployment-oriented simulation validation
        ↓
    sim-to-sim validation
        ↓
    guarded hardware bring-up
        ↓
    real-robot evaluation

## V3 future-work status

    G2 unseen known-objective combinations   BLOCKED / FUTURE WORK
    G3 unseen objective identity             BLOCKED / FUTURE WORK

These stages are not marked FAILED.

They require a semantically stable variable-set learning substrate that V3-G1 did not establish.
## Method-search stop rule

From this closure onward, do not:

- add another V3 gradient correction;
- change scalarization to rescue V3;
- add objective-specific V3 losses;
- redesign the objective-set architecture;
- reopen critic/MC/trajectory mechanism mining;
- use failed held-out-cardinality cells as evidence against cardinality generalization.

A new V3 branch requires an externally motivated formulation hypothesis, not another post-hoc continuation of the current audit chain.

## Thesis narrative

Recommended structure:

### Main contribution — Phase 1 / V2

Develop and validate a reliable one-model preference-conditioned MORL controller for a fixed objective vocabulary, including:
- authority isolation;
- authority durability;
- critic compatibility;
- robustness validation;
- semantic validation;
- explicit scoped limitation for Smoothness.

### Extension — Phase 2 / V3

Test whether the Phase-1 controller principles extend from fixed-index preferences to variable objective sets.

Results:
- representation/interface generalization succeeds;
- learning across changing objective sets does not preserve semantic validity;
- no compact repair mechanism is identified under the preregistered causal ladder.

This negative result motivates future work on semantically stable objective-set learning rather than weakening the Phase-1 deployment contribution.

## Source-of-truth status

Use the following chain as canonical:

Phase 1:
- docs/verdicts/authority/authority-isolated-h2a-u30-semantic-validity-verdict.md
- docs/verdicts/authority/authority-isolated-s-trajectory-temporal-decomposition-verdict.md

V3:
- docs/verdicts/objective_set/objective-set-g0-verdict.md
- docs/verdicts/objective_set/objective-set-g1-current-verdict.md
- docs/verdicts/objective_set/objective-set-g1-credit-vs-mc-audit-verdict.md
- docs/verdicts/objective_set/objective-set-g1-trajectory-semantic-credit-verdict.md
- this closure document

README and earlier final-controller documents remain historical unless explicitly superseded.
