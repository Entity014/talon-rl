# Phase 5 P5-E2 — Plant-Ensemble Training Screen Verdict

Status: **FROZEN — FAIL AT SOURCE-DOMAIN NO-REGRESSION**
Date: 2026-09-26

## Decision

The first preregistered Phase-5 plant-ensemble training screen is complete.

Both treatment and matched control were trained from the same frozen u20 source state to fixed u30 using the same seed and continuation budget.

The ensemble treatment fails the source-domain no-regression gate because Orientation semantics regress on the canonical source evaluation.

Therefore:

    P5-E0  ensemble freeze             PASS / FROZEN
    P5-E1  baseline characterization   COMPLETE / FROZEN
    P5-E2  plant-ensemble screen       FAIL
    P5-E3  held-out validation         BLOCKED
    P5-E4  MuJoCo validation           BLOCKED
    D4     hardware                    BLOCKED

Machine-readable verdict:

    artifacts/phase5_e2_screen_verdict.json
## Frozen initialization

Common source:

    runs/authority_isolated_coverage2_control-2026-09-25/resume_state.pt
    update = 20

Seed:

    75001

Budget:

    exact 10 updates
    u20 -> u30

Checkpoint rule:

    fixed u30 only
    no best checkpoint
    no semantic checkpoint selection

Control:

    canonical Phase-1 source training distribution

Treatment:

    frozen P5-E0 plant ensemble
    plant parameters hidden from policy

No reward, objective, actor/critic architecture, preference protocol, or authority-retention method was changed.
## Treatment sampler provenance

Realized plant draws:

    657

Draw indices:

    0 .. 656

Duplicate draw IDs:

    0

Realized support:

    mass delta       -1.4976 .. +2.9935 kg
    passive blend     0.00066 .. 0.99812
    contact blend     0.00156 .. 0.99941

Minimum L-infinity distance from any frozen held-out tuple:

    0.04776

Held-out leakage:

    none

Thus the treatment executed the intended bounded ensemble and did not train on E3 held-out tuples.
## Training invariants

Matched control u30:

    pairwise authority retention     1.1467
    tangent authority retention      0.9044
    ratio max error                  0
    max termination fraction         0

Ensemble treatment u30:

    pairwise authority retention     1.0363
    tangent authority retention      1.0650
    ratio max error                  0
    max termination fraction         0

Both arms preserve the fixed-probe authority geometry sufficiently for the E2 source-domain semantic comparison.

The treatment failure is therefore not an authority-collapse or PPO-invariant failure.
## Canonical source-domain reference

Frozen Phase-1 reference:

    T semantics       PASS
    A semantics       PASS
    O semantics       PASS
    S semantics       FAIL / known limitation

    critic            PASS
    center            PASS
    continuum mono    PASS
    endpoint-between  PASS

This is the no-regression target.

## Matched nominal control at u30

    T semantics       PASS
    A semantics       FAIL
    O semantics       PASS
    S semantics       FAIL

    critic            FAIL
    center            FAIL
    continuum mono    PASS
    endpoint-between  FAIL

The matched control demonstrates that this stochastic continuation is not benign: A and critic validity regress even without ensemble exposure.
## Ensemble treatment at u30

    T semantics       PASS
    A semantics       FAIL
    O semantics       FAIL
    S semantics       FAIL

    critic            PASS
    center            PASS
    continuum mono    PASS
    endpoint-between  FAIL

The treatment therefore exhibits a mixed effect:

Positive:
- critic validity is recovered relative to the matched control;
- pairwise and tangent authority remain preserved;
- T semantics remain valid;
- center compromise remains valid.

Negative:
- O semantics regress from PASS to FAIL;
- O also passes in the matched control, making this regression treatment-specific under the matched screen;
- A is not rescued;
- endpoint-between structure remains invalid.

## Primary gate

The Phase-5 contract requires source-domain preservation of:

    T semantics
    O semantics
    global authority
    tangent authority
    critic validity

Treatment result:

    T preserved             PASS
    O preserved             FAIL
    pairwise authority      PASS
    tangent authority       PASS
    critic validity         PASS

Therefore:

    SOURCE-DOMAIN GATE      FAIL
## Stop-rule interpretation

The failure is not:

- insufficient plant exposure;
- held-out leakage;
- PPO ratio drift;
- global preference-authority collapse;
- critic collapse.

It is a semantic no-regression failure.

The ensemble exposure improves one substrate property—critic validity—but does so while losing a required semantic property, Orientation.

That violates the Phase-5 core requirement:

> semantic robustness must improve without sacrificing validated source-domain MORL properties.

Under the preregistered stop rule, source T/O semantic regression stops Phase-5 escalation.

## Not reached

Because source no-regression fails:

    post-training in-ensemble semantic robustness   NOT REACHED
    P5-E3 held-out semantic validation              NOT REACHED
    P5-E4 MuJoCo cross-engine validation            NOT REACHED

These gates are not marked FAILED; they remain BLOCKED.

No post hoc plant-range expansion, objective weighting, O-specific repair, critic-specific repair, extra updates, or checkpoint search is authorized.
## Scientific result

Phase 5 provides a useful negative result:

> Direct exposure to a bounded, evidence-derived plant ensemble can preserve preference authority and improve critic validity, yet still alter the semantic mapping of a previously valid objective on the source domain.

This extends the earlier distinction between authority and semantic correctness.

The result shows that:

    plant diversity
        != automatic semantic robustness

and that:

    critic robustness
        != semantic no-regression

for preference-conditioned MORL.

## Final status

    Phase 5 method escalation     STOPPED
    first E2 formulation          FAIL
    E3                            BLOCKED
    E4                            BLOCKED
    hardware D4                   BLOCKED

Any future attempt to rescue Phase 5 requires a new explicit method contract rather than tuning this screen post hoc.
