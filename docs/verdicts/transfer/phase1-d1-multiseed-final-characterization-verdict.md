# Phase-1 D1 — Multi-Seed Final Characterization Verdict

Status: **FROZEN — CHARACTERIZATION COMPLETE**
Date: 2026-09-26

## Scope

D1 characterizes three independent stochastic u20 -> u30 continuations of the frozen final Phase-1 method.

Seeds:

    73001
    73002
    73003

All three runs:
- started from the same validated u20 source;
- used the exact frozen Phase-1 continuation method;
- reached fixed u30;
- completed without training termination;
- preserved PPO ratio invariants;
- did not modify the canonical artifact.

This is continuation-seed characterization, not end-to-end retraining from random initialization.

## Reproducibility summary

    pairwise authority retention       3 / 3
    tangent authority retention        3 / 3
    strict all-edge floor              1 / 3

    T semantic endpoint                3 / 3
    A semantic endpoint                2 / 3
    O semantic endpoint                3 / 3
    S semantic endpoint                1 / 3

    center compromise                  3 / 3
    continuum aggregate               3 / 3

    fresh critic validity              3 / 3
    fresh-reset robustness             1 / 3
    held-out robustness                2 / 3

    PPO invariant                      3 / 3
    training termination-free          3 / 3
## Authority and durability

Using the exact formal Phase-1 matched-support evaluator:

Seed 73001:

    pairwise retention    0.9969
    tangent retention     1.0827
    minimum edge          O-C = 0.9150
    authority gate        PASS
    edge gate             PASS

Seed 73002:

    pairwise retention    1.0540
    tangent retention     1.1643
    minimum edge          S-C = 0.8047
    authority gate        PASS
    edge gate             FAIL

Seed 73003:

    pairwise retention    0.9476
    tangent retention     1.0777
    minimum edge          O-C = 0.8732
    authority gate        PASS
    edge gate             FAIL

Interpretation:

Global preference authority and local tangent authority are strongly reproducible across all three continuations.

However the stronger claim that every individual simplex edge remains above the 0.90 floor is not robustly reproducible.

Therefore the final thesis should distinguish:

    preference authority durability        strongly reproducible
    strict all-edge geometry durability    seed-sensitive / not robustly reproducible
## Semantic reproducibility

### Tracking

    PASS 3 / 3

Tracking endpoint semantics are strongly reproducible.

### Angular stability

    PASS 2 / 3

Seed 73002 fails:

    objective correctness    0.75
    physical correctness     0.50
    endpoint survival        0.875

Because this seed also exhibits broader fresh/held-out robustness failures, the A result is partially engineering-confounded rather than a clean isolated semantic reversal.

Angular semantics are therefore partially reproducible / seed-sensitive.

### Orientation

    PASS 3 / 3

Orientation endpoint semantics are strongly reproducible.

### Smoothness

Seed outcomes:

    73001   S-semantic-inconsistent
    73002   S-valid
    73003   S-engineering-confounded

Therefore Smoothness is strongly seed-sensitive.

The Phase-1 canonical S limitation remains valid for the deployment checkpoint, but it is not a universal cross-seed property of the final formulation.

The correct statement is not:

    S always fails

but:

> Smoothness semantic validity is seed-sensitive under the frozen final formulation; the canonical deployment checkpoint exhibits a clean semantic inconsistency, while another continuation realizes valid S semantics and a third is robustness-confounded.
## Center and continuum

All three seeds pass:

    center compromise >= 0.75
    continuum monotonicity >= 0.65
    continuum endpoint-between >= 0.65

Cross-seed descriptive ranges:

    continuum monotonicity    0.7448 .. 0.7604
    continuum between         0.6667 .. 0.6910
    center compromise         0.875  .. 1.000

These higher-level preference-family structures are strongly reproducible.

## Critic validity

Fresh value validity passes all three seeds.

Fresh H32 EV mean:

    73001   0.3378
    73002   0.3285
    73003   0.3780

Fresh MC64 EV mean:

    73001   0.2849
    73002   0.2986
    73003   0.3116

All seeds satisfy:
- positive H32/MC64 EV mean;
- negative-EV fraction <= 0.25.

Therefore critic value validity is strongly reproducible.

This must be kept separate from robustness failures in the same fresh-reset suite.
## Robustness

Fresh-reset survival:

    73001   1.000   PASS
    73002   0.875   FAIL
    73003   0.875   FAIL

Held-out survival:

    73001   1.000   PASS
    73002   0.875   FAIL
    73003   1.000   PASS

Therefore:

    fresh-reset robustness       1 / 3   not robustly reproducible
    held-out robustness          2 / 3   partially reproducible / seed-sensitive

This is the main D1 reproducibility red flag.

It weakens any method-level claim that the final training formulation robustly reproduces the canonical robustness regime across stochastic continuations.

It does not invalidate the canonical deployment checkpoint, which independently passes its frozen robustness evidence.

## D1 claim update

Canonical controller claims remain checkpoint-specific:

    preference authority              VALID
    authority durability              VALID
    critic compatibility              VALID
    T/A/O canonical semantics         VALID
    S canonical semantic limitation   VALID

Cross-seed formulation claims must be narrower:

    pairwise/tangent authority         strongly reproducible
    strict all-edge floor              not robustly reproducible
    T semantics                        strongly reproducible
    A semantics                        seed-sensitive
    O semantics                        strongly reproducible
    S semantics                        strongly seed-sensitive
    critic validity                    strongly reproducible
    robustness                         seed-sensitive / weak on fresh resets
## Thesis interpretation

The final Phase-1 formulation does not produce a single deterministic semantic/robustness phenotype across continuation seeds.

What reproduces most strongly is:

- causal preference authority;
- local tangent authority;
- Tracking semantics;
- Orientation semantics;
- center/continuum preference-family structure;
- critic value validity;
- PPO/training engineering invariants.

What is seed-sensitive is:

- exact all-edge simplex geometry;
- Angular semantic validity;
- Smoothness semantic validity;
- reset robustness.

This strengthens the thesis distinction between:

    architecture-level authority
    versus
    realized semantic/robustness phenotype

The method should therefore not be described as universally reproducing all canonical endpoint properties across stochastic continuations.

## D2 decision

D1 method tuning is CLOSED.

No seed result authorizes:
- changing retention strength;
- changing rewards;
- extending training;
- selecting another seed as the new canonical controller;
- reopening S or V3 method search.

The original frozen canonical seed 73001 remains the deployment candidate.

D2 deployment-oriented simulation validation is **AUTHORIZED FOR THE CANONICAL CONTROLLER**, with explicit awareness that robustness and some semantic properties are not fully reproducible across continuation seeds.

D2 must test the frozen canonical controller, not optimize across seeds.
## Source artifacts

Runner:

    scripts/rl/phase1_d1_isolated_runner.py

Characterization:

    scripts/rl/phase1_d1_characterize.py
    scripts/rl/phase1_d1_formal_authority.py

Seed outputs:

    runs/phase1_d1_seed_73001/
    runs/phase1_d1_seed_73002/
    runs/phase1_d1_seed_73003/

Aggregate raw characterization:

    runs/phase1_d1_characterization/multiseed_characterization.json

Corrected taxonomy:

    runs/phase1_d1_characterization/multiseed_characterization_v2.json

Formal authority:

    runs/phase1_d1_characterization/formal_authority_multiseed.json

Canonical freeze:

    docs/protocols/freeze/phase1-d0-canonical-freeze-manifest.md

D1 contract:

    docs/contracts/transfer/phase1-d1-multiseed-final-characterization-contract.md
