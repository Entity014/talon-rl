# Phase-1 D1 — Multi-Seed Final Characterization Contract

Status: **PREDECLARED — CHARACTERIZATION ONLY**
Date: 2026-09-25

## Scientific question

How reproducibly does the frozen final Phase-1 formulation recover the validated qualitative result across independent stochastic continuations of the final u20 -> u30 training stage?

This is **not** method selection and **not** optimization.

The question is not:

> How can every seed be made to pass 4/4 objectives?

The question is:

> Given the frozen final formulation, which validated properties reproduce across seeds, and which limitations are seed-sensitive?

## Independence scope

The required D1 runs are independent **final-stage continuation seeds** conditioned on the same validated u20 source state.

Common frozen source:

    runs/authority_isolated_coverage2_control-2026-09-25/resume_state.pt

Common frozen authority reference:

    runs/authority_isolated_u20_authority_support-2026-09-25/u20_authority_support.npz

Each seed independently executes the exact final u20 -> u30 simplex-edge-retention treatment.

This must be described in the thesis as:

    independent stochastic continuation seeds from a common validated u20 source

and not as:

    independent end-to-end training seeds from random initialization

unless a separate end-to-end replication study is later performed.

## Seeds

Predeclared continuation seeds:

    73001   canonical/reference seed
    73002   replication seed 1
    73003   replication seed 2

No seed may be dropped, replaced, or rerun with changed hyperparameters because its result is unfavorable.
## Frozen training budget and method

For every seed:

    start update          u20
    final update          u30
    continuation updates  10

Frozen method:
- authority-isolated actor;
- wide critic;
- projected PPO/tail repair;
- lambda = .95;
- kappa = .05;
- all-simplex-edge asymmetric retention;
- gamma_edge = .90;
- rho = .25;
- beta0 = 2.497041993384243;
- same preference schedule;
- same critic support/head-refresh contract;
- same optimizer configuration;
- same reset/training distribution.

The sole varying factor is stochastic seed.

## Fixed checkpoint policy

The characterization checkpoint is:

    u30

for every seed.

u20 and intermediate snapshots may be reported diagnostically but cannot replace u30.

Semantic endpoint results, critic scores, survival, or authority scores may not select a different checkpoint.

## Output isolation

Every seed must write to a separate run directory.

Required naming:

    runs/phase1_d1_seed73001-2026-09-26/
    runs/phase1_d1_seed73002-2026-09-26/
    runs/phase1_d1_seed73003-2026-09-26/

No run may overwrite the canonical Phase-1 artifact.

The canonical checkpoint remains frozen even if a replication seed performs better.
## Characterization group A — preference authority

At u20 and u30 report:

1. pairwise action-distance authority;
2. simplex-tangent Jacobian authority;
3. all ten T/A/O/S/C edge-energy retentions;
4. heavy-heavy edge retention;
5. heavy-center edge retention;
6. minimum individual edge retention.

Primary durability quantities:

    pairwise retention = authority_u30 / authority_u20
    tangent retention  = tangent_u30 / tangent_u20

Frozen reference gates inherited from the final method:

    pairwise retention >= .90
    tangent retention  >= .90
    minimum important edge retention >= .90

For characterization report:
- per-seed values;
- mean;
- standard deviation;
- min/max;
- pass count out of 3.

Do not alter edge-retention strength if one seed fails.

## Characterization group B — semantics

At fixed u30, run the exact frozen H2a protocol.

Endpoint preferences:

    T = [.7,.1,.1,.1]
    A = [.1,.7,.1,.1]
    O = [.1,.1,.7,.1]
    S = [.1,.1,.1,.7]
    C = [.25,.25,.25,.25]

Per objective report:
- normalized-objective heavy-vs-center delta;
- physical-proxy heavy-vs-center delta;
- correctness fraction across matched suites;
- endpoint survival.

Frozen endpoint validity criterion:

    normalized direction correctness >= .75
    physical direction correctness   >= .75
    endpoint survival                >= .95

T/A/O are reproducibility targets.

S is **characterized, not used as a required acceptance target**.
## S limitation characterization

For every seed classify S at u30 as:

    S-valid:
        both normalized and physical correctness >= .75

    S-semantic-inconsistent:
        engineering gates valid,
        but normalized and/or physical semantic correctness < .75

    S-engineering-confounded:
        endpoint survival < .95,
        critic invalid,
        or another broad engineering failure prevents clean semantic interpretation

No category is treated as an optimization target.

Interpretation:

- if S passes in some seeds, report seed sensitivity;
- if S remains semantically inconsistent with engineering validity, the scoped limitation reproduces;
- if S failure becomes engineering-confounded, report the limitation as less cleanly isolated in that seed.

Do not require S to reproduce the exact canonical 0.50/0.50 correctness fractions.

## Center and continuum

Run the exact six heavy-heavy paths:

    T-A
    T-O
    T-S
    A-O
    A-S
    O-S

with:

    alpha = 0,.25,.5,.75,1

Report per seed:

    aggregate monotonicity
    aggregate endpoint-between fraction
    center-compromise fraction

Frozen reference gates:

    monotonicity >= .65
    endpoint-between >= .65
    center compromise >= .75

Also report S-containing and non-S-containing paths separately as descriptive analysis.
## Characterization group C — critic and engineering validity

### Critic

Run fresh current-policy value validation.

Required metrics:
- fresh H32 EV mean;
- fresh H32 negative-EV fraction;
- fresh H32 bias;
- MC64 validation when available under the validated critic protocol.

Frozen H32 gate:

    EV mean > 0
    negative-EV fraction <= .25

### PPO contract

Require:

    max ratio invariant error <= 1e-4

### Robustness

Report:
- endpoint-suite minimum survival;
- continuum-suite minimum survival;
- held-out/fresh-reset survival;
- termination topology;
- non-finite action/value count.

Frozen reference:
- semantic endpoint survival >= .95;
- held-out/fresh-reset minimum survival >= .95;
- no new broad termination topology;
- no non-finite outputs.

The known canonical T-S continuum seed with survival .875 is retained as a documented canonical robustness exception and must not be silently removed from comparison.

## Characterization group D — limitation reproducibility

The final cross-seed interpretation must answer:

1. Are T/A/O semantics reproducible?
2. Is finite preference authority durable?
3. Is critic validity reproducible?
4. Is robustness broadly reproducible?
5. Is S primarily:
   - consistently semantically limited,
   - seed-sensitive,
   - or engineering-confounded?
6. Does any new objective besides S develop a recurrent semantic limitation?
## Red-flag rules

These rules do not trigger method repair. They change the thesis characterization.

### Major reproducibility red flag

Flag if at least 2 of 3 seeds show any of:

- T semantic endpoint FAIL;
- A semantic endpoint FAIL;
- O semantic endpoint FAIL;
- pairwise authority retention < .90;
- tangent authority retention < .90;
- critic gate FAIL;
- held-out/fresh-reset survival < .95.

Interpretation:

    the final formulation is not robustly reproducible for that claimed property

Do not tune the method after observing this.

### Isolated-seed instability

If exactly 1 of 3 seeds fails a required T/A/O/authority/critic/robustness property:

    report seed sensitivity / limited reproducibility

Do not discard the seed and do not replace it.

### S outcome

S alone never triggers a major red flag unless its result is engineering-confounded by a broad failure.

## Aggregate thesis interpretation

The characterization is considered **strongly reproducible** for a property when all 3 seeds pass its frozen gate.

It is **partially reproducible / seed-sensitive** when 2 of 3 pass.

It is **not robustly reproducible** when fewer than 2 of 3 pass.

These labels summarize evidence only; they do not select a new controller.

The original canonical u30 checkpoint remains the deployment candidate unless a deployment-specific safety gate later rejects it.
## Statistical reporting

For continuous metrics report:

    n = 3 seeds
    mean
    standard deviation
    median
    min
    max

Do not present n=3 as high-powered population inference.

For binary gates report:

    pass_count / 3

For semantic correctness fractions report both:
- per-seed suite fraction;
- cross-seed descriptive summary.

No significance testing is required for D1.

## Forbidden adaptations

During D1 do not:
- modify architecture;
- modify rewards;
- modify semantic proxies;
- change retention gamma/rho/beta0;
- change critic fitting;
- change PPO/tail repair;
- replace seeds;
- extend training past u30 because a seed looks bad;
- choose u20/u25/u29 because semantic results look better;
- tune reset suites;
- remove failed suites;
- reopen S mechanism search;
- reopen V3.

## D1 completion rule

D1 is complete when:
1. all three predeclared seed continuations reach u30;
2. all frozen authority/semantic/engineering evaluations are run;
3. no seed is omitted;
4. one consolidated cross-seed verdict is frozen.

D1 completion does **not** require every property to pass.

After the consolidated verdict:

    D2 deployment-oriented simulation validation

is authorized if the canonical controller remains safe enough for deployment testing under the already validated canonical evidence and D1 does not reveal a new broad catastrophic instability.

## Source-of-truth

Canonical controller freeze:

    docs/protocols/freeze/phase1-d0-canonical-freeze-manifest.md

Canonical semantic result:

    docs/verdicts/authority/authority-isolated-h2a-u30-semantic-validity-verdict.md

Canonical S limitation:

    docs/verdicts/authority/authority-isolated-s-trajectory-temporal-decomposition-verdict.md

V3 closure:

    docs/closures/phase_closures/v3-phase-closure-verdict.md
