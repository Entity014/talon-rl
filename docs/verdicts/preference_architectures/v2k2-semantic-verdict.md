# V2-K2 Semantic Accumulation Verdict

Status: **FROZEN — V2-K2 FAIL; V2-K3 NOT AUTHORIZED; V2-K BRANCH CLOSED**

Date: 2026-09-24

## Question

After V2-K1 demonstrated genuine preference-dependent private residual
subspaces, do semantic competencies accumulate across objectives under the
exact frozen Foundation-V2 semantic contract?

This is the first semantic evaluation in the V2-K branch.

## Frozen evaluation contract

V2-K2 uses the exact V2-B2 semantic evaluator:

- matched-reset endpoint suites
- T/A/O/S-heavy versus matched center
- 4 suites
- 64-step evaluation horizon
- unchanged objective and physical semantic definitions
- unchanged endpoint PASS threshold: objective >= 0.75 and physical >= 0.75
- unchanged survival threshold: >= 0.95
- unchanged continuum paths: T-A, T-O, T-S
- unchanged interpolation alphas: 0, 0.25, 0.5, 0.75, 1
- unchanged continuum threshold: >= 0.65
- unchanged critic guardrail
- measurement only, zero additional training updates

Checkpoint:
- V2-K1 update 75, seed 73001

## Endpoint result

| Axis | Objective correctness | Physical correctness | Survival | PASS |
|---|---:|---:|---:|---|
| Tracking | 0.75 | 0.75 | 1.00 | **PASS** |
| Angular | 0.25 | 0.50 | 1.00 | FAIL |
| Orientation | 0.25 | 0.25 | 1.00 | FAIL |
| Smoothness | 0.00 | 0.00 | 1.00 | FAIL |

Only Tracking passes.

Detailed mean heavy-vs-center changes:

- Tracking objective delta = **+7.97e-5**
- Tracking physical delta = **+7.08e-4**
  - despite the positive mean physical delta, three of four matched suites have
    the correct lower-is-better direction, satisfying the frozen 0.75 fraction
    threshold.

- Angular objective delta = **+9.97e-5**
- Angular physical delta = **-0.00677**

- Orientation objective delta = **-2.59e-4**
- Orientation physical delta = **+0.03952**

- Smoothness objective delta = **-2.78e-4**
- Smoothness physical delta = **+0.03316**

## Continuum result

- monotonicity fraction = **0.60417**
- required = **0.65**
- endpoint-between fraction = **0.39583**
- required = **0.65**

Both continuum criteria fail.

## Foundation / collateral guardrails

The semantic failure is not accompanied by a foundation collapse.

- minimum survival = **1.0**
- maximum tracking ratio to center = **1.0009**
- critic H32 EV mean = **0.3145**
- critic negative fraction = **0.10**
- critic mean absolute bias = **0.02887**

All non-semantic guardrails pass.

## Comparison with V2-B

| Architecture | Endpoint PASS axes | Continuum monotonicity | Endpoint-between | Survival |
|---|---|---:|---:|---:|
| V2-B | Angular only | 0.64583 | 0.52083 | 1.0 |
| V2-K | Tracking only | 0.60417 | 0.39583 | 1.0 |

V2-K therefore does **not** convert specialization authority into accumulated
semantic competence.

Instead, semantic success rotates again:
- V2-B final semantic winner: **Angular**
- V2-K final semantic winner: **Tracking**

and the continuum metrics are lower than V2-B.

## Why this result is important

V2-K1 had already removed the authority-stage objections that blocked V2-C and
V2-H.

V2-K1 established:

- preference-separated coefficient vectors;
- rank-3 coefficient geometry;
- preference-separated private residual directions;
- mean residual direction diversity (1 - cosine) = **0.05069**;
- rank-3 centered residual manifold;
- strongly preference-dependent module contribution shares;
- causal private-path action authority;
- preserved Foundation V2.

Therefore K2 failure cannot be explained by:
- a uniform router;
- dead private modules;
- a rank-1 coefficient map;
- almost-parallel private residual directions;
- a disconnected modular path;
- PPO / critic / survival collapse.

The branch successfully creates the intended preference-private architecture,
but the semantic endpoint result still does not accumulate across objectives.

## Interpretation

This is stronger evidence than the separate V2-C and V2-H failures.

V2-C showed:
> private capacity without preference allocation is insufficient.

V2-H showed:
> conditional parameter variation dominated by a common direction is
> insufficient.

V2-K showed:
> even when continuous preference conditioning drives genuinely distinct
> private residual subspaces with causal action authority, semantic competence
> still does not accumulate under the current validated training objective.

The observed winner rotation therefore cannot be attributed solely to the lack
of spontaneously learned parameter specialization.

The likely limitation has moved beyond the narrow architecture-sharing
hypothesis tested by C/H/K.

## Decision

**V2-K2 FAIL.**

Per the predeclared stop rule:

- V2-K3 retention/path audit is **NOT AUTHORIZED**
- no multi-seed V2-K confirmation
- no deployment benchmark
- no module-count tuning
- no coefficient normalization/tuning
- no diversity or orthogonality regularizer
- no objective-to-module assignment
- no learning-rate / lambda / entropy tuning

The V2-K branch is closed.

The final thesis reference remains:

> **V2-B + Foundation V2 + GAE lambda = 0.95 + no retention intervention**

## Scientific implication

The architecture study now supports a more general limitation statement:

> Increasing conditioning authority, adding private capacity, learning
> multidimensional preference-conditioned parameter variation, and even
> combining continuous preference coefficients with genuinely distinct private
> residual subspaces were all insufficient to produce robustly accumulated
> four-objective semantic competence under the validated shared MORL training
> objective.

This is a scope-bounded empirical conclusion, not a universal claim against
modular or hypernetwork policies.

Primary artifacts:
- `scripts/rl/v2k2_semantic_eval.py`
- `runs/v2k2_semantic_eval-2026-09-24/semantic_report.json`
- `runs/v2k2_semantic_eval-2026-09-24/PROVENANCE_MANIFEST.json`
- `docs/verdicts/preference_architectures/v2k1-private-subspace-verdict.md`
- `docs/contracts/preference_architectures/v2k-contract.md`
