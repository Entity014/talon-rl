# V2-B1 FiLM Authority Verdict

Status: FROZEN — V2-B1 PASS; V2-B2 AUTHORIZED

Date: 2026-09-23

## Scope

V2-B1 tested FiLM authority only. Semantic correctness was not judged.

Foundation V2, PPO/action-logprob semantics, critic body, critic head contract, expanded current-policy support, seed 73001, and 75-update budget were held fixed.

## FiLM authority result

- FiLM weight norm: 0 -> 0.31775
- FiLM bias norm: 0 -> 0.18069
- max FiLM weight gradient norm: 0.21151
- median FiLM weight gradient norm: 0.11615
- max FiLM bias gradient norm: 0.31258
- median FiLM bias gradient norm: 0.16069
- mean |gamma| at center preference: 0.01278
- mean |beta| at center preference: 0.01860
- gamma pairwise preference distance: 0.05312
- beta pairwise preference distance: 0.05732

## Incremental authority over V2-A path

- total pairwise action distance: 0.06248
- FiLM-masked pairwise action distance: 0.04746
- total preference Jacobian norm: 0.02910
- FiLM-masked preference Jacobian norm: 0.02319
- mean FiLM action authority: 0.05088

Thus masking FiLM removes measurable preference-conditioned action separation and preference Jacobian authority while retaining the RV1 direct + V2-A embedding paths.

## Foundation guardrails

- early critic EV mean: 0.53551
- early negative fraction: 0.05
- late critic EV mean: 0.28067
- late negative fraction: 0.05
- combined negative fraction: 0.05
- PPO max ratio error: 1.53e-5
- last-10 termination fraction: 0.0

All predeclared V2-B1 criteria passed.

## Decision

V2-B1 PASS.

The single-site FiLM branch acquires nonzero learned modulation and causal action authority beyond the retained V2-A path while preserving Foundation V2.

V2-B2 exact semantic evaluation is authorized.

No multi-site FiLM, adapters, routing, or auxiliary semantic mechanism is authorized before the V2-B2 result.