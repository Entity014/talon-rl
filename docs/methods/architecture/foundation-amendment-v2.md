# Foundation Amendment V2 — Architecture-Agnostic Critic Support

Status: PREDECLARED BEFORE SEMANTIC RE-EVALUATION

Date: 2026-09-23

## Motivation

The selected-12 critic support contract was sufficient for repaired RV1 but became marginal under V2-A visitation. Expanded current-policy support passed the late critic gate without changing actor, critic body, PPO, rewards, objectives, or evaluation semantics.

Therefore selected-12 is superseded as the final architecture-comparison foundation by an architecture-agnostic expanded current-policy support rule.

## Frozen amended rule

- Actor initialization: architecture-specific, each function-preserving from the same frozen RV1-A initialization.
- Objectives: validated normalized 4D objective vector, unchanged.
- PPO/action-logprob semantics: repaired version, unchanged.
- Critic body: shared frozen body during actor training, unchanged.
- Critic heads: objective-specific linear heads, ridge lambda = 1.
- Support: current-policy reset-diverse expanded pool, same construction for every architecture.
- Anchor seed specs: same fixed 6 specs chosen from the same 12 initialization candidates.
- Adaptive supports: all recent 24 current-policy support candidates retained.
- Phase coverage: both early-H32 and late-H32 support units retained.
- Head refresh cadence: every actor update, before advantage computation.
- Checkpoint rule: after actor update, a final current-policy head refresh MUST be performed before saving evaluation checkpoints.
- Seed: 73001 for one-seed method-selection screen.
- Updates: 75.
- Semantic evaluator: unchanged RV1-C endpoint + continuum matched reset suites and thresholds.
- No FiLM, adapters, routing, auxiliary semantic losses, curriculum, rehearsal, or coefficient guidance.

## Fairness requirement

The amended foundation must be applied identically to RV1 and V2-A. No semantic architecture comparison is valid across different foundation versions.

## Decision matrix

- RV1 FAIL, V2-A PASS -> embedding architecture supported.
- RV1 PASS, V2-A PASS -> architecture escalation not necessary.
- RV1 FAIL, V2-A FAIL -> V2-A insufficient; V2-B may be justified.
- RV1 PASS, V2-A FAIL -> embedding treatment not supported.

## Stop rule

No V2-B/FiLM experiment is authorized until RV1 and V2-A have both been trained and semantically evaluated under this amended foundation.