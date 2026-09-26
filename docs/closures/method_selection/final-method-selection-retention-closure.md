# Retention-Branch Closure and Provisional Method Selection

> **Supersession note (2026-09-24):** Retention remains closed and V2-C closed at its authority gate. The later V2-H preference-conditioned policy-manifold branch passed exact H0 function preservation but failed H1 because generated T/A/O/S-heavy weight deltas remained dominated by a common parameter direction despite nontrivial centered rank and measurable action authority. H2 semantics was not authorized. V2-B therefore remains the final reference.

Status: **SOURCE OF TRUTH — RETENTION, V2-C, AND V2-H CLOSED; FINAL REFERENCE FIXED**

Date: 2026-09-24

## 1. Final reference configuration

The thesis reference configuration is:

- architecture: **V2-B single-site FiLM preference-conditioned policy**
- training foundation: **Foundation V2**
- validated normalized 4D objectives
- repaired squashed-action PPO / matched action-log-prob semantics
- shared critic body with objective-specific heads
- representative reset-diverse critic supervision
- critic ridge regularization lambda = 1 where the repaired critic fit uses ridge
- GAE lambda = 0.95
- **no retention intervention**
- no rehearsal, semantic auxiliary loss, hard retention constraint, or outcome surrogate in the final reference method

This configuration is selected because it is the clean validated architecture/foundation baseline after the causal-repair sequence, while no tested retention intervention demonstrated robust multi-seed superiority.

## 2. Retention ladder — final status

| Candidate | Optimization behavior | Retention result | Final status |
|---|---|---|---|
| Gradient retention | technically feasible | insufficient semantic preservation | CLOSED |
| Hard action retention | preserves retained behavior locally | restricts plasticity | CLOSED |
| Hard short-trajectory retention | stronger local preservation | stronger plasticity collapse; still incomplete | CLOSED |
| Soft Delta-a rehearsal, fixed beta | full step magnitude retained | rehearsal gradient grows and hijacks update direction | REJECTED |
| Bounded soft rehearsal | stable optimization geometry | solves gradient dominance, not semantic forgetting by itself | **MECHANISM VALIDATED** |
| Bounded Delta-a rehearsal | stable estimator and bounded update | single-seed gain, no robust multi-seed advantage | **CONTROLLED BASELINE ONLY** |
| Semantic-outcome score-function rehearsal | bounded strength | gradient direction non-reproducible | CLOSED |
| Semantic-outcome local pathwise FD | estimator family changed only | gradient direction still non-reproducible | CLOSED |
| Semantic-outcome deterministic surrogate | more reproducible surrogate gradients | held-out semantic fidelity fails across seeds | CLOSED |

No additional retention-content search is authorized within the current thesis scope.

## 3. What was actually solved

The rehearsal stability/plasticity problem was solved.

For bounded rehearsal,

alpha_t = min(beta_cap, rho * ||g_mixed|| / (||g_rehearsal|| + eps))

with rho = 0.25.

Across the three-seed bounded-Delta-a validation:
- mean cosine between mixed and total update directions = 0.9745
- minimum observed cosine = 0.9698
- preference separation remained effectively unchanged from control
- active rehearsal updates were normally capped at 0.25 of the mixed-gradient norm
- no rehearsal-gradient hijack was observed
- no parameter-step collapse was observed
- no preference-authority collapse was observed

Therefore failures of semantic retention after bounded rehearsal cannot be attributed to the earlier stability/plasticity implementation failure.

## 4. What was not solved

Preference-conditioned Delta-a memory did not produce a robust semantic-retention advantage.

Three paired seeds:

| Seed | Control PASS events | Bounded Delta-a PASS events | Paired effect |
|---|---:|---:|---:|
| 980001 | 4 | 6 | +2 |
| 981001 | 9 | 7 | -2 |
| 982001 | 7 | 6 | -1 |

Aggregate:
- control PASS events = 20
- bounded Delta-a PASS events = 19
- mean paired effect = -0.333
- sample SD = 2.082
- mean semantic-score effect = -0.00781
- positive semantic-score effect in 1/3 seeds

The original positive 6-vs-4 single-seed result is retained as a genuine positive instance, but it is insufficient for a method-level superiority claim.

## 5. Semantic-outcome branch closure

The semantic-outcome branch was tested through three estimator/representation families while keeping the retained semantic object fixed.

### 5.1 Score-function estimator

CRN, centering, and replica averaging improved some magnitude statistics but did not recover a reproducible normalized direction.

### 5.2 Local pathwise finite-difference estimator

Replacing likelihood-ratio estimation with CRN two-sided local finite differences and chain propagation did not recover a reproducible gradient direction at u4 or u7.

### 5.3 Deterministic local surrogate

Surrogate-derived gradients became more reproducible, but held-out semantic prediction failed.

Minimal ridge held-out EV was negative for nearly all S/O/T objective and physical targets across u4/u7. A richer temporal representation increased in-sample fit without repairing held-out generalization, and nonlinear capacity did not rescue the mapping.

Therefore the current semantic-outcome formulation lacks a validated seed-generalizable local geometry suitable for rehearsal.

## 6. Scientific conclusion

Preferred thesis wording:

> The observed semantic forgetting could not be resolved reliably by retaining local gradients, policy actions, short-horizon trajectory outcomes, or preference-conditioned action-response differences. Hard retention increasingly restricted optimization plasticity, while soft rehearsal required explicit gradient budgeting to remain stable. Bounding rehearsal-gradient contribution robustly preserved optimization geometry, but the retained preference-conditioned action-response proxy did not produce consistent semantic-retention gains across training seeds. Attempts to replace that proxy with direct semantic-outcome rehearsal were limited by non-reproducible local gradients or poor seed-generalization of learned outcome surrogates.

A shorter contribution statement:

> The study separates rehearsal stability from memory-content sufficiency: bounded gradient budgeting is a validated optimization mechanism, whereas the tested retained representations are insufficient for robust semantic retention.

## 7. Thesis interpretation

Bounded Delta-a rehearsal should appear as:
- an ablation / attempted retention mechanism,
- evidence that replay-gradient budgeting can stabilize rehearsal,
- a negative or limited retention result.

It should **not** appear as:
- the final proposed successful retention method,
- a robustly superior baseline,
- evidence that semantic forgetting is solved.

The final reference method is V2-B + Foundation V2 + GAE lambda = 0.95 with no retention intervention.

## 8. Scope boundary

The following are explicitly outside the current method-selection branch:
- latent semantic rehearsal,
- learned invariant trajectory memory,
- larger replay memories,
- adaptive memory selection,
- longer semantic memories,
- new surrogate representations,
- new retention coefficients or larger rho,
- new outcome-estimator families.

Any of these would constitute a new research hypothesis rather than completion of the current branch.

## 9. Next authorized work

Only consolidation/confirmation work is authorized:

1. freeze final reference configuration and provenance;
2. consolidate multi-seed architecture/foundation results;
3. consolidate retention negative results and mechanism evidence;
4. prepare thesis figures/tables;
5. write contribution, limitation, and discussion sections;
6. run only predeclared final confirmation experiments if a thesis claim still lacks direct multi-seed evidence.

No exploratory retention debugging is authorized.

## 10. Supersession note

This document supersedes the **authorization state and method-selection status** in earlier master synthesis documents that were written before RV1/V2-B and the retention ladder completed.

Earlier diagnostic evidence remains valid unless explicitly superseded by later verdict artifacts.

Primary closing artifacts:
- docs/verdicts/preference_architectures/v2b-bounded-deltaa-multiseed-verdict.md
- docs/verdicts/preference_architectures/v2b-semantic-outcome-estimator-repair-verdict.md
- docs/verdicts/preference_architectures/v2b-semantic-outcome-pathwise-estimator-verdict.md
- docs/verdicts/preference_architectures/v2b-semantic-outcome-surrogate-gate-verdict.md
- runs/v2b_bounded_deltaa_multiseed-2026-09-24/aggregate.json
- runs/v2b_bounded_deltaa_multiseed-2026-09-24/FINAL_PROVENANCE_MANIFEST.json
- runs/v2b_semantic_outcome_rehearsal_gate_v2-2026-09-24/SURROGATE_GATE_PROVENANCE.json
