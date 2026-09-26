# Appendix Traceability Map

Status: THESIS-PRODUCTION SUPPORT DOCUMENT

This appendix preserves experiment-level provenance without requiring the main thesis narrative to follow the chronological debug sequence.

## A. Foundation repair

| Thesis claim | Primary evidence artifact |
|---|---|
| Objective/reward semantics required repair before preference interpretation | `docs/contracts/general/master-synthesis-repaired-foundation-contract.md` and repaired T3/T4 artifacts |
| PPO action and log-probability semantics were repaired | `docs/contracts/general/master-synthesis-repaired-foundation-contract.md` |
| Critic freshness and support coverage were required for valid objective-level credit | `docs/verdicts/preference_architectures/rv1-method-selection-verdict.md` and critic repair artifacts |
| Corrected stochastic/deterministic gradient comparison converges in the zero-variance limit | repaired C40-C43 artifacts listed in `docs/contracts/general/master-synthesis-repaired-foundation-contract.md` |

## B. Architecture ladder

| Thesis claim | Primary evidence artifact |
|---|---|
| RV1 direct conditioning is semantically insufficient on the validated foundation | `docs/verdicts/preference_architectures/rv1-method-selection-verdict.md` |
| V2-A embedding acquires causal action authority | `docs/verdicts/preference_architectures/v2a1-authority-foundation-verdict.md` |
| V2-B FiLM acquires incremental causal authority | `docs/verdicts/preference_architectures/v2b1-authority-verdict.md` |
| V2-B remains semantically incomplete | `docs/verdicts/preference_architectures/v2b2-semantic-verdict.md` |

## C. Temporal-credit and path diagnosis

| Thesis claim | Primary evidence artifact |
|---|---|
| λ=1.0 is locally closer to MC32 than λ=0.95 | `docs/verdicts/preference_architectures/v2b-lambda-horizon-verdict.md` |
| Static mixed-batch geometry alone does not explain collapse | `docs/verdicts/preference_architectures/v2b-mixed-batch-geometry-verdict.md` |
| Optimizer history contributes to path dependence but is not a complete explanation | `docs/verdicts/preference_architectures/v2b-optimizer-history-isolation-verdict.md` |
| Parameter-path divergence is not reducible to simple step-length amplification | `docs/verdicts/preference_architectures/v2b-parameter-path-verdict.md` |

## D. Retention ladder

| Thesis claim | Primary evidence artifact |
|---|---|
| Gradient retention is insufficient | `docs/verdicts/preference_architectures/v2b-reference-gradient-retention-gate-verdict.md` |
| Hard trajectory retention strengthens preservation but worsens plasticity | `docs/verdicts/diagnostics/v2b-trajectory-semantic-retention-gate-verdict.md` |
| Fixed soft rehearsal can hijack the update direction | `docs/verdicts/preference_architectures/v2b-soft-semantic-rehearsal-gate-verdict.md` |
| Bounded auxiliary-gradient control solves optimization takeover | `docs/verdicts/preference_architectures/v2b-bounded-semantic-rehearsal-gate-verdict.md` |
| Bounded Δa lacks robust multi-seed efficacy | `docs/verdicts/preference_architectures/v2b-bounded-deltaa-multiseed-verdict.md` |
| Score-function semantic-outcome gradients are unstable | `docs/verdicts/preference_architectures/v2b-semantic-outcome-estimator-repair-verdict.md` |
| Local pathwise semantic-outcome gradients are also unstable | `docs/verdicts/preference_architectures/v2b-semantic-outcome-pathwise-estimator-verdict.md` |
| Deterministic semantic surrogates fail held-out fidelity | `docs/verdicts/preference_architectures/v2b-semantic-outcome-surrogate-gate-verdict.md` |

## E. Literature-guided round

| Thesis claim | Primary evidence artifact |
|---|---|
| DER-style diverse replay adaptation fails the short semantic gate | `docs/closures/method_selection/literature-guided-retention-closure.md` and L1 report |
| Multi-timescale Policy Consolidation adaptation fails the short semantic gate | `docs/closures/method_selection/literature-guided-retention-closure.md` and L2 report |
| Literature-guided retention round does not change final method selection | `docs/closures/method_selection/literature-guided-retention-closure.md` |

## F. Final architecture-limitation branch

| Thesis claim | Primary evidence artifact |
|---|---|
| V2-C modular residual initialization is exactly function-preserving relative to V2-B | `docs/verdicts/preference_architectures/v2c0-function-preserving-verdict.md` |
| V2-C private experts become active but the router remains near-uniform and fails preference-dependent specialization | `docs/verdicts/preference_architectures/v2c1-modular-authority-verdict.md` |
| V2-C semantic gate was not authorized because the authority/specialization mechanism failed | `docs/verdicts/preference_architectures/v2c1-modular-authority-verdict.md` |

## G. Policy-parameter-manifold branch

| Thesis claim | Primary evidence artifact |
|---|---|
| V2-H low-rank hypernetwork initialization is exactly function-preserving relative to V2-B | `docs/verdicts/preference_architectures/v2h0-function-preserving-verdict.md` |
| V2-H learns preference-dependent coefficients and multidimensional centered parameter variation, but generated absolute weight deltas remain nearly parallel across heavy preferences | `docs/verdicts/preference_architectures/v2h1-parameter-manifold-verdict.md` |
| V2-H semantic accumulation gate was not authorized because the predeclared direction-diversity criterion failed | `docs/verdicts/preference_architectures/v2h1-parameter-manifold-verdict.md` |

## H. Combination-rescue branch

| Thesis claim | Primary evidence artifact |
|---|---|
| V2-K combines continuous preference coefficients with four unlabeled private residual modules and is exactly function-preserving at initialization | `docs/verdicts/preference_architectures/v2k0-function-preserving-verdict.md` |
| V2-K learns preference-separated coefficients, rank-3 private-residual geometry, distinct residual directions, preference-dependent module shares, and causal private-path action authority | `docs/verdicts/preference_architectures/v2k1-private-subspace-verdict.md` |
| Despite successful private-subspace specialization, V2-K fails the frozen semantic gate and rotates the single endpoint success back to Tracking | `docs/verdicts/preference_architectures/v2k2-semantic-verdict.md` |
| V2-K continuum metrics are below V2-B and K3 is not authorized | `docs/verdicts/preference_architectures/v2k2-semantic-verdict.md` |

## I. Competence-floor / max-min optimization branch

| Thesis claim | Primary evidence artifact |
|---|---|
| Norm-matched max-min combination repairs negative local per-objective PPO gains at u50/u75 without step collapse | `docs/verdicts/general/competence-floor-maxmin-audit-verdict.md` |
| Paired u50→u75 pilot preserves non-negative predicted gains on every treatment update while Foundation V2 remains valid | `docs/verdicts/general/competence-floor-pilot-verdict.md` |
| Despite repaired local gradient geometry, acquired Orientation semantic competence is not retained in the floor arm | `docs/verdicts/general/competence-floor-pilot-verdict.md` |
| Competence-floor branch is closed in the tested formulation and no further tuning is authorized | `docs/verdicts/general/competence-floor-pilot-verdict.md` |

## J. Trajectory-level objective sufficiency audit

| Thesis claim | Primary evidence artifact |
|---|---|
| Local objective validity does not imply trajectory-level semantic sufficiency | `docs/verdicts/diagnostics/trajectory-objective-sufficiency-verdict.md` |
| Across 64 finite axis-transitions, Delta training return has weak rank alignment with semantic objective and physical margins | `docs/verdicts/diagnostics/trajectory-objective-sufficiency-verdict.md` |
| Positive training-return changes frequently coincide with semantic deterioration and can accompany PASS→FAIL transitions | `docs/verdicts/diagnostics/trajectory-objective-sufficiency-verdict.md` |
| No new method is authorized until the missing trajectory-level semantic statistic is identified | `docs/verdicts/diagnostics/trajectory-objective-sufficiency-verdict.md` |

## K. Trajectory-information attribution audit

| Thesis claim | Primary evidence artifact |
|---|---|
| Heavy-only scalar objective return discards relational temporal information relevant to semantic competence | `docs/verdicts/diagnostics/trajectory-information-attribution-verdict.md` |
| Late, final-window, worst-window, and persistence heavy-vs-center advantages track semantic-margin changes far better than J_i alone | `docs/verdicts/diagnostics/trajectory-information-attribution-verdict.md` |
| Late relational advantage deteriorates in all observed PASS→FAIL transitions | `docs/verdicts/diagnostics/trajectory-information-attribution-verdict.md` |
| Generic heavy-only tail statistics are weaker than matched relational trajectory descriptors | `docs/verdicts/diagnostics/trajectory-information-attribution-verdict.md` |
| Reward redesign remains unvalidated; held-out relational-persistence validation is required first | `docs/verdicts/diagnostics/trajectory-information-attribution-verdict.md` |

## L. Held-out relational-persistence validation

| Thesis claim | Primary evidence artifact |
|---|---|
| Heavy-versus-center relational alignment generalizes to three unseen training seeds | `docs/verdicts/diagnostics/relational-persistence-heldout-verdict.md` |
| Relational mean signals outperform heavy-only total and heavy-only late controls on held-out semantic correctness changes | `docs/verdicts/diagnostics/relational-persistence-heldout-verdict.md` |
| Temporal relational descriptors reproduce across all three seeds and are highly sensitive to PASS→FAIL events | `docs/verdicts/diagnostics/relational-persistence-heldout-verdict.md` |
| Temporal persistence does not outperform relational mean by the frozen global-correlation margin | `docs/verdicts/diagnostics/relational-persistence-heldout-verdict.md` |
| Reward redesign remains blocked; relationality is primary and temporal persistence secondary | `docs/verdicts/diagnostics/relational-persistence-heldout-verdict.md` |

## M. Relational-objective design audit

| Thesis claim | Primary evidence artifact |
|---|---|
| Pure heavy-versus-center relation remains substantially more aligned with semantic correctness than heavy-only return | `docs/verdicts/diagnostics/relational-objective-design-audit-verdict.md` |
| Scalar `J_H + eta(J_H-J_C)` improves alignment but no frozen eta satisfies all authorization criteria | `docs/verdicts/diagnostics/relational-objective-design-audit-verdict.md` |
| Moderate eta controls most center-degradation false gains, but a heavy-performance guardrail does not eliminate semantic PASS→FAIL false approvals | `docs/verdicts/diagnostics/relational-objective-design-audit-verdict.md` |
| Some semantic forgetting occurs even when both absolute heavy return and relational margin improve | `docs/verdicts/diagnostics/relational-objective-design-audit-verdict.md` |
| Direct relational reward training remains blocked | `docs/verdicts/diagnostics/relational-objective-design-audit-verdict.md` |

## N. Semantic-gate factorization audit

| Thesis claim | Primary evidence artifact |
|---|---|
| Mean relational deterioration accompanies most held-out semantic forgetting events | `docs/verdicts/diagnostics/semantic-gate-factorization-verdict.md` |
| Worst-suite or phase collapse occurs in 85.7% of held-out PASS→FAIL events | `docs/verdicts/diagnostics/semantic-gate-factorization-verdict.md` |
| Every held-out PASS→FAIL event contains a context-level sign loss | `docs/verdicts/diagnostics/semantic-gate-factorization-verdict.md` |
| Two semantic failures occur despite both objective and physical mean relational margins improving | `docs/verdicts/diagnostics/semantic-gate-factorization-verdict.md` |
| Distributional context failure is real but not established as the dominant mechanism; distributional-objective training remains blocked | `docs/verdicts/diagnostics/semantic-gate-factorization-verdict.md` |

## O. Prospective context-incremental retention audit

| Thesis claim | Primary evidence artifact |
|---|---|
| Same-checkpoint context correctness is avoided as a tautological predictor of PASS | `docs/verdicts/diagnostics/context-incremental-retention-verdict.md` |
| Only 18 PASS-origin held-out transitions are available, making the prospective test underpowered by the frozen rule | `docs/verdicts/diagnostics/context-incremental-retention-verdict.md` |
| Source suite×phase context features do not improve leave-one-seed-out retention prediction beyond mean relation | `docs/verdicts/diagnostics/context-incremental-retention-verdict.md` |
| Context structure remains descriptively real but is not a validated causal intervention target | `docs/verdicts/diagnostics/context-incremental-retention-verdict.md` |
| No context/distributional training intervention is authorized | `docs/verdicts/diagnostics/context-incremental-retention-verdict.md` |

## P. Phase-1 semantic-target design audit

| Thesis claim | Primary evidence artifact |
|---|---|
| A conjunctive objective+physical heavy-versus-center target is robustly aligned with semantic changes | `docs/verdicts/transfer/phase1-semantic-target-design-verdict.md` |
| The conjunctive target does not materially outperform simpler relational controls | `docs/verdicts/transfer/phase1-semantic-target-design-verdict.md` |
| PASS→FAIL false approval remains 14.3% and is not reduced by conjunction | `docs/verdicts/transfer/phase1-semantic-target-design-verdict.md` |
| Residual false approvals are the same context-sensitive failures where mean relations improve but semantic competence disappears | `docs/verdicts/transfer/phase1-semantic-target-design-verdict.md` |
| Stage A fails; no physical-decomposition Stage B or training pilot is authorized | `docs/verdicts/transfer/phase1-semantic-target-design-verdict.md` |

## Q. Preference-to-behavior ordering audit

| Thesis claim | Primary evidence artifact |
|---|---|
| Exact five-point continuum ordering measurement completed over all 25 held-out checkpoints | `docs/verdicts/preference_control/preference-behavior-ordering-audit-verdict.md` |
| Primary joint interior ordering deterioration occurs in 71.4% of PASS→FAIL events, below the frozen 75% gate | `docs/verdicts/preference_control/preference-behavior-ordering-audit-verdict.md` |
| Ordering is less sensitive to forgetting than the validated mean heavy-versus-center relation | `docs/verdicts/preference_control/preference-behavior-ordering-audit-verdict.md` |
| No residual failure with non-decreasing objective+physical mean relations is explained by ordering deterioration | `docs/verdicts/preference_control/preference-behavior-ordering-audit-verdict.md` |
| The two residual counterexamples actually improve joint ordering while losing semantic PASS | `docs/verdicts/preference_control/preference-behavior-ordering-audit-verdict.md` |
| Ranking/monotonicity intervention is not authorized | `docs/verdicts/preference_control/preference-behavior-ordering-audit-verdict.md` |

## R. Semantic-gate margin / robustness audit

| Thesis claim | Primary evidence artifact |
|---|---|
| The frozen 3-of-4 semantic PASS rule has an exact continuous gate-margin representation | `docs/verdicts/diagnostics/semantic-gate-robustness-audit-verdict.md` |
| Median PASS→FAIL gate margin moves from +0.459 to -2.474 | `docs/verdicts/diagnostics/semantic-gate-robustness-audit-verdict.md` |
| 64.3% of forgetting events survive ±0.25 normalized threshold perturbation | `docs/verdicts/diagnostics/semantic-gate-robustness-audit-verdict.md` |
| 28.6% of forgetting events are near-boundary with clearance <0.10 | `docs/verdicts/diagnostics/semantic-gate-robustness-audit-verdict.md` |
| Most events remain more likely than not under exact paired reset bootstrap | `docs/verdicts/diagnostics/semantic-gate-robustness-audit-verdict.md` |
| Semantic forgetting is heterogeneous: mostly substantive, with a material evaluator-sensitive subset | `docs/verdicts/diagnostics/semantic-gate-robustness-audit-verdict.md` |

## S. Final selection

Primary source of truth:

- `docs/closures/method_selection/final-method-selection-retention-closure.md`
- `docs/thesis/thesis-results-consolidation.md`
- `runs/final_method_selection-2026-09-24/FINAL_METHOD_SELECTION_MANIFEST.json`

Final reference:

> V2-B + Foundation V2 + GAE λ = 0.95 + no retention intervention.

The main thesis should cite experiment families by scientific purpose. Internal C-number identifiers belong in this appendix or repository traceability only.
## Z. One-model explicit policy-family H1

| Thesis claim | Primary evidence artifact |
|---|---|
| V2-PF begins exactly function-equivalent to V2-B | `runs/v2pf_h0_function_preservation-2026-09-25/v2pf_h0_report.json` |
| The larger generated policy block learns a multidimensional centered parameter family | `docs/verdicts/preference_architectures/v2pf-h1-authority-verdict.md` |
| Centered parameter and functional specific-energy fractions are both about 26% at H1 final | `docs/verdicts/preference_architectures/v2pf-h1-authority-verdict.md` |
| Family-block causal preference authority is transient and falls below the frozen final 5% masking criterion | `docs/verdicts/preference_architectures/v2pf-h1-authority-verdict.md` |
| Foundation V2 critic, PPO ratio, survival, and gradient-path checks remain valid | `docs/verdicts/preference_architectures/v2pf-h1-authority-verdict.md` |
| H2 semantic accumulation is not authorized and the policy-family escalation is closed without retuning | `docs/verdicts/preference_architectures/v2pf-h1-authority-verdict.md` |

