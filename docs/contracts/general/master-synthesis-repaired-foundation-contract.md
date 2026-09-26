# Master Synthesis / Repaired-Foundation Contract

> **Supersession note (2026-09-24):** The repaired-foundation evidence in this document remains valid, but its later authorization/method-selection state is superseded by `docs/closures/method_selection/final-method-selection-retention-closure.md`, which reflects the completed RV1/V2-B and retention branches.

Status: **HISTORICAL FOUNDATION SOURCE — SCIENTIFIC DIAGNOSIS RETAINED; METHOD-SELECTION AUTHORIZATION SUPERSEDED**

Date: 2026-09-23

Purpose: consolidate the evidence chain from the original V1/V2 preference-conditioned failures through C43, separate retained observations from invalidated causal conclusions, freeze the validated training foundation, and define the next minimal method-selection experiment. This document supersedes informal causal interpretations that conflict with later evidence.

---

## 0. Status vocabulary

- **RETAIN** — still supported and may be used in the thesis or next experiment.
- **SUPERSEDED** — the observation may remain valid, but a later artifact provides the preferred interpretation or implementation.
- **INVALIDATED** — the claim must not be used as evidence because a later audit showed the comparison, estimator, or implementation was invalid.
- **REJECT** — explicitly tested and not supported under the stated conditions.

Mandatory distinction:

> An observed failure can be RETAINED while the architectural conclusion drawn from that failure is INVALIDATED or SUPERSEDED.

---

# 1. Evidence ledger

## 1.1 Historical locomotion / reward foundation

| Finding / historical claim | Status | Current interpretation | Source / superseding evidence |
|---|---|---|---|
| Original deterministic locomotion milestone failed to produce a seed-robust deployable baseline. | **RETAIN** | Historical result is real under the then-frozen pipeline. | docs/methods/general/consolidated-locomotion-failure-analysis.md |
| Reward semantics permitted survived-but-nonfunctional behavior and did not cleanly encode directed locomotion. | **RETAIN** | Reward/objective semantics were a genuine confound in early preference conclusions. | docs/baselines/multiobjective_bridge/reward-objectives-revision-r1-design.md; later T3 objective reconstruction |
| Command exposure/history affected deterministic evaluation behavior. | **RETAIN** | Distribution exposure mattered diagnostically, but exposure-only repair was insufficient. | docs/methods/general/consolidated-locomotion-failure-analysis.md |
| A single fixed action-subspace, clipping, or global shrink intervention was identified as the locomotion fix. | **REJECT** | D1 did not find a repeatable global action signature. | docs/methods/general/consolidated-locomotion-failure-analysis.md |
| Early V1 preference failure proves direct preference conditioning is architecturally insufficient. | **INVALIDATED AS ARCHITECTURAL INFERENCE** | The observed failure remains historical evidence, but it occurred before reward/PPO/critic foundation repairs; it cannot justify architecture escalation. | repaired-foundation chain through T3/T4/T5 and C23–C43 |
| Early V2 complexity is justified because V1 failed. | **INVALIDATED AS DECISION RULE** | V2 must earn justification only after repaired minimal V1 fails on the validated foundation. | this contract |

## 1.2 Preference-conditioned formulation

| Finding / claim | Status | Current interpretation | Evidence |
|---|---|---|---|
| Preference must be explicit policy input; no hidden inference from history. | **RETAIN** | Remains part of the method contract. | docs/baselines/multiobjective_bridge/m0-2-preference-conditioned-moppo-design-draft.md |
| Objective-separated critic values are required for objective-level credit analysis. | **RETAIN** | Semantically collapsed heads cannot support valid preference-credit claims. | M0.2 design + T5 critic diagnostics |
| Old five-objective formulation should be carried forward unchanged. | **SUPERSEDED** | Final repaired branch uses validated normalized **4D** objectives. | post_v2_t3a3_objective_selection; post_v2_t3b_scaling; post_v2_t4_repaired |
| Historical V1/V2 semantic endpoint failures can directly rank V1 vs V2 architecture. | **INVALIDATED AS ARCHITECTURE EVIDENCE** | Those outcomes are confounded by now-known pipeline issues. | repaired foundation below |

## 1.3 Objective and action-semantics repair

| Finding / claim | Status | Current interpretation | Evidence |
|---|---|---|---|
| Reward/objective regrouping was necessary to obtain physically coherent 4D objectives. | **RETAIN** | Clean 4D objective semantics are part of the foundation. | T3a/T3b runs |
| PPO action/log-prob semantics were previously consistent with tanh-squashed execution. | **INVALIDATED** | Repaired branch must use corrected squashed-policy action/log-prob path. | repaired PPO branch leading into T4/T5 |
| External action semantics/clamping may silently differ from the policy distribution. | **RETAIN AS GUARDRAIL** | Every new experiment must audit exact actor output → tanh → env processing. | C40 |
| IsaacLab joint-position processing applies an approximately 0.25 linear scale after policy action. | **RETAIN** | Expected environment semantics, not a hidden nonlinear mismatch. | C40 |

## 1.4 Critic foundation

| Finding / claim | Status | Current interpretation | Evidence |
|---|---|---|---|
| Scalar-copy / inherited objective heads are a safe neutral initialization. | **REJECT / SUPERSEDED** | Objective heads use the validated zero-head initialization contract. | C0/C1 and later repair sequence |
| Shared critic body can be retained. | **RETAIN** | Shared representation is permitted with objective-specific heads and valid supervision. | C18–C25 |
| Objective-specific critic heads are required. | **RETAIN** | Each objective retains its own head. | T5 repair sequence |
| Short/recent-only support is sufficient critic supervision. | **REJECT** | It produced poor fresh-state generalization / coverage. | C20/C21 |
| Representative reset-diverse support repairs fresh value generalization. | **RETAIN** | This is the validated critic supervision contract. | C22/C23 |
| Ridge lambda = 1 is retained for repaired critic fitting. | **RETAIN** | Validated setting in the successful support branch. | C8/C23 sequence |
| H32 critic target is retained. | **RETAIN** | Repaired validation uses H32. | C10 and later branch |
| Critic-side repair survives actor updates sufficiently to reopen semantic-credit analysis. | **RETAIN** | C25 validated the foundation used by the later causal branch. | post_v2_t5_c25_actor_updating25-2026-09-23 |

## 1.5 Angular semantic-credit branch

| Finding / claim | Status | Current interpretation | Evidence |
|---|---|---|---|
| Angular semantic weakness remained after critic repair and required local causal analysis. | **RETAIN** | Motivated C26 onward. | C26–C38 |
| C39 proved stochastic geometry remains structurally different from deterministic geometry as sigma → 0. | **INVALIDATED** | Deterministic comparator aggregated action gradients before state-dependent chain propagation. | C40–C41 |
| Post-tanh FD, pre-tanh zero-variance FD, and explicit tanh chain rule are different derivative objects. | **REJECT** | They agree numerically when evaluated per environment before aggregation. | C40 |
| Deterministic batch gradient may be constructed as E[J]^T E[g_a]. | **INVALIDATED** | Correct construction is E[J_i^T g_a,i]. | C40–C41 |
| Corrected deterministic comparator restores substantial stochastic alignment. | **RETAIN** | Repairs the C38–C39 narrative. | C41 |
| Score-function and pathwise estimators disagree structurally at high sample count. | **REJECT** | They converge strongly to each other. | C42 |
| u25 residual at current variance is pure Monte Carlo / weak-gradient noise. | **REJECT** | High-sample CI narrows while systematic residual remains. | C42 |
| Finite policy variance produces a real stochastic-vs-mean-action geometry shift at u25. | **RETAIN** | Property of stochastic expected objective, not implementation defect. | C42–C43 |
| Corrected stochastic gradient fails to return to deterministic target as sigma → 0. | **REJECT** | It converges smoothly to corrected deterministic target. | C43 |
| Residual approximately follows O(sigma^2). | **RETAIN, CONSERVATIVE WORDING** | Evidence is consistent with finite-variance smoothing over a nonlinear action–reward landscape. | C43; R2 about 0.996 and 0.999 |
| PPO must be modified to force finite-variance stochastic gradient to match deterministic mean-action gradient. | **REJECT AS CURRENT RATIONALE** | No PPO correctness defect remains from this branch. | C40–C43 |

---

# 2. Frozen scientific conclusion: C39–C43

Preferred thesis wording:

> The apparent stochastic–deterministic gradient mismatch was initially amplified by an invalid batch-gradient construction that aggregated action gradients before propagation through state-dependent policy Jacobians. After correcting the per-state chain rule, a smaller but systematic residual remained at finite policy variance. This residual was consistently reproduced by both likelihood-ratio and pathwise estimators, was not explained by Monte Carlo uncertainty, and decayed smoothly toward the corrected deterministic gradient as the policy variance approached zero. The residual approximately followed an O(sigma^2) trend, consistent with finite-variance smoothing over a nonlinear action–reward landscape.

Frozen causal status:

- **C39 — INVALIDATED:** old deterministic comparator was not the intended batch derivative.
- **C40 — PASS:** coordinate and derivative equivalence confirmed on the actual action path.
- **C41 — PASS:** correct per-env chain-then-aggregate comparator restores alignment.
- **C42 — PASS:** finite-sample-only explanation rejected.
- **C43 — PASS:** residual tends toward zero as variance shrinks, approximately O(sigma^2).

**Stop rule:** C39–C43 is frozen. No C44+ diagnostic is authorized merely to continue investigation.

---

# 3. Validated repaired foundation

## 3.1 Objectives

    objective space      = validated normalized 4D
    objective semantics  = repaired / physically interpretable
    preference vector    = explicit 4D conditioning input
    normalization        = fixed by validated T3b scaling contract

Do not restore old reward buckets or the historical five-objective decomposition for RV1.

## 3.2 PPO / action semantics

    policy distribution  = repaired squashed-action semantics
    actor mean path      = pre-tanh mean -> tanh -> policy action
    joint log-prob       = matched to the squashed executed action
    external clamp       = no untracked clamp allowed
    env processing       = explicit IsaacLab JointPositionAction path

Actor action, stored old log-prob, PPO ratio, and executed action must refer to the same stochastic transformation.

## 3.3 Critic

    body                 = shared critic body retained
    heads                = objective-specific
    head init            = zero objective heads
    target horizon       = H32
    regularization       = ridge lambda = 1 where repaired fit uses ridge
    supervision support  = representative reset-diverse support
    fresh validation     = mandatory

Prohibited critic regressions:

- scalar-copy objective heads;
- identical heads presented as semantic decomposition;
- recent-only support treated as sufficient;
- actor conclusions drawn from a critic that fails fresh H32 / MC validation.

## 3.4 Actor

    architecture         = minimal shared baseline actor
    conditioning         = direct preference conditioning only
    evaluation action    = deterministic tanh(mean)

The next experiment must not import V2 complexity.

## 3.5 Monitoring / evaluation

Preserve:

- deterministic evaluation at fixed declared preferences;
- objective rewards/returns/advantages;
- physical semantic metrics;
- survival/failure metrics;
- tracking/locomotion metrics;
- action and saturation statistics;
- explicit preference labels;
- non-mutation of evaluation-only monitors.

---

# 4. Prohibited regressions / unauthorized mechanisms for RV1

Do not add before repaired V1 fails:

- V2 latent preference representation;
- FiLM;
- adapters;
- routing / experts;
- preference curriculum;
- rehearsal;
- semantic auxiliary losses;
- coefficient or gradient guidance;
- adaptive preference shaping;
- extra reconstruction losses;
- architecture-specific residual branches;
- post-hoc exploration-variance reduction.

Also prohibited:

- old reward buckets;
- old 5D objective contract;
- scalar-copy critic heads;
- invalid action/log-prob pairing;
- untracked external action clamp;
- aggregate-g_a-before-chain deterministic comparator;
- historical V1 failure used as proof that V1 architecture is insufficient.

---

# 5. Historical-result reinterpretation

## 5.1 Old V1

**Observed result: RETAIN**

Old V1 did not demonstrate required semantic preference behavior under its historical pipeline.

**Architectural conclusion: INVALIDATED**

It is no longer valid to conclude that direct preference conditioning is insufficient, because the historical result preceded objective, PPO action/log-prob, critic initialization, and critic-support repairs.

Preferred thesis wording:

> The original V1 configuration failed to demonstrate valid semantic preference response; however, this outcome cannot isolate architectural insufficiency because the training foundation was later found to contain independent reward, action-semantics, and critic-supervision confounds.

## 5.2 Old V2

**Observed result: RETAIN**

V2 runs remain valid descriptions of those systems.

**Necessity claim: INVALIDATED / UNPROVEN**

V2 complexity is not currently justified as necessary because its motivation inherited the confounded V1 failure.

Decision rule:

> V2 is reopened only if repaired minimal V1 fails under the validated foundation.

## 5.3 Critic and Angular history

T5 critic repair and C26–C43 remain valid mechanism evidence, but they are not authorization to add complexity to the next actor.

---

# 6. Repaired V1 (RV1) experiment contract

## 6.1 Single causal question

> **Can direct preference conditioning alone produce valid continuous 4D semantic preference response once the training foundation is valid?**

No other architecture question is authorized in RV1.

## 6.2 Treatment

The only method-level treatment is:

    4D preference w
          |
          v
    direct conditioning into minimal shared actor

Everything else inherits the repaired foundation.

## 6.3 Fixed foundation

RV1 must use:

    validated normalized 4D objectives
    repaired squashed PPO action/log-prob semantics
    shared baseline actor
    shared critic body
    objective-specific zero-initialized critic heads
    representative reset-diverse critic supervision
    validated critic target / regularization contract
    fixed evaluation preferences
    frozen semantic and physical metrics

## 6.4 Explicit exclusions

RV1 must not include latent preference encoders, FiLM, adapters, routing, experts, curriculum, rehearsal, semantic auxiliary losses, coefficient/gradient guidance, Angular-specific fixes, variance interventions, V2 warm-starts, or architecture-specific rescue logic.

If any are introduced, the run is **not RV1**.

---

# 7. RV1 ladder

Training remains unauthorized until RV1-A is frozen and passes.

## RV1-A — implementation / function-preserving gate

Required checks:

1. Fixed probe observations and preferences produce finite actions.
2. Deterministic action equals tanh(actor_mean(obs,w)).
3. Stochastic sample and joint log-prob use the same transformed distribution.
4. PPO old/new log-prob identity holds with unchanged parameters.
5. Preference columns affect only declared conditioning paths.
6. Critic output shape is exactly 4 objective heads.
7. Objective heads start at validated zero-head initialization.
8. Objective order and normalization match frozen 4D contract.
9. Reset-diverse critic-support loader reproduces validated support.
10. Evaluation monitor does not mutate actor, critic, optimizer, RNG, or normalizer.

Numerical equality tolerances must be frozen before smoke; reuse existing 1e-6-class checks where applicable.

**PASS → RV1-B may run.**  
**FAIL → implementation repair only; no training interpretation.**

## RV1-B — one-seed short preference-sensitivity screen

Purpose: cheaply reject obvious non-learning / conditioning-dead cases without final semantic claims.

Required telemetry:

- policy sensitivity to preference;
- objective reward/return/advantage by preference;
- critic fresh-state validity;
- survival and early locomotion stability;
- action saturation;
- deterministic fixed-preference monitor;
- physical semantic metrics.

Important:

> Nonzero action derivative with respect to preference is necessary instrumentation only. It is not RV1 success.

Stop conditions:

- conditioning numerically dead;
- critic fresh generalization collapses;
- locomotion foundation catastrophically collapses;
- invalid/NaN PPO semantics;
- fixed preferences remain behaviorally indistinguishable.

**PASS → RV1-C.**

## RV1-C — semantic response gate

Required fixed preference set:

- central/reference preference;
- heavy preference for each of the 4 objectives;
- intermediate/interpolated preferences between selected endpoints.

Behavior-level criteria:

1. **Objective response** — increasing preference for objective k produces expected directional change in objective outcome.
2. **Physical semantic response** — response appears in predeclared physical metrics, not reward bookkeeping alone.
3. **Cross-preference differentiation** — heavy preferences produce distinct behavior.
4. **Intermediate / monotonic response** — interpolated preferences show coherent intermediate behavior where a trade-off is expected; perfect linearity is not required.
5. **Persistence** — differentiation persists beyond a transient checkpoint/window.
6. **Locomotion and safety guardrails** — survival, tracking, catastrophic-tail failure, and saturation remain acceptable.

Semantic gain cannot be rescued by unusable locomotion.

**PASS → RV1-D.**  
**FAIL → repaired V1 is experimentally insufficient; only then may V2 be reopened.**

## RV1-D — confirmatory validation

Authorized only after RV1-C passes.

Purpose:

- multi-seed confirmation;
- full-duration training;
- frozen preference evaluation matrix;
- held-out/intermediate preference checks;
- final physical semantic and locomotion validation.

No architecture changes are allowed between RV1-C and RV1-D.

---

# 8. RV1 decision rule

## Repaired V1 PASS

> **STOP ARCHITECTURE ESCALATION.**

Conclusion:

> Direct preference conditioning is sufficient on the validated foundation; V2 complexity is not required by current evidence.

The simpler method becomes the primary thesis candidate.

## Repaired V1 FAIL

> **V2 BECOMES EXPERIMENTALLY JUSTIFIED.**

Failure must first be described behaviorally and mechanistically. V2 then answers a new causal question; it is not reopened merely because historical V1 failed.

---

# 9. Global stop rule

> **No architectural complexity before repaired minimal V1 fails on the validated pipeline.**

A new experiment must have decision value. Diagnostic curiosity alone is insufficient authorization.

---

# 10. Thesis consolidation map

Main-text scientific sequence:

    Historical V1/V2 preference failures
                 |
                 v
          foundation audit
                 |
                 v
      reward/objective repair
                 |
                 v
     PPO action/log-prob repair
                 |
                 v
    critic init/support repair
                 |
                 v
      Angular credit analysis
                 |
                 v
    apparent stochastic/deterministic mismatch
                 |
                 v
      batch comparator bug found
                 |
                 v
       corrected comparator
                 |
                 v
    finite-variance residual confirmed
                 |
                 v
 residual -> deterministic limit as sigma -> 0
          approximately O(sigma^2)
                 |
                 v
     scientific diagnosis frozen
                 |
                 v
 method selection restarts from repaired minimal V1

Recommended main-text figures:

1. Pipeline / root-cause map.
2. Critic repair progression.
3. Angular causal chain.
4. Apparent mismatch → corrected comparator story.
5. 1 - cosine(g_sigma, g_det) vs sigma^2 for u25/s0 and u25/s1 with regression.

C-series identifiers belong primarily in methods/traceability/appendix, not as the main chapter narrative.

---

# 11. Evidence artifacts to preserve

Core repaired-objective / foundation artifacts:

    runs/post_v2_t3a_regrouping-2026-09-23
    runs/post_v2_t3a2_atomic_coherence-2026-09-23
    runs/post_v2_t3a3_objective_selection-2026-09-23
    runs/post_v2_t3a4_controllability-2026-09-23
    runs/post_v2_t3a5_effort-2026-09-23
    runs/post_v2_t3b_scaling-2026-09-23
    runs/post_v2_t4_repaired-2026-09-23

Core critic artifacts:

    runs/post_v2_t5_c0_zero_critic-2026-09-23
    runs/post_v2_t5_c20_coverage-2026-09-23
    runs/post_v2_t5_c22_diverse12-2026-09-23
    runs/post_v2_t5_c23_reset_diverse-2026-09-23
    runs/post_v2_t5_c25_actor_updating25-2026-09-23

Frozen gradient-mechanism artifacts:

    runs/post_v2_t5_c39_variance_curvature-2026-09-23
    runs/post_v2_t5_c40_derivative_equivalence-2026-09-23
    runs/post_v2_t5_c41_corrected_gradient_compare-2026-09-23
    runs/post_v2_t5_c42_highsample_confidence-2026-09-23
    runs/post_v2_t5_c43_corrected_variance_limit-2026-09-23

Historical context documents:

    docs/methods/general/consolidated-locomotion-failure-analysis.md
    docs/baselines/multiobjective_bridge/reward-objectives-revision-r1-design.md
    docs/baselines/multiobjective_bridge/m0-2-preference-conditioned-moppo-design-draft.md
    docs/baselines/multiobjective_bridge/m0-2a-function-preserving-staged-redesign.md
    docs/closures/synthesis/b1-closure-and-formulation-decision.md

---

# 12. Authorization state

    C39-C43 diagnostic branch   = CLOSED / FROZEN
    C44 variance intervention   = NOT AUTHORIZED
    V2 architecture escalation  = NOT AUTHORIZED
    RV1-A implementation gate   = NEXT AUTHORIZED WORK
    RV1-B/C/D training          = BLOCKED until preceding gate passes

The next code change should be limited to an RV1 manifest / implementation-equivalence gate. No new training run is authorized by this document alone.
