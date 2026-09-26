# Method

## 1. Overview

This study investigates preference-conditioned multi-objective reinforcement learning (MORL) for quadruped locomotion. The objective is not only to optimize several rewards under a user-specified preference vector, but to determine whether changes in that preference produce reliable and physically meaningful changes in locomotion behavior.

The policy is conditioned explicitly on a four-dimensional preference vector:

`a_t ~ π_θ(a_t | s_t, w),    w ∈ Δ⁴`

The four objective dimensions correspond to Tracking, Angular stability, Orientation, and Smoothness. A valid preference-conditioned controller is expected to preserve usable locomotion while producing a measurable semantic response when one objective is emphasized.

The experimental procedure was organized into three stages. First, the training and evaluation foundation was repaired and validated so that later semantic failures could not be attributed to reward, PPO, critic, or evaluation confounds. Second, preference-conditioning architectures were compared under the same validated foundation. Third, after semantic acquisition followed by forgetting was observed, a controlled retention study was performed.

## 2. Validated MORL Foundation

### 2.1 Objective formulation

The final formulation uses four normalized objectives with fixed physical interpretation. Earlier reward decompositions were not carried forward because they mixed effects that could not support reliable preference-level interpretation.

All architecture and retention comparisons use the same four-objective order, normalization, and preference semantics.

### 2.2 PPO action semantics

The stochastic actor follows a tanh-squashed action distribution. Executed action, stored old log-probability, and PPO likelihood ratio refer to the same stochastic transformation.

The deterministic evaluation action is:

`a = tanh(μ_θ(s, w))`

No untracked nonlinear clamp is inserted between this action and the declared environment processing path.

### 2.3 Critic structure and supervision

The critic uses a shared representation with objective-specific value heads. Objective heads are interpreted separately rather than as copies of one scalar value estimate.

Critic supervision uses representative current-policy and reset-diverse support. Fresh-state validation is required because strong fit on recently collected states alone does not guarantee usable objective-level credit.

### 2.4 Semantic evaluation

Reward improvement alone is not considered semantic success. Each objective-heavy preference is evaluated against a center preference under matched reset suites.

For each objective, evaluation includes objective-direction correctness, a predeclared physical semantic metric, survival and locomotion guardrails, and matched-reset comparisons. Continuum tests additionally measure monotonicity and endpoint-between behavior for intermediate preferences.

A semantic endpoint passes only when both objective and physical criteria satisfy the frozen thresholds while locomotion remains valid.

## 3. Preference-Conditioning Architecture Ladder

### 3.1 RV1: direct conditioning

RV1 is the minimal repaired architecture. The four-dimensional preference vector is provided directly to the shared actor. Its purpose is to test whether direct conditioning alone is sufficient once the training foundation is valid.

### 3.2 V2-A: learned preference embedding

V2-A retains the RV1 direct path and adds a learned preference embedding. The embedding is first tested for causal action authority before semantic behavior is interpreted.

### 3.3 V2-B: single-site FiLM modulation

V2-B adds one function-preserving FiLM modulation site to the shared actor while retaining both the RV1 direct path and the V2-A embedding.

The FiLM generator is initialized to preserve the preceding function at initialization. This permits a controlled test of whether one additional modulation mechanism improves preference-conditioned behavior without changing the validated foundation.

V2-B is retained as the final reference architecture because it produced the strongest tested preference-conditioned action differentiation and continuum behavior.

## 4. Semantic Forgetting Diagnosis

Checkpoint-level evaluation showed that individual semantic competencies could become valid and later disappear under continued shared-policy updates. This motivated a second-stage retention study.

The retention question is distinct from the acquisition question:

> Can a shared preference-conditioned policy preserve previously acquired semantic competence while continuing to optimize the current MORL objective?

The retention study separates two possible failure sources: instability caused by the auxiliary retention objective dominating current learning, and insufficiency of the retained memory representation itself.
## 5. Retention Mechanisms

### 5.1 Hard retention

Three hard-retention families were examined: local gradient retention, action retention, and short-trajectory retention. These methods test whether increasingly strong local anchors can preserve semantic competence.

### 5.2 Soft action-response rehearsal

The principal soft memory object is the preference-conditioned action-response difference:

`Δa_i(s) = π_θ(s, w_i) - π_θ(s, w_C)`

where `w_i` is an objective-heavy preference and `w_C` is the center preference.

A fixed-coefficient version was evaluated first, followed by a bounded-gradient formulation.

### 5.3 Bounded auxiliary-gradient budget

The common safety mechanism limits auxiliary influence according to:

`α_t = min(β_max, ρ ||g_current|| / (||g_aux|| + ε)),    ρ = 0.25`

The purpose of this rule is to preserve the current-learning direction and prevent auxiliary-gradient takeover. It is treated as an optimization safeguard rather than as a semantic-memory solution.

### 5.4 Semantic-outcome rehearsal

To move closer to behavior-level semantics, a second memory family retained heavy-versus-center semantic outcome margins over matched multi-step rollouts.

Three gradient constructions were tested while holding the semantic target fixed:

- likelihood-ratio score-function estimation;
- local pathwise finite-difference estimation;
- deterministic local surrogate modeling.

Each family was required to pass fixed-policy estimator or fidelity gates before training was authorized.

### 5.5 Literature-guided adaptations

A final bounded literature-guided round tested two principles selected for their relevance to the diagnosed failure mode.

The DER-style adaptation treated complete trajectories as memory units, described them using discounted four-objective returns and preference, and selected a diverse subset through crowding distance. Historical transitions were not inserted into the PPO likelihood objective.

The Policy Consolidation adaptation retained historical policy snapshots at multiple timescales and imposed a bounded KL-based auxiliary gradient on current on-policy states. The total consolidation gradient remained subject to the same `ρ = 0.25` safety budget.

## 6. Final Reference Configuration

The final thesis reference system is:

- V2-B single-site FiLM preference-conditioned actor;
- Foundation V2;
- normalized four-objective formulation;
- repaired squashed-action PPO;
- validated critic support and freshness protocol;
- GAE λ = 0.95;
- no retention intervention.

Retention mechanisms are therefore analyzed as controlled method studies rather than included in the final reference training method.
