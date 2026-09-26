# Discussion

## 1. Interpreting the Architecture Results

The architecture progression shows that preference-conditioning capacity and semantic correctness are related but not equivalent.

RV1 demonstrated that direct conditioning can influence the policy without producing valid four-objective semantic control. V2-A and V2-B increased preference-conditioned action differentiation and improved continuum interpolation, confirming that richer conditioning mechanisms provide additional control authority. However, semantic endpoint success did not accumulate across objectives.

This distinction is important. A policy can respond strongly to preference while responding in the wrong behavioral direction, or while improving one objective at the expense of previously acquired semantic competence on another. Preference sensitivity is therefore necessary instrumentation, but not sufficient evidence of correct MORL semantics.

## 2. Semantic Acquisition Versus Semantic Retention

A central finding of the study is that semantic acquisition and semantic retention are separate problems.

Several axes became semantically valid at intermediate checkpoints before later disappearing. This shows that the optimization process can reach locally correct semantic behaviors. The difficulty is that subsequent shared-policy updates reorganize the policy in a way that does not preserve those behaviors consistently.

The observed failure is therefore not well described as a complete inability to learn semantic preference response. It is better described as unstable preservation of multiple semantic competencies within one shared nonlinear policy.

This finding also explains why architecture escalation alone had diminishing decision value. Additional conditioning capacity can improve control authority, but it does not directly solve interference among already acquired semantic behaviors.

## 3. Stability-Plasticity in Retention

The retention experiments exposed two distinct mechanisms.

Hard retention directly protects previous behavior but progressively restricts plasticity. Gradient, action, and short-trajectory anchors therefore improved local preservation at the cost of limiting adaptation.

Soft rehearsal removed the hard constraint but created a second failure mode: an unconstrained auxiliary gradient could grow until it dominated the current-learning direction.

The bounded-gradient rule successfully separated these effects. By limiting auxiliary-gradient magnitude relative to the mixed current-learning gradient, rehearsal no longer hijacked optimization. Mixed-total cosine remained close to one, parameter-step collapse disappeared, and preference authority was preserved.

This is a positive mechanism result.

However, optimization stability did not imply semantic retention. Once gradient takeover was removed, the remaining failure could be attributed more directly to the retained object or the geometry through which it was used.

## 4. Why the Tested Memory Objects Were Insufficient

The tested memory objects increase progressively in semantic richness:

- local gradients;
- action snapshots;
- short trajectories;
- preference-conditioned action-response differences;
- direct semantic rollout outcomes.

None provided robust multi-objective semantic retention.

This suggests that semantic competence in the shared policy is not localized in one simple stable object. Two policies can remain locally similar in action space while reorganizing higher-level objective trade-offs. Likewise, preserving a short trajectory or local gradient does not guarantee preservation of behavior over the broader state distribution induced by future policy updates.

The Δa representation is especially informative. It is a stable and well-behaved local proxy for preference-conditioned action differentiation, but the multi-seed result shows that preserving this proxy is not sufficient to preserve semantic competence reliably.

## 5. Semantic-Outcome Geometry

Direct semantic-outcome rehearsal was intended to move beyond proxy retention.

The estimator ladder showed that the initial failure was not merely a poor choice of gradient estimator. Score-function gradients were unstable, but replacing them with local pathwise finite differences did not recover a reproducible direction.

A deterministic surrogate made the gradient more reproducible, yet the underlying mapping did not generalize across held-out reference seeds.

This separates gradient reproducibility from semantic validity. A smooth and repeatable gradient is only useful if it belongs to a model that predicts the semantic quantity of interest under relevant distribution shift.

The surrogate result therefore provides evidence that the current semantic-outcome representation does not expose a sufficiently stable seed-generalizable local geometry for rehearsal.

## 6. Interpretation of the Literature-Guided Results

The literature-guided round was deliberately narrow and should be interpreted as an adaptation study rather than as a claim that the original algorithms fail.

The DER-style adaptation tested whether a more diverse memory distribution could repair the weakness of a static reference set. It changed which competencies were represented and sometimes improved local forgetting metrics, but did not improve overall semantic performance.

The Policy Consolidation adaptation tested a different hypothesis: perhaps preserving policy distributions over multiple timescales is more effective than preserving explicit action memories. This also failed the semantic short gate despite retaining safe optimization geometry.

Together, these results strengthen one conclusion:

> The remaining difficulty is not explained only by an overly narrow replay buffer or by the lack of historical policy regularization.

They do not imply that all continual-RL or MORL replay methods are ineffective. The result is specific to the tested adaptations, V2-B architecture, Foundation-V2 training setting, and frozen semantic evaluation contract.

## 7. What the Study Successfully Establishes

The study provides several positive findings even though the final semantic-retention problem remains open.

### 7.1 Foundation validity is essential

Architecture conclusions drawn before reward, PPO, critic, and evaluation repair were not reliable. Once these confounds were removed, later failures became scientifically interpretable.

### 7.2 Conditioning authority can be measured separately from semantic correctness

V2-A and V2-B acquire real causal preference authority, yet remain semantically incomplete. This distinction provides a useful evaluation principle for preference-conditioned MORL.

### 7.3 Semantic forgetting is observable as a distinct phenomenon

The policy can acquire semantic behavior and lose it later. This identifies retention, not only acquisition, as a core challenge in shared preference-conditioned control.

### 7.4 Bounded auxiliary-gradient budgeting is a validated optimization mechanism

The bounded rule consistently prevents rehearsal-gradient takeover without collapsing plasticity. This mechanism remains useful independently of whether a particular memory representation succeeds.

## 8. Limitations

The conclusions should be interpreted within the tested scope.

First, the experiments use one quadruped morphology and simulation domain. The generality of semantic forgetting across different robots or tasks is not established.

Second, the objective formulation contains four declared semantic dimensions. Other objective decompositions may produce different interference and retention geometry.

Third, the architecture ladder is finite. The results do not establish that no larger or modular architecture could solve the problem.

Fourth, the DER and Policy Consolidation experiments are principled adaptations rather than exact reproductions of the original algorithms.

Fifth, several retention and mechanism experiments use controlled finite-update or short-gate protocols. These are appropriate for causal method selection but are not substitutes for full-scale evaluation of entirely new algorithms.

Sixth, no sim-to-real semantic-retention experiment was performed.

Finally, the final V2-B reference itself remains semantically incomplete. The thesis therefore characterizes the problem and its tested mechanisms rather than claiming complete semantic preference control.

## 9. Future Work

Future work should introduce genuinely new hypotheses rather than continue coefficient tuning within the closed branch.

Promising directions include:

- invariant trajectory-level semantic representations;
- modular or objective-specialized policy structures;
- explicit skill decomposition;
- broader-support model-based semantic prediction;
- continual-MORL algorithms designed jointly with on-policy optimization;
- longer-duration multi-seed confirmation of future candidates;
- sim-to-real evaluation of semantic retention.

A particularly important direction is to investigate representations that preserve behavior-level objective trade-offs across state-distribution change rather than only local action similarity.
