# Results

## 1. Foundation Validation

The first result is that semantic preference behavior became interpretable only after independent training and evaluation confounds were removed.

The final foundation corrected objective semantics, PPO action/log-probability consistency, critic-head interpretation, critic support coverage, and fresh-policy evaluation. After these repairs, later semantic failures occurred while survival, critic validity, and the action-likelihood path remained acceptable. The remaining failures therefore could not be attributed to an invalid optimization pipeline.

## 2. Architecture Progression

Table 1 summarizes the semantic progression from direct conditioning to the final V2-B architecture.

| Architecture | Preference mechanism | Endpoint passes | Continuum monotonicity | Endpoint-between | Survival |
|---|---|---:|---:|---:|---:|
| RV1 | direct preference conditioning | 0/4 | 0.60938 | 0.40278 | 1.0 |
| V2-A | direct path + learned embedding | 1/4 (Tracking) | 0.62500 | 0.47917 | 1.0 |
| V2-B | direct path + embedding + one-site FiLM | 1/4 (Angular) | 0.64583 | 0.52083 | 1.0 |

The architecture ladder shows a consistent increase in preference-conditioned expressivity. V2-A and V2-B both acquired measurable causal preference authority, while the continuum metrics improved from RV1 to V2-B.

However, endpoint correctness did not accumulate across objectives. RV1 passed none of the four semantic endpoints. V2-A passed Tracking, whereas V2-B passed Angular. The strongest architecture therefore remained semantically incomplete.

For V2-B, the semantic failure was not accompanied by locomotion or critic collapse. Survival remained 1.0, the maximum tracking ratio relative to center preference was 1.0281, critic H32 explained variance was 0.3335, critic negative fraction was 0.0875, and mean absolute critic bias was 0.0344.

These results indicate that added conditioning capacity improved behavioral differentiation without guaranteeing globally correct four-objective semantic control.

## 3. Temporal Credit and Optimization Geometry

A temporal-credit audit showed that λ = 1.0 more closely matched MC32 local gradient geometry than the reference λ = 0.95 setting. For Orientation, cosine alignment to MC32 increased from 0.838 at λ = 0.95 to 0.926 at λ = 1.0. Similar improvements were observed for Angular and Smoothness.

This diagnostic finding established that temporal credit matters locally, but subsequent training evidence did not justify replacing the final reference with λ = 1.0. The final evaluated configuration therefore remains at λ = 0.95.

Additional audits rejected simple explanations based on static objective conflict, mixed-batch geometry, a monotonic collapse of state-fixed local credit, or optimizer history alone. Visitation and parameter-path dependence remained relevant, but no single local mechanism explained the full semantic failure.

## 4. Semantic Acquisition and Forgetting

Checkpoint-level evaluation revealed a recurring pattern: individual semantic competencies could become valid and later disappear under continued shared-policy optimization.

This separates two failure modes:

- **acquisition failure**, in which a semantic axis never becomes valid;
- **retention failure**, in which a valid semantic axis is acquired and subsequently lost.

The observed forgetting occurred after preference authority and local learning signals had been validated. This motivated a dedicated retention study rather than further architecture escalation.

## 5. Custom Retention Study

### 5.1 Hard retention

Gradient retention did not preserve semantic competence sufficiently.

Hard action retention preserved local behavior more directly but increasingly restricted policy plasticity. Short-trajectory retention strengthened local preservation further but produced a stronger plasticity cost and still did not solve semantic retention globally.

These experiments established a stability-plasticity trade-off: stronger local anchors could preserve selected behavior, but increasingly interfered with continued policy adaptation.

### 5.2 Fixed-weight soft rehearsal

Soft action-response rehearsal avoided a hard constraint but introduced a different failure mode. As training progressed, the rehearsal gradient could dominate the current-learning direction even when the total parameter-step magnitude remained nonzero.

This showed that preserving step magnitude alone is insufficient; the update direction must also remain controlled.

### 5.3 Bounded-gradient rehearsal

The bounded auxiliary-gradient rule solved the optimization-stability problem.

Across three seeds in the bounded Δa experiment:

- mean cosine between current-learning and total update direction was 0.9745;
- minimum observed cosine was 0.9698;
- preference separation remained effectively unchanged;
- active auxiliary updates were bounded near `ρ = 0.25`;
- no parameter-step collapse was observed;
- no preference-authority collapse was observed.

However, semantic-retention efficacy did not reproduce across seeds.

| Seed | Control PASS events | Bounded Δa PASS events | Paired effect | Mean semantic-score effect |
|---|---:|---:|---:|---:|
| 980001 | 4 | 6 | +2 | +0.00391 |
| 981001 | 9 | 7 | -2 | -0.00781 |
| 982001 | 7 | 6 | -1 | -0.01953 |
| Aggregate | 20 | 19 | mean -0.333 | mean -0.00781 |

The first seed produced a genuine positive instance, but this benefit did not reproduce. The bounded-gradient rule is therefore supported as a stability mechanism, whereas bounded Δa is not supported as a robust semantic-retention method.

## 6. Semantic-Outcome Retention Study

The action-response memory was replaced by a more direct semantic-outcome target to test whether memory content, rather than gradient budgeting, was the remaining limitation.

### 6.1 Score-function estimator

Common random numbers, centering, and replica averaging reduced some norm variability, but normalized gradient directions remained poorly reproducible.

Even when magnitude-based SNR approached approximately one after replica averaging, independently averaged gradient directions remained nearly orthogonal.

### 6.2 Local pathwise finite-difference estimator

A local pathwise finite-difference estimator was then used to remove likelihood-ratio estimation as the main explanation. It also failed the fixed-policy direction-reproducibility gate at the tested checkpoints.

The instability therefore could not be attributed to score-function variance alone.

### 6.3 Deterministic local surrogate

A deterministic surrogate produced more reproducible gradients, but held-out semantic prediction failed. Minimal and richer temporal representations achieved substantially better in-sample fit than held-out fit, while nonlinear capacity did not restore seed-generalization.

A reproducible gradient of a poorly generalizing surrogate was therefore not considered a valid semantic rehearsal signal.

## 7. Literature-Guided Retention Round

### 7.1 DER-style diversity-based replay

The DER-inspired adaptation used trajectory-level diversity in objective-return and preference space while preserving PPO on-policy semantics.

On the short gate:

- control: 4 PASS events, mean semantic score 0.4531;
- static bounded Δa: 6 PASS events, mean semantic score 0.4570;
- DER-diverse replay: 5 PASS events, mean semantic score 0.4453.

The DER adaptation changed which competencies were acquired or preserved, but did not improve overall semantic performance. Optimization geometry remained stable, so the short-gate failure was not caused by gradient takeover.

### 7.2 Multi-timescale Policy Consolidation

The Policy Consolidation adaptation used historical teachers at 1-, 2-, and 4-update timescales with KL regularization on current on-policy states. The total consolidation gradient remained bounded by the same safety mechanism.

On the same short gate:

- control: 4 PASS events, mean semantic score 0.4531;
- static bounded Δa: 6 PASS events, mean semantic score 0.4570;
- Policy Consolidation: 4 PASS events, mean semantic score 0.3906.

The consolidation arm preserved optimization geometry but did not preserve semantic competence. Tracking and Angular competencies could be acquired and then lost, while Orientation never passed the endpoint criterion in this short gate.

## 8. Final Method Selection

Neither the custom retention methods nor the two literature-guided adaptations provided sufficient evidence to justify inclusion in the final reference method.

The final thesis reference therefore remains:

> **V2-B + Foundation V2 + GAE λ = 0.95 + no retention intervention.**

This choice does not imply that semantic retention was solved. It reflects the strongest validated architecture/foundation configuration without adding a retention mechanism whose superiority was not supported by the evidence.
