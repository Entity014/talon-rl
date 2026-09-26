# Literature-Guided Retention Round — Final Closure

Status: **FROZEN — L1 AND L2 FAIL THE PREDECLARED SHORT RETENTION GATE; RETENTION SEARCH CLOSED**

Date: 2026-09-24

## Motivation

The custom retention ladder established two separate facts:

1. retention interventions can destabilize current learning unless their gradient contribution is explicitly bounded;
2. after gradient-budget control is imposed, the retained local Delta-a representation does not provide a robust multi-seed semantic-retention advantage.

Before final method selection, one bounded literature-guided round was authorized to test two established retention principles that align most closely with the diagnosed failure modes.

No open-ended literature-method search was authorized.

## Frozen foundation

Both literature-guided candidates retained:

- V2-B single-site FiLM preference-conditioned actor
- Foundation V2
- validated normalized 4D objectives
- repaired squashed PPO action/log-prob semantics
- GAE lambda = 0.95
- same nominal finite update norm
- same endpoint semantic evaluator and thresholds
- same evaluation suites
- same preference set
- bounded auxiliary-gradient budget rho = 0.25
- no old transitions inserted into the PPO objective

The bounded-gradient rule remained:

alpha_t = min(beta_cap, rho * ||g_mixed|| / (||g_aux|| + eps))

Only the auxiliary retention principle changed.

---

# L1 — DER-style diverse semantic replay adaptation

## Literature principle

Adapted from Abels et al. (2019), Dynamic Weights in Multi-Objective Deep Reinforcement Learning.

The relevant DER principles are:

- trajectories are atomic memory units;
- diversity replaces pure recency as the memory-selection criterion;
- discounted multi-objective return vectors provide trajectory signatures;
- crowding distance promotes coverage of diverse return regions.

## PPO-safe adaptation

The original work uses off-policy value-learning replay. That mechanism was not copied into PPO.

Instead, DER was adapted only as a memory-selection strategy for the already validated auxiliary action-response rehearsal:

- retained object remains frozen preference-conditioned Delta-a;
- candidate memory consists of complete H32 trajectories;
- each trajectory signature concatenates discounted 4D return and its conditioning preference;
- classic per-dimension normalized crowding distance selects memory units;
- 32 trajectory units per acquired axis are retained;
- 8 states per trajectory are stored, giving 256 rehearsal states per axis, approximately matching the previous static reference-set state count;
- when an axis is semantically competent again, new trajectories are proposed and diversity pruning replaces FIFO/recency pruning;
- the auxiliary rehearsal gradient is bounded at rho = 0.25.

This isolates memory-distribution diversity without increasing nominal memory-state count or introducing off-policy PPO updates.

## Short-gate result — seed 980001

| Arm | PASS events | Mean semantic score | Mean mixed-total cosine | Mean weighted auxiliary ratio |
|---|---:|---:|---:|---:|
| Control | 4 | 0.453125 | 1.0000 | 0 |
| Static bounded Delta-a | 6 | 0.457031 | 0.9739 | 0.21875 |
| DER-diverse bounded Delta-a | 5 | 0.445313 | 0.9739 | 0.21875 |

Axis observations:
- DER acquired T but retained-pass fraction after acquisition was 0.
- DER acquired A and retained it better locally than static Delta-a on this seed.
- O worst-case forgetting improved from 0.50 control to 0.375 under DER, but O never passed the endpoint criterion.
- S retained-pass fraction was 0.25, below static Delta-a at 0.375.

Interpretation:

The DER adaptation changed which competencies were visited/preserved, but did not improve the overall semantic-retention gate. Its mean semantic score was below both control and the static Delta-a baseline.

Optimization safety remained intact, so the failure is not attributable to gradient-budget instability.

### L1 decision

**FAIL short gate.**

Do not tune:
- buffer size,
- number of stored states,
- crowding metric,
- return/preference scaling,
- refresh frequency,
- additional preference samples.

No multi-seed expansion is authorized for L1.

---

# L2 — multi-timescale policy consolidation adaptation

## Literature principle

Adapted from Kaplanis et al. (2019), Policy Consolidation for Continual Reinforcement Learning.

The relevant principles are:

- retain policy history at multiple timescales;
- regularize current policy by historical policies through KL constraints;
- deeper/slower historical policies impose longer-timescale memory;
- policy-level consolidation extends PPO-like policy-space regularization beyond only the previous update.

## V2-B adaptation

A compact three-timescale variant was used for the eight-update short gate:

- historical teacher periods: 1, 2, 4 updates;
- relative internal teacher weights: 1, 2, 4;
- teacher snapshots are updated only after the student update at their declared period;
- consolidation uses diagonal Gaussian pre-tanh policy distributions;
- KL direction: D_KL(current || historical teacher), consistent with the reverse-KL-style policy-history constraint used in the policy-consolidation formulation;
- KL is evaluated on the current on-policy PPO observation/preference batch;
- no historical transition is used in the PPO likelihood objective;
- the total consolidation gradient is bounded by rho = 0.25.

This is an adaptation of the multi-timescale policy-consolidation principle, not a claim of exact reproduction of the original full cascade architecture.

## Short-gate result — seed 980001

| Arm | PASS events | Mean semantic score | Mean mixed-total cosine | Mean weighted auxiliary ratio |
|---|---:|---:|---:|---:|
| Control | 4 | 0.453125 | 1.0000 | 0 |
| Static bounded Delta-a | 6 | 0.457031 | 0.9739 | 0.21875 |
| Policy consolidation | 4 | 0.390625 | 0.9789 | 0.18750 |

Axis observations under policy consolidation:
- T acquired at u3 but retained-pass fraction afterward = 0.
- A acquired at u4 but retained-pass fraction afterward = 0.
- O never passed and ended at semantic score 0.
- S retained-pass fraction = 0.25.
- preference-conditioned action separation remained intact.

Interpretation:

Multi-timescale policy-history regularization did not improve semantic retention under the validated V2-B pipeline. The bounded budget successfully prevented update hijacking, but policy-history proximity itself was not sufficient to preserve the desired semantic competencies and reduced the mean semantic score substantially relative to control.

### L2 decision

**FAIL short gate.**

Do not tune:
- cascade length,
- teacher periods,
- KL direction,
- teacher weighting,
- teacher refresh timing,
- consolidation coefficient beyond the already validated bounded-gradient mechanism.

No multi-seed expansion is authorized for L2.

---

# Literature-guided round conclusion

The literature-guided round was deliberately narrow:

1. DER-style diversity-based trajectory memory;
2. multi-timescale policy consolidation.

Neither candidate passed the same first-stage retention criterion.

Therefore the absence of a successful retention method is no longer based only on custom mechanisms. Two closely related published retention principles were also adapted under the validated preference-conditioned MORL pipeline and did not show sufficient short-gate efficacy to justify confirmatory expansion.

This does not contradict the original papers:
- DER was developed with off-policy value learning in a dynamic-weight MORL setting and is adapted here only at the memory-selection level;
- Policy Consolidation was developed as a multi-policy cascade for continual RL and is represented here by a bounded compact multi-timescale distillation adaptation.

The result is specific to the present V2-B / Foundation-V2 setting and adaptation contract.

## Final causal status

- custom retention-mechanism search: CLOSED
- literature-guided retention round: CLOSED
- bounded auxiliary-gradient budget: VALIDATED
- static bounded Delta-a: CONTROLLED RETENTION BASELINE ONLY
- DER-style diverse semantic replay adaptation: REJECTED AT SHORT GATE
- multi-timescale policy consolidation adaptation: REJECTED AT SHORT GATE
- semantic-outcome rehearsal formulation: CLOSED
- final retention intervention: NONE

## Final method-selection implication

The final thesis reference configuration remains:

**V2-B + Foundation V2 + GAE lambda = 0.95 + no retention intervention.**

Bounded gradient budgeting remains a supported methodological finding, not part of the final reference training method.

## Thesis-level wording

> A final literature-guided retention round tested diversity-based trajectory memory inspired by Diverse Experience Replay and multi-timescale policy-history regularization inspired by Policy Consolidation. Both adaptations retained the previously validated bounded auxiliary-gradient budget and avoided off-policy PPO updates. Although optimization geometry remained stable, neither adaptation passed the short semantic-retention gate. These results reinforce the distinction between stable rehearsal optimization and the harder problem of preserving preference-conditioned semantic competence.

## Stop rule

No further retention method, memory representation, replay rule, consolidation schedule, or retention coefficient is authorized within the current thesis scope.

Further work belongs in future work, not the present method-selection branch.
