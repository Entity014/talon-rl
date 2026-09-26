# Relational-Persistence Held-Out Validation Verdict

Status: **FROZEN — PARTIAL / INCONCLUSIVE; RELATIONAL CONTRAST VALIDATED, TEMPORAL PERSISTENCE NOT YET INCREMENTALLY VALIDATED**

Date: 2026-09-24

## Question

Does the discovery-set finding that semantic competence is better tracked by matched heavy-versus-center temporal persistence than by the heavy-policy scalar objective return reproduce on independent training seeds and policy paths?

## Strict held-out set

Primary validation uses only the ordinary weighted-PPO CONTROL arms from the existing bounded-Delta-a multi-seed experiment:

- training seed 980001
- training seed 981001
- training seed 982001

For each seed:

    shared base checkpoint -> control u1 -> ... -> control u8

Primary sample size:
- 3 independent training seeds
- 24 finite policy transitions
- 96 axis-transition samples
- 25 unique policy checkpoints after hash deduplication

None of these control paths were part of the original trajectory-attribution discovery set.

The action-response rehearsal arms were excluded from the primary validation.

## Frozen evaluation

Every checkpoint was re-evaluated using the unchanged semantic endpoint protocol:
- 8 environments
- 64 steps
- 4 matched suites
- reset seeds 840001..840004
- T/A/O/S-heavy and center preferences
- unchanged V2-B objective normalization and physical proxies

No optimizer step, reward change, descriptor refit, threshold refit, or window selection occurred.

Semantic targets were intentionally non-tautological:

    objective correctness fraction
    physical correctness fraction
    semantic score = 0.5 * (objective correctness + physical correctness)
    frozen endpoint PASS

## Formal gate result

The predeclared held-out gate returns:

> **PARTIAL / INCONCLUSIVE**

Seven of eight criteria pass.

Passed:
- at least 4/6 relational-temporal descriptor groups reproduce: **PASS (6/6)**
- median relational-temporal |Spearman| exceeds heavy-only J by >=0.15: **PASS**
- matched relational late descriptors beat heavy-only late controls by >=0.10 on both objective and physical sides: **PASS**
- key temporal descriptors have consistent correlation sign across all 3 training seeds: **PASS**
- at least 4 PASS->FAIL events exist: **PASS (14 events)**
- both late objective and late physical relational advantage deteriorate in >=75% of PASS->FAIL events: **PASS (14/14 = 100%)**
- heavy-only J sensitivity is >=0.15 lower than late relational sensitivity: **PASS (64.3% vs 100%)**

Failed:
- temporal descriptors must add >=0.10 absolute Spearman beyond the corresponding matched heavy-versus-center mean relation on both objective and physical sides: **FAIL**

Therefore reward redesign remains blocked under the original contract.

## Baseline and mechanism-control correlations

### Heavy-only summaries

| Descriptor | Target | Spearman rho |
|---|---|---:|
| heavy total objective return J_i | objective correctness | **0.409** |
| heavy objective late mean | objective correctness | **0.420** |
| heavy physical late mean | physical correctness | **0.358** |

Temporal position alone does not solve the mismatch.

### Matched heavy-versus-center mean relation

| Descriptor | Target | Spearman rho |
|---|---|---:|
| mean objective advantage | objective correctness | **0.732** |
| mean physical advantage | physical correctness | **0.754** |

This is a large and consistent held-out improvement over the heavy-only scalar return.

Per-seed Spearman for mean objective advantage:
- seed 980001: **0.678**
- seed 981001: **0.735**
- seed 982001: **0.751**

Per-seed Spearman for mean physical advantage:
- seed 980001: **0.813**
- seed 981001: **0.690**
- seed 982001: **0.782**

Thus the **relational heavy-versus-center component generalizes strongly across training seeds**.

### Frozen relational-temporal candidates

| Descriptor | Spearman rho |
|---|---:|
| objective late advantage | **0.658** |
| physical late advantage | **0.681** |
| objective late-minus-early | **0.672** |
| physical late-minus-early | **0.653** |
| objective slope | **0.669** |
| physical slope | **0.680** |
| objective final-16 | **0.622** |
| physical final-16 | **0.688** |
| objective worst-16 | **0.559** |
| physical worst-16 | **0.630** |
| objective positive-advantage fraction | **0.697** |
| physical positive-advantage fraction | **0.716** |

All six paired descriptor groups exceed |rho|=0.50 on both objective and physical targets.

Median absolute Spearman across all 12 relational-temporal descriptors:

    0.671

versus heavy-only scalar J:

    0.409

Difference:

    +0.262

Thus the discovery-set relational-temporal signal clearly reproduces in held-out data.

## Why the formal gate does not fully pass

The held-out mechanism controls separate three hypotheses:

    heavy-only total
        -> heavy-only late
        -> heavy-center mean
        -> heavy-center temporal persistence

Observed pattern:

    heavy-only total             weak/moderate
    heavy-only late              weak/moderate
    heavy-center mean            strong
    heavy-center temporal        strong

but the temporal descriptors do **not** exceed the matched relational mean by the predeclared +0.10 margin.

Indeed, mean relation is slightly stronger globally:

    objective mean relation rho = 0.732
    objective late relation rho = 0.658

    physical mean relation rho  = 0.754
    physical late relation rho  = 0.681

Therefore the held-out evidence validates **relation** much more strongly than it validates **additional temporal information beyond relation**.

The correct diagnosis is not:

> temporal persistence is unnecessary.

Rather:

> temporal persistence reproduces strongly and is especially sensitive to forgetting events, but its incremental information beyond the matched heavy-versus-center relation is not established by global held-out correlation.

## PASS -> FAIL behavior

There are **14 held-out PASS -> FAIL events**.

Deterioration fractions:

| Descriptor | Fraction deteriorating |
|---|---:|
| heavy-only J | **64.3%** |
| heavy objective late | 71.4% |
| heavy physical late | 78.6% |
| mean objective relation | 85.7% |
| mean physical relation | 85.7% |
| late objective relation | **100%** |
| late physical relation | **100%** |
| objective late-minus-early | **100%** |
| physical late-minus-early | 92.9% |
| objective slope | **100%** |
| physical slope | 92.9% |
| objective final-16 | 92.9% |
| physical final-16 | **100%** |
| objective worst-16 | 85.7% |
| physical worst-16 | 78.6% |

This reproduces the discovery observation that late relational structure is highly sensitive to semantic forgetting.

However, because the relational mean itself already identifies 85.7% of PASS->FAIL events, the held-out evidence does not establish that temporal persistence is required for the continuous semantic target generally.

## Scientific interpretation

The held-out audit sharpens the previous diagnosis.

### Validated

The current training objective is missing a **relational preference effect**:

> semantic competence depends strongly on how the heavy-preference policy behaves relative to the matched center policy, whereas the current training objective primarily optimizes the absolute heavy-policy return.

This relation generalizes across three independent training seeds and 96 held-out axis-transitions.

### Strong but not incrementally validated

Temporal persistence remains important for the specific forgetting phenomenon:
- late relation deteriorates in 14/14 PASS->FAIL events;
- slope and late-minus-early descriptors show nearly the same event sensitivity;
- all relational-temporal descriptor groups reproduce strongly.

But global held-out correlation does not show a >=0.10 improvement over the matched relational mean.

Therefore the evidence supports:

    RELATIONAL CONTRAST        VALIDATED
    TEMPORAL PERSISTENCE       STRONG SECONDARY / EVENT-SENSITIVE SIGNAL
    TEMPORAL INCREMENTAL VALUE NOT YET VALIDATED

## Decision

**HELD-OUT RESULT: PARTIAL / INCONCLUSIVE under the predeclared reward-redesign gate.**

Reward redesign remains **NOT AUTHORIZED** by this contract.

No coefficient, reward term, architecture, optimizer, lambda, replay, or semantic threshold is changed.

If further research is pursued, the next question should be narrower than another broad attribution sweep:

> Can the matched heavy-versus-center relational mean be made into a causal training signal without pathological improvement through degrading the center policy, and does adding a persistence component provide incremental semantic-retention benefit beyond that relational mean?

This should be tested as a controlled objective-design experiment with an absolute-heavy-performance guardrail, not by further descriptor selection.

## Updated thesis wording

Recommended wording:

> Held-out validation across three independent training seeds confirmed that matched heavy-versus-center behavioral contrast generalized substantially better than the heavy-policy return alone as an indicator of semantic competence. Mean relational advantages achieved rank correlations around 0.73--0.75, compared with 0.41 for the heavy-only objective return. Temporal relational descriptors also reproduced strongly and deteriorated in every observed PASS-to-FAIL event, but they did not provide a predeclared incremental correlation gain over the relational mean. The evidence therefore validates missing relational information and identifies temporal persistence as an event-sensitive secondary structure rather than a fully isolated causal factor.

Primary artifacts:
- `docs/contracts/diagnostics/relational-persistence-heldout-contract.md`
- `scripts/rl/relational_persistence_heldout_audit.py`
- `runs/relational_persistence_heldout-2026-09-24/heldout_report.json`
