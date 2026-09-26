# Phase-1 Semantic Target Design Audit Verdict

Status: **FROZEN — STAGE A FAIL; CONJUNCTIVE MEAN RELATIONAL TARGET NOT AUTHORIZED**

Date: 2026-09-24

## Question

Can a semantic target that combines objective and physical heavy-versus-center relations conjunctively align more closely with semantic competence than the validated objective relation alone, while avoiding scalar-compensation and center-degradation pathologies?

## Candidate

Discovery-only scale calibration was used per axis:

    z_obj  = m_obj  / median_discovery(|m_obj|)
    z_phys = m_phys / median_discovery(|m_phys|)

Primary candidate:

    C_sem = -log(0.5 * [exp(-z_obj) + exp(-z_phys)])

This is a smooth minimum-like conjunction:
- both objective and physical relations contribute;
- the weaker channel has greater influence;
- heavy=center maps to zero relational competence;
- no mixing coefficient or temperature is tuned.

Validation uses only the strict three-seed held-out CONTROL corpus:
- 24 policy transitions
- 96 axis-transition samples
- 14 PASS->FAIL events
- 16 FAIL->PASS events

No training or new rollout was performed.

## Baselines

| Score | Meaning | Spearman vs Delta semantic score |
|---|---|---:|
| B0 | absolute heavy return `J_H` | **0.421** |
| B1 | standardized objective relation | **0.804** |
| B2 | arithmetic objective+physical relation | **0.815** |
| C_sem | conjunctive smooth-min relation | **0.813** |

The candidate is substantially better than absolute heavy-only return but does not materially improve over the already validated objective relation or the simple arithmetic two-channel relation.

## Event behavior

### PASS -> FAIL false approval

| Score | False approval |
|---|---:|
| B0 | 35.7% |
| B1 | **14.3%** |
| B2 | **14.3%** |
| C_sem | **14.3%** |

The conjunctive target does not reduce semantic forgetting false approval beyond either relational control.

### FAIL -> PASS true approval

- B0: 75.0%
- B1: 93.8%
- B2: 87.5%
- C_sem: 87.5%

### Positive-score semantic deterioration

Among positive transitions:
- B0: 27.3% semantically deteriorate
- B1: **3.8%**
- B2: **4.0%**
- C_sem: **4.0%**

Again, conjunction is informative but adds no measurable benefit over the simpler relational controls.

## Cross-seed robustness

C_sem has positive held-out Spearman in all three independent training seeds:
- seed 980001: 0.830
- seed 981001: 0.763
- seed 982001: 0.850

Thus the target is not noisy or non-generalizing. Its failure is one of **incremental sufficiency**, not statistical instability.

## Objective-side center-degradation pathology

Among 50 C_sem-positive transitions:
- 24% have Delta J_H < 0
- 40% are objective-center dominated under the frozen decomposition
- strict Delta J_H >= 0 guardrail retains 76% of candidate-positive transitions

However, after restricting to PASS->FAIL events admissible under the heavy guardrail, false approval is **40%**.

Therefore an absolute-heavy guardrail does not rescue the target's semantic sufficiency.

Stage B physical heavy/center decomposition is not authorized because Stage A fails.

## Decisive false approvals

The only two C_sem-positive PASS->FAIL events are the same residual counterexamples already identified by semantic-gate factorization.

### Seed 980001, u3 -> u4, Smoothness

    Delta J_H > 0
    Delta objective relation > 0
    Delta C_sem > 0

but:

    Delta semantic score = -0.25
    PASS -> FAIL

### Seed 981001, u4 -> u5, Tracking

Again:

    Delta J_H > 0
    Delta objective relation > 0
    Delta C_sem > 0

but:

    Delta semantic score = -0.25
    PASS -> FAIL

Adding the physical relational channel conjunctively does not eliminate these failures.

## Predeclared gate

Passed:
- C_sem Spearman >=0.65
- C_sem beats absolute heavy-only B0 by >=0.20
- PASS->FAIL false approval <=0.15
- FAIL->PASS true approval >=0.60
- positive-score semantic deterioration <=0.15
- positive correlation in all three held-out seeds

Failed:
- C_sem does not beat B1 by >=0.05
- C_sem does not reduce PASS->FAIL false approval by >=0.10 versus B2
- strict heavy guardrail does not preserve/improve PASS->FAIL behavior

Formal verdict:

> **STAGE A FAIL**

## Interpretation

The audit establishes a sharper hierarchy:

    absolute heavy return
        -> weak

    objective heavy-vs-center relation
        -> strong

    objective + physical mean relation
        -> slightly stronger descriptively

    conjunctive objective+physical mean relation
        -> no additional semantic-retention benefit

Therefore the missing semantic structure is not recovered simply by combining the two validated mean relational channels with an AND-like scalar aggregation.

The residual false approvals are precisely the context-sensitive failures where mean relations improve but semantic correctness across suites collapses.

This is consistent with the prior conclusion that context structure contains real residual information, while the prospective evidence remains insufficient to justify a context intervention.

## Decision

No Stage B physical-decomposition replay is run.

No semantic-target training pilot is authorized.

No coefficient, temporal term, context term, PPO rule, architecture, optimizer, or replay change is authorized from this result.

The Phase-1 controller-fix branch should not proceed by further scalar combinations of mean objective and physical relations.

## Thesis wording

> A conjunctive semantic target combining normalized objective and physical heavy-versus-center margins was robustly aligned with semantic-score changes, but it did not improve meaningfully over simpler relational controls and did not reduce PASS-to-FAIL false approvals. The remaining false approvals were the same context-sensitive counterexamples in which mean relational quantities improved while semantic competence was lost. Thus combining mean objective and physical relations was informative but still insufficient as a semantic training target.

Primary artifacts:
- `docs/contracts/transfer/phase1-semantic-target-design-contract.md`
- `scripts/rl/phase1_semantic_target_design_audit.py`
- `runs/phase1_semantic_target_design_audit-2026-09-24/phase1_semantic_target_design_report.json`
