# Update × Visitation Interaction Audit Verdict

Status: **FROZEN — NO INTERACTION EVIDENCE**

Date: 2026-09-24

## Question

Does robust semantic collapse arise because the finite update interacts with the target policy's new closed-loop state distribution, producing preference-response damage that is not present under either update-only or visitation-only conditions?

## Corpus and design

Primary corpus is the independent v18 source-PASS set:

    total eligible transitions = 22
    robust collapse            = 10
    retained / non-robust      = 12

For every transition-axis, exact heavy-preference closed-loop state trajectories were cached for both source and target checkpoints:

    38 unique checkpoint-axis state caches
    4 reset suites
    64 steps
    8 environments

Cache integrity:

    expected = 38
    files    = 38
    missing  = 0
    extra    = 0
    invalid  = 0

The same source and target policies were then evaluated offline in the 2x2 design:

    Sπ / Ss    source policy on source states
    Tπ / Ss    target policy on source states
    Sπ / St    source policy on target states
    Tπ / St    target policy on target states

## Primary interaction definition

For response/Jacobian destruction metrics `D = 1-cosine`, interaction is:

    I = D(Tπ,St) - D(Tπ,Ss) - D(Sπ,St) + D(Sπ,Ss)

with `D(Sπ,Ss)=0` by definition.

Positive interaction would indicate additional destruction that only appears when the updated policy is evaluated on the target closed-loop visitation distribution.

## Primary result: response rotation interaction

### Q1

    interaction AUC = 0.692

### Q4

    interaction AUC = 0.408

    robust median   = -0.00159
    retained median = -0.00094

Only 3/10 robust events exceed the retained Q4 median.

Interaction growth:

    AUC(Q4 - Q1) = 0.267

Seed direction is inconsistent.

The interaction becomes **more negative**, not more positively destructive, in robust collapse.

Therefore response-space update × visitation amplification is not supported.

## Primary result: preference-Jacobian interaction

### Q1

    interaction AUC = 0.667

### Q4

    interaction AUC = 0.333

    robust median   = -0.00210
    retained median = -0.00095

Only 2/10 robust events exceed the retained Q4 median.

Interaction growth:

    AUC(Q4 - Q1) = 0.350

Direction fails consistently across seeds.

Thus Jacobian-space interaction is also contrary to the predicted positive-amplification mechanism.

## Frozen gate

Neither primary family satisfies any meaningful late-interaction pattern:

| Criterion | Response rotation | Jacobian rotation |
|---|---:|---:|
| Q4 AUC >= .75 | FAIL (.408) | FAIL (.333) |
| >=.5 retained-IQR median effect | FAIL | FAIL |
| >=8/10 robust above retained median | FAIL (3/10) | FAIL (2/10) |
| 3-seed consistent direction | FAIL | FAIL |
| interaction-growth AUC >=.70 | FAIL (.267) | FAIL (.350) |
| beats v19 update-only AUC by >=.05 | FAIL | FAIL |

Formal verdict:

> **NO INTERACTION EVIDENCE**

## 2x2 cell decomposition

The failure of the interaction hypothesis is informative because the four cells show a strong but largely non-specific visitation effect.

### Heavy-vs-center action-response rotation

Median destruction relative to Sπ/Ss:

#### Q1

    update only:      Tπ/Ss
      robust AUC 0.667
      median robust 0.00254
      median retained 0.00177

    visitation only:  Sπ/St
      median robust 0.0278
      median retained 0.0322

    target closed-loop: Tπ/St
      median robust 0.0338
      median retained 0.0339

#### Q4

    update only:      Tπ/Ss
      AUC 0.742
      median robust 0.00179
      median retained 0.00136

    visitation only:  Sπ/St
      AUC 0.367
      median robust 0.5108
      median retained 0.5292

    target closed-loop: Tπ/St
      AUC 0.367
      median robust 0.5123
      median retained 0.5297

Thus visitation shift produces a very large phase-dependent response rotation in **both** outcomes. The target-policy/target-state cell is almost identical to visitation-only, rather than showing extra collapse-specific damage.

## Preference-Jacobian cell decomposition

The same pattern appears for the preference Jacobian.

### Q4

    update only Tπ/Ss
      AUC 0.717
      median robust   0.00181
      median retained 0.00141

    visitation only Sπ/St
      AUC 0.325
      median robust   0.5198
      median retained 0.5399

    target closed-loop Tπ/St
      AUC 0.317
      median robust   0.5202
      median retained 0.5397

Again, the dominant effect is the ordinary state-distribution change along the rollout, and it is at least as large in retained transitions as in collapse transitions.

## Secondary post-hoc observation: source-visitation-local update effect

One descriptive signal is worth retaining without promoting it to a mechanism.

Update-only effect on **source-policy visited states** (`Tπ/Ss`) becomes moderately discriminative in Q3:

    response rotation Q3 AUC = 0.767
    Jacobian rotation Q3 AUC = 0.783

By Q4 this falls to:

    response AUC = 0.742
    Jacobian AUC = 0.717

This was not the predeclared primary interaction metric and Q3 was not selected prospectively. It therefore cannot rescue the failed interaction hypothesis or authorize an intervention.

It does suggest that function-space update effects may depend on *which task-relevant source states are probed*, rather than on a universal fixed-state probe set. This is a future hypothesis only.

## Secondary magnitude interactions

Interaction of response magnitude and Jacobian norm remains weak:

    response-norm Q4 AUC = 0.550
    Jacobian-norm Q4 AUC = 0.508

No magnitude interaction separates robust collapse.

## Scientific interpretation

v19 suggested that fixed-state function changes alone were only modestly different. The present audit tests the natural next explanation: perhaps those updates become destructive only on the state distribution generated by the target policy.

That explanation is not supported.

The target-state visitation shift strongly rotates heavy-vs-center responses and preference Jacobians over the rollout, but it does so in retained transitions as strongly as—or more strongly than—in robust collapses.

Moreover:

    Tπ/St ≈ Sπ/St

for late-phase response/Jacobian destruction, producing a negative or near-zero difference-in-differences interaction.

Therefore:

> **The large closed-loop state-distribution shift is real but not collapse-specific, and robust semantic collapse is not explained by a super-additive interaction between policy update and target visitation under the tested action-response/Jacobian representations.**

## Decision

No visitation regularizer, Jacobian penalty, state-distribution matching objective, or update×visitation intervention is authorized.

The interaction branch is closed under this representation and corpus.

The remaining evidence does not support a simple low-dimensional mechanism among the tested source descriptors, update magnitudes, static functional changes, or update×visitation interaction terms.

## Thesis wording

> A 2×2 policy-by-visitation audit found no evidence that robust semantic collapse was caused by a super-additive interaction between the finite policy update and the subsequent target-state distribution. Closed-loop visitation itself produced large phase-dependent rotations of heavy-versus-center action responses and preference Jacobians, reaching approximately 0.51 destruction by the final quarter, but these shifts were equally large or larger in retained transitions. The difference-in-differences interaction was non-discriminative and often negative (Q4 AUC 0.408 for response rotation and 0.333 for Jacobian rotation). Thus target visitation strongly changes the local policy-response geometry, but this change is not specific to semantic collapse.
