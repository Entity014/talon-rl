# Competence-Floor / Max-Min Optimization Audit Contract

Status: **PREDECLARED — DIAGNOSTIC ONLY, NO TRAINING**

Date: 2026-09-24

## Motivation

Architecture and retention search are closed.

The remaining empirical failure is winner rotation under successive shared-policy optimization: semantic competence can be acquired, but a later weighted-sum PPO update path can improve another objective while previously acquired competence disappears.

V2-K is decisive for this branch because it established genuine preference-dependent private subspaces yet still reproduced winner rotation. Therefore the next question concerns the optimization criterion rather than additional representational capacity.

## Literature principle

This audit borrows only the optimization principle from max-min MORL and conflict-averse multi-task optimization:

> prefer a joint update whose worst local objective improvement is improved, rather than accepting the weighted-average direction whenever one objective has negative local improvement.

No published algorithm is copied verbatim in this gate.

## Frozen setting

Architecture:
- V2-B single-site FiLM
- Foundation V2
- GAE lambda = 0.95
- no retention intervention

Frozen checkpoints:
- runs/v2b_lambda095_pilot-2026-09-23/model_50.pt
- runs/v2b_lambda095_pilot-2026-09-23/model_75.pt

These checkpoints are chosen because the frozen semantic path shows winner rotation:
- u50: Orientation passes
- u75: Angular passes while Orientation collapses

Audit batches:
- exact training-style mixed endpoint preferences
- 8 environments, two instances of each T/A/O/S-heavy preference
- fresh on-policy rollouts
- four fixed diagnostic seeds per checkpoint
- H32, GAE lambda = 0.95
- same repaired transformed-action PPO semantics

No optimizer step is allowed.

## Directions

For objective i, let q_i be the actor ascent gradient of its unscalarized PPO surrogate on the same mixed batch.

Normalize:

    u_i = q_i / (||q_i|| + eps)

The current training direction is the exact ascent direction of the existing weighted MORL PPO objective:

    d_mix = - grad L_MORL

For fair first-order comparisons, define normalized predicted gain:

    I_i(d) = u_i^T (d / ||d||)

Thus I_i(d) is the cosine-like first-order improvement for objective i, independent of objective gradient magnitude.

## Max-min candidate

Search deterministic simplex weights alpha over four objectives:

    v(alpha) = sum_i alpha_i u_i
    alpha_i >= 0
    sum_i alpha_i = 1

Choose:

    alpha* = arg max_alpha min_i I_i(v(alpha))

Then rescale to the exact mixed-direction norm:

    d_floor = ||d_mix|| * v(alpha*) / ||v(alpha*)||

The max-min candidate therefore has exactly the same first-order step norm budget as the current mixed PPO direction.

Simplex search resolution is fixed at 0.02 (50 integer units).

## Primary diagnostics

For every checkpoint x seed case report:
- per-objective gradient norms
- pairwise objective-gradient cosine matrix
- mixed-direction normalized gains I_T/I_A/I_O/I_S
- floor-direction normalized gains I_T/I_A/I_O/I_S
- worst gain for mixed and floor directions
- mean gain for mixed and floor directions
- number of negative-gain objectives
- floor simplex weights alpha*
- cosine(d_floor, d_mix)
- norm ratio ||d_floor|| / ||d_mix||
- improvement in worst normalized gain

Aggregate separately for u50 and u75 and across all eight cases.

## Predeclared diagnostic PASS gate

The branch is authorized for a short training pilot only if all of the following hold:

1. **Norm budget exactness**
   - mean and max absolute error of ||d_floor||/||d_mix|| - 1 <= 1e-5.

2. **Worst-objective improvement**
   - floor direction improves the worst normalized gain over mixed direction in at least 7/8 checkpoint-seed cases.

3. **Meaningful gain**
   - mean improvement in worst normalized gain across all cases >= 0.05.

4. **Conflict repair**
   - whenever the mixed direction has at least one negative predicted objective gain, the floor direction reduces the number of negative axes in at least 75% of such cases.

5. **Non-degenerate update**
   - floor direction is finite and nonzero in every case.
   - mean cosine with mixed direction > 0.10.
   - no case has cosine with mixed direction < -0.50.

6. **Checkpoint relevance**
   - at both u50 and u75, aggregate worst normalized gain under the floor direction is greater than under the mixed direction.

A stronger diagnostic outcome is recorded separately if the floor direction achieves non-negative predicted gain for all four objectives.

## Decision rule

PASS:
- authorize one short, bounded competence-floor training pilot;
- architecture, critic, lambda, PPO semantics, data schedule, and evaluation contract remain frozen.

FAIL:
- close the competence-floor branch without training.

No semantic threshold tuning, architecture changes, replay, retention, gradient-memory terms, optimizer hyperparameter tuning, or additional regularizers are authorized in this diagnostic gate.
