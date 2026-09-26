# Relational-Objective Design Audit Contract

Status: **PREDECLARED — OFFLINE DIAGNOSTIC ONLY, NO TRAINING**

Date: 2026-09-24

## Motivation

Held-out validation established that matched heavy-versus-center relational information generalizes substantially better than absolute heavy-policy return as an indicator of semantic competence.

The next question is not whether to add a temporal reward. It is:

> Can a relational objective be added without creating false improvement by degrading the center policy?

No optimizer, PPO rule, reward, architecture, or semantic threshold is changed in this audit.

## Frozen data

Use only the strict held-out control data from:

`runs/relational_persistence_heldout-2026-09-24/heldout_report.json`

This contains:
- 3 independent weighted-PPO training seeds
- 24 finite policy transitions
- 96 axis-transition samples
- matched endpoint semantic correctness and PASS transitions

No new rollouts are required.

## Reconstructed objective quantities

For every checkpoint and axis i:

    J_H = heavy-policy normalized objective return
    R   = J_H - J_C = matched heavy-versus-center mean objective relation

Therefore:

    J_C = J_H - R

For each finite transition compute:

    Delta J_H
    Delta J_C
    Delta R = Delta J_H - Delta J_C

## Frozen candidate family

For eta in the predeclared set:

    eta in {0.25, 0.50, 1.00}

score the unconstrained relational candidate:

    J_cand(eta) = J_H + eta * (J_H - J_C)

so:

    Delta J_cand
      = (1 + eta) Delta J_H - eta Delta J_C

No eta is selected or tuned after observing the result. Robustness across all three is the target.

## Absolute-heavy guardrail

A relational gain is considered **admissible** only if:

    Delta J_H >= 0

with numerical tolerance 1e-10.

This is a strict diagnostic guardrail; no epsilon is tuned.

The audit separately reports unconstrained and guardrailed behavior.

## Semantic targets

Primary continuous target:

    Delta objective correctness fraction

because the candidate objective is built from objective returns.

Secondary target:

    Delta semantic score
      = 0.5 * (Delta objective correctness
               + Delta physical correctness)

Primary event target:

    PASS -> FAIL

## Baselines / controls

Compare:

1. `Delta J_H` — current absolute heavy-only return
2. `Delta R` — pure heavy-versus-center relation
3. `Delta J_cand(eta)` — absolute + relational candidate
4. guardrailed candidate — same candidate, but positive relational gains are considered usable only when `Delta J_H >= 0`

The pure relation control is diagnostic only; it is not a proposed reward because it can be increased by degrading center performance.

## Required pathology diagnostics

For every eta report:

### Center-degradation false gain
Fraction of transitions satisfying:

    Delta J_cand > 0
    AND Delta J_H <= 0

This is the cleanest indicator that the relational candidate can call an update "good" despite non-improving heavy performance.

### Center-driven contribution share
For candidate-positive transitions decompose:

    heavy contribution  = (1 + eta) Delta J_H
    center contribution = -eta Delta J_C

Report fraction where:

    center contribution > max(heavy contribution, 0)

### Semantic false gain
Fraction of candidate-positive transitions with:

    Delta objective correctness < 0

and separately with:

    Delta semantic score < 0

### PASS -> FAIL false approval
Among PASS -> FAIL transitions, fraction where:

    Delta J_cand > 0

and the same fraction after the strict heavy guardrail.

## Alignment metrics

For `Delta J_H`, `Delta R`, and each `Delta J_cand(eta)` report:
- Spearman with Delta objective correctness
- Pearson with Delta objective correctness
- sign agreement
- Spearman with Delta semantic score
- per-seed Spearman
- per-axis Spearman

For the guardrailed subset (`Delta J_H >= 0`) report the same correlations descriptively, together with sample count.

## Predeclared authorization gate

A minimal relational-objective design branch is authorized only if **all** hold for every eta in {0.25, 0.50, 1.00}:

1. `Delta J_cand` Spearman with Delta objective correctness exceeds the `Delta J_H` baseline by >= 0.10.
2. Candidate semantic false-gain fraction on objective correctness is <= 0.25.
3. Center-degradation false-gain fraction is <= 0.10.
4. After applying the strict `Delta J_H >= 0` guardrail, PASS -> FAIL false approval is <= 0.20.
5. Guardrailed candidate retains at least 60% of candidate-positive transitions; the guardrail must not make the candidate nearly unusable.
6. Candidate sign relationship is consistent across all 3 training seeds.

Additionally:
- pure relational `Delta R` must outperform `Delta J_H` by >=0.15 Spearman, confirming that the held-out relational signal remains present in this exact offline design sample.

## Interpretation

### AUTHORIZED
Absolute+relational scoring improves alignment robustly, and the heavy-performance guardrail controls center-degradation pathology without discarding most candidate-positive updates.

This authorizes a separate minimal objective-design branch, not full training.

### RELATION VALID BUT CANDIDATE PATHOLOGICAL
Pure relation is informative, but adding it directly creates too many false gains from center degradation or semantic deterioration. Reward redesign remains blocked until a different constrained formulation is specified.

### NO BENEFIT
The relational candidate does not improve alignment beyond heavy-only return. Reward redesign remains blocked.

No temporal term is introduced in this audit.
