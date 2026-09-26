# Phase-1 Semantic Target Design Audit Contract

Status: **PREDECLARED — OFFLINE TARGET DESIGN, NO TRAINING**

Date: 2026-09-24

## Base controller is frozen

All future controller-fix evidence in this branch uses:
- V2-B single-site FiLM actor
- Foundation V2 critic/freshness contract
- GAE lambda=.95
- repaired transformed-action PPO semantics
- unchanged architecture / optimizer / replay / retention

This audit changes no controller parameter.

## Evidence motivating the target

Validated:
- absolute heavy-only return is not sufficient;
- matched heavy-versus-center relation generalizes across held-out seeds;
- objective-only scalar relational augmentation improves alignment but is still insufficient;
- semantic PASS requires both objective and physical directional correctness.

Therefore the target candidate should represent **objective + physical relational competence conjunctively**, without adding temporal/context structure that has not been incrementally validated.

## Data split

### Calibration only — discovery corpus
Use only:
`runs/trajectory_information_attribution_audit-2026-09-24/trajectory_information_attribution_report.json`

This corpus is used only to fix per-axis scale constants.

### Validation — strict held-out control corpus
Use only:
- `runs/semantic_gate_factorization_audit-2026-09-24/semantic_gate_factorization_report.json`
- `runs/relational_persistence_heldout-2026-09-24/heldout_report.json`

Validation consists of:
- 3 independent weighted-PPO control seeds
- 24 transitions
- 96 axis-transition samples

No scale or formulation is refit on held-out data.

## Per-axis relational margins

At checkpoint theta:

    m_obj_i(theta)  = mean heavy-minus-center normalized objective margin
    m_phys_i(theta) = mean center-minus-heavy physical margin

Both use higher-is-better sign convention.

## Frozen scale calibration

For each axis independently, using discovery checkpoints only:

    s_obj_i  = median(|m_obj_i|)  + 1e-8
    s_phys_i = median(|m_phys_i|) + 1e-8

Then:

    z_obj_i  = m_obj_i / s_obj_i
    z_phys_i = m_phys_i / s_phys_i

No learned calibration model is fit.

## Candidate and controls

### Current baseline

    B0 = J_H

absolute heavy-policy normalized objective return.

### Relational objective-only control

    B1 = z_obj

### Two-channel arithmetic relational control

    B2 = 0.5 * (z_obj + z_phys)

This allows one channel to compensate for the other.

### Primary semantic candidate — conjunctive smooth minimum

    C_sem = -log(0.5 * [exp(-z_obj) + exp(-z_phys)])

Properties:
- C_sem = 0 when both normalized margins are zero;
- improving the weaker channel has greater marginal effect;
- a strongly positive objective relation cannot freely compensate for a negative physical relation, or vice versa;
- heavy=center produces zero relational competence rather than a positive target, so preference collapse is not rewarded by construction.

Numerically stable log-sum-exp implementation is required.

No temperature or mixing coefficient is tuned.

## Held-out transition quantities

For every finite transition compute:

    Delta B0
    Delta B1
    Delta B2
    Delta C_sem

Targets:
- Delta semantic score = 0.5*(Delta objective correctness + Delta physical correctness)
- PASS -> FAIL
- FAIL -> PASS

## Primary metrics

For all four scores report:
- Spearman with Delta semantic score
- Pearson with Delta semantic score
- sign agreement
- per-seed Spearman
- per-axis Spearman

Event metrics:
- PASS -> FAIL false approval: fraction with Delta score > 0
- FAIL -> PASS true approval: fraction with Delta score > 0
- among Delta score > 0 cases, fraction with Delta semantic score < 0

## Objective-side center-degradation guardrail

From held-out reports:

    J_C_obj = J_H - m_obj

For every candidate-positive transition report whether:

    Delta J_H < 0

and whether positive relational improvement is center-driven.

A strict diagnostic guardrail is also evaluated:

    Delta J_H >= 0

Physical heavy/center absolute decomposition is not present in the frozen report corpus. It is therefore **not inferred**. If and only if the semantic candidate passes the primary offline gate, a second measurement-only replay is authorized to collect absolute physical heavy/center values and close the physical center-degradation check before any training.

## Predeclared primary target gate

C_sem passes Stage A only if all hold:

1. Spearman(Delta C_sem, Delta semantic score) >= 0.65.
2. C_sem Spearman exceeds B0 by >=0.20.
3. C_sem Spearman exceeds B1 by >=0.05.
4. C_sem PASS->FAIL false approval <=0.15.
5. C_sem PASS->FAIL false approval is at least 0.10 lower than B2.
6. C_sem FAIL->PASS true approval >=0.60 when at least 4 such events exist.
7. Among Delta C_sem > 0 cases, semantic-score deterioration fraction <=0.15.
8. Correlation sign is positive in all 3 held-out training seeds.
9. Under strict Delta J_H >=0 guardrail, PASS->FAIL false approval does not increase and at least 60% of candidate-positive transitions remain admissible.

If Stage A fails, semantic-target training remains blocked and no physical-decomposition rerun is performed.

If Stage A passes, authorize Stage B measurement only:
- rerun held-out checkpoints;
- record absolute physical heavy and center values;
- reject the target if >10% of candidate-positive transitions are explainable by physical-center degradation while heavy physical performance does not improve.

Only after Stage A + Stage B pass may a short paired training pilot be designed.

## Decision scope

This contract authorizes no reward coefficient, PPO update, temporal term, context term, architecture change, optimizer change, or training run.
