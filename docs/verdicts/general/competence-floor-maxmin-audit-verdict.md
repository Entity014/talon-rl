# Competence-Floor / Max-Min Fixed-Policy Audit Verdict

Status: **FROZEN — DIAGNOSTIC PASS; SHORT PAIRED PILOT AUTHORIZED**

Date: 2026-09-24

## Question

Can a norm-matched max-min actor update repair the local first-order failure of the current weighted MORL PPO direction at frozen checkpoints where semantic winners rotate?

## Frozen diagnostic

Architecture / training foundation:
- V2-B + Foundation V2
- GAE lambda = 0.95
- no retention
- no architecture change

Checkpoints:
- u50: Orientation endpoint PASS
- u75: Angular endpoint PASS, Orientation collapsed

For each checkpoint, four fresh mixed training-style on-policy batches were collected. Per-objective PPO ascent gradients were normalized, and a deterministic simplex search selected the direction maximizing the minimum normalized first-order gain. The candidate was then rescaled to the exact norm of the current mixed PPO direction.

No optimizer step was taken.

## Result

All predeclared audit criteria pass.

Across 8 checkpoint x seed cases:
- mixed worst normalized gain mean = **-0.2896**
- floor worst normalized gain mean = **+0.2791**
- mean worst-gain improvement = **+0.5687**
- mixed mean gain = **0.3467**
- floor mean gain = **0.3379**

Thus the floor direction strongly improves the worst local objective while preserving nearly the same mean normalized gain.

Conflict structure:
- 7/8 mixed-direction cases have at least one negative objective gain
- floor direction reduces negative-axis count in **7/7** conflict cases
- floor direction has **zero negative objective gains in 8/8 cases**

Step-budget / non-collapse:
- mean floor/mixed norm ratio = **1.00000002**
- maximum norm-ratio error is below the frozen 1e-5 tolerance
- mean cosine(floor, mixed) = **0.4343**
- minimum cosine(floor, mixed) = **0.1953**
- floor direction is finite and nonzero in every case
- PPO ratio invariant passes

Both u50 and u75 separately show higher aggregate worst normalized gain under the floor direction than under the mixed direction.

## Interpretation

The current weighted PPO direction frequently accepts a locally negative first-order change for one objective in exchange for average progress.

A norm-matched max-min combination can repair that local geometry without shrinking the update to near zero and without reversing the whole mixed direction.

This establishes an optimization-level mechanism worth testing in training.

It does **not** establish semantic retention or semantic accumulation.

## Decision

**DIAGNOSTIC PASS.**

A single bounded paired short training pilot is authorized from the same u50 checkpoint:
- control = current weighted PPO
- treatment = norm-matched max-min floor direction
- fresh Adam state in both arms
- same rollout/support/reset schedule
- 25 updates corresponding to global u51..u75

Primary artifacts:
- `docs/contracts/preference_architectures/competence-floor-maxmin-contract.md`
- `scripts/rl/v2b_competence_floor_maxmin_audit.py`
- `runs/v2b_competence_floor_maxmin_audit-2026-09-24/competence_floor_maxmin_report.json`
