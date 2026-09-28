# Teacher V4 — F1 Locomotion Viability of the Frozen T/D/O Formulation

Status: **FROZEN 2026-09-28 in the commit that adds this file and `f1_viability.py`, before any F1 execution.** Read-only and offline. It trains nothing, re-weights nothing and changes no divisor.
Branch: `v4-c2-semantic-preservation`
Follows: [substrate attribution verdict](../../verdicts/teacher_v4/teacher-v4-c3-substrate-attribution-verdict.md) (V4-C3 6/6 and V4-C 15/16 centers stand)

## Question

Does the frozen raw-to-objective mapping (T, D = A, O with the T3-B divisors)
give useful locomotion a viable preference region, rather than trivial
low-motion behavior?

This is a validation of the objective abstraction, not a diagnostic of the
V4 trainer.

## Behavior bank

A candidate is a behavior, (policy/run, preference condition), not a
policy: one V4 policy gives several conditions, and M0 appears once per run
(same policy, different snapshot sets). Every (run, condition) steady-window level from the substrate-attribution
outputs, with no new rollout:

- `runs/teacher_v4_c3_substrate-2026-09-28/*/substrate_attribution.json`
  (6 runs × C, T⁺, A⁺, O⁺, M0);
- `runs/teacher_v4_c_substrate-2026-09-28/*/substrate_attribution.json`
  (16 runs × C, T⁺, A⁺, O⁺, S⁺, M0).

That is 126 candidate behaviors. All share one env (Isaac-Talon-A1-V4C-v0), the same
512 snapshots per run, and the same 96-step steady window.

## Behavioral classes (behavioral observables only)

Class assignment uses only the touchdown-step fraction (td) and the weighted
`track_lin_vel_xy_exp` (tl). It never uses an objective score.

| class | rule |
|---|---|
| standing | td < 0.02 and tl < 0.40 (the frozen standing criterion) |
| step-in-place | td ≥ 0.02 and tl < 0.40 |
| partial locomotion | 0.40 ≤ tl < 1.00 |
| established locomotion | tl ≥ 1.00 |

Non-locomoting = standing ∪ step-in-place. Locomoting = partial ∪
established.

Disclosure: the tl ≥ 1.00 boundary was set after the tracking values were
seen (non-M0 max 0.63, M0 min 1.30). Any boundary in (0.63, 1.30) gives the
same classes.

## Objective vector and score

R(π) = [T, D, O] = [(track_lin + track_ang) / 1.71946, ang_vel_xy_l2 /
0.15591, flat_orientation_l2 / 0.01563], using the weighted steady-window
levels and the frozen `NORMALIZATION_DIVISORS`. J(π, w) = wᵀR(π).

For each w, the observed behavioral envelopes are J_L*(w) = max over
locomoting behaviors of wᵀR, and J_N*(w) = the same over non-locomoting
behaviors. These are the best **observed** candidates, not estimates of a
population optimum.
Locomotion **wins** at w if J_L*(w) − J_N*(w) > m, with **m = 0.05**. It
**loses** if J_L*(w) − J_N*(w) < −m. Otherwise the point is a tie.

m is a practical margin in normalized-score units, not a confidence bound:
F1 is a deterministic geometry audit on mean behavior vectors. Primary m =
0.05. Sensitivity (descriptive, frozen now): m ∈ {0, 0.025, 0.05, 0.10}.

## Preference support

Named points, (T, D, O):

| cardinality | points |
|---|---|
| m = 3 | C = (1/3, 1/3, 1/3); T⁺ = (.70, .15, .15); D⁺ = (.15, .70, .15); O⁺ = (.15, .15, .70) |
| m = 1 | {T} = (1, 0, 0); {D} = (0, 1, 0); {O} = (0, 0, 1) |
| m = 2 | TD (.70, .30, 0), (.30, .70, 0); TO (.70, 0, .30), (.30, 0, .70); DO (0, .70, .30), (0, .30, .70) |

Distributions (100 000 draws each, `sample_objective_sets`, K = 3, seed 0),
reported separately:

- m = 1, m = 2, m = 3 alone;
- the G1-1 mixture (2, 3) and the G1-2 mixture (1, 3);
- the uniform simplex (grid step 0.01, 5151 points).

Each m-specific distribution is also split into T-containing and T-free
sets. With w_T = 0, no objective rewards locomotion, so a low-motion winner
there follows from the formulation by construction. T-free points are
reported and never gate.

## Pre-declared reading (primary)

| outcome | reading |
|---|---|
| locomotion wins at all of C, T⁺, D⁺, O⁺ **and** does not lose at any T-containing m = 1 / m = 2 named point ({T}, TD, TO endpoints) | **Full support.** The standing collapse is an optimization or gait-discovery failure (gait-shaping contract next). |
| locomotion loses at C **and** wins on < 50 % of both training mixtures | **Not supported over the intended support.** Standing is the objective's own answer, not only an optimizer failure. |
| otherwise | **Region-limited support.** Report the region per cardinality. |

No threshold is set on simplex or training fractions beyond the < 50 %
above. The fractions characterize the region quantitatively.

**Limit.** The bank contains only behaviors these policies produced. A
"does not support" reading means no existing locomoting behavior beats the
existing non-locomoting ones. A gentler, unobserved gait could still win.
For the best locomoting behavior, F1 reports the D and O cost reduction it
would need to win at C.

## Secondary (descriptive)

- The 4-class winner map b*(w) = argmax over classes of the best candidate
  in each class, as region shares of the uniform simplex and of each
  training distribution.
- At the named points, the per-class best score. Is J monotone along
  standing → step-in-place → partial → established?
- The class count per source (V4-C3, V4-C, M0).
- The primary reading and all region fractions under each sensitivity margin.

Script: `f1_viability.py`. Output: `runs/teacher_v4_f1-2026-09-28/f1.json`.
