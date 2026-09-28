# Teacher V4 — F6 Conditional Raw-Feature Screen (gentle vs D-costly locomotion)

Status: **FROZEN 2026-09-28 (r2) in the commit that sets this line, before any D-costly (F4) per-step trace was generated.**
Branch: `v4-c2-semantic-preservation`
Method: [objective selection](../../methods/general/objective-selection.md), layers 2–3 (conditional selection)

## Question

Among locomoting behaviors already stratified by the current A-based D
definition, which raw terms provide additional structure beyond A?

The L / H labels use R_D, which **is** A (`ang_vel_xy_l2`). So A is tied to
the label by construction, and F6 cannot find an *alternative* to A. That
would need a phenotype defined independently of A, or a new round with
several gentle policies and an outcome criterion that does not use the
candidate feature. Allowed outputs: baseline A, candidate complement to A,
redundant with A, nonspecific, unresolved.
**F6 never outputs "D_v2 selected".** The gentle side comes from one policy
(s73102). A D_v2 needs gentle locomotion from more than one independent
policy, then a fresh realization round validated on new seeds.

Guiding rule: **window-level evidence proposes a feature; behavior/policy-
level consistency decides whether it survives as a candidate.**

## Behavior bank (frozen)

| behavior | source | condition | required phenotype | independence |
|---|---|---|---|---|
| L_C | V4-C G1-2 s73102 model_300 | C | L | same policy as L_T+ |
| L_T+ | V4-C G1-2 s73102 model_300 | T⁺ | L | same policy as L_C |
| H_74102 | F4 B0 s74102 model_600 | T⁺ | H_D | own policy |
| H_74103 | F4 B0 s74103 model_600 | T⁺ | H_D | own policy |
| ref_M0 | M0, branches in the s73102 run | — | any | reference |
| ref_standing | V4-C G1-2 s73101 model_300 | C | standing | reference |
| ref_step_74102 / 74103 | F4 s74102 / s74103 model_600 | C | step-in-place | reference |

The primary analysis is L vs H only. References answer whether a feature
separates gentle vs costly locomotion, or only motion vs no motion.

**Inclusion:** a behavior enters only if its phenotype, recomputed in this
protocol, matches the required one:

- L: tl ≥ 0.40, R_D ≥ −0.23, R_O ≥ −0.48;
- H_D (D-costly): tl ≥ 0.40, R_D < −0.23, R_O ≥ −0.48;
- other-costly: tl ≥ 0.40, R_O < −0.48, excluded from the primary screen;
- standing: tl < 0.40 and td < 0.02; step-in-place: tl < 0.40 and td ≥ 0.02. Failures are excluded and reported.
If no L or no H behavior remains, F6 is unresolved.

## Trace protocol (frozen before any costly trace)

`substrate_attribution.py` (K from the checkpoint), unchanged except for a
per-env survival mask added to the trace file. With the mask, the s73102
run reproduced its earlier JSON and traces bit for bit.

- Branches C, T⁺, A⁺, O⁺, M0 from 512 center-warm-up snapshots (env seeds
  0, 1 × 256 envs). Discarded burn-in, cyclic position balance, 128-step
  deterministic branches.
- Traces: the first 64 envs of each env seed, all 128 steps. The steady
  window is steps 33–128. Envs that terminated in any branch are dropped
  (mask).
- Raw terms (weighted, as in the reward manager, higher-is-better):
  track_lin_vel_xy_exp, track_ang_vel_z_exp, lin_vel_z_l2, ang_vel_xy_l2 (A),
  dof_torques_l2, dof_acc_l2, action_rate_l2, feet_air_time,
  flat_orientation_l2. `feet_air_time` is only non-zero at touchdowns, so it
  is read through window means. `dof_pos_limits` is in the library but has
  weight 0 (a constant-zero term), so it cannot separate anything and is not
  screened: 9 of the 10 library terms.

Runs, into `runs/teacher_v4_f6-2026-09-28/traces/`: v4c_g1_2_s73102,
v4c_g1_2_s73101, f4_s74102, f4_s74103.

## Resolutions

- per step: descriptive distributions and Spearman ρ only;
- 32-step non-overlapping windows (3 per env): the within-behavior local
  statistic;
- per-behavior means: the mandatory cross-behavior check.

The 64 envs estimate a behavior's distribution. They never make the gentle
side 64 independent policies. No p-value is computed from step or window
counts.

## Metrics per raw term j

1. Variance within locomotion, per behavior.
2. S_LH: standardized mean difference of window means, L windows vs H
   windows (pooled SD).
3. Behavior ordering: consistent if every L behavior mean lies on one side
   of every H behavior mean.
4. Spearman ρ(r_j, r_A) per behavior and pooled. Pooled alone never decides
   (Simpson's paradox). Policy-level summary: s73102 = max(|ρ_L_C|, |ρ_L_T+|),
   and |ρ| for each H policy. It is reported, not gated.
5. A-conditioned residual separation. Restrict window means to the common A
   support of L and H. Split into 4 equal-count bins, [left, right) except
   the last, which is closed. In each bin with ≥ 10 windows per class,
   Δ_j = mean_L − mean_H. The pooled residual survives if ≥ 2 bins are
   evaluable and at least max(2, ⌈0.75 × evaluable⌉) have the sign of the
   overall L − H difference. With fewer than 2 evaluable bins the result is
   "insufficient common support". No extrapolation. Coverage c_L, c_H (the
   share of each class's windows inside the common support) is reported
   descriptively.
6. Per-H-policy residual: the same procedure for L vs H_74102 and L vs
   H_74103 separately. Direction-compatible if the mean of its evaluable bin
   deltas has the overall sign. If it has fewer than 2 evaluable bins, it is
   reported as unevaluable.
7. S_motion: standardized difference, locomotion windows (L ∪ H) vs
   ref_standing.

## Pre-declared classes (no ranking)

Pre-declared **screening** thresholds, not universal statistical cutoffs:
|S| ≥ 0.8 is strong screening separation, |S| < 0.5 weak screening
separation.

| condition (applied in order) | class |
|---|---|
| term is A | baseline (A) |
| \|S_LH\| < 0.5 | nonspecific (motion feature if \|S_motion\| ≥ 0.8) |
| \|S_LH\| < 0.8, or behavior ordering not consistent | unresolved (weak or conflicting) |
| fewer than 2 evaluable pooled A bins | unresolved (insufficient common A support) |
| pooled residual does not survive | redundant with A |
| pooled residual survives, and every evaluable H policy is direction-compatible (at least one evaluable) | **candidate complement to A** |
| otherwise | unresolved (policy-dependent residual) |

A candidate complement means that r_j *may carry information beyond the
current A*. It never means r_j is a better replacement for A.

The tracking terms are expected to separate L and H partly, because both
translate at different speeds. They are reported, and are not D candidates.

## Leakage

F2 / F4 / M0 traces are selection data. Any D candidate that comes out of
F6 is validated only on new policies and seeds. F5's paused partial run is
not used.

Code: `f6_features.py` (offline). Output: `runs/teacher_v4_f6-2026-09-28/f6_features.json`.
