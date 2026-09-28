# Teacher V4 — F6 Conditional Raw-Feature Screen Verdict

Status: **FROZEN. No candidate complement to A. lin_vel_z and action_rate separate L from H very strongly (|S| 3.7–4.0), and their pooled residual beyond A survives, but both are "unresolved (policy-dependent residual)". L and H barely overlap in A (9 % of H windows lie in the common support), so no single costly policy has two evaluable A bins. feet_air_time is redundant with A. Descriptive: the low-D "gentle" gait is a vertically bouncing gait with flight phases (|v_z| ~20× the D-costly gaits). A does not measure that.**
Date: 2026-09-28
Branch: `v4-c2-semantic-preservation`
Contract: [teacher-v4-f6-conditional-feature-contract.md](../../contracts/teacher_v4/teacher-v4-f6-conditional-feature-contract.md) (r2, frozen at `3f4cc5f`, before the costly traces)
Output: `runs/teacher_v4_f6-2026-09-28/f6_features.json`, traces in `traces/`

## Inclusion

| behavior | phenotype here | tl | R = [T, D, O] | included |
|---|---|---|---|---|
| L_C (s73102) | L | 0.57 | [0.72, −0.04, −0.03] | yes |
| L_T+ (s73102) | L | 0.63 | [0.78, −0.07, −0.07] | yes |
| H_74102 | H_D | 0.64 | [0.78, −0.36, −0.07] | yes |
| H_74103 | H_D | 0.82 | [0.88, −0.49, −0.08] | yes |
| ref_M0 | other-costly | 1.34 | [1.11, −0.46, −0.89] | reference |
| ref_standing (s73101 C) | standing | 0.30 | [0.43, −0.00, −0.04] | yes |
| ref_step_74102 / 74103 | standing in this protocol | 0.30–0.31 | — | excluded (required step-in-place) |

The primary L-vs-H analysis is complete. Only the step-in-place references
dropped out.

## Registered classes

| term | S_LH (window) | ordering consistent | pooled residual | class |
|---|---|---|---|---|
| ang_vel_xy_l2 (A) | +3.00 | yes | — | baseline (A) — circular with the labels |
| lin_vel_z_l2 | **−3.98** | yes | survives (3/3 bins) | unresolved (policy-dependent residual) |
| action_rate_l2 | **+3.67** | yes | survives | unresolved (policy-dependent residual) |
| feet_air_time | +1.16 | yes | does not survive | redundant with A |
| dof_acc_l2 | +0.51 | yes | — | unresolved (weak) |
| track_lin_vel_xy_exp | −0.31 | yes | — | nonspecific / motion (tracking, not a D candidate) |
| track_ang_vel_z_exp | −0.20 | no | — | nonspecific / motion (tracking) |
| dof_torques_l2 | +0.24 | no | — | nonspecific / motion |
| flat_orientation_l2 | +0.18 | no | — | nonspecific |

Sign convention is L − H on higher-is-better terms. S_LH is negative for
lin_vel_z: **L pays much more vertical-velocity penalty than H.**

**Why lin_vel_z and action_rate are unresolved, not candidates.** The
common A support holds 100 % of L windows but only 9 % of H windows (the two
classes are almost disjoint in A). Per costly policy, only one A bin is
evaluable, so direction compatibility cannot be assessed for either H
policy. The contract's "otherwise" branch applies. Descriptively, the single
bins agree in sign with the pooled residual for both terms and both
policies. This is not a conflict; it is missing support.

## Descriptive: what the two phenotypes are

Steady window, surviving traced envs:

| behavior | all feet in air | 4 feet down | bound-like pair sync | \|v_z\| | ‖ω_xy‖ | action_rate term |
|---|---|---|---|---|---|---|
| L_C | 1.3 % | 65 % | 9 % | 0.31 | 0.29 | −0.004 |
| L_T+ | 9.2 % | 57 % | 7 % | 0.35 | 0.42 | −0.006 |
| H_74102 | 0 % | 45 % | 20 % | 0.017 | 1.02 | −0.162 |
| H_74103 | 0 % | 23 % | 25 % | 0.013 | 1.23 | −0.192 |
| M0 | 0 % | 75 % | 2 % | 0.064 | 1.03 | −0.055 |

- **L (low-D) is a vertically bouncing gait with flight phases.** Roll/pitch
  rate is low, but vertical velocity is about 20× that of H and 5× M0's.
- **H (D-costly) is a grounded, bound-like gait** with high roll/pitch rate
  and a very high action rate. Action rate tracks A within H (policy-level
  |ρ| 0.85 / 0.60), consistent with the earlier A–S redundancy.
- The low-D phenotype that F1 and F2-A treated as objective-compatible
  owes its low D cost partly to A's blind spot. A measures roll/pitch rate
  only, and the gait moves its dynamics into the vertical axis.

## Reading

- F6 finds no term that qualifies as a complement to A under the frozen
  rules. The two strongest separators (lin_vel_z, action_rate) are blocked
  by thin common A support, not by conflicting evidence.
- The descriptive result reshapes the D question. "Low D" under the current
  A realization is achieved by a bouncing gait. Whether Dynamic Stability
  *should* penalize vertical oscillation is an objective-realization
  question for a new selection round. F6 cannot answer it, because the
  labels are built from A, and the gentle side is one policy.
- This also explains the F3 and dose-audit result: `lin_vel_z` in R_shared
  directly penalizes the only observed low-D gait.

**Limits.** One gentle policy. Two costly policies. Labels are defined by
A. Nothing here selects a D_v2. Any realization change needs fresh
policies and seeds for validation.
