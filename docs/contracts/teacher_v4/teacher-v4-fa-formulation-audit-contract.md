# Teacher V4 — FA Formulation Audit (read-only)

Status: **FROZEN 2026-09-29 before the added traces and before any computation. Descriptive, no gates, nothing trained.**
Branch: `v4-c2-semantic-preservation`
Follows: [F8 verdict](../../verdicts/teacher_v4/teacher-v4-f8-vertical-controllability-verdict.md)

## Question

Should tracking be an objective negotiated on equal terms with the
stability axes, or a task-feasibility requirement, with R / V / O
negotiated inside the locomoting manifold?

The suspected structural issue: under J(w) = w_T T + w_R R + w_V V + w_O O,
standing is a natural optimum of R, V and O. Only T asks the robot to move.
F3's R_shared was still an additive scalar and did not test a
feasibility-first formulation.

## Behavior bank

Every (run, condition) steady-window level from substrate-attribution runs
that log the measurement library:

- existing (F7 traces): V4-C G1-2 s73102 (C, T⁺, A⁺, O⁺, M0), F4 s74102 and
  s74103 model_600 (C, T⁺, A⁺, O⁺);
- added now: V4-C G1-2 s73101 model_300 (the standing reference, C, T⁺, A⁺,
  O⁺), and the six F8 model_600 runs (C, T⁺, A⁺, O⁺, V⁺).

M0 enters once (from the s73102 run). Classes use the F1 rules (td, tl).
Locomoting = tl ≥ 0.40. Non-locomoting = standing ∪ step-in-place.

## Objective components (higher is better)

T = (track_lin + track_ang), R = ang_vel_xy_l2, V1 = lin_vel_z_l2 (weighted),
V3 = −body_height_osc_l2, O = flat_orientation_l2.

Formulations: K3 = (T, R, O); K4-V1 = (T, R, V1, O); K4-V3 = (T, R, V3, O).

Scalings (the same behaviors, no retraining):

1. **T3-B** (current): divide by the frozen divisors (T 1.71946, R 0.15591,
   O 0.01563, V1 0.11905, V3 0.0024435).
2. **Stock-relative:** weighted terms without divisors. V3 has no stock
   weight; weight −1 is used and flagged as arbitrary.
3. **Bank-range:** divide each component by its (max − min) over the bank.
   Data-dependent, reported as a sensitivity only.

## Analyses

1. **Break-even w_T.** Preferences w = (w_T, (1 − w_T)/(K − 1), …), with w_T on
   a 0.01 grid. For each locomoting behavior g: the smallest w_T at which
   J(g) exceeds the best non-locomoting J. Reported per behavior,
   formulation and scaling, with the median over locomoting behaviors. Also
   reported: the share of the full simplex (0.02 grid) where the best
   locomoting behavior beats the best non-locomoting one.
2. **Normalization sensitivity:** how break-even w_T and the simplex share
   move across the three scalings.
3. **Locomotion-conditioned Pareto geometry:** restricted to locomoting
   behaviors, non-dominated sets on (R, V1, O), (R, V3, O) and (T, R, V, O);
   Spearman ρ between components across locomoting behaviors. Question: do
   R / V / O trade off inside the locomoting set?

## What it can and cannot say

Descriptive, on observed behaviors only. A high break-even w_T says the
objective geometry, as written, makes locomotion win only when T dominates.
It does not prove that a feasibility-first formulation would train better.
That needs its own contract.

Script: `fa_geometry.py`. Output: `runs/teacher_v4_fa-2026-09-29/fa_geometry.json`.
