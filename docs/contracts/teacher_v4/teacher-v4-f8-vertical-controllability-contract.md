# Teacher V4 — F8 Vertical-Stability Realization and Controllability Screen

Status: **DRAFT 2026-09-28. Not frozen. No F8 training run.** Pipeline smoke-tested only (seed 1, 512 envs, 50 iterations, not F8 data).
Branch: `v4-c2-semantic-preservation`
Follows: [F7 stage 2 verdict](../../verdicts/teacher_v4/teacher-v4-f7-stage2-vertical-realization-verdict.md)

## Question

Which vertical signal, **once it has preference authority**, gives a
vertical-stability axis that is controllable, distinct from R, and
task-viable? F8 is a realization-selection plus controllability screen,
not final validation. V1 and V3 enter as **equal candidates**; neither is
the default.

## Arms (paired, everything else identical)

| arm | T | R | V | O |
|---|---|---|---|---|
| F8-V1 | tracking | ang_vel_xy_l2 | lin_vel_z_l2 (stock, weight −2.0) | flat_orientation_l2 |
| F8-V3 | tracking | ang_vel_xy_l2 | body_height_osc_l2 (weight −1.0; env `Isaac-Talon-A1-V4C-V3-v0`) | flat_orientation_l2 |

Objective order is TAOV (A = R). Divisors are T3-B abs-means under M0
model_299, measured on the V3 env in the same run that reproduced
T/A/O/S exactly: **V1 0.11904645, V3 0.00244346**
(`objectives.V_REALIZATIONS`). TeacherV4 K = 4, the V4-C PPO shell, all stop
gates, the post-fix normalizer, no R_shared.

- Training support: cardinalities **{2, 3, 4}**. m = 4 (the anchor) is
  included. Singletons are excluded because T-free singletons ({R}, {V},
  {O}) make standing optimal by construction.
- **600 iterations** from scratch, checkpoints every 50 (F4: 300 censors
  locomotion discovery).
- Seeds: **76101, 76102, 76103** for both arms (6 runs).

## Measurement

Replay of every checkpoint with the F2-A protocol (`f2a_bifurcation.py
--mode replay --conds f8`): the same reset suite, deterministic, plus
vertical metrics. Physical response vector per condition:
y = [tl, ‖ω_xy‖, RMS v_z (world), vertical excursion, tilt (guardrail)].
**No objective return is compared across arms** (V1 and V3 have different
units).

Condition sets (heavy = `heavy_w(4, i)`: 0.70 / 0.10 / 0.10 / 0.10):

- **Primary: T-anchored.** T55 = (T .55, R .15, O .15, V .15), T55R = (.55,
  .35, .05, .05), T55V = (.55, .05, .05, .35). w_T is equal across the
  three, so the tracking pressure is held constant and only the R ↔ V lean
  changes. Rationale: in every V4 run so far, C rarely translates (F2–F4).
  A C-based comparison would mostly compare standing behaviors.
- **Secondary:** C, R⁺ (= A⁺), V⁺. T⁺ and O⁺ are replayed for context.

## Per-seed criteria (at checkpoints 550 **and** 600)

With base / R-lean / V-lean = T55 / T55R / T55V (primary), or C / A⁺ / V⁺
(secondary):

1. **Task viable:** tl(base) ≥ 0.40.
2. **V controllable:** RMS v_z and excursion each ≥ 10 % lower under the
   V-lean than under base, **and** tl(V-lean) ≥ 0.40 (not by stopping).
3. **R controllable:** ‖ω_xy‖ ≥ 10 % lower under the R-lean than under base,
   and tl(R-lean) ≥ 0.40.
4. **Double dissociation:** V-lean has lower RMS v_z and excursion than
   R-lean, **and** R-lean has lower ‖ω_xy‖ than V-lean.

A seed passes if all four hold at both checkpoints. An arm passes if at
least 2/3 seeds pass.

## Pre-declared reading (primary set; the secondary set is reported the same way)

| outcome | reading |
|---|---|
| exactly one arm passes | **V\* = that arm**, frozen for F9 |
| both pass | V\* = the arm with the larger median relative vertical improvement (V-lean vs base); a difference < 0.05 → both carried forward |
| neither passes, and < 2 seeds per arm are task-viable | case 3 not supported *at this stage*: task not viable at base |
| neither passes, otherwise | case 3 not supported at this stage: V not controllable or not distinct from R. Cases 1 and 2 are reconsidered |

**Leakage.** F8 policies are selection data for V. Validation of V\* (R / V
independence, joint viability) uses new seeds and policies only (F9).
Locomoting F8 policies may extend the behavior bank for later Layer-4
selection, never the F9 validation.

Code: `--objectives TAOV --v-objective V1|V3` in `train_v4c.py`,
`--conds f8 --v-objective` in `f2a_bifurcation.py`, aggregate
`f8_screen.py`. Runs: `runs/teacher_v4_f8-2026-09-28/<arm>_seed<seed>/`.
