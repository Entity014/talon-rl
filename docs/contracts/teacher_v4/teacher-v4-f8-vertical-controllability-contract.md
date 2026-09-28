# Teacher V4 — F8 Vertical-Stability Realization and Controllability Screen

Status: **FROZEN 2026-09-28 (r2) in the commit that sets this line, before any F8 training run.** Pipeline smoke-tested only (seed 1, 512 envs, 50 iterations, not F8 data).
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

- Training support: cardinalities **{2, 3, 4}**, **task-anchored: T is in
  every sampled set** (`--require-objective T`): {T,R}, {T,O}, {T,V}; {T,R,O},
  {T,R,V}, {T,O,V}; {T,R,O,V}. The unrestricted sampler also draws T-free
  pairs and triples, which make standing optimal by construction, exactly
  like T-free singletons. F8 asks whether V can change vertical dynamics
  while the locomotion objective is active, so it trains only on that
  support. This is an **F8 training support**, not a rule of the final MORL
  architecture. Without the flag the sampler's draws are unchanged (tested).
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

- **Primary: T-anchored**, in TAOV order: T55 = (.55, .15, .15, .15),
  T55R = (.55, .25, .15, .05), T55V = (.55, .05, .15, .25). w_T = .55 and
  w_O = .15 are fixed, and only R ↔ V moves (.15/.15 → .25/.05 or .05/.25).
  The treatment is weaker than a heavy preference, but its reading is
  clean: no orientation relaxation is mixed in. Rationale: in every V4 run so far, C rarely translates (F2–F4).
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
| V1 passes, V3 fails | carry V1 to F9 |
| V3 passes, V1 fails | carry V3 to F9 |
| both pass | **carry both to F9**; F8 cannot resolve the realization. No averaged-vertical tie-break: averaging RMS v_z and excursion would be a hidden α = 0.5 |
| neither passes, and < 2 seeds per arm task-viable at base | task viability unresolved; case 3 is **not** rejected |
| neither passes, base locomotes | no evidence of a distinct controllable V under this formulation. Case 3 is not supported *at this stage*. That is not a falsification: the failure could come from training or representation |

**Leakage.** F8 policies are selection data for V. Validation of V\* (R / V
independence, joint viability) uses new seeds and policies only (F9).
Locomoting F8 policies may extend the behavior bank for later Layer-4
selection, never the F9 validation.

Code: `--objectives TAOV --v-objective V1|V3` in `train_v4c.py`,
`--conds f8 --v-objective` in `f2a_bifurcation.py`, aggregate
`f8_screen.py`. Runs: `runs/teacher_v4_f8-2026-09-28/<arm>_seed<seed>/`.
