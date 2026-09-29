# Teacher V4 — FB-2a Online-Surrogate Calibration

Status: **FROZEN 2026-09-29 in the commit that adds this file, before any FB-2a training.**
Branch: `v4-c2-semantic-preservation`
Follows: [FB-2 verdict](../../verdicts/teacher_v4/teacher-v4-fb2-lagrangian-task-verdict.md). FB-2's online tl ran 0.05–0.13 above the replay tl, so λ fell to 0 while the replay sat at the boundary.

## The one change

| | FB-2 | FB-2a |
|---|---|---|
| online dual target T_min^online | 0.40 | **0.52** |
| everything else | — | identical |

**0.52 is a calibration target for the stochastic online surrogate. The
physical / evaluation feasibility requirement stays replay tl ≥ 0.40.** The
meaning of "viable" does not change.

Rule for δ, fixed before training: T_min^online = 0.40 + δ, where δ is the
80th percentile of the online − replay tl gap over the 15 FB-2
(seed, region) pairs. Computed: median 0.078, 80th percentile **0.125**
(linear interpolation), 90th 0.134, max 0.143. δ = 0.12 (two decimals, as
decided before training) covers 11/15 of the observed gaps. It was not
chosen by trying targets.

Unchanged: five w_R regions; λ₀ = 0.786; η = 0.15; cap 20 (numerical
bound); λ floor 0 (it can still fall to 0); T_lin task stream (yaw in no
stream); R/O realizations; T3-B divisors; PPO shell; 600 iterations;
cardinalities {1, 2}; the FB-1/FB-2 gate (five-point replay viability, R⁺ /
O⁺ authority at 550 and 600, joint ≥ 2/3); the FB-2 reading table and the
dual-vs-replay diagnostic. Fresh seeds **79101, 79102, 79103**.

No λ floor, no replay-driven dual update, no yaw constraint: one confound
at a time.

## Reading (as FB-2, plus the R diagnosis this round is meant to separate)

- viability recovered, and R⁺ still does not lower ‖ω_xy‖: the task
  controller is calibrated, but `ang_vel_xy_l2` does not produce a
  meaningful rotational-stability preference inside feasible locomotion.
  Back to the R realization.
- joint ≥ 2/3: the constrained formulation is supported. R semantics were
  masked by the task conflict.
- viability still < 2/3: inspect the R-vertex λ and tracking trend. The
  dual mechanism or R compatibility stays unresolved. "R incompatible" is
  not claimed from 600 iterations alone.

Command: `--lagrange-tmin 0.52` (other flags as FB-2). Runs:
`runs/teacher_v4_fb-2026-09-29/fb2a/seed<seed>/`. Aggregate:
`fb1_screen.py --root …/fb2a --seeds 79101,79102,79103`. The screen's
viability threshold is the replay 0.40, independent of the training target.
