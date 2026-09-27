# Teacher V4 — V4-C3 Three-Objective Training Contract

Status: **PREDECLARED — FROZEN 2026-09-28 before any V4-C3 training. The evaluation contract is frozen separately, before any evaluation.**
Branch: `v4-c2-semantic-preservation`

## Decision

The old four-objective formulation {T, A, O, S} is frozen as is (V4-C,
V4-C2*). The new objective set drops S as an independent objective:

    T — Tracking            track_lin_vel_xy_exp + track_ang_vel_z_exp
    A — Angular Stability   ang_vel_xy_l2         (suppress body roll/pitch rate)
    O — Orientation         flat_orientation_l2   (stay upright)

Basis: V4-C2F and V4-C2S-R1. S (action rate, and action jerk even when
trained on it) is subsumed by A (ρ 0.79–0.92, PC1 0.94–0.99, SS − SA not
positive). A and O are behaviorally distinct (step ρ 0.16, off-diagonal
0.46). S is removed, not blended into A: there is no A+S composite and no
new coefficient. Smoothness is not claimed as an independent objective on
this task and substrate.

## Unchanged

Env `Isaac-Talon-A1-V4C-v0`. The T/A/O kernels, weights and T3-B divisors
(not re-tuned). TeacherV4 architecture (objective vocabulary of size 3). The
objective-set sampler (V3 modes, zero-weight padding). Objective-set PPO in
the M0 shell. 4096 × 24, 5 × 4, 300 iterations = 29.49M samples. Checkpoint
iteration 300. Stop gates: non-finite, log-std outside [−5, 2], and the new
**dead-critic gate** (5 consecutive iterations with critic feature std
< 1e-5). A stopped run counts as a failure and is not replaced. No
specificity loss, no critic change, and no architecture change in this
round.

## Folds and seeds

    G1-1   train cardinalities {2, 3}   held out {1}
    G1-2   train cardinalities {1, 3}   held out {2}
    seeds  73101, 73102, 73103          (6 runs)

m = 3 {T, A, O} is in training in both folds and is the canonical anchor,
as m = 4 was before.

`train_v4c.py --objectives TAO --cardinalities {2,3 | 1,3}` (the code path
is covered by `test_three_objective_set_runs_end_to_end`).
Runs: `runs/teacher_v4_c3_g1_{1,2}_seed{73101,73102,73103}-2026-09-28/`.
