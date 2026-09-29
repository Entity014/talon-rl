# Teacher V4 — FC-0 Rotational-Stability Realization Audit (read-only)

Status: **FROZEN 2026-09-29 (r2), together with `fc0_audit.py`, before any FC-0 trace was generated.** Trace and analysis code smoke-tested on a scratch run only.
Branch: `v4-c2-semantic-preservation`
Follows: [FB-2a verdict](../../verdicts/teacher_v4/teacher-v4-fb2a-surrogate-calibration-verdict.md). The task mechanism works well enough for this screen; O works as a preference inside locomotion; R (`ang_vel_xy_l2`) does not. FB-2a does not show the task mechanism is complete: the dual did not settle, and yaw is unconstrained.

## Question

Inside feasible locomotion, what is the physical signature of the rotation
we want R to reduce? Does `ang_vel_xy_l2` measure it, or does it penalize
motion the gait needs, or reward a posture bias?

F2-A and F8 lacked locomoting policies for this question. FB-2a has them.

## Behavior bank (behaviors with replay tl ≥ 0.40 only; no standing)

- **Primary (decides the class):** FB-2a seeds 79101–79103, checkpoint 600.
  Conditions: R vertex, R⁺, C, O⁺, O vertex (up to 15 behaviors). It is the
  dataset that matches the question: calibrated task mechanism,
  locomoting, the same R/O preference space.
- **Secondary (robustness, descriptive only; never flips a primary
  class):** FB-2 seeds 78101–78103 at 600; F4 B0 s74102 / s74103 at 600; V4-C
  G1-2 s73102 at 300. These come from other formulations, so pooling them
  would add distribution shift.
- A behavior is included only if its tl ≥ 0.40 in this protocol.

Protocol: the F2-A replay (same reset suite, deterministic), with
`--traces --only-checkpoint`. It saves per step, for the first 64 envs per
reset seed: roll, pitch (wrapped), ω_x, ω_y (base), v_x, v_y (base),
commanded v_x, v_y, weighted track_lin, and foot contacts, plus the
survival mask.

## Features (per 32-step window of steps 33–128, surviving envs)

| id | feature | meaning |
|---|---|---|
| F_O | mean ‖θ_xy‖ (roll, pitch) | absolute posture (O-like reference, not an R candidate) |
| **F_rate** | mean ‖ω_xy‖² | **current R** (`ang_vel_xy_l2`, unweighted) |
| **F_osc** | RMS ‖θ_xy − EMA₀.₅ₛ(θ_xy)‖, EMA started at step 1 | candidate: oscillation about the body's own posture |
| **F_acc** | mean ‖Δω_xy / dt‖² | candidate (secondary): angular acceleration; may be smoothness rather than stability |
| F_p2p | peak-to-peak of linearly detrended roll plus pitch | descriptive excursion |

Covariates: **command magnitude** v_cmd = mean ‖(v_x,cmd, v_y,cmd)‖ (the task
demand); speed = mean ‖v_xy‖ (base, a policy output); window tl; cadence =
touchdowns per second (descriptive / mechanism only: cadence is part of the
gait, not a task requirement).

Phase-conditioned candidates (removing expected gait rotation by phase) are
out of scope: too complex, and tied to the gaits seen so far.

## Analyses and staged classification (primary bank; the secondary is reported the same way)

Behavior ID = policy × condition (e.g. 79101/R⁺). η² is computed within 4
equal-count bins of a conditioning variable, over groups with ≥ 5 windows,
with at least 2 such groups per bin, and averaged over the bins.

Screening thresholds (pre-declared, not universal statistical cutoffs):
η² ≥ 0.10 is minimal informativeness; median |ρ| ≥ 0.5 with v_cmd or speed
is the task-association flag; median |ρ| ≥ 0.7 with F_O is the
O-association flag. Correlation alone never rejects.

**Stage 1 — informativeness.** η²(behavior) within window-tl bins
(matched task) ≥ 0.10? No → **unresolved**.

**Stage 2 — task confounding.** Task-associated (flag)? If so, remove a
linear fit on [v_cmd, speed] and recompute the matched-task η²(behavior).
If it is < 0.10 → **reject (task-confounded)**. Otherwise continue with
the flag.

**Stage 3 — O independence.** O-associated (flag)? If so, compute
η²(behavior) within F_O bins (posture-matched). If it is < 0.10 →
**reject (O-redundant)**. Otherwise continue with the flag.

Passing all stages → **candidate** (plus any flags).

Also reported:

- η²(policy) within tl bins: a policy-dependence diagnostic, not
  informativeness;
- the per-policy Spearman ρ with v_cmd, speed, cadence and F_O;
- **posture-bias mechanism (FB-2a):** R⁺ vs C relative change of each
  feature and of F_O. The flag "improves while posture bias worsens" never
  removes a candidate on its own: the association does not show that
  posture bias is the cause.

F_rate (the current R) goes through the same stages. This is the mechanism
audit of FB-2a's R failure.

## Leakage

FC-0 may propose an R candidate. It never declares R_v2 from FB-2a data
and validates on the same data. A candidate goes to FC-1: new training,
fresh seeds, under the FB-2a task mechanism.

Scripts: `f2a_bifurcation.py --traces --only-checkpoint` (collection),
`fc0_audit.py` (analysis, written before the traces are opened). Output:
`runs/teacher_v4_fc-2026-09-29/fc0/`.
