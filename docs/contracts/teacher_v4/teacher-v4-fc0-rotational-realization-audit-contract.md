# Teacher V4 — FC-0 Rotational-Stability Realization Audit (read-only)

Status: **DRAFT 2026-09-29. Not frozen. No FC-0 trace generated.** Trace code smoke-tested on a scratch run only.
Branch: `v4-c2-semantic-preservation`
Follows: [FB-2a verdict](../../verdicts/teacher_v4/teacher-v4-fb2a-surrogate-calibration-verdict.md). The task mechanism works well enough for this screen; O works as a preference inside locomotion; R (`ang_vel_xy_l2`) does not. FB-2a does not show the task mechanism is complete: the dual did not settle, and yaw is unconstrained.

## Question

Inside feasible locomotion, what is the physical signature of the rotation
we want R to reduce? Does `ang_vel_xy_l2` measure it, or does it penalize
motion the gait needs, or reward a posture bias?

F2-A and F8 lacked locomoting policies for this question. FB-2a has them.

## Behavior bank (behaviors with replay tl ≥ 0.40 only; no standing)

- **Primary:** FB-2a seeds 79101–79103, checkpoint 600. Conditions: R
  vertex, R⁺, C, O⁺, O vertex (up to 15 behaviors).
- **Secondary:** FB-2 seeds 78101–78103 at 600 (same conditions); F4 B0
  s74102 / s74103 T⁺ at 600; V4-C G1-2 s73102 C and T⁺ at 300. Each is
  included only if its tl ≥ 0.40 in this protocol.

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

Covariates: speed = mean ‖v_xy‖ (base); cadence = touchdowns per second;
window tl.

Phase-conditioned candidates (removing expected gait rotation by phase) are
out of scope: too complex, and tied to the gaits seen so far.

## Analyses

Policy = training run. The two V4-C behaviors are one policy.

1. **Task coupling:** per policy, Spearman of the feature with speed and
   with cadence over its windows. Median |ρ| across policies.
2. **O redundancy:** per policy, Spearman of the feature with F_O. Median
   |ρ|.
3. **Posture-bias exploitability:** in FB-2a, the relative change R⁺ vs C
   of the feature and of F_O. Does the feature improve when posture bias
   grows?
4. **Matched-tracking discrimination:** pool the windows, split them into 4
   equal-count bins of window tl, and within each bin compute η² of the
   feature by policy. Mean over bins. The same with FB-2a condition
   identity. It asks whether the feature separates behaviors at comparable
   task performance, and is not just "prefers slower walking".

## Pre-declared classification (per feature; F_rate assessed the same way)

- **task-coupled** if median |ρ_speed| ≥ 0.5 or median |ρ_cadence| ≥ 0.5;
- **O-redundant** if median |ρ_O| ≥ 0.7;
- **discriminative** if mean matched-bin η²(policy) ≥ 0.10.

| condition | class |
|---|---|
| not task-coupled, not O-redundant, discriminative | **candidate** R realization |
| task-coupled or O-redundant | **reject** (penalizes gait motion, or duplicates O) |
| otherwise (not discriminative) | **unresolved** |

Reported for F_rate: whether the current R is task-coupled, O-redundant,
or exploitable by posture bias. This is the mechanism audit of FB-2a's R
failure.

## Leakage

FC-0 may propose an R candidate. It never declares R_v2 from FB-2a data
and validates on the same data. A candidate goes to FC-1: new training,
fresh seeds, under the FB-2a task mechanism.

Scripts: `f2a_bifurcation.py --traces --only-checkpoint` (collection),
`fc0_audit.py` (analysis, written before the traces are opened). Output:
`runs/teacher_v4_fc-2026-09-29/fc0/`.
