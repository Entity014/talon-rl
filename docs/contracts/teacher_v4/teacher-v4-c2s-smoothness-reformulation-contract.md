# Teacher V4 — V4-C2S Smoothness Reformulation Audit Contract

Status: **PREDECLARED — FROZEN 2026-09-27 before data collection (descriptive audit, no training)**
Branch: `v4-c2-semantic-preservation`

## Question

V4-C2F found the current S (action rate, ‖a_t − a_{t−1}‖) subsumed by A
(ρ 0.84–0.92, PC1 0.96–0.99). The concept to keep is "actuation does not
jerk", not "actuation barely changes". Which smoothness formulation keeps a
clear physical, hardware-relevant meaning and is **not** nearly the same
objective as A (or as T or O)?

Decision (user): keep four objectives and redefine S. Do not merge A and S.

## Candidates (per step, higher is better)

    S0  = −‖a_t − a_{t−1}‖                          current: action rate
    S1  = −‖a_t − 2a_{t−1} + a_{t−2}‖               action second difference (action jerk)
    S2  = −‖q̇_t − 2q̇_{t−1} + q̇_{t−2}‖               joint-velocity second difference (joint jerk)
    S3  = −‖τ_t − τ_{t−1}‖                           applied-torque rate

a is the policy action, q̇ the joint velocities, and τ the applied joint
torques (`robot.data.applied_torque`), all at the 0.02 s policy step.

## Data

The 10 primary V4-C `model_300.pt` checkpoints. The same snapshot and switch
protocol as V4-C2F: 512 snapshots per checkpoint, a discarded burn-in, then
five 128-step branches (center and heavy T/A/O/S) with cyclic position
balance. Steady window 33–128. Obs corruption off. Episodes ending in any
branch are excluded. Per step, S_T, S_A, S_O and S0–S3 are recorded.

Limitation stated in advance: these policies were trained with S0 in the
reward. The audit measures how the candidates co-vary with A on the
behavior these policies produce, which is not necessarily the behavior a
policy trained on the candidate would produce.

## Measures (per candidate, against A, O and T; medians over checkpoints)

Layer 1: Spearman ρ, per step and on 32-step window means, and the
median-split off-diagonal mass. Layer 2: PC1 share of the standardized
steady-window snapshot cloud, and of the per-(checkpoint × preference)
policy means. Layer 3 is not used: these policies were never trained on the
candidates.

## Selection rule (fixed now)

1. Eligible: the candidate's |ρ| (per step and per window) with each of T and
   O is not higher than the A–O values from V4-C2F (0.16 per step, 0.23 per
   window). The new S must not simply become a copy of T or O.
2. Among eligible candidates, rank by redundancy with A on four measures:
   |ρ| per step, |ρ| per window, off-diagonal mass (higher is better), and
   PC1 on snapshots. The candidate with the best mean rank is selected.
3. The selected candidate must beat S0 on all four A-measures. Otherwise no
   candidate is selected and that is reported.
4. Ties: prefer, in order, S3 (hardware-relevant torque), S1, S2.

The selected S′ is then frozen into the objective set before any V4-C2P
training contract is written. Its T3-B-style divisor is measured on the M0
policy at that point, as for the other objectives.
