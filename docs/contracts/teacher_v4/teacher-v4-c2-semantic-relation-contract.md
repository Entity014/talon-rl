# Teacher V4 — V4-C2 Semantic Relation Contract (definition + baseline measurement)

Status: **DRAFT — freeze before running**
Branch: `v4-c2-semantic-preservation` (V4-C is frozen at tag `v4-c-frozen`; nothing here re-scores it)
Date: 2026-09-27

## Purpose

V4-C showed that preference authority does not imply semantic correctness:
π(w⁺) ≠ π(w⁰) held everywhere, but for A the behavior moved the wrong way
in most seeds. V4-C2 makes semantic correctness a first-class property,
for every objective i, with no objective-specific code.

This contract does two things, both before any new loss is written:

1. It defines the semantic relation that V4-C2 must preserve.
2. It measures that relation on the frozen V4-C checkpoints (the baseline),
   per objective and per horizon, which also shows where the relation breaks
   (local, long-horizon, or only across training).

The choice of mechanism (pairwise margin, ranking, semantic trust region,
trajectory constraint) comes after this baseline and has its own contract.

## Semantic score (higher is better for every objective)

    S_T = −(|v_x − v_x^cmd| + |ω_z − ω_z^cmd|)     tracking
    S_A = −‖ω_xy‖                                    angular stability
    S_O = −tilt (deg)                                orientation
    S_S = −‖a_t − a_{t−1}‖                           smoothness

These are the G1 physical metrics with the sign flipped. The objective-space
score J_i, the normalized objective reward (already higher is better), is
reported alongside.

## Semantic relation

For objective i in active set O, a preference w_i⁺ that raises i's weight
(heavy: 0.70 on i, the rest split evenly) versus the reference w⁰ (center),
from the same state, same plant and same command, over horizon H:

    ΔS_i(H) = S_i(w_i⁺; s, H) − S_i(w⁰; s, H) > 0

V4-C2 must eventually show this for every i, across seeds.

## Measurement: same-state twins

`Isaac-Talon-A1-V4C-v0`, 2N envs. All envs run w⁰ (deterministic
`act_inference`) for W warm-up steps. Then, for each pair (j, j+N), the full
state of env j is copied into env j+N: root pose and velocity, joint position
and velocity, trunk mass, previous action (for `last_action` and
`action_rate`), velocity command with its resampling timer and heading
target, and episode step counter. From there, env j keeps w⁰ and env j+N
switches to w_i⁺. Both run 32 steps. One twin block runs per objective i
(the m = 4 set), all from the same warm-up states.

- N = 256 pairs per checkpoint and objective, W = 100, env seed fixed and
  shared by all checkpoints.
- **Twin-fidelity gate (must pass before any result is read):** with w⁰ on
  both twins, the maximum absolute difference in every S_i, joint positions
  and rewards over 32 steps ≤ 1e-4. If it fails, no result is used.

Per pair, objective i and H ∈ {1, 4, 8, 16, 32}: ΔS_i(H), ΔJ_i(H)
(discounted sum, γ = 0.99), and the cross-effects ΔS_k(H) for k ≠ i. Pairs
that terminate inside H are excluded at that H and counted. Per checkpoint:
mean, 95% bootstrap interval, and the fraction of pairs with ΔS_i > 0.

## Checkpoints

The 16 V4-C `model_300.pt`. Primary: the 10 seed-sensitivity runs
(73104–73108 × two folds); the 6 G1 runs are reported alongside. No training.

## Interpretation (fixed now), per objective i and checkpoint

A horizon is "correct" when the mean ΔS_i > 0 with the 95% interval
excluding 0, "wrong" when the mean is < 0 with the interval excluding 0, and
"flat" otherwise.

- wrong or flat already at H = 1–4 → the relation fails locally: objective /
  control formulation;
- correct at H = 1–4, wrong at H = 16–32 → long-horizon / basin;
- correct at every H while the m = 4 endpoint for i still fails →
  optimization across updates / policy-family training.

Reported as counts over the 10 primary checkpoints for each objective, and
cross-tabulated with each checkpoint's endpoint result from the
seed-sensitivity evaluation. T/O/S serve as positive controls: their
endpoints pass 9–10/10, so the relation should hold for them. If it does
not, the measurement is suspect before A is interpreted. If no class
dominates for A, that is reported, not resolved by picking one.
