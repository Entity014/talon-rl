# Teacher V4 — V4-C2 A Same-State Counterfactual Audit Contract

Status: **DRAFT — freeze before running**
Branch: `v4-c2-a-repair` (V4-C is frozen at tag `v4-c-frozen`; nothing here re-scores it)
Date: 2026-09-27

## Question

From the same state, does switching the preference from center to A-heavy
improve closed-loop angular motion, and at which horizon does it stop doing
so? The answer decides what V4-C2 repairs: the A reward, the long-horizon
optimization, or the policy-family training.

## Checkpoints

The 16 V4-C `model_300.pt` checkpoints (primary: the 10 seed-sensitivity
runs 73104–73108 × two folds; the 6 G1 runs reported alongside). No training.

## Same-state branching ("twins")

`Isaac-Talon-A1-V4C-v0` with 2N envs. All envs run the center preference
(m = 4, deterministic `act_inference`) for a warm-up of W steps. Then, for
each pair (i, i+N), the full state of env i is copied into env i+N:
root pose and velocity, joint position and velocity, trunk mass, previous
action (the obs `last_action` and the `action_rate` term), velocity command
and its resampling timer and heading target, and episode step counter.
From that step, env i keeps the center preference and env i+N switches to
A-heavy (0.70 on A, 0.10 on the others). Both are rolled for 32 steps.

- N = 256 pairs per checkpoint, W = 100, env seed fixed (the same for all
  checkpoints).
- **Twin-fidelity gate (must pass before any result is read):** with the
  same preference on both twins, the maximum absolute difference in ‖ω_xy‖,
  joint positions and rewards over 32 steps must be ≤ 1e-4. If it fails, no
  result from this audit is used.

## Measurements per pair, for H ∈ {1, 4, 8, 16, 32} (A-heavy minus center)

- Δ discounted A reward sum (γ = 0.99), normalized objective units;
- Δ mean ‖ω_xy‖ (negative = A-heavy better);
- Δ mean tilt, Δ mean tracking error, Δ mean ‖action‖, Δ mean ‖torque‖;
- pairs that terminate inside H are excluded at that H and counted.

Per checkpoint: mean and 95% bootstrap interval over pairs, and the
fraction of pairs with Δ‖ω_xy‖ < 0.

## Interpretation (fixed now)

A horizon is "correct" for a checkpoint when the mean Δ‖ω_xy‖ < 0 and its
95% interval excludes 0; "wrong" when the mean is > 0 with the interval
excluding 0; otherwise "flat". Per checkpoint:

- wrong or flat already at H = 1–4 → **local objective / control formulation problem**;
- correct at H = 1–4, wrong at H = 16–32 → **long-horizon / basin problem**;
- correct at every H while the m = 4 A endpoint still fails → **optimization-across-updates /
  policy-family training problem**.

Reported as counts over the 10 primary checkpoints, and cross-tabulated
against each checkpoint's A endpoint result from the seed-sensitivity
evaluation. The class that covers most of the failing checkpoints sets
V4-C2's repair target; if none does, that is reported, not resolved by
picking one.
