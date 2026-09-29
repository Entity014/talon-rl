# Teacher V4 — FB-1 Fixed-Task Feasibility Screen (T task, R/O preferences)

Status: **DRAFT 2026-09-29. Not frozen. No FB-1 training run.** Pipeline smoke-tested (seed 1, 512 envs, 50 iterations; not FB-1 data).
Branch: `v4-c2-semantic-preservation`
Design: [behavior–preference hierarchy](../../methods/general/behavior-preference-hierarchy.md). Task pressure: [FB-0](teacher-v4-fb0-task-pressure-calibration-contract.md), **α_T = 0.44** (λ_T = 0.786), frozen at `0b2e433` before this contract.

## Question

If tracking is taken off the preference simplex and made a fixed-weight
task term, do policies locomote across the preference space **and** keep
R / O authority inside that locomotion?

**The only change from V4-C3 is the formulation.** No V (its realization is
unresolved), no new normalization, no R_shared.

## Formulation

    J(w) = α_T · T̃ + (1 − α_T) · (w_R R̃ + w_O Õ),   w_R + w_O = 1,   α_T = 0.44

- T is the task. It is **never in the conditioning set**: the actor and critic
  objective-set encoders see only {(R, w_R), (O, w_O)}.
- Critic: three streams, queried by objective id (T, R, O). The T stream
  shares the critic body (conditioned on the preference encoding) and has its
  own query embedding.
- Actor loss: the per-stream PPO surrogate with loss weights [α_T, (1 − α_T)
  w_R, (1 − α_T) w_O] over all three streams. The task gradient does not
  shrink when the preference moves (tested). Advantage normalization uses
  these loss weights.
- Unchanged: T3-B divisors for T, R, O, TeacherV4, PPO shell and adaptive
  KL (desired_kl 0.01), stop gates, post-fix normalizer, V4-C env.
- Preference support: cardinalities {1, 2} over (R, O): the vertices {R},
  {O}, and pairs.
- **600 iterations** from scratch, checkpoints every 50. Fresh seeds
  **77101, 77102, 77103**.

Code: `--objectives TAO --task-alpha 0.44 --cardinalities 1,2` in
`train_v4c.py`. The optional `loss_w` / `loss_mask` / `query_ids` keys in
`objective_set_ppo.update` default to the old path.

## Measurement

The F2-A replay protocol (the same reset suite, deterministic) on every
checkpoint. Conditions over (R, O): C (.5, .5), R⁺ (.7, .3), O⁺ (.3, .7), and
the R / O vertices (descriptive). Physical metrics: tl, ‖ω_xy‖, tilt, RMS v_z.

## Per-seed criteria

1. **Task viability:** translation (tl ≥ 0.40) persistent on at least two
   consecutive checkpoints through 600 at **each** of R⁺, C and O⁺.
2. **Preference authority inside feasible locomotion**, at checkpoints 550
   and 600: ‖ω_xy‖(R⁺) ≤ 0.9 ‖ω_xy‖(C); tilt(O⁺) ≤ 0.9 tilt(C); and
   tl(R⁺), tl(O⁺) ≥ 0.40.

## Pre-declared reading

| viability (≥ 2/3 seeds) | authority (≥ 2/3 seeds) | reading |
|---|---|---|
| pass | pass | **abstraction supported** (behavior requirement ≠ preference objective); FB-2 adds V |
| fail | — | fixed task pressure insufficient; a constrained formulation (T ≥ T_min) becomes the candidate |
| pass | fail | task pressure dominates the preferences; the fixed scalar still has the tension |

Reported descriptively: cross-effects (R⁺ on tilt, O⁺ on ‖ω_xy‖), the
vertices, C / R⁺ / O⁺ ladders, and online T / R / O rewards and EV.

Aggregate: `fb1_screen.py --root runs/teacher_v4_fb-2026-09-29/fb1`. Runs:
`runs/teacher_v4_fb-2026-09-29/fb1/seed<seed>/` (+ `replay/`).
