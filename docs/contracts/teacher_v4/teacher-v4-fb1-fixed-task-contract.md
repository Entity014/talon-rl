# Teacher V4 — FB-1 Fixed-Task Feasibility Screen (T task, R/O preferences)

Status: **FROZEN 2026-09-29 (r2) in the commit that sets this line, before any FB-1 training run.** Pipeline smoke-tested (seed 1, 512 envs, 50 iterations; not FB-1 data).
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
  w_R, (1 − α_T) w_O] over all three streams. Moving the preference never
  reallocates weight away from the task: the explicit T loss coefficient
  stays α_T (tested). The absolute task gradient can still change through the
  state distribution, the task advantage, the shared normalization scale and
  the policy itself. Advantage normalization uses
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
   consecutive checkpoints through 600 at **each** of the five points: R
   vertex (1, 0), R⁺ (.7, .3), C (.5, .5), O⁺ (.3, .7), O vertex (0, 1). FB-0
   calibrated α_T against the whole segment, vertices included. At a vertex,
   the training objective is still 0.44 T + 0.56 R (or O). The vertex means
   "the whole preference budget goes to R while still locomoting", not "R
   without the task".
2. **Preference authority inside feasible locomotion**, at checkpoints 550
   and 600: ‖ω_xy‖(R⁺) ≤ 0.9 ‖ω_xy‖(C); tilt(O⁺) ≤ 0.9 tilt(C); and
   tl(R⁺), tl(O⁺) ≥ 0.40. Measured on the interior points only, to avoid
   boundary effects.
3. **Joint:** the seed passes only if it passes 1 **and** 2. The same policy
   must show both.

## Pre-declared reading

Primary: the number of seeds with **joint** pass. Viability and authority
counts are reported separately as diagnostics.

| outcome | reading |
|---|---|
| joint ≥ 2/3 | **the fixed-task formulation supports the coexistence of locomotion viability and R / O preference semantics** under this screen; FB-2 adds V |
| viability < 2/3 | fixed task pressure insufficient; a constrained formulation (T ≥ T_min) becomes the candidate |
| viability ≥ 2/3, authority < 2/3 | task pressure dominates the preferences; the fixed scalar still has the tension |
| both ≥ 2/3 but joint < 2/3 | no joint support: viability and authority occur in different policies |

FB-1 is an existence / compatibility test. It makes no claim that FB-1 is
better than V4-C3: there is no matched concurrent comparison. A superiority
claim needs its own comparative contract.

Reported descriptively: cross-effects (R⁺ on tilt, O⁺ on ‖ω_xy‖), the
vertices, C / R⁺ / O⁺ ladders, and online T / R / O rewards and EV.

Aggregate: `fb1_screen.py --root runs/teacher_v4_fb-2026-09-29/fb1`. Runs:
`runs/teacher_v4_fb-2026-09-29/fb1/seed<seed>/` (+ `replay/`).
