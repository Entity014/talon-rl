# Teacher V4 — FB-2 Preference-Conditioned Task Constraint (Lagrangian)

Status: **FROZEN 2026-09-29 (r2) in the commit that sets this line, before any FB-2 training run.** Pipeline smoke-tested (seed 1, 512 envs, 50 iterations; not FB-2 data).
Branch: `v4-c2-semantic-preservation`
Follows: [FB-1 verdict](../../verdicts/teacher_v4/teacher-v4-fb1-fixed-task-verdict.md) (fixed α_T = 0.44 insufficient; only the O vertex translates; tracking falls monotonically toward R)

## Question

If the task becomes an explicit constraint whose pressure adapts per
preference region, instead of a fixed price, do policies translate across the
whole R/O preference segment **and** keep R/O semantics inside that
locomotion?

The only change from FB-1: fixed α_T → per-region Lagrange multipliers.

## Formulation

    maximize   w_R J_R + w_O J_O            (w_R + w_O = 1)
    subject to linear tracking ≥ T_min,     for every preference region

    L(w) = w_R R̃ + w_O Õ + λ_{region(w)} · (J_lin − T_min)
    actor:  w_R R̃ + w_O Õ + λ_{region(w)} · T̃_lin      (−λ T_min has no policy gradient)

- **T_min = 0.40** on track_lin_vel_xy_exp (weighted step value, the same
  units and threshold as the replay gate). It is a *linear-motion
  feasibility* constraint, not the full task specification.
- **The multiplied stream is the constrained quantity.** The task stream is
  T_lin = track_lin_vel_xy_exp only, with its own critic query and
  advantage, divided by the same T divisor (1.71946) so that λ₀ keeps its
  FB-1 meaning. **Yaw tracking is in no stream in FB-2.** A vector constraint
  (τ_lin, τ_yaw) is future work.
- **Regions by w_R, frozen:** O vertex [0, 0.15), O⁺ [0.15, 0.40), C
  [0.40, 0.60), R⁺ [0.60, 0.85), R vertex [0.85, 1]. Edges belong to the
  region above (tested). Training still samples the continuum
  (cardinalities {1, 2}). Each sample uses its region's λ.
- **Projected dual update** (descent in λ under max_π L = J_pref + λ(J_lin −
  T_min)), once per iteration after the PPO update: λ_j ← clip(λ_j + η (T_min − tl_j), 0, λ_max). tl_j is the online
  mean track_lin over that region's rollout samples (stochastic policy, all
  episode phases). A region with no samples keeps its λ. λ can fall when the
  region satisfies the constraint.
- **λ_j(0) = 0.786** for all regions (= 0.44 / 0.56 from FB-0), so FB-2
  starts at the FB-1 weighting. **η = 0.15** (a persistent deficit of 0.1
  adds about 0.75 in 50 iterations). **λ_max = 20** (α ≈ 0.95), a numerical
  safety bound with no semantic meaning. λ can rise and fall: it acts as
  the constraint's controller (up on violation, down when comfortably
  satisfied, settling near the boundary).
- Loss weights per sample: [λ_region, w_R, w_O] over the streams T, R, O.
  Advantage normalization uses them. Everything else as FB-1: T outside the
  conditioning set, three critic streams, T3-B divisors, PPO shell,
  desired_kl 0.01, stop gates, V4-C env, no V, no R_shared.
- **600 iterations** from scratch, fresh seeds **78101, 78102, 78103**.

Disclosed: the online tl comes from the stochastic policy over all episode
phases, and is typically below the deterministic steady-window replay tl.
λ can therefore keep rising after the replay gate is met. The λ trajectories
are reported.

Code: `--objectives TAO --cardinalities 1,2 --lagrange-tmin 0.40`
(`--lagrange-lambda0 0.786 --lagrange-eta 0.15 --lagrange-cap 20`) in
`train_v4c.py`. λ and the per-region online tl are logged per iteration and
saved in checkpoints.

## Gate (identical to FB-1)

Per seed: viability, meaning persistent tl ≥ 0.40 at the R vertex, R⁺, C,
O⁺ and O vertex. Authority at 550 and 600: ‖ω_xy‖(R⁺) ≤ 0.9 C, tilt(O⁺)
≤ 0.9 C, both translating. **Joint** = viability ∧ authority, and ≥ 2/3
joint seeds is required.

## Pre-declared reading

| outcome | reading |
|---|---|
| joint ≥ 2/3 | the constrained task requirement and R/O preference semantics coexist under this screen; FB-3 adds V |
| viability ≥ 2/3, authority < 2/3 | the constraint holds, but task pressure masks the preference |
| both ≥ 2/3, joint < 2/3 | no joint support |
| viability < 2/3, and in ≥ 2/3 seeds an R-side dual (R⁺ or R vertex) is at the cap in a non-viable seed | the online constraint was not recovered under this dual mechanism / budget. Before attributing incompatibility, check the replay agreement and the tracking trend. Incompatibility of the R realization is suggested (evidence, not proof) only if λ is at the cap, the online violation is persistent, the deterministic replay is non-viable **and** there is no upward tracking trend |
| viability < 2/3 otherwise | dual adaptation or target insufficient |

Reported: per region, λ_final, the online tl (mean of the last 50
iterations) and the replay tl at 600, side by side (how differently the
dual and the evaluator see the task). Also the λ and online-tl
trajectories, and the FB-1-style A–E breakdown.

Aggregate: `fb1_screen.py --root runs/teacher_v4_fb-2026-09-29/fb2 --seeds 78101,78102,78103`.
