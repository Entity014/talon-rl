# Teacher V4 — FB-2 Lagrangian Task-Constraint Verdict

Status: **FROZEN. Registered reading: "FAIL viability, duals not at cap: dual adaptation / target insufficient". Joint 0/3, viability 0/3, authority 0/3. Descriptively, the interior of the segment now translates persistently (R⁺ and C 3/3, O⁺ 2/3). Three things break the gate. (1) Once the online constraint is met, λ falls to 0, and replay tracking then decays below 0.40 at some points: the online (stochastic) tl runs 0.05–0.13 above the replay tl. (2) The R vertex never translates in 2/3 seeds, with λ still rising (7.4–7.7, cap 20). (3) R⁺ never lowers ‖ω_xy‖ against C (0/3). O⁺ lowers tilt 3/3.**
Date: 2026-09-29
Contract: [teacher-v4-fb2-lagrangian-task-contract.md](../../contracts/teacher_v4/teacher-v4-fb2-lagrangian-task-contract.md) (r2, frozen at `cdaa683`, before training)
Runs: `runs/teacher_v4_fb-2026-09-29/fb2/seed{78101,78102,78103}/`, aggregate `fb1_screen.json`

All three runs completed 600 iterations with no stop gate. EV (T_lin, R,
O) at 301–600 is 0.97 / 0.99 / 0.99–1.00. LR 4.8–8.2e-4, KL 0.013–0.014,
clip fraction 0.17–0.19.

## Viability (persistent tl ≥ 0.40, first checkpoint)

| seed | R vertex | R⁺ | C | O⁺ | O vertex |
|---|---|---|---|---|---|
| 78101 | 550 | 450 | 450 | 400 | **—** (0.52 at 450 → 0.38 at 600) |
| 78102 | **—** (0.26–0.28 throughout) | 400 | 350 | 400 | 400 |
| 78103 | **—** (0.26–0.30 throughout) | 450 | 450 | **—** (0.44 at 450 → 0.37 at 600) | 350 |

## Dual vs evaluator (checkpoint 600)

| seed | region | λ_final | online tl (last 50) | replay tl 600 |
|---|---|---|---|---|
| 78101 | O vertex / O⁺ / C / R⁺ / R vertex | 0 / 0 / 0 / 0 / 2.61 | .50 / .54 / .60 / .62 / .49 | .38 / .40 / .47 / .48 / .43 |
| 78102 | same order | 0 / 0 / 0 / 0 / **7.42** | .53 / .49 / .51 / .51 / .34 | .48 / .40 / .42 / .43 / .28 |
| 78103 | same order | 0 / 0 / 0 / .53 / **7.73** | .53 / .49 / .55 / .54 / .32 | .50 / .37 / .48 / .49 / .26 |

- λ rises in every region early (to about 2.5–4.5 by iteration 300). It
  drops to 0 once the online tl clears 0.40.
- **The online tl is 0.05–0.13 above the replay tl.** With λ at 0 the dual
  sees the constraint as satisfied, while the deterministic replay sits at
  0.37–0.40 at some points (78101 O vertex, 78103 O⁺, O⁺ at 600 in all
  seeds). Tracking then decays toward the boundary, because nothing pushes
  it above.
- R vertex (78102, 78103): the online violation persists (0.32–0.35), the
  replay stays non-viable (0.26–0.28), there is no upward trend, and λ is
  still rising (6.0 → 7.4 / 7.7 over iterations 450–600), below the cap.
  Three of the four contract conditions for suggesting incompatibility hold.
  The fourth (λ at cap) does not, so no incompatibility is attributed.

## Authority (550 and 600)

- O⁺ vs C: tilt −14 % to −32 %, in 3/3 seeds.
- R⁺ vs C: ‖ω_xy‖ **not lower** in any seed (+1 % to +22 %). R⁺ raises tilt
  (+3 % to +33 %).
- R⁺ and O⁺ translate at 550 and 600 in 78101 and 78102, and at 550 in
  78103 (O⁺ 0.37 at 600).

## Reading

- As registered: the dual mechanism with T_min = 0.40 on the online signal
  does not hold the deterministic replay above 0.40 everywhere, and does
  not recover the R vertex within 600 iterations.
- Descriptively, FB-2 changes the picture from FB-1. The interior translates
  in every seed (FB-1: never), and the O side too in most seeds. The
  constraint mechanism restores task feasibility over most of the segment.
  This is not a superiority claim (no matched comparison), but the shift is
  large.
- **Two remaining problems are separable.** (a) A calibration gap: the dual
  targets the online tl, which runs above the evaluator's tl, and a
  projected λ of 0 removes all task pressure at the boundary. (b) R
  semantics: with the task enforced, the R preference no longer lowers
  roll/pitch rate. At the R vertex, the task stays infeasible under the
  pressure reached so far. Both point to R (`ang_vel_xy_l2`) as the
  objective in tension with translation (FA ρ(T, R) = −0.56, FB-1
  monotonicity).

**Limits.** 3 seeds, 600 iterations, λ at the R vertex still rising, one
T_min, and yaw tracking unrewarded in FB-2 by design.
