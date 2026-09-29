# Teacher V4 — FB-1 Fixed-Task Feasibility Verdict

Status: **FROZEN. Registered reading: "FAIL viability: fixed task pressure insufficient; constrained formulation becomes the candidate". Joint 0/3, viability 0/3, authority 0/3. Only the O vertex translates (3/3 seeds). Tracking falls monotonically as the preference moves toward R, and the R vertex never translates. The preferences do steer semantics (O⁺ lowers tilt 3/3, R⁺ lowers ‖ω_xy‖ 2/3), but not inside translation.**
Date: 2026-09-29
Contract: [teacher-v4-fb1-fixed-task-contract.md](../../contracts/teacher_v4/teacher-v4-fb1-fixed-task-contract.md) (r2 frozen at `9343029`, erratum `bcb9dd2` before any result). α_T = 0.44 from FB-0 (`0b2e433`).
Runs: `runs/teacher_v4_fb-2026-09-29/fb1/seed{77101,77102,77103}/`, aggregate `fb1_screen.json`

All three runs completed 600 iterations with no stop gate. "Translates"
means tl ≥ 0.40 in this protocol's command regime (standing gives 0.29). It
does not mean moving at every instant.

## A. Five-point viability

tl at checkpoint 600 (R vertex / R⁺ / C / O⁺ / O vertex) and the first
persistent-translation checkpoint:

| seed | R vertex | R⁺ | C | O⁺ | O vertex | persistent from |
|---|---|---|---|---|---|---|
| 77101 | 0.28 | 0.29 | 0.29 | 0.38 | **0.64** | O vertex only (250) |
| 77102 | 0.27 | 0.32 | 0.34 | 0.36 | **0.57** | O vertex only (400) |
| 77103 | 0.28 | 0.30 | 0.31 | 0.35 | **0.58** | O vertex only (400) |

Tracking is ordered monotonically from the R vertex to the O vertex in every
seed. Interior tracking was still rising at 600 (C: 0.29–0.34, O⁺:
0.35–0.38).

## B. Joint seeds: 0/3

## C. Semantic direction (550 and 600, interior)

- O⁺ vs C: tilt −17 % to −24 %, in 3/3 seeds.
- R⁺ vs C: ‖ω_xy‖ −14 % to −20 % in 77101 and 77103. In 77102 it is −4 to −8 %,
  below the 10 % bar.
- Cross-effects: R⁺ raises tilt (+0 % to +22 %). O⁺ raises ‖ω_xy‖ (+49 % to
  +86 %).
- The R vertex adopts a strongly tilted posture (6.5–12.8°) with low
  roll/pitch rate, stepping in place.

## D. Tracking preservation

Authority failed only on "R⁺ and O⁺ translate" (tl < 0.40 at both). The
semantic responses exist, but not inside feasible locomotion.

## E. Critic and optimizer

EV (T, R, O) at iterations 301–600 is 0.97 / 1.00 / 0.99–1.00 in all seeds.
LR 2.9–3.9e-4, KL 0.011–0.012, clip fraction 0.15–0.17. Preference
authority is 0.94–1.35. Nothing points to an optimization or critic
failure. The symptom is the formulation: R pressure trades against
translation.

## Reading

- The fixed task term keeps T in the loss, but at α_T = 0.44 the R
  preference still wins against translation. This matches the FA geometry
  (ρ(T, R) = −0.56 across locomoting gaits). O does not oppose translation:
  the O vertex walks in every seed.
- FB-0 calibrated α_T against the **observed** bank, whose best locomoting
  behaviors included the low-R bouncing gait. FB-1 policies did not find
  low-R gaits, so the bank-based α_T was optimistic. FB-0 had declared this
  limit ("necessary, not sufficient").
- As registered, the next candidate is a **constrained formulation**
  (T ≥ T_min), which enforces the requirement instead of pricing it.

**Limits.** 3 seeds, 600 iterations with tracking still rising at the
interior, one α_T.
