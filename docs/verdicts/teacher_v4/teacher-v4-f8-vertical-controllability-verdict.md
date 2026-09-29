# Teacher V4 — F8 Vertical Controllability Screen Verdict

Status: **FROZEN. Registered reading (primary and secondary): "task viability unresolved: base mostly non-locomoting; case 3 not rejected". 0/3 seeds pass in either arm. The T-anchored base translates at both 550 and 600 in no seed (tl 0.31–0.41 at 600). Descriptive: the R lean lowers ‖ω_xy‖ in 6/6 seeds, but at a tracking cost. The V lean cannot show vertical control, because these gaits never bounce (vertical levels are already at the floor), a floor effect like D's in V4-C3.**
Date: 2026-09-29
Contract: [teacher-v4-f8-vertical-controllability-contract.md](../../contracts/teacher_v4/teacher-v4-f8-vertical-controllability-contract.md) (r2, frozen at `a84b0ad`, before training)
Runs: `runs/teacher_v4_f8-2026-09-28/{V1,V3}_seed{76101,76102,76103}/`, aggregate `f8_screen.json` (`f8_screen.py`)

All 6 runs completed 600 iterations with no stop gate. The primary set is
T55 / T55R / T55V (TAOV: .55/.15/.15/.15, .55/.25/.15/.05, .55/.05/.15/.25).

## Checkpoint 600, primary set

| arm seed | T55 tl / ‖ω‖ / RMS v_z / exc | T55R tl / ‖ω‖ | T55V tl / ‖ω‖ / RMS v_z / exc |
|---|---|---|---|
| V1 76101 | 0.41 / 0.69 / 0.017 / 1.0 mm | 0.37 / 0.58 | 0.44 / 0.89 / 0.017 / 0.9 mm |
| V1 76102 | 0.36 / 0.69 / 0.024 / 2.3 mm | 0.32 / 0.55 | 0.40 / 0.88 / 0.017 / 1.5 mm |
| V1 76103 | 0.35 / 0.52 / 0.034 / 2.3 mm | 0.31 / 0.42 | 0.38 / 0.69 / 0.035 / 2.2 mm |
| V3 76101 | 0.33 / 0.87 / 0.044 / 2.7 mm | 0.28 / 0.48 | 0.39 / 1.19 / 0.039 / 1.8 mm |
| V3 76102 | 0.31 / 0.63 / 0.029 / 1.5 mm | 0.27 / 0.27 | 0.38 / 1.06 / 0.026 / 1.5 mm |
| V3 76103 | 0.35 / 0.39 / 0.069 / 5.9 mm | 0.31 / 0.32 | 0.39 / 0.59 / 0.060 / 5.5 mm |

Viability (tl ≥ 0.40 at 550 **and** 600) holds in no seed. V1 76101
reaches 0.41 at 600 only. Under the secondary set (C / A⁺ / V⁺), every
condition stands or steps in place (tl 0.26–0.28).

## Descriptive (not gated)

- **R responds, at a tracking cost.** T55R lowers ‖ω_xy‖ against T55 in
  6/6 seeds (−16 % to −57 %), but tl falls in 6/6 (to 0.27–0.37).
- **V shows no vertical control, because there is no vertical motion to
  remove.** None of these gaits bounces. RMS v_z is 0.017–0.069 m/s and the
  excursion 1–6 mm, against 0.34 m/s and 5 cm for the F2-A bouncing walker.
  T55V barely changes the vertical metrics (sometimes lower, sometimes
  higher). This is a floor effect, as with D in V4-C3.
- **Moving weight from R to V raises tracking.** tl(T55V) > tl(T55) in
  6/6 seeds, and ‖ω_xy‖ rises. The V lean mostly acts as relaxing R.
- **Double dissociation in the ordering sense** (V-lean vertical < R-lean,
  R-lean ‖ω‖ < V-lean) holds in 5/6 at 600. It is driven by the R side and
  by R-lean's slight vertical increase, not by a V-side reduction.
- **Budget again.** T⁺ tracking rises steadily to 600 (0.35–0.44). It
  reaches partial translation in 2/6 only at 550–600. C stands throughout.

## Reading

F8 cannot select a V realization and does not test case 3 fairly. The
policy family does not reliably translate at the base condition. The
grounded gaits it finds have no vertical oscillation for V to act on. As
registered, case 3 is **not rejected**. V's controllability remains
untested where it matters: on gaits that have vertical motion to trade.

Two structural facts carry forward:

1. V can only show authority on a behavior bank that contains vertical
   motion. The only such gait seen so far is the F2-A bouncing walker.
2. Viability (translation at the base) is still the binding constraint.
   600 iterations take the T-anchored base only to tl ≈ 0.31–0.41.

**Limits.** 3 seeds per arm, 600 iterations, one T-anchored dose (±0.10 on
R / V).
