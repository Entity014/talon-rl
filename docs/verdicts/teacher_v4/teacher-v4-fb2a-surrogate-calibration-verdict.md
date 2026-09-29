# Teacher V4 — FB-2a Online-Surrogate Calibration Verdict

Status: **FROZEN. Viability 2/3, authority 0/3, joint 0/3. Registered reading (FB-2 table): "viability ≥ 2/3, authority < 2/3: the constraint holds, but task pressure masks the preference". The FB-2a chain reading: "viability recovered, and R⁺ still does not lower ‖ω_xy‖. The task controller is calibrated, but `ang_vel_xy_l2` does not produce a meaningful rotational-stability preference inside feasible locomotion. Back to the R realization." O⁺ lowers tilt in 3/3.**
Date: 2026-09-29
Contract: [teacher-v4-fb2a-surrogate-calibration-contract.md](../../contracts/teacher_v4/teacher-v4-fb2a-surrogate-calibration-contract.md) (frozen at `d16ee89`, before training). The only change from FB-2: online dual target 0.40 → 0.52. The replay requirement stays 0.40.
Runs: `runs/teacher_v4_fb-2026-09-29/fb2a/seed{79101,79102,79103}/`, aggregate `fb1_screen.json`

All runs completed 600 iterations with no stop gate. EV (T_lin, R, O) at
301–600 is 0.97 / 0.97–0.99 / 0.98–0.99. LR 4.2–5.3e-4, KL 0.014, clip
fraction 0.19.

## Viability (first persistent checkpoint; replay tl at 600)

| seed | R vertex | R⁺ | C | O⁺ | O vertex | viable |
|---|---|---|---|---|---|---|
| 79101 | **—** (0.35; rising from 0.27 at 500) | 300 (0.84) | 300 (0.86) | 300 (0.83) | 300 (0.70) | no |
| 79102 | 400 (0.96) | 200 (0.61) | 200 (0.57) | 200 (0.61) | 200 (0.46) | **yes** |
| 79103 | 300 (0.73) | 200 (1.03) | 200 (0.93) | 200 (0.83) | 200 (0.67) | **yes** |

Tracking is far above the requirement (up to about 1.0, near M0's 1.33),
and the R vertex translates in 2/3.

## Dual vs evaluator (600)

- λ ends at 0 in every region except the 79101 R vertex (16.9, approaching
  the cap of 20; online tl 0.41, replay 0.35).
- The online − replay gap is now mixed in sign (79103 replay > online).
- After λ reaches 0, tracking erodes (e.g. 79102 C 1.01 → 0.57, O vertex
  0.97 → 0.46 between 300 and 600) but stays above 0.40 at 600. The
  controller has not settled within 600 iterations.

## Authority (550 and 600)

| seed | ‖ω_xy‖ R⁺ / C | tilt O⁺ / C | R⁺ tilt / C |
|---|---|---|---|
| 79101 | 1.01–1.02 | 0.81–0.87 | 1.09–1.18 |
| 79102 | 1.13–1.15 | 0.80–0.83 | 1.22 |
| 79103 | 1.01–1.05 | 0.80–0.87 | 1.20–1.24 |

- **O⁺ lowers tilt (−13 % to −20 %) in 3/3 seeds**, while translating.
- **R⁺ never lowers ‖ω_xy‖ (0/3).** It is equal or higher, and it raises
  tilt.
- R⁺ and O⁺ both translate at 550 and 600 in every seed. The authority
  failure is entirely on R.

## Reading

- Calibration worked on the task side. With the online target moved to
  0.52, the replay requirement holds over the segment in 2/3 seeds,
  including the R vertex, which never translated in FB-1 and translated in
  only 1/3 in FB-2.
- With the task met, the R preference still does not reduce roll/pitch
  rate. It tilts the body more instead, the same signature as the FB-1 R
  vertex. O works as a preference inside locomotion, and R does not. The
  remaining failure belongs to the **R realization** (`ang_vel_xy_l2`), not
  to the task mechanism. This matches every earlier signal: FA ρ(T, R) = −0.56,
  the FB-1 monotonicity, and the FB-2 R⁺ result.

**Limits.** 3 seeds, 600 iterations, the dual not settled (tracking still
eroding after λ reaches 0), yaw unrewarded, one R realization.
