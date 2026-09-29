# Teacher V4 — FC-B R Semantic Trajectory Audit Verdict

Status: **FROZEN. Registered verdict: case B in 3/3 seeds. R⁺ is never inverted while λ is high. Inversion starts after λ has fallen (u* = 500 / 350 / 400), and it persists to the endpoint in 2/3 seeds. In the third seed (79103), the endpoint is flat (+2.9 %). Timing only; no causal claim.**
Date: 2026-09-29
Contract: [teacher-v4-fcb-r-trajectory-contract.md](../../contracts/teacher_v4/teacher-v4-fcb-r-trajectory-contract.md) (frozen with `fcb_trajectory.py` at `e87add5`, before any trace was read; timing disclosure in the contract)
Output: `runs/teacher_v4_fc-2026-09-29/fcb/` (`traces/seed<s>/`, `fcb_trajectory.json`, `fcb.out`)

## Registered result

The table shows Δ_R = F_rate(R⁺)/F_rate(C) − 1 at viable checkpoints. The
λ phase (h / m / l) and the mean of λ_R⁺ and λ_C over the prior 50
iterations are given after the slash. "n/a" means not viable (R⁺ or C has
tl < 0.40).

| it | 79101 | 79102 | 79103 |
|---|---|---|---|
| 150 | n/a / h 4.4 | n/a / h 4.1 | n/a / h 4.1 |
| 200 | n/a / h 5.6 | flat −2 % / h 4.6 | **correct −47 %** / h 4.8 |
| 250 | n/a / h 6.6 | flat −5 % / h 4.0 | correct −5 % / h 4.3 |
| 300 | **correct −58 %** / h 6.9 | flat +3 % / m 1.6 | correct −11 % / m 2.4 |
| 350 | correct −56 % / h 6.9 | **inverted +9 %** / l 0.0 | flat −3 % / m 0.15 |
| 400 | correct −64 % / h 6.1 | inverted +12 % / l | **inverted +9 %** / l 0.0 |
| 450 | correct −18 % / h 3.9 | inverted +5 % / l | inverted +9 % / l |
| 500 | **inverted +5 %** / m 1.2 | inverted +27 % / l | inverted +17 % / l |
| 550 | inverted +10 % / l 0.03 | inverted +33 % / l | inverted +11 % / l |
| 600 | inverted +13 % / l 0.0 | inverted +49 % / l | flat +3 % / l |
| case | **B** | **B** | **B** |

Answers to the three questions:

1. **When does inversion start?** At 500, 350 and 400. In every seed this is
   at or after the first checkpoint whose λ window is low or near zero.
2. **What is λ doing at onset?** It has already fallen. At u*, the λ window
   is 1.2 (mid, falling), 0.01 and 0.00. No high-λ checkpoint is inverted in
   any seed.
3. **Does R recover after λ falls?** No, not within 600 iterations. The
   inversion grows in 79102 (+9 % to +49 %) and in 79101 (+5 % to +13 %).
   In 79103 it peaks at 500 (+17 %) and returns to flat at 600 (+3 %), which
   is not correct.

## Descriptive (not registered)

- **Early "correct" readings are confounded with tracking.** At every
  correct checkpoint, R⁺ tracks worse than C: Δtl_R is −0.03 to −0.25 in
  79101 at 300–450, and −0.03 to −0.05 in 79103 at 200–300. So in the high-λ
  phase, the lower rotation at R⁺ comes with less translation. This is not
  clean rotational authority. After λ falls, Δtl_R turns positive in 79102
  and 79103 (+0.04 to +0.10). There, R⁺ tracks better *and* rotates more.
  The sign of Δ_R moves with the sign of Δtl_R in 79102 and 79103. It
  does not in 79101, which inverts while Δtl_R stays negative (−0.03).
- **Rotation falls at all preferences, not only at R⁺.** F_rate(C) peaks and
  then falls: 0.87 → 0.42 (79101), 0.46 → 0.065 (79102) and 0.56 → 0.16
  (79103). The policy does reduce ω_xy over training, but R⁺ does not stay
  below C. The R term acts as a shared cost, not as a
  preference-specific axis.
- **O positive control holds.** Δ_O is negative at most viable checkpoints,
  and at 600 it is −30 %, −25 % and −15 %. The one exception is 79102 at
  350–450 (−0.1 % to +8 %). The preference channel works for O through the
  same training period.
- **F_osc follows F_rate.** Δ_R(F_osc) is positive at every inverted
  checkpoint (+2 % to +24 %). So the inversion is not specific to the
  ‖ω_xy‖² realization.
- **Replay tl and online tl agree roughly.** Per region, they are within
  about 0.2 of each other at viable checkpoints (the largest gap is 79103 C
  at 300: 1.09 vs 0.90). They stay separate evidence, as in the contract.

## Reading

- As registered: case B in all seeds. The high-λ phase does not coincide
  with inversion. So early task-dominated shaping (case A) is not what the
  timing shows. The pattern is consistent with drift under the accumulated
  preference-phase updates, after the task constraint became slack.
- The descriptive points narrow this. When λ ≈ 0, R lowers rotation for the
  whole policy (at C as well), but R⁺ does not keep a lower level than C.
  And the R⁺ gait gains tracking along with rotation. One candidate is that
  the preference gradient in that phase favors a faster or more
  tracking-efficient R⁺ gait, whose rotation cost R does not offset at
  w_R 0.7. Another is that the w-conditioned difference is not learned
  while the shared component is. FC-B cannot tell these apart.
- Per the contract's mapping, case B points to a multi-update probe from the
  checkpoint before u*: 450 (79101), 300 (79102) and 350 (79103). That probe
  is a new question. It needs its own contract, with its step count, gate
  and tracking control fixed before it runs. FC-B does not decide it.

**Limits.** 50-iteration checkpoint spacing, so onset is known only to
within 50 iterations. A deterministic 128-step replay per checkpoint.
Viability starts at 200–300, so the earliest high-λ phase is not readable.
The ±5 % state threshold puts two onsets (+5.2 %, +5.1 %) just above the line.
