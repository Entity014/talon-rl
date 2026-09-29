# Teacher V4 — FC-A R Credit / Gradient / Virtual-Step Audit Verdict

Status: **FROZEN. Registered verdict: "unresolved" (3/3 seeds). No virtual step up to KL 0.02 moves F_rate by 5 % (max |ΔF_rate| 2.1 %), in any direction. The null repeat is exact (0.0 %), so this is a real sub-threshold effect, not noise. Descriptively: no credit weakness (s_R 0.51–0.69 at R⁺), no R/O interference at 600, and the R-ascent step leans the right way in 2/3 seeds but is tiny. The local update barely moves physical rotation at PPO-sized steps.**
Date: 2026-09-29
Contract: [teacher-v4-fca-r-credit-audit-contract.md](../../contracts/teacher_v4/teacher-v4-fca-r-credit-audit-contract.md) (r2, frozen with both scripts at `47fbfda`, before any FB-2a checkpoint was audited)
Output: `runs/teacher_v4_fc-2026-09-29/fca/` (`seed<s>/`, `seed<s>_300/`, `fca_aggregate.json`)

## Checkpoint 600, R⁺, locomoting slice (primary)

| seed | loco fraction | s_R | cos(g_R, g_O) | ‖g_R,logσ‖/‖g_R‖ | ΔF_rate R-ascent @0.01 / 0.02 | R-descent | mixed |
|---|---|---|---|---|---|---|---|
| 79101 | 0.86 | 0.51 | +0.57 | 0.09 | −0.0 % / −1.6 % | +1.0 / +0.9 % | −0.1 / +1.7 % |
| 79102 | 0.70 | 0.69 | +0.15 | 0.05 | −0.3 / −0.4 % | +0.3 / +0.9 % | −0.1 / −1.4 % |
| 79103 | 0.92 | 0.68 | −0.14 | 0.09 | −0.2 / +0.2 % | −0.6 / −0.4 % | +0.7 / +0.9 % |

At 600, λ = 0 at R⁺ in every seed, so g_T = 0 there, as expected. Null
repeat 0.0 % in all seeds. The virtual steps change tl by at most 2 %.

## Descriptive

- **Credit strength is adequate.** At R⁺, R gets 51–69 % of the preference
  contribution, in line with its 0.7 weight. No credit-weak flag.
- **No R/O interference at 600.** cos(g_R, g_O) at R⁺ is +0.57 / +0.15 /
  −0.14. At O⁺ it is −0.36 / −0.15 / −0.52, a mild conflict on the O side.
- **Direction.** R-ascent lowers F_rate slightly and R-descent raises it in
  79101 and 79102, the right sign but under 2 %. 79103 shows no consistent
  sign.
- **The full gradient is mostly mean.** log_std carries only 5–9 % of
  ‖g_R‖.
- **Checkpoint 300 (secondary, λ active).** T dominates the contribution
  where λ is large: 82–94 % at the R vertex, and 82–86 % in every region
  of 79101. cos(g_R, g_T) is mixed (−0.47 to +0.62). There is no consistent
  R/T conflict, but at 300 the R signal is a small share (5–11 %) wherever
  λ ≳ 7.

## Reading

- As registered: unresolved. FC-A cannot place the R failure at a single
  local update. At KL ≤ 0.02, the local mean change moves roll/pitch rate by
  at most about 2 %, well under the pre-declared 5 %.
- What it rules out, descriptively: at the endpoint, R credit is not weak,
  R is not cancelled by O, and the log_std component is not absorbing the
  update. So none of the single-update mechanisms (credit strength,
  interference, variance absorption) explains FB-2a's R⁺ result.
- What is left: the R effect, if any, must accumulate across many updates,
  where R⁺ ends up with *more* rotation (FC-0). Also possible: the
  checkpoint-300 phase, where T dominates the loss (λ ≈ 7–17) and R has
  about 5–11 % of the credit, shaped the gait before λ fell. Once λ fell,
  per-update R steps are too small to undo it.
- A larger-step or multi-update probe is a new question. It needs its own
  contract (thresholds and step sizes fixed before it runs). The 5 % gate
  and the KL ladder here are not revised.

**Limits.** One checkpoint per phase, fixed-w batches, KL ≤ 0.02, and a
deterministic 128-step replay.
