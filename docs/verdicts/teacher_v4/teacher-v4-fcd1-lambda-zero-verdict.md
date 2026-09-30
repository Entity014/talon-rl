# Teacher V4 — FC-D1 Task-Pressure Ablation Verdict

Status: **FROZEN. Registered verdict: "task-dominated updates in other regions contribute" (3/3 seeds). The mixed continuation with the task actor weight at 0 everywhere (D1) does not invert in any seed. FC-C arm A, which is identical except for the loss, inverts in all three.**
Date: 2026-09-30
Contract: [teacher-v4-fcd1-lambda-zero-contract.md](../../contracts/teacher_v4/teacher-v4-fcd1-lambda-zero-contract.md) (frozen with `fcd1_probe.py` at `72fc53b`, before any D1 branch was trained)
Output: `runs/teacher_v4_fc-2026-09-30/fcd1/` (`seed<s>/D1/`, `fcd1_probe.json`, `fcd1.out`)

**Check passed.** D1's k = 0 replay equals A's (difference 0.0) in every seed.

## Registered result

Endpoint = mean over k = 30, 40, 50. Values are raw Δ_R / matched Δ_R.

| seed (u0) | A (FC-C) | D1: λ ≡ 0 in loss | reading |
|---|---|---|---|
| 79101 (450) | +14.1 / +20.0 % inverted | −26.0 / −11.8 % not inverted | contributes |
| 79102 (300) | +7.6 / +10.2 % inverted | +1.4 / +3.5 % not inverted | contributes |
| 79103 (350) | +10.1 / +9.4 % inverted | −7.8 / −9.1 % not inverted | contributes |

Every D1 endpoint checkpoint is viable (tl ≥ 0.74 at R⁺ and C).

## Descriptive (not registered)

- **Where A's task pressure is.** These are the mean λ over A's 50
  iterations:

  | seed | O vertex | O⁺ | C | R⁺ | R vertex |
  |---|---|---|---|---|---|
  | 79101 | 0.00 | 0.01 | 0.16 | 2.22 | 14.58 |
  | 79102 | 0.00 | 0.00 | 0.00 | 0.02 | 8.89 |
  | 79103 | 0.00 | 0.00 | 0.00 | 0.00 | 7.78 |

  In 79102 and 79103, the only region with real task pressure is the **R
  vertex**. So there, D1 differs from A essentially only in the R-vertex
  task updates. In 79101, D1 also removes task pressure at R⁺ and C.
- **The R vertex does not reach the tracking target.** In D1 its online tl
  is 0.32–0.47 (A's dual stays high for the same reason). In A, the
  R-vertex updates are therefore task-dominated: λ ≈ 8–15 against w_R = 1.
- **Mixed sampling without task pressure does not invert within 50
  iterations.** D1 keeps A's preference distribution and R⁺ data share. So
  neither updates at other preferences alone nor dilution alone is enough
  here. This narrows FC-C's items 1 and 2 to "not sufficient without item
  3". They may still interact with it.
- **Tracking in 79101 D1.** R⁺ tracks worse than C (Δtl ≈ −0.15). So raw
  Δ_R (−26 %) overstates; matched Δ_R (−12 %) is the fairer number, and it
  is still clearly not inverted.
- **Shared reduction continues.** F_rate at C falls 39–57 % in D1, against
  25–42 % in A.
- **O control holds in D1.** At k = 50, Δ_O is −12 %, −21 % and −10 %.

## Reading

- As registered: in the mixed continuation, the R⁺ inversion depends on the
  task-dominated updates. In 2/3 seeds these come essentially from the R
  vertex alone.
- A candidate mechanism (not tested): at the R vertex, the policy stands or
  barely tracks, so λ stays high. Those updates push "track harder" in the
  R-heavy part of the preference space. Through shared parameters and the
  neighbouring conditioning, that push reaches R⁺ (w_R 0.7) more than C, and
  raises R⁺'s rotation relative to C. This fits FC-B only partly. FC-B's
  onsets were after λ_R⁺ and λ_C fell, but FC-B never looked at λ at the R
  vertex.
- So the chain now reads:
  - FC-C: mixed training is needed.
  - FC-D1: the task pressure it carries is needed, mostly from the R vertex.
  - What remains is to show which region's task updates are enough, and
    through which parameters they act.
- Candidate follow-ups (each a new contract, not decided here):
  1. **task pressure at the R vertex only**: the A loss with λ = 0
     everywhere except the R vertex, to test sufficiency;
  2. a matching region-drop on the O side, as a control;
  3. parameter-group attribution of the R-vertex task update.

  If the R-vertex requirement (tl ≥ 0.52 at w_R = 1) is itself infeasible,
  that is also a formulation question. It concerns the per-region
  constraint, not the R reward.

**Limits.** One branch per arm per seed. 50 iterations. In 79101, D1 removes
task pressure from R⁺ and C as well as the R vertex. The registered claim is
"necessary in this continuation", not a claim about FB-2a's full history.
