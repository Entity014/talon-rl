# Teacher V4 — F4 Budget Audit Verdict (B0, 300 → 600)

Status: **FROZEN. Registered reading: NO BUDGET SIGNAL. C translates persistently in 0/3 seeds by 600 (all 3 valid, LR floor ≤ 1.3 %). Not gated, but important: T⁺ becomes budget-sensitive. T⁺ translates persistently in 2/3 (from 350 and from 500), with tracking still rising at 600, while C only moves from standing to stepping in place.**
Date: 2026-09-28
Branch: `v4-c2-semantic-preservation`
Contract: [teacher-v4-f4-budget-audit-contract.md](../../contracts/teacher_v4/teacher-v4-f4-budget-audit-contract.md) (frozen at `0cd87d6`, before training)
Runs: `runs/teacher_v4_f4-2026-09-28/none_seed{74101,74102,74103}/`, aggregate `f4_budget.json` (`f4_budget.py`)

All three controlled restarts completed 301–600 with no stop gate. The LR
sat at the floor on 1.3 %, 0.3 % and 0 % of the added iterations, so all
three seeds are valid.

## Registered reading

| seed | C t_step | C t_translate | C at 600 (class, tl) |
|---|---|---|---|
| 74101 | 50 | — | step-in-place, 0.286 |
| 74102 | 450 | — | step-in-place, 0.277 |
| 74103 | 550 | — | step-in-place, 0.278 |

0/3 → **no budget signal.** Per the contract, the actor/exploration
contract opens next.

## Descriptive (not gated)

T⁺ ladder, tl per checkpoint (300 is the F3 end, the restart is at 301):

| seed | 300 | 350 | 400 | 450 | 500 | 550 | 600 | t_translate (persistent) |
|---|---|---|---|---|---|---|---|---|
| 74101 | 0.277 | 0.277 | 0.292 | 0.308 | 0.318 | 0.338 | 0.367 | — |
| 74102 | 0.324 | 0.361 | 0.385 | 0.398 | **0.445** | 0.498 | 0.528 | 500 |
| 74103 | 0.397 | **0.462** | 0.513 | 0.549 | 0.606 | 0.649 | 0.699 | 350 |

- **T⁺ is budget-sensitive:** 2/3 translate persistently by 600, and the
  third rises every checkpoint from 350. Under the registered rule it would
  read as budget-sensitive. It is secondary, not gated.
- **C lags on the same staircase:** standing → stepping in place in all
  three (74101 from 50, 74102 from 450, 74103 from 550). C tl moves only
  0.266 → 0.286. T⁺ before C, as in F2-A.
- **The T⁺ gait is D-costly.** Final T⁺ R: 74103 [0.80, −0.48, −0.08],
  74102 [0.70, −0.34, −0.08]. That is at or near M0's D cost (−0.46), with
  low O cost. This is not the low-cost gentle gait of F2-A (T⁺ D −0.07 at
  s73102).
- C at 600 stays near standing cost (D −0.02 to −0.04, O −0.03 to −0.11).

## Reading

Extra optimization is enough for T-heavy preferences to find translation,
but the gait found costs a lot of D. At the center, 300 more iterations only
produce stepping in place, with no translation. This fits F1: with a
D-costly gait on offer, the center scores standing and stepping at least as
high as walking. The center's stall may therefore reflect the objective
balance at C, given the gait the optimizer finds, rather than a lack of
exploration.

That is an interpretation, not an F4 result. It bears on the design of the
next contract. The registered next step is the actor/exploration contract.
The descriptive result suggests it should ask about the **gait type found**
(low D-cost vs D-costly), not only whether translation appears.

**Limits.** 3 seeds. A controlled restart, not a continuation. The ladder
ends at 600 while T⁺ tracking is still rising.
