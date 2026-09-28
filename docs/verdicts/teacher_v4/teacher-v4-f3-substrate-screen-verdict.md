# Teacher V4 — F3 Substrate Screen Verdict

Status: **FROZEN. No arm is screen-positive (0/3 C locomotion in every arm, B0 included). By the registered rule, all four arms read as collapse. For `linz` and `all` this is strong (T⁺ stands in 3/3 against 0/3 for B0). For `torque_acc` and `air` it rests on a single T⁺-standing run against B0's zero. No confirmation stage runs.**
Date: 2026-09-28
Branch: `v4-c2-semantic-preservation`
Contract: [teacher-v4-f3-substrate-screen-contract.md](../../contracts/teacher_v4/teacher-v4-f3-substrate-screen-contract.md) (r2, frozen at `8c1e654`, before training)
Runs: `runs/teacher_v4_f3-2026-09-28/<arm>_seed{74101,74102,74103}/`, aggregate `f3_screen.json` (`f3_screen.py --stage screen`)

All 15 runs completed 300 iterations with no stop gate. The reset
fingerprints are identical across all F3 and F2-A replays (one fingerprint
set).

## Registered reading

| arm | C locomotes (≥ 2 consecutive checkpoints) | T⁺ standing at 300 | reading |
|---|---|---|---|
| B0 none | 0/3 | 0/3 | — |
| B1 linz | 0/3 | 3/3 | collapse |
| B2 torque_acc | 0/3 | 1/3 | collapse (1 vs 0) |
| B3 air | 0/3 | 1/3 | collapse (1 vs 0) |
| B4 all | 0/3 | 3/3 | collapse |

No arm is screen-positive, so the guardrail and confirmation do not apply.
Per the contract: "If every arm fails, an actor/exploration contract opens
separately."

## Descriptive

Final class at 300 (C / T⁺) and first stepping checkpoint:

| arm | s74101 | s74102 | s74103 |
|---|---|---|---|
| B0 | step / step (C steps from 50) | stand / step | stand / step (T⁺ tl 0.40, one hundredth below partial) |
| B1 linz | stand / stand | stand / stand | stand / stand |
| B2 torque_acc | stand / step | stand / stand | stand / step |
| B3 air | stand / step | stand / stand | step / step (C steps from 250) |
| B4 all | stand / stand | stand / stand | stand / stand |

- T⁺ steps in place at 300: B0 3/3, torque_acc 2/3, air 2/3, linz 0/3,
  all 0/3. No T⁺ translates in any arm (tl ≤ 0.40).
- `linz` and `all` suppress stepping. The dose audit recorded this
  direction before training: the only observed gentle gait pays a large
  lin_vel_z penalty.
- `air` gives the only non-B0 C stepping (s74103, from 250). One run.
- Online preference authority (last 10 iterations): 0.99–1.56 in every arm
  except all s74103 (0.59).
- New-seed B0 reaches C locomotion in 0/3. This fits the rarity seen in V4-C
  (1/16).

## Reading

At this dose (λ = 1/d_T, stock relative magnitudes), a preference-invariant
stock substrate does not get V4 policies from stepping in place to
translation in 300 iterations. The vertical-velocity constraint actively
suppresses stepping, both alone and in the stock combination. Torque/acc
regularization and the air-time prior leave stepping roughly at the B0 level
but add no translation. The screen gives no evidence that R_shared at stock
scale addresses the step → translate stall.

**Limits.** 3 seeds per arm. One dose, fixed in advance. 300 iterations.
The collapse reading for torque_acc and air rests on one run each. A zero
count in B0 makes the "more T⁺-standing than B0" rule trigger on any single
standing run.

Next, per the contract: an actor/exploration contract, opened separately.
