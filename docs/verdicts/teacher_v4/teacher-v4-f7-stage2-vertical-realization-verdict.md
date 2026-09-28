# Teacher V4 — F7 Stage 2 Verdict: Vertical-Axis Realization

Status: **FROZEN. Registered reading: "vertical realization unresolved on current selection bank (default V1 lin_vel_z_l2)". All three candidates are admissible (low redundancy with R and O in every policy) and order the 5 translating behaviors perfectly (τ = 1.0). V2 (vertical acceleration) drops out at the policy-blocked step. V1 and V3 tie there (0.691 vs 0.703) and separate only on pooled windows, which the rule does not accept as a decision.**
Date: 2026-09-28
Contract: [teacher-v4-f7-stage2-vertical-realization-contract.md](../../contracts/teacher_v4/teacher-v4-f7-stage2-vertical-realization-contract.md) (r2, frozen at `3fcd9ac`, before traces and target)
Output: `runs/teacher_v4_f7-2026-09-28/f7_stage2.json` (`f7_stage2.py`), traces in `traces/`

## Target

Median per-window vertical excursion (world root height, peak-to-peak,
detrended):

| behavior | excursion |
|---|---|
| s73102 C | 5.1 cm |
| s73102 T⁺ | 6.3 cm |
| F4 s74102 T⁺ | 0.16 cm |
| F4 s74103 T⁺ | 0.07 cm |
| M0 | 0.92 cm |

## Rule, step by step

| candidate | admissible (max \|ρ_R\|, max \|ρ_O\|) | τ behavior | policy-blocked median ρ | pooled ρ |
|---|---|---|---|---|
| V1 lin_vel_z_l2 | yes (0.23, 0.26) | 1.0 | 0.691 | 0.965 |
| V2 base_lin_acc_z_l2 | yes (0.33, 0.14) | 1.0 | **0.333** | 0.906 |
| V3 body_height_osc_l2 | yes (0.16, 0.17) | 1.0 | 0.703 | 0.934 |

- Step 2: all tie at τ = 1.0.
- Step 3: V3 − V1 = 0.012 < 0.02, a tie. V2 is out.
- Step 4 (pooled) picks V1, so the rule's unresolved clause applies.

Policy-blocked within-behavior ρ:

| | s73102 | s74102 | s74103 | M0 |
|---|---|---|---|---|
| V1 | 0.94 | 0.63 | 0.51 | 0.75 |
| V2 | 0.66 | 0.29 | 0.17 | 0.38 |
| V3 | 0.98 | 0.69 | 0.71 | 0.45 |

## Reading

- **V is non-redundant with R and O on every policy.** Per-step |ρ| with
  ang_vel_xy is ≤ 0.33 and with orientation ≤ 0.26, for all three
  candidates. This is the first direct support, at the signal level, that a
  vertical axis carries information the current objectives do not (case 3).
- **Vertical acceleration (V2) represents excursion worst locally.** It
  still orders behaviors correctly.
- **V1 and V3 are not distinguishable on this bank.** V3 tracks excursion
  better within the two grounded costly policies (0.69 / 0.71 vs 0.63 /
  0.51) and within s73102. V1 does better on M0 (0.75 vs 0.45). The
  policy-level median ties. V1 stays as the default only.

The next steps are the same either way. F8 (controllability of V on fresh
training and seeds) can run with V1 as default, or with V1 and V3 as two
arms. A larger policy bank (b) is what could separate them.

**Limits.** 4 policies, 5 behaviors, selection data only.
