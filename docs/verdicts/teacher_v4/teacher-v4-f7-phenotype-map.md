# Teacher V4 — F7 Stage 1: Dynamic-Stability Phenotype Map

Status: **DESCRIPTIVE, 2026-09-28. Among the translating behaviors, rotational and vertical dynamics order in opposite directions. No observed gait is low on both. Dynamic stability is at least two-dimensional on this behavior bank, and one A-like axis cannot rank these gaits.**
Contract: [teacher-v4-f7-dynamic-stability-semantics-contract.md](../../contracts/teacher_v4/teacher-v4-f7-dynamic-stability-semantics-contract.md) (frozen at `1cdb92d`, before computation)
Output: `runs/teacher_v4_f7-2026-09-28/f7_phenotype.json` (`f7_phenotype.py`)

Medians over 32-step windows (10–90 % range in the JSON). No raw reward
term defines any column.

| behavior | tl | rot RMS ‖ω_xy‖ | vert RMS \|v_z\| | vert excursion (m) | flight fraction | tilt ° (guardrail) |
|---|---|---|---|---|---|---|
| s73102 C (low-A) | 0.55 | 0.33 | **0.34** | **0.051** | 0.00 (q90 0.06) | 0.67 |
| s73102 T⁺ (low-A) | 0.62 | 0.48 | **0.41** | **0.063** | 0.06 (q90 0.25) | 1.01 |
| F4 s74102 T⁺ (high-A) | 0.64 | **1.12** | 0.019 | 0.002 | 0 | 0.89 |
| F4 s74103 T⁺ (high-A) | 0.82 | **1.32** | 0.012 | 0.001 | 0 | 0.99 |
| M0 | 1.35 | 1.18 | 0.072 | 0.010 | 0 | 3.72 |
| standing s73101 C | 0.27 | 0.08 | 0.056 | 0.012 | 0 | 0.74 |
| F4 s74102 / s74103 C | 0.28 | 0.13–0.17 | 0.04 | 0.006 | 0 | 0.52–0.66 |

## Reading

- **The translating gaits sit on a rotational–vertical trade-off.** The
  low-A gait oscillates vertically (~5–6 cm, |v_z| ~0.35 m/s) with low
  roll/pitch rate. The high-A gaits are almost vertically still (~1–2 mm)
  with high roll/pitch rate. M0 sits between them vertically and is high
  rotationally. No translating gait is low on both axes. Only standing is,
  and it does not translate.
- **Flight is secondary.** The bounce is mostly vertical oscillation with
  contact. Flight appears only in a minority of windows (s73102 T⁺ q90
  25 %).
- **Tilt barely separates the V4 behaviors** (0.5–1.0°). It separates M0
  (3.7°). Keeping tilt out of D costs nothing here.
- So "low D" under A does not mean "dynamically stable" in a
  physics-neutral sense. It picks one side of a trade-off that A cannot see.

This supports treating cases 1–3 of the contract as live (A + vertical
constituent, vertical as a constraint, or two objectives). Case 4 (vertical
bounce acceptable) is a normative choice the data cannot make.

**Limits.** Selection data only. Body-frame v_z, so the excursion is
approximate. One low-A policy, two high-A policies, and M0.
