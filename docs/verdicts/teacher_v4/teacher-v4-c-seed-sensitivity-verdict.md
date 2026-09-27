# Teacher V4 — V4-C Seed-Sensitivity Verdict

Status: **FROZEN — A semantic direction passes in 2/10 new seeds → "effectively a systematic A failure" (band fixed in advance); T 10/10, O 9/10, S 9/10; critic A 0/10, S 0/10**
Date: 2026-09-27
Contract: [teacher-v4-c-seed-sensitivity-contract.md](../../contracts/teacher_v4/teacher-v4-c-seed-sensitivity-contract.md) (frozen at `979cb94` / `f5f861d`, before training)

## Runs and evaluation

Ten new runs, `runs/teacher_v4_c_g1_{2,3}_seed{73104…73108}-2026-09-27/`,
unchanged recipe (`train_v4c.py`; normalizer fix `61a483d` as declared), all
300 iterations, no stop gate, none excluded or replaced. Evaluated at
`model_300.pt` with `anchor_evaluate.py` (commit `7a142d5`), output
`anchor_evaluation.json` per run.

Evaluator gate: on the six existing runs it reproduced every stored G1 m = 4
endpoint value (fractions, deltas, survival) with zero mismatches. One
evaluation process hung at simulation start and was killed and rerun; the
rerun completed normally, and every result below comes from a completed
run.

## Per run (m = 4, iteration 300; * = new)

| fold | seed | T A O S endpoint | critic A / S (MC256 EV mean) | ‖ω_xy‖ C / A-heavy / O-heavy | ΔJ_A |
|---|---|---|---|---|---:|
| G1-2 | 73101 | ✓ ✗ ✓ ✓ | ✗ −2.6 / ✗ −30.8 | 0.331 / 0.314 / 0.303 | +0.00002 |
| G1-2 | 73102 | ✓ ✓ ✓ ✓ | ✗ −1.7 / ✗ −6.4 | 0.495 / 0.332 / 0.419 | +0.00113 |
| G1-2 | 73103 | ✓ ✗ ✓ ✓ | ✗ −17.7 / ✗ −61.2 | 0.288 / 0.295 / 0.267 | +0.00001 |
| G1-2 | 73104* | ✓ ✓ ✓ ✓ | ✗ −10.2 / ✗ −55.1 | 0.456 / 0.408 / 0.429 | +0.00021 |
| G1-2 | 73105* | ✓ ✗ ✓ ✓ | ✗ −45.0 / ✗ −396 | 0.389 / 0.456 / 0.365 | −0.00070 |
| G1-2 | 73106* | ✓ ✗ ✓ ✗ | ✗ −90.7 / ✗ −298 | 0.358 / 0.363 / 0.336 | −0.00009 |
| G1-2 | 73107* | ✓ ✗ ✓ ✓ | ✗ −14.3 / ✗ −253 | 0.397 / 0.475 / 0.373 | −0.00084 |
| G1-2 | 73108* | ✓ ✓ ✓ ✓ | ✗ −4.0 / ✗ −24.1 | 0.329 / 0.311 / 0.346 | +0.00007 |
| G1-3 | 73101 | ✓ ✓ ✓ ✓ | ✗ −9.2 / ✗ −26.8 | 0.336 / 0.329 / 0.317 | +0.00015 |
| G1-3 | 73102 | ✓ ✗ ✓ ✓ | ✗ −25.7 / ✗ −330 | 0.349 / 0.362 / 0.342 | +0.00019 |
| G1-3 | 73103 | ✓ ✗ ✓ ✗ | ✗ −12.5 / ✗ −102 | 0.311 / 0.336 / 0.288 | −0.00039 |
| G1-3 | 73104* | ✓ ✗ ✓ ✓ | ✗ −11.3 / ✗ −2619 | 0.449 / 0.478 / 0.381 | −0.00035 |
| G1-3 | 73105* | ✓ ✗ ✓ ✓ | ✗ −46.9 / ✗ −165 | 0.313 / 0.304 / 0.282 | −0.00009 |
| G1-3 | 73106* | ✓ ✗ ✗ ✓ | ✗ −2.1 / ✗ −12.7 | 0.370 / 0.430 / 0.347 | −0.00031 |
| G1-3 | 73107* | ✓ ✗ ✓ ✓ | ✗ −3.3 / ✗ −13.2 | 0.421 / 0.490 / 0.379 | −0.00056 |
| G1-3 | 73108* | ✓ ✗ ✓ ✓ | ✗ −40.7 / ✗ −244 | 0.362 / 0.378 / 0.283 | +0.00010 |

Authority retention vs V3 G0 is 1.20–1.97 in every run. Endpoint survival
is 1.00 everywhere except G1-3 s73103 (0.75, S endpoint).

## Aggregates

| | T | A | O | S | critic A | critic S | critic T | critic O |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **10 new (primary)** | 10 | **2** | 9 | 9 | **0** | **0** | 10 | 3 |
| new G1-2 (5) | 5 | 2 | 5 | 4 | 0 | 0 | 5 | 2 |
| new G1-3 (5) | 5 | 0 | 4 | 5 | 0 | 0 | 5 | 1 |
| all 16 (mixed normalizer) | 16 | 4 | 15 | 14 | 0 | 0 | 16 | 4 |

## Verdict

A passes in **2/10** new seeds, which falls in the band fixed in advance as
**"effectively a systematic A failure"** under the current formulation. The
earlier 2/6 was not an unlucky draw. In 7 of the 10 new seeds, A-heavy has a
*higher* ‖ω_xy‖ than center (ΔJ_A < 0), and O-heavy reaches a lower ‖ω_xy‖
than A-heavy in 13 of the 16 runs.

T (10/10), O (9/10) and S (9/10) endpoint semantics are reproducible. The
critic is valid for T in every run, for O in 3/10, and for A and S in none.
Authority exceeds the Phase-1 reference everywhere, so the policy does
respond to preferences; what fails is specifically the A direction.

Together with the closed A diagnostics (not sign, not proxy, not
under-powering, not credit quality as a seed separator), this is now
characterized as a systematic A limitation of V4-C's formulation, not seed
bad luck.

## Next decision (per contract)

Accept the limitation and write V4-C up as is, or open a redesign (V4-D /
V5) aimed at A specifically. Undecided.
