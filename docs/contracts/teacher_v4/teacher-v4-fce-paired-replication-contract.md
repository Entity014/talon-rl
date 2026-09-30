# Teacher V4 — FC-E Paired Causal Replication of Task-Pressure Contrasts

Status: **FROZEN 2026-09-30, with `fce_probe.py` (sequential rule and its self-check) and `train_v4c.py --branch-seed`, before any FC-E branch was trained.**
Branch: `v4-c2-semantic-preservation`
Follows: [FC-D2 verdict](../../verdicts/teacher_v4/teacher-v4-fcd2-rvertex-sufficiency-verdict.md). Two near-identical branches differed by about 7 points of Δ_R, while the effects under study are about 10 points or more. The current claim is: "task pressure during mixed-preference training is involved in the R inversion. The region-specific mechanism is not yet stable enough to name." No further mechanism or attribution step (FC-D3, parameter groups) runs until these contrasts replicate.

## Contrasts

The start checkpoints u0 are the FC-C ones: 79101 → 450, 79102 → 300,
79103 → 350. Branches run 50 iterations and are saved every 10.

| contrast | seed | arm 1 | arm 2 | role |
|---|---|---|---|---|
| task@79101 | 79101 | A (λ all regions) | D1 (λ ≡ 0 in loss) | primary |
| task@79103 | 79103 | A | D1 | primary |
| region@79101 | 79101 | D2p (λ R⁺ only) | D2 (λ R vertex only) | primary |
| task@79102 | 79102 | A | D1 | secondary (original effect inside the noise band) |

The arm flags are as in FC-C, D1 and D2 (`--loss-arm pref`,
`--lambda-regions R+|R-vertex`).

## Pairing (common random numbers)

Repeat r ∈ {1, …} runs **both arms of a contrast with the same branch seed**
(`--branch-seed r`, run seed = seed + 10⁶ + 1000 r). That fixes the env
reset sequence, the first rollout, the initial preference samples and the
generator stream. The arms differ only in the loss mask.

*Disclosed:* the random streams stay aligned only until the trajectories
diverge. The number of resampled episodes differs, so later draws shift.
Pairing reduces the noise; it does not remove it.

The original branches (r = 0: FC-C / D1 / D2) have already been seen. They
are reported alongside and are **not counted** among the repeats.

## Endpoint

Per branch, the endpoint is Δ_R averaged over k = 30, 40, 50 of the viable
checkpoints, computed by `fcc_probe.arm_audit` (unchanged), once raw and
once matched.

Per repeat, the **paired effect** is Δ_R(arm 1) − Δ_R(arm 2), in
percentage points. **Matched is primary**, and raw is reported alongside.
A pair is invalid if either branch collapses or has no matched value.
Binary inversion labels are reported but do not decide anything.

## Sequential rule (frozen)

**Stage 1**: repeats r = 1, 2, 3.

| condition on the 3 paired matched effects | class |
|---|---|
| all 3 valid, all > 0, median > +10 pp | replicated (stop) |
| all 3 valid, all < 0, median < −10 pp | reversed (stop) |
| otherwise (mixed sign, median ≤ 10 pp, or an invalid pair) | open: expand |

**Stage 2**: only for the open contrasts, add r = 4, 5. Classify on all 5:

| condition | class |
|---|---|
| ≥ 4 valid, ≥ 4 positive, median > +10 pp | replicated |
| ≥ 4 valid, ≥ 4 negative, median < −10 pp | reversed |
| otherwise | not replicated |

There is no stage 3. The rule is applied by the script and never extended
after results are seen.

## Pre-declared reading

- **Task pressure matters:** named if task@79101 **and** task@79103 are
  both replicated. If only one is, the reading is "seed-dependent". If
  neither is, "not replicated".
- **R⁺ pressure over R-vertex pressure (79101):** named if region@79101 is
  replicated.
- task@79102 is secondary. It is reported, but it is not needed for either
  reading.

Descriptive only:
- the SD of the paired effects;
- raw vs matched agreement;
- the r = 0 originals against the repeats;
- per-arm inversion labels, Δtl, Δ_O and λ trajectories.

Next-step mapping (not decided here): the region contrast replicated leads
to region-level mechanism / parameter attribution for R⁺ task pressure.
The task contrast replicated without the region contrast leads to a
formulation-level treatment of task pressure under mixed training. Neither
replicated means the diagnosis stops at "mixed training is needed" (FC-C),
with the task-pressure role unresolved.

Scripts: `train_v4c.py --resume <u0> --iterations u0+50 --save-every 10 --branch-seed r [arm flags]`, `f2a_bifurcation.py --traces`, `fce_probe.py --stage 1|2`.
Output: `runs/teacher_v4_fc-2026-09-30/fce/seed<s>/<arm>_r<r>/`, `fce_stage1.json`, `expand.txt`, `fce_stage2.json`, `fce.out`.
