# Teacher V4 — FC-D1 Task-Pressure Ablation of the Mixed Continuation

Status: **FROZEN 2026-09-30, with `fcd1_probe.py`, before any D1 branch was trained.** No trainer code change: D1 uses the existing FC-C flag `--loss-arm pref`.
Branch: `v4-c2-semantic-preservation`
Follows: [FC-C verdict](../../verdicts/teacher_v4/teacher-v4-fcc-continuation-verdict.md). The mixed continuation (arm A) inverts R⁺ in 3/3 seeds. No fixed-R⁺ arm does. The verdict listed three confounds between A and B. This contract isolates item 3: task pressure in other regions (λ at the R vertex is 8–14 at u0).

## Question

Is the task-dominated update in other regions necessary for the R⁺
inversion under mixed training?

## Treatment (one change against FC-C arm A)

| | FC-C arm A | FC-D1 arm D1 |
|---|---|---|
| start checkpoint u0 | 450 / 300 / 350 | same |
| conditioning w | FB-2a mixed sampler | same |
| actor loss weights [T_lin, R, O] | [λ_region, w_R, w_O] | **[0, w_R, w_O] in every region** |
| critic | fits T, R, O | same |
| run seed, PPO config, 50 iterations, save every 10 | as FC-C | same |

The dual variables are still computed and logged, but they have no effect
on the update. That is equivalent to λ ≡ 0 in the loss. Since the run seed
is the same, the first rollout of D1 is identical to A's. The two branches
differ only through the loss from the first update on.

Arm A is not re-run. Its FC-C replay is the comparison.

## Measurement and endpoint

Same as FC-C, reusing `fcc_probe.arm_audit` unchanged:
- raw and matched Δ_R, averaged over k = 30, 40, 50;
- inverted, not inverted, split, or unresolved / collapse by the FC-C rule
  (±5 %, both measures needed).

Check: D1's k = 0 replay must equal A's (difference 0.0), or nothing is
read.

## Pre-declared reading (per seed)

| A (FC-C) | D1 | reading |
|---|---|---|
| inverted | inverted | task pressure in other regions is **not necessary**; what remains is cross-preference updates / dilution |
| inverted | not inverted | task-dominated updates in other regions **contribute** |
| inverted | split / collapse / unresolved | unresolved |

A reading is named when ≥ 2/3 seeds give it. Otherwise the per-seed
readings are reported.

Disclosed:
- **λ differs by seed at u0.** In 79103, only the R vertex has λ > 0 (7.96).
  In 79101, λ is also nonzero at C (1.07) and R⁺ (3.68). So D1 removes task
  pressure from more regions in 79101.
- **One branch per arm, with no repeat.** A single D1 at the gate is
  reported as such, not re-run.
- **Removing T at every region also removes it where tracking is weak.**
  Tracking may erode (FC-C C/D in 79101). A collapse is "unresolved", never
  "not inverted".

Descriptive only: the λ trajectory in A over the branch, S_shared, Δtl,
Δ_O, and online tl per region.

Next-step mapping (not decided here): if task pressure is not necessary, the
next step is dilution vs specific interference (R⁺ quota, leave-one-region-
out). If it contributes, the next step is to locate which region's task
updates matter.

Scripts: `train_v4c.py --resume <u0> --iterations u0+50 --save-every 10 --loss-arm pref` (FB-2a flags otherwise), `f2a_bifurcation.py --traces`, `fcd1_probe.py`.
Output: `runs/teacher_v4_fc-2026-09-30/fcd1/seed<s>/D1/`, `fcd1_probe.json`, `fcd1.out`.
