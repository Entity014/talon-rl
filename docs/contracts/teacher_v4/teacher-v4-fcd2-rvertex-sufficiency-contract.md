# Teacher V4 — FC-D2 R-Vertex Task-Pressure Sufficiency Test

Status: **FROZEN 2026-09-30, with `fcd2_probe.py` and the `train_v4c.py --lambda-regions` flag, before any D2 branch was trained.**
Branch: `v4-c2-semantic-preservation`
Follows: [FC-D1 verdict](../../verdicts/teacher_v4/teacher-v4-fcd1-lambda-zero-verdict.md). Removing all task pressure from the mixed continuation removes the inversion (3/3). In 79102 and 79103, the task pressure in A sits almost entirely at the R vertex. In 79101, A also has λ_R⁺ ≈ 2.2 and λ_C ≈ 0.2.

## Question

Is R-vertex task pressure alone, within mixed training, sufficient to
reproduce the R⁺ inversion?

This is separate from whether the R-vertex target (tl ≥ 0.52 at w_R = 1) is
feasible. That question is FC-D3, with its own contract and gate.

## Treatment (one change against FC-C arm A)

| | A | **D2 (primary)** | D2p (secondary, 79101 only) |
|---|---|---|---|
| sampler, u0, run seed, PPO config, 50 iterations | FC-C | same | same |
| λ in the actor loss | every region | **R vertex only** | R⁺ only |
| dual updates | every region | every region (logged; only the listed λ is used) | same |

Flag: `--lambda-regions R-vertex` / `R+`. The default (all regions) is
bitwise the old path.

**Disclosed before running.** Over A's branch, λ outside the R vertex was:
- 79103: 0 except one iteration at 0.002 (O vertex);
- 79102: nonzero in 6 iterations (λ_R⁺ ≤ 0.36);
- 79101: nonzero in all 50 (λ_R⁺ ≤ 3.68, λ_C ≤ 1.07).

So in 79102 and 79103, D2 is a **near-repeat of A**: the losses differ by a
negligible amount, but chaotic divergence makes the branch a real second
sample. **Only 79101 tests sufficiency.** D2 in 79102 and 79103 is read as
a consistency / branch-noise check, not as extra votes.

## Measurement and endpoint

FC-C protocol, `fcc_probe.arm_audit` unchanged. Raw and matched Δ_R are
averaged over k = 30, 40, 50 and classified by the FC-C rule. Check: D2's
k = 0 replay must equal A's (difference 0.0).

## Pre-declared reading

| 79101 A | 79101 D2 | reading |
|---|---|---|
| inverted | inverted | **R-vertex task pressure sufficient** (within mixed training). If the D2 near-repeats in 79102 / 79103 are not both inverted, this is appended: "branch-sensitive" |
| inverted | not inverted | **not sufficient alone in 79101**: needs pressure in other regions (R⁺ / C) or A's dual interaction |
| inverted | split / collapse / unresolved | unresolved |

Descriptive only:
- D2p (R⁺ only, 79101): does the R⁺ pressure alone invert?
- The D2 near-repeats vs A in 79102 / 79103 (branch noise at fixed loss).
- λ trajectories per arm, Δtl, S_shared, Δ_O.

Next-step mapping (not decided here): if sufficient, the next steps are
FC-D3 (R-vertex feasibility) and then parameter-group attribution of the
R-vertex task update. If not sufficient, the next step is the R⁺/C pressure
interaction in 79101.

Scripts: `train_v4c.py --resume <u0> --iterations u0+50 --save-every 10 --lambda-regions R-vertex|R+` (FB-2a flags otherwise), `f2a_bifurcation.py --traces`, `fcd2_probe.py`.
Output: `runs/teacher_v4_fc-2026-09-30/fcd2/seed<s>/D2/`, `seed79101/D2p/`, `fcd2_probe.json`, `fcd2.out`.
