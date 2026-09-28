# Teacher V4 — F1 Locomotion Viability Verdict

Status: **FROZEN — REGION-LIMITED SUPPORT (all four margins). Locomotion never loses at the primary margin. It wins where T dominates and ties elsewhere. The best observed gentle gait scores above standing at C, so standing is not the objective's optimum against every gait. It is against M0's gait.**
Date: 2026-09-28
Branch: `v4-c2-semantic-preservation`
Contract: [teacher-v4-f1-locomotion-viability-contract.md](../../contracts/teacher_v4/teacher-v4-f1-locomotion-viability-contract.md) (frozen at `d8f322d`, before execution)
Output: `runs/teacher_v4_f1-2026-09-28/f1.json` (`f1_viability.py`, offline, 9 s)

## Behavior bank

126 behaviors. V4-C3: 19 standing, 4 step-in-place, 1 partial. V4-C: 70
standing, 7 step-in-place, 3 partial. M0: 22 established. There are only
**4 partial locomotion behaviors**, from 3 policies.

## Primary (m = 0.05)

| point | L − N | outcome | best per class: standing / step / partial / established |
|---|---|---|---|
| C | +0.031 | tie | 0.140 / 0.186 / 0.217 / −0.080 |
| T⁺ | +0.168 | **win** | 0.351 / 0.403 / 0.523 / 0.571 |
| D⁺ | −0.002 | tie | 0.060 / 0.078 / 0.077 / −0.287 |
| O⁺ | +0.003 | tie | 0.052 / 0.078 / 0.081 / −0.526 |
| {T} | +0.524 | **win** | |
| TD (.70/.30), (.30/.70) | +0.233, +0.022 | win, tie | |
| TO (.70/.30), (.30/.70) | +0.119, +0.028 | win, tie | |
| {D}, {O}, DO endpoints (T-free) | −0.04 to −0.03 | tie | reported, never gates |

Full support fails: C, D⁺ and O⁺ are ties, not wins. "Not supported" fails:
locomotion does not lose at C. **Reading: region-limited support.**

| support | win | tie | lose |
|---|---|---|---|
| uniform simplex | 32 % | 68 % | 0 % |
| m = 1, T-containing | 100 % | 0 | 0 |
| m = 2, T-containing | 63 % | 37 % | 0 |
| m = 3 | 27 % | 73 % | 0 |
| G1-1 mixture | 34 % | 66 % | 0 |
| G1-2 mixture | 30 % | 70 % | 0 |
| T-free sets (all m) | 0 | 100 % | 0 |

Sensitivity (the reading is region-limited under every margin):

| m | C | T⁺ | D⁺ | O⁺ | simplex win / lose |
|---|---|---|---|---|---|
| 0 | win | win | lose (−0.002) | win | 73 % / 27 % |
| 0.025 | win | win | tie | tie | 49 % / 4 % |
| 0.05 | tie | win | tie | tie | 32 % / 0 % |
| 0.10 | tie | win | tie | tie | 15 % / 0 % |

## Descriptive

- **A gentle gait exists that the objective does not penalize.** V4-C G1-2
  s73102 at C tracks 0.57 with R = [0.72, −0.04, −0.03], against M0's
  [1.09, −0.46, −0.95]. Its D and O cost is about a tenth of M0's. At C it
  beats the best standing behavior by 0.077 and the best non-locomoting one
  (a step-in-place behavior) by 0.031.
- **Score is monotone up to partial, then drops.** At C and at O⁺:
  standing < step-in-place < partial, and established (M0) is the lowest.
  At D⁺, partial ≈ step-in-place. The objective prefers moderate
  locomotion to both standing and M0's aggressive gait.
- **4-class winner map, uniform simplex:** partial 59 %, step-in-place 21 %,
  established 13 %, standing 7 %.
- **Step-in-place scores high on T** (best T 0.58, above standing's best of
  0.55), probably through yaw tracking in place. This was not decomposed.

## What this changes

The substrate-attribution verdict said standing is the objective's better
answer over most of the simplex. That was computed against M0's gait only,
and the verdict flagged that a gentler gait might score better. F1 shows one
does. Against the observed envelope, the frozen T/D/O formulation never
prefers non-locomotion beyond the margin. It prefers locomotion clearly
only where T dominates. Elsewhere the landscape is **nearly flat** between
standing, stepping in place and gentle walking (|L − N| ≤ 0.04).

So the V4 standing collapse is not explained by the objective alone. The
more likely reading is a weak locomotion incentive (a flat landscape at
most weights) combined with gait discovery. Gentle walking was reached by
only 3 of 22 policies. This favors the gait-shaping / optimization branch
(b) over a formulation rewrite (a), with the caveat below.

**Caveats.** The locomoting envelope rests on 4 behaviors from 3 policies,
and one of them (V4-C G1-2 s73102) decides most of it. The ties are small
in absolute terms: a margin of 0.05 is a practical choice, not a
statistical bound, and at m = 0 the map flips at D⁺ by 0.002.
