# Teacher V4 — V4-C2R Switch-Controlled Semantic Effect Verdict

Status: **FROZEN — at steady state with the switch controlled, A is "correct" in 7/7 valid primary checkpoints (12/12 valid overall); T 7/7, S 7/7, O 5/7. Descriptive caveat: A-heavy beats T-heavy on A but is indistinguishable from O-heavy and S-heavy**
Date: 2026-09-27
Branch: `v4-c2-semantic-preservation`
Contract: [teacher-v4-c2r-switch-controlled-contract.md](../../contracts/teacher_v4/teacher-v4-c2r-switch-controlled-contract.md) (frozen at `45db3a8`, δ = 0.00209)
Script: `switch_semantics.py --mode test`; output `switch_semantics_test.json` in each V4-C run directory.

## Registered result

Fidelity invalid (|repeat-null mean| > δ): 4/16. These are G1-2 s73105,
G1-2 s73106 and G1-3 s73107 (primary), and G1-3 s73101. Valid: 7/10 primary,
12/16 overall.

Steady window (steps 33–128), D_i = mean over j ≠ i of S_i(w_i⁺) − S_i(w_j⁺),
simultaneous one-sided 95% bounds over the four objectives. Classes (+
correct, − wrong), D̄, and the m = 4 endpoint result [P/F]:

| run | T | A | O | S |
|---|---|---|---|---|
| G1-2 s73104* | + .486 [P] | **+ .092 [P]** | + .184 [P] | + .149 [P] |
| G1-3 s73104* | + .166 [P] | **+ .042 [F]** | + .664 [P] | + .040 [P] |
| G1-3 s73105* | + .449 [P] | **+ .083 [F]** | + .164 [P] | + .083 [P] |
| G1-3 s73106* | + .216 [P] | **+ .050 [F]** | − .364 [F] | + .040 [P] |
| G1-2 s73107* | + .338 [P] | **+ .065 [F]** | − .208 [P] | + .086 [P] |
| G1-2 s73108* | + .291 [P] | **+ .054 [P]** | + .462 [P] | + .119 [P] |
| G1-3 s73108* | + .131 [P] | **+ .029 [F]** | + .520 [P] | + .021 [P] |
| G1-2 s73101 | + .288 [P] | + .026 [F] | + .441 [P] | + .064 [P] |
| G1-2 s73102 | + .499 [P] | + .165 [P] | + .768 [P] | + .312 [P] |
| G1-3 s73102 | + .232 [P] | + .061 [F] | + .749 [P] | + .062 [P] |
| G1-2 s73103 | + .367 [P] | + .072 [F] | + .673 [P] | + .140 [P] |
| G1-3 s73103 | + .138 [P] | + .046 [F] | + .679 [P] | − .033 [F] |

(* primary. G1-3 s73103 excluded 73 snapshots that fell within 128 steps;
G1-2 s73107 excluded 2.)

| | T | A | O | S |
|---|---|---|---|---|
| valid primary (7) | 7 correct | **7 correct** | 5 correct, 2 wrong | 7 correct |
| all valid (12) | 12 correct | **12 correct** | 10 correct, 2 wrong | 11 correct, 1 wrong |

A class × endpoint (all valid): correct & endpoint pass 3, **correct &
endpoint fail 9**.

Per the interpretation fixed in advance: the positive controls pass (T, S,
and O in most checkpoints). **A is correct → the earlier A failures (m = 4
endpoints vs center, and the center-comparator test) came from the
center-relative comparison and the switch transient, not from A's own
steady-state direction.**

## Descriptive caveat: A's advantage comes from T, not from O or S

D_A averages three pairwise differences. Split, over the 12 valid
checkpoints:

| pairwise, steady | median | min | count < 0 |
|---|---:|---:|---:|
| S_A(A⁺) − S_A(T⁺) | +0.170 | +0.051 | 0 / 12 |
| S_A(A⁺) − S_A(O⁺) | −0.0003 | −0.031 | 6 / 12 |
| S_A(A⁺) − S_A(S⁺) | +0.0095 | −0.011 | 2 / 12 |

Asking for A gives far better angular stability than asking for T. It gives
the **same** angular stability as asking for O, and about the same as
asking for S. So the policy family separates "track fast" from "be stable",
but it has no A-specific direction beyond what O and S already produce. This
matches the earlier outcome-level A–O and A–S entanglement, and explains why
A-heavy stalls near ‖ω_xy‖ ≈ 0.33 rad/s in every run. The gate as frozen
(mean over j) counts that as correct. It is reported here, not re-gated.

## Other observations

- Transient window (steps 1–32) D is also positive for T and mostly for A
  and S under the switch control. Once every branch switches, the switching
  cost largely cancels.
- O is wrong at steady state in 2 checkpoints (G1-3 s73106, G1-2 s73107),
  and in 1 of them the O endpoint still passes.
- Fidelity invalid rose to 4/16 (from 3/16 in the center test). The longer
  128-step branches accumulate more position bias.

## Consequence for V4-C2

The V4-C "A fails 2/10" result does not mean A-heavy moves angular motion
the wrong way at steady state. It means that, relative to the center and
at endpoint scale, A adds nothing distinguishable from the stability that
O and S already buy. The semantic-preservation target is therefore
**objective specificity**, S_i(w_i⁺) > S_i(w_j⁺) for *every* j ≠ i, not just
direction against the center. For A that means specifically separating A
from O and S.
