# Teacher V4 — V4-C3 Verdict ({T, D, O} objective set)

Status: **FROZEN — OBJECTIVE LAYER NOT VALID. B passes in 0/6 runs because the MC256 critic is invalid for D (= A) in 6/6 runs. Without the critic gate, the semantic part of B (B1 + B2) passes in 2/3 (G1-1) and 1/3 (G1-2), and seen-support C in 0/3 and 1/3. Held-out D: 0/6.**
Date: 2026-09-28
Branch: `v4-c2-semantic-preservation`
Contracts: [training](../../contracts/teacher_v4/teacher-v4-c3-training-contract.md), [evaluation](../../contracts/teacher_v4/teacher-v4-c3-evaluation-contract.md) (frozen at `b258386`, δ₃ = 0.00260 from the K = 3 null before any semantics result)
Runs: `runs/teacher_v4_c3_g1_{1,2}_seed{73101,73102,73103}-2026-09-28/`. Aggregate: `runs/teacher_v4_c3-2026-09-28/aggregate.json` (`c3_aggregate.py`).

## Gates (frozen hierarchy)

A fidelity-invalid run is not interpreted and counts as not passing.

| run | fidelity | B1 + B2 anchor | B3 critic T / A / O | **B** | C seen | D held-out |
|---|---|---|---|---|---|---|
| G1-1 s73101 | **invalid** (O null 0.0050) | — | ✗ / ✗ / ✓ | ✗ | — | — |
| G1-1 s73102 | ok | ✓ | ✓ / ✗ / ✗ | ✗ | ✗ (m = 2) | ✗ (m = 1) |
| G1-1 s73103 | ok | ✓ | ✓ / ✗ / ✗ | ✗ | ✗ (m = 2) | ✗ (m = 1) |
| G1-2 s73101 | ok | ✓ | ✓ / ✗ / ✗ | ✗ | ✓ (m = 1) | ✗ (m = 2) |
| G1-2 s73102 | ok | ✗ (A: A⁺ − C = 0.000) | ✓ / ✗ / ✗ | ✗ | ✗ (m = 1) | ✗ (m = 2) |
| G1-2 s73103 | **invalid** (O null 0.0037) | — | ✓ / ✗ / ✓ | ✗ | — | — |

    A  training integrity          6/6 (training contract)
    B  full-set validity           G1-1 0/3, G1-2 0/3
    C  seen-support validity       G1-1 0/3, G1-2 1/3
    E  B and C in >= 2/3, both folds   FAIL
    D  held-out cardinality        G1-1 0/3, G1-2 0/3   (separate claim) FAIL

Objective-layer validity (A ∧ B ∧ C per E): **not established.**
Cardinality generalization: **not established.**

Sensitivity (post-hoc, registered): excluding G1-1 s73102 leaves B and C at
0 in G1-1. The verdict does not change.

## What fails

**1. The D (= A) critic, in every run.** MC256 EV for A: −7.69, −0.74, −2.75,
−19.06, −0.49, −8.26; negative fraction 0.44–0.81. This is the same failure
as V4-C ([G1-R](teacher-v4-c-g1r-critic-verdict.md)). Removing S did not fix
it. The O critic is also invalid in 4/6 runs (EV −2.14 to −0.65), including
all three semantically valid runs with B1 + B2 passing. The T critic is valid
in 5/6.

**2. D self-direction off the anchor.** B2 (identifiability) passes in every
group of every run (18/18): preferences never collapse. The B1 failures in C
and D are concentrated on D. Its singleton {A} fails S_A({A}) − S_A(C) > δ₃
in 3 of the 4 fidelity-valid runs (means −0.090 to +0.001). In m = 2, the
endpoints that raise A also fail B1 for A in 3/4 valid runs. At the anchor,
A⁺ − C on S_A is small even where it passes (+0.011, +0.026, +0.087, and 0.000
in the failing G1-2 s73102).

**3. Fidelity.** 2/6 runs are fidelity invalid, both on O (repeat null 0.0050
and 0.0037 against δ₃ = 0.0026). δ₃ came from two checkpoints. The repeat
null of O varies more between checkpoints than those two showed.

## Descriptive (not gated)

Anchor interaction matrix, fidelity-valid runs, steady window,
r_j = S(j⁺) − S(C) as [ΔS_T, ΔS_A, ΔS_O]:

| run | T⁺ | A⁺ | O⁺ |
|---|---|---|---|
| G1-1 s73102 | +0.19, −0.58, −0.22 | −0.01, +0.01, −0.22 | +0.02, −0.15, +0.15 |
| G1-1 s73103 | +0.56, −0.70, −0.10 | −0.07, +0.03, −0.03 | −0.02, −0.02, +0.37 |
| G1-2 s73101 | +0.35, −0.38, −0.16 | −0.06, +0.09, −1.25 | −0.03, +0.07, +0.22 |
| G1-2 s73102 | +0.05, −0.07, −0.45 | +0.02, 0.00, −0.42 | −0.07, +0.03, +0.19 |

- T and O have clear self-direction. D's own response is an order of
  magnitude smaller than T's and O's.
- T⁺ costs D in every run (conflict). A⁺ costs O in every run, up to −1.25
  (conflict), although A–O were distinct in the selection audit.
- Continuity (V4-C endpoint protocol, heavy vs center): T passes 6/6, O 5/6,
  A 0/6 (mean objective delta ≈ 0). V4-C had A at 2/10.
- Preference authority vs V3 G0: pairwise retention 1.24–1.88, tangent
  1.29–2.21, in all runs.

## Conclusion

Dropping S did not repair D. The model still has authority, and the three
preferences stay distinguishable (B2 18/18). But D has no reliable
self-direction off the anchor, and its critic cannot predict D returns in any
run. The failure moved with A from V4-C to V4-C3 unchanged. It is a property
of the D objective as currently realized (`ang_vel_xy_l2`, divisor 0.156) in
this training setup, not of the S objective or of the fourth objective slot.

This result does not re-select objectives
([no leakage](../../methods/general/objective-selection.md#separation-of-selection-and-evaluation-no-leakage)).
Any change to D, its realization or its critic target needs a new contract.
