# Teacher V4 — V4-C2 Null-Calibrated Preference-Effect Verdict

Status: **FROZEN — registered result reported as-is; A and S results confounded by a preference-switch transient the design did not control (found after the run); T positive control passes**
Date: 2026-09-27
Branch: `v4-c2-semantic-preservation`
Contract: [teacher-v4-c2-semantic-relation-contract.md](../../contracts/teacher_v4/teacher-v4-c2-semantic-relation-contract.md) (frozen at `c1380f3`)
Script: `preference_effect.py` (commit `fa2dcba`); output `preference_effect.json` in each V4-C run directory.

## Registered result

Fidelity: 3/16 checkpoints are **fidelity invalid** (|null mean| > 0.002):
G1-2 s73104 (primary), G1-2 s73101 and G1-2 s73102. All three are in the
noisier G1-2 fold. They are not interpreted. Valid: 9/10 primary, 13/16 all.

Per-H class (H = 1, 2, 4, 8, 16, 32; + correct, − wrong, . flat), with the
checkpoint's m = 4 endpoint result (P/F) from the seed-sensitivity
evaluation:

| run | T | A | O | S |
|---|---|---|---|---|
| G1-2 s73104* | invalid | | | |
| G1-3 s73104* | `++++++` P | `------` F | `------` P | `------` P |
| G1-2 s73105* | `++++++` P | `------` F | `----..` P | `-----.` P |
| G1-3 s73105* | `++++++` P | `------` F | `------` P | `------` P |
| G1-2 s73106* | `++++++` P | `------` F | `------` P | `------` F |
| G1-3 s73106* | `++++++` P | `------` F | `------` F | `----.+` P |
| G1-2 s73107* | `++++++` P | `------` F | `+++.--` P | `------` P |
| G1-3 s73107* | `++++++` P | `------` F | `++++++` P | `------` P |
| G1-2 s73108* | `++++++` P | `------` P | `+.----` P | `------` P |
| G1-3 s73108* | `++++++` P | `------` F | `++++++` P | `------` P |
| G1-3 s73101 | `++++++` P | `------` P | `+....-` P | `------` P |
| G1-3 s73102 | `++++++` P | `------` F | `-.++++` P | `------` P |
| G1-2 s73103 | `++++++` P | `------` F | `++++++` P | `------` P |
| G1-3 s73103 | `++++++` P | `------` F | `++++++` P | `------` F |

(* primary)

Frozen ladder classes, over the 9 valid primary checkpoints:

| objective | class |
|---|---|
| T | local correct, persistent: **9/9** |
| A | "no detectable semantic effect": 9/9. More precisely, significantly **wrong** at all six H in 9/9 |
| O | no effect 5, locally correct then destroyed 2, persistent 2 |
| S | no effect 8 (wrong at most H), emerges from multi-step dynamics 1 |

Against the across-objective rules: T is detected, so the measurement is not
"suspect" under the frozen rule. The "T/O/S correct, A not" pattern does not
hold, because O is mixed and S is mostly wrong.

## Confound found after the run: the preference-switch transient

The effect curves for A and S have the same shape in every valid checkpoint:
largest at H = 1 and decaying monotonically toward zero by H = 32. Examples of
D̄ over H = 1…32:

    G1-3 s73105   A: −0.226 −0.204 −0.138 −0.086 −0.050 −0.025   S: −0.799 −0.472 −0.266 −0.142 −0.069 −0.029
    G1-2 s73108   A: −0.138 −0.100 −0.071 −0.060 −0.057 −0.036   S: −0.624 −0.424 −0.257 −0.139 −0.064 −0.009
    G1-3 s73101   A: −0.109 −0.101 −0.102 −0.073 −0.056 −0.026   S: −0.299 −0.203 −0.136 −0.090 −0.050 −0.023

The treatment branch *switches* preference at the snapshot, and the center
branches do not. The switch produces an immediate action jump, which S
scores directly as roughness, and a body-rate transient, which A scores as
‖ω_xy‖. Both decay over about 0.6 s. The null (center vs center) contains no
switch, so it cannot remove this. The clearest evidence is S: in 10/13
checkpoints its endpoint passes (heavy S is smoother than center over
constant-preference rollouts), yet here it is "wrong" at almost every H. T
is immune because the tracking effect is immediate and large (+0.18 at
H = 1, growing).

Consequence: **for A and S, this test measured the switch transient plus
the semantic effect, not the semantic effect alone.** The registered A
result ("wrong at every H") cannot be read as "A-heavy makes angular motion
worse at steady state". H = 32 (0.64 s) does not reach past the transient.
The design flaw is the missing control for the switch itself. The
simulator's noise was not the problem: fidelity held in 13/16.

## What remains valid

- The branch-noise calibration works: 13/16 checkpoints met δ = 0.002, and
  T is detected at every H in every valid checkpoint.
- For T, same-state preference authority is semantically correct, locally
  and over 32 steps.

## Next (requires a new, preregistered contract)

Remove the switch confound by design, not by reinterpretation. Options:

1. **Switch in every branch.** Compare w_i⁺ with another switched preference
   (e.g. S_i(w_i⁺) vs S_i(w_j⁺) for j ≠ i, or vs a switched center-like
   control), so both sides carry the transient.
2. **Measure past the transient.** Score only a post-switch window (e.g.
   steps 32–128 after the switch), with a null for that window.
3. Both.
