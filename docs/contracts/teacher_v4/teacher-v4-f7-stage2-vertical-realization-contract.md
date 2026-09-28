# Teacher V4 — F7 Stage 2: Vertical-Axis Realization (selection, read-only)

Status: **FROZEN 2026-09-28 (r2) in the commit that sets this line, before the stage-2 traces and before any target computation.**
Branch: `v4-c2-semantic-preservation`
Follows: [F7 stage 1 map](../../verdicts/teacher_v4/teacher-v4-f7-phenotype-map.md)

## Semantic decision recorded (2026-09-28, user)

Working semantic hypothesis: **case 3**. Dynamic Stability splits into
**R** (rotational stability) and **V** (vertical stability), with candidate
set {T, R, V, O}. This is not a frozen objective set. F7 stage 1 shows R
and V are phenomenologically distinct (a trade-off, not redundancy). It
does not show that V is preference-controllable. Cases 1 (α-blend) and 2
(vertical constraint) are fallbacks if the controllability screen (F8)
fails. Case 1 is not chosen, because α would hide a preference decision
inside the objective. Case 2 is not chosen, because there is no requirement
that forbids vertical motion, and dense lin_vel_z at stock scale suppressed
all stepping (F3). A future case 2 would use a bound (|v_z| < v_max), not a
dense penalty.

R realization: A = `ang_vel_xy_l2`, unchanged. It measures rotational
dynamics by definition.

## Question (stage 2)

Which per-step signal should realize V, chosen against a physical vertical
target that no candidate defines?

## Candidates

| id | per-step signal | note |
|---|---|---|
| V1 | lin_vel_z_l2 = v_z² (body frame, stock term) | stock; the simplest |
| V2 | base_lin_acc_z_l2 = (Δv_z world / dt)² | measurement library |
| V3 | body_height_osc_l2 = (z − EMA₀.₅ₛ(z))² | measurement library; oscillation, not posture. It replaces base_height_l2, which penalizes a steady offset (O-like) and whose default target is the spawn height |

The traces also log the whole measurement library
(`talon_rl/rewards/measurements.py`: slip, impact, joint velocity, action
magnitude, joint power) for the later feature/role screen. Stage 2 uses
only V1–V3.

## Target and data

Target: per 32-step window, the peak-to-peak of the linearly detrended
**world root height z** (simulator state, not derived from any candidate).
V1 and the target are physically linked (height is the integral of v_z).
That is physics, not label circularity: no candidate is used to assign
labels.

Traces: the substrate-attribution protocol plus the measurement-library
columns (root z world, v_z world, and the candidates). Checked: the existing
columns reproduce the F6 s73102 traces bit for bit. Behaviors: the same 8 as F7 stage 1 (selection data).

**Disclosure.** The reproduction check (s73102 run) printed steady means of
the new columns for s73102 C, s73102 T⁺ and M0, so 3 of the 5 translating
behaviors' candidate levels have been seen: base_lin_acc_z_l2 51 / 62 / 8.7,
body_height_osc_l2 3.1e-4 / 4.1e-4 / 0.8e-4. The F4 behaviors and the target
(root-z excursion) have not been computed for any candidate.

## Pre-declared selection rule

Candidates are read as non-negative costs. Translating behaviors: s73102 C,
s73102 T⁺, F4 s74102 T⁺, F4 s74103 T⁺, M0. Policies: s73102, s74102,
s74103, M0 (s73102's two behaviors are one policy).

1. **Admissibility:** per policy (s73102 = max over its two behaviors),
   per-step |Spearman(candidate, ang_vel_xy cost)| < 0.7 **and**
   |Spearman(candidate, flat_orientation cost)| < 0.7.
2. **Primary: behavior-level Kendall τ** between candidate and target
   behavior medians over the 5 translating behaviors. Its question is
   cross-gait / cross-policy ordering. With 5 points, τ is coarse and ties
   are expected. That reflects thin cross-policy evidence and is **not**
   repaired with window pseudo-samples.
3. Tie on τ → **policy-blocked within-behavior Spearman.** Per behavior,
   Spearman(candidate window mean, target window) is computed. Each policy is
   the median of its behaviors, and the median over the 4 policies decides.
   A difference < 0.02 counts as a tie.
4. Still tied → pooled-window Spearman (a descriptive tie-break only).
5. Still tied → the stock term V1.

If the choice is made only at step 4 or 5, or the chosen τ ≤ 0, the verdict
is **"vertical realization unresolved on current selection bank"**, with V1
named only as the default. If no candidate is admissible, the verdict is
"no admissible V realization", and the case-1 / case-2 fallbacks are
reconsidered.

Also reported: per-step density (share of non-negligible values) and
kurtosis (dense-reward conditioning).

The chosen V is a **candidate realization** that goes to F8 (controllability
on fresh training and seeds), with its divisor set by the T3-B protocol
before training. It is not an objective until F8 passes.
