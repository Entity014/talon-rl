# Teacher V4 — V4-C2F Objective Specificity / Identifiability Audit Contract

Status: **PREDECLARED — FROZEN 2026-09-27 before data collection (descriptive audit, no training)**
Branch: `v4-c2-semantic-preservation`

## Question

Do A (angular stability, ‖ω_xy‖) and O (orientation, tilt) correspond to
controllably distinct behaviors on the feasible locomotion distribution, or
does one subsume the other? The same measurements are made for every
objective pair, so the A–O answer is judged *relative to the other pairs*,
not against an absolute threshold.

## Data (same protocol as V4-C2R)

The 16 V4-C `model_300.pt` checkpoints. Per checkpoint: 512 snapshots
(256 envs × env seeds 0, 1) after 100 center warm-up steps, obs corruption
off. Branches are restored in place: a discarded burn-in, then five
128-step branches, center and heavy T/A/O/S, ordered by a cyclic shift of
[C, T⁺, A⁺, O⁺, S⁺] by (env + seed) mod 5. Every branch switches at t = 0
(center → center counts as a switch with no change). Steady window: steps
33–128. Snapshots with an episode end in any branch are excluded.

Scores: S_T, S_A, S_O, S_S (higher is better) per step.

## Layer 1 — state-space separability (per pair of objectives)

On per-step samples in the steady window (every 4th step, all branches):

- Spearman ρ between the two scores;
- median-split quadrant mass. Off-diagonal mass = fraction of samples good
  on one and bad on the other (0.5 = independent, 0 = identical ordering).

The same statistics are also computed on 32-step window means, since A is a
rate and O a level and their relation may show only over time.

## Layer 2 — feasible outcome frontier (per pair)

Points: per (checkpoint, branch preference, snapshot), the steady-window
mean of the two scores, standardized over the pooled cloud. Fraction of
variance on the first principal component (0.5 = isotropic 2-D, 1.0 = a
line). Also computed across policies only (per checkpoint × preference
means).

## Layer 3 — cross-objective controllability matrix

M[k, j] = mean steady-window S_k under heavy j, with the center as a column,
per checkpoint, plus bootstrap 95% CIs over snapshots. Specificity entries:
S_i(i⁺) − S_i(j⁺) for every ordered pair. For A/O, the block
AA − AO and OO − OA. Counts over checkpoints of CI > 0, CI < 0 or
overlapping 0.

## Reading, fixed now

- **B — A subsumed by O:** A–O is the most redundant pair on layers 1 and 2
  (highest |ρ|, lowest off-diagonal mass, highest PC1 among the six pairs),
  AA − AO is not positive in most checkpoints, and OO − OA is positive. → Do
  not add a specificity loss. Merge A into a stability objective, or
  redefine A so that it is orthogonal to O.
- **A — distinct but not realized:** A–O is not the most redundant pair on
  layers 1 and 2 (a separable feasible region exists), yet AA − AO ≤ 0 in
  most checkpoints. → A semantic-specificity loss is justified.
- **C — regime-dependent:** the layers disagree, or separability exists
  only in some checkpoints/windows. → The likely lever is state coverage or
  curriculum, not a loss.

The same reading is reported for A–S, which also showed low separation.
