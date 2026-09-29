# Teacher V4 — FB-0 Task-Pressure Calibration (read-only)

Status: **FROZEN 2026-09-29 before computation.** Read-only on the FA behavior bank. It fixes α_T for FB-1 before any FB-1 training. α_T is never revised on FB-1 results.
Branch: `v4-c2-semantic-preservation`
Design: [behavior–preference hierarchy](../../methods/general/behavior-preference-hierarchy.md)

## Formulation (fixed-task)

    J(w) = α_T · T̃ + (1 − α_T) · Σ_{i∈P} w_i · R̃_i,     Σ w_i = 1,  P = {R, O}

α_T is a **system parameter, not a user preference**. λ_T = α_T / (1 − α_T).
T̃, R̃, Õ use the frozen T3-B divisors (T 1.71946, R 0.15591, O 0.01563).
No new normalization. V is not in FB (unresolved realization, F8).

## Data

The FA bank: 47 behaviors, 10 locomoting (tl ≥ 0.40), steady-window levels
(`runs/teacher_v4_fa-2026-09-29/fa_geometry.json` sources).
Non-locomoting = standing ∪ step-in-place.

## Rule

α on a 0.01 grid. For each α, compute J at three preference points: the
R vertex (1, 0), the O vertex (0, 1) and the center (0.5, 0.5). The point is
feasible if the best locomoting J exceeds the best non-locomoting J by more
than **m = 0.05** (the F1 practical margin, in normalized units).

**α_T = the smallest α at which all three points are feasible.**

Reported descriptively: the smallest α per point, the smallest α that
makes the whole w_R ∈ [0, 1] segment (0.01 grid) feasible, and λ_T.

## Limits

Observed behaviors only. Feasibility over the bank is necessary, not
sufficient, for FB-1 policies to locomote. It sets the task pressure
without looking at any preference-authority result.

Script: `fb0_alpha.py`. Output: `runs/teacher_v4_fb-2026-09-29/fb0_alpha.json`.
