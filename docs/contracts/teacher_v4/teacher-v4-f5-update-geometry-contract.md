# Teacher V4 — F5 Actor-Update Geometry Screen (A0 vs A2)

Status: **FROZEN 2026-09-28, then PAUSED by user decision before any result.** One partial run (A0 s75101, stopped at iteration 69, model_50 only) exists and is not used. The objective-layer feature-selection round comes first; F5 resumes, if at all, from a fresh launch.
Branch: `v4-c2-semantic-preservation`
Follows: [F4 verdict](../../verdicts/teacher_v4/teacher-v4-f4-budget-audit-verdict.md), [F5-0 dose audit](../../verdicts/teacher_v4/teacher-v4-f5-0-actor-dose-audit.md)

## Question

Does an actor-update treatment raise the chance of discovering an
**objective-compatible low-D gait**, or does it only produce more
D-costly translation?

F4 separated locomotion discovery from useful locomotion discovery. With
more budget, T⁺ translated in 2/3 runs, but at R_D −0.34 / −0.48; the rare
walker s73102 had about −0.07, and C did not follow.

## Arms

| arm | desired_kl | entropy_coef | everything else |
|---|---|---|---|
| A0 | 0.01 | 0.01 | frozen |
| A2 | **0.02** | 0.01 | identical to A0 |

A2 relaxes the adaptive-KL controller in the direction associated with the
rare walker (higher LR, KL and clip fraction than its matched controls in
F5-0), allowing larger policy updates. The rule (rsl_rl, verbatim) multiplies
LR by 1.5 when KL < d/2 and divides it by 1.5 when KL > 2d. So A2 moves
the band from (0.005, 0.02) to (0.01, 0.04). **The achieved LR, KL and clip
fraction are measured outcomes. A2 does not claim to reproduce the walker's
trace, and matching it is not a success criterion.**

No stochasticity arm: F2-A found log_std and entropy nonspecific, and F5-0
found the walker *less* stochastic than its controls.

Frozen: {T, D = A, O}, T3-B divisors, TeacherV4, PPO shell (except
desired_kl), all stop gates, the post-fix normalizer, V4-C env, no R_shared,
G1-1 cardinalities {2, 3}, K = 3. **600 iterations from scratch**, 4096 envs,
checkpoints every 50. Training starts from scratch because the F2-A /
F5-0 difference lies at iterations ~20–40.

Seeds: screen **75101–75105** for both arms (paired: same seed per pair,
nominally matched init and env RNG; not bit-paired under GPU physics).
Confirmation (only if screen-positive): **75201–75205**, both arms.

## Phenotype (thresholds reused from F1 / F3; none new)

Per checkpoint and condition, from the F2-A replay:

| phenotype | rule |
|---|---|
| N (no translation) | tl < 0.40 |
| H (costly translation) | tl ≥ 0.40 and (R_D < −0.23 or R_O < −0.48) |
| L (low-cost, objective-compatible translation) | tl ≥ 0.40, R_D ≥ −0.23, R_O ≥ −0.48 |

**Primary: persistent L at C**, meaning L on at least two consecutive
checkpoints through 600 (F3 r2 rule). A run stopped by an integrity gate
counts as not-L and is reported separately.

## Pre-declared reading

Let k_A0 and k_A2 be the number of seeds (out of 5) with persistent L at C.

| outcome | reading |
|---|---|
| k_A2 ≥ 2 and k_A2 ≥ k_A0 + 2 | **A2 screen-positive**, go to confirmation on 75201–75205 (same rule) |
| k_A2 ≤ k_A0 | **update-geometry hypothesis not supported** |
| otherwise | **inconclusive** |

## Secondary (mechanism, descriptive)

T⁺ final and persistent phenotype, N / H / L counts per arm, and C and T⁺
ladders per seed, read as follows:

- A2: T⁺ H rises, C stays N → larger updates speed discovery but reach the
  wrong basin;
- T⁺ H → L → update geometry affects the gait phenotype;
- T⁺ L, then C L → strongest support;
- T⁺ L, C N → the gait exists, but transfer across w stalls;
- A2 ≈ A0 → not supported.

**Integrity reporting** (window means 1–100, 101–300, 301–600): KL, clip
fraction, LR, log_std, termination fraction, preference authority. The
existing stop gates stay.

Code: `--desired-kl` in `train_v4c.py`, aggregate `f5_screen.py`. Runs:
`runs/teacher_v4_f5-2026-09-28/<arm>_seed<seed>/` (+ `replay/`).
