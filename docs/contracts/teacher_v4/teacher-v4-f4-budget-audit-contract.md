# Teacher V4 — F4 Budget Adequacy Audit (B0, 300 → 600)

Status: **FROZEN 2026-09-28 in the commit that adds this file, before any F4 training.**
Branch: `v4-c2-semantic-preservation`
Follows: [F3 verdict](../../verdicts/teacher_v4/teacher-v4-f3-substrate-screen-verdict.md) (the F3 verdict is unchanged)

## Question

Does additional optimization under the unchanged B0 formulation, starting
from the 300-iteration training state, recover translation?

Why 300 is a suspect boundary: the F2-A walker (s73102) first translated
at checkpoint 300, and the new-seed B0 runs step in place at T⁺ 3/3, with
s74103 at tl 0.40, one hundredth below partial.

F3 asked whether a shared substrate helps at a fixed budget (no). F4 asks
whether the fixed budget itself censors late gait discovery.

## Treatment

B0 of F3 only (`--shared none`), seeds 74101, 74102, 74103. Everything else
is frozen: {T, D, O}, divisors, PPO shell, log_std and entropy settings,
cardinalities {2, 3}, env, normalizer behavior.

**Controlled restart, not an exact continuation.** `train_v4c.py --resume`
loads model, actor and critic optimizers, LR and the e_t normalizer from
`model_300.pt`, then trains iterations 301–600 (checkpoints 350 … 600).
Env state, the realized preference sequence and RNG state are not in the
checkpoint. The restart draws env and sampler randomness from seed +
1 000 000, and every env starts a fresh episode at the restart. A positive
result means "300 updates were insufficient from that state". It never
means "the original uninterrupted run would have walked".

Smoke (4096 envs, s74101, 10 iterations): the LR dips briefly after the
restart (min 3.4e-5) and returns to the original run's final range
(1.7e-4 – 5.8e-4) within 4 iterations. Critic EV stays > 0.8.

## Measurement

The F2-A replay protocol on checkpoints 350 … 600, joined with the F3 B0
replays of 50 … 300. The ladder is 50 … 600. Persistence follows the F3 r2
rule: at least two consecutive checkpoints through 600. Translation at 600
only is "late, persistence unresolved" and does not count.

**Integrity (added, can only invalidate):** a seed whose LR sits at the
adaptive floor (≤ 1.1e-5) for more than half of the 300 added iterations is
optimizer-stalled and excluded. If fewer than 2 seeds remain, the audit is
invalid.

## Pre-declared reading (primary = C)

| valid seeds with persistent C translation by 600 | reading |
|---|---|
| ≥ 2/3 | **budget-sensitive** |
| 1/3 | **rare-late** |
| 0/3 | **no budget signal**, so the actor/exploration contract opens |

Also reported: t_step → t_translate per seed, T⁺ vs C order, tl per
checkpoint (to see approach to the 0.40 boundary), and the final R.

An extension to 1000 iterations is not part of F4. It needs its own
contract, and is considered only if the result is rare-late or clearly
approaching the boundary.

Code: `--resume` in `train_v4c.py`, aggregate `f4_budget.py`. Runs:
`runs/teacher_v4_f4-2026-09-28/none_seed<seed>/` (+ `replay/`).
