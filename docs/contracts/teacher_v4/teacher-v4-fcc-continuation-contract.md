# Teacher V4 — FC-C Short Counterfactual Continuation (diagnostic branches)

Status: **FROZEN 2026-09-29, with `fcc_probe.py` and the `train_v4c.py` FC-C flags, before any FC-C branch was trained.** Pipeline smoke-tested on a scratch 2-iteration R-only branch, which was not replayed.
Branch: `v4-c2-semantic-preservation`
Follows: [FC-B verdict](../../verdicts/teacher_v4/teacher-v4-fcb-r-trajectory-verdict.md). Case B in 3/3 seeds: R⁺ inverts only after λ has fallen. R reduces rotation for the whole policy, and C drops as much as R⁺. So R behaves as a shared cost, not as a preference axis.

## Question

Which accumulated interaction turns R⁺ from not-inverted to inverted
relative to C? The candidates are the R stream itself, R/O interaction,
task interaction, and training across the mixed preference distribution.

No formulation change and no reward change. These are diagnostic branches.
They are not candidate policies.

## Branches

Start from each FB-2a seed's checkpoint before FC-B's u*: **79101 → 450,
79102 → 300, 79103 → 350**. Each branch runs **50 iterations** of real
training (fresh rollout, then PPO update, then fresh rollout), with the FB-2a
PPO config (4096 envs, desired KL 0.01). It restores the model, optimizers,
LR, e_t normalizer and **the checkpoint's λ vector** (`--resume`, now also
restoring λ). Checkpoints are saved every 10 iterations.

| arm | conditioning w | actor loss weights [T_lin, R, O] | question |
|---|---|---|---|
| A | FB-2a mixed sampler | [λ_region, w_R, w_O], live dual | does the plain continuation reproduce the inversion? |
| B | fixed R⁺ (0.7, 0.3) | [λ_R⁺, 0.7, 0.3], live dual | is training across preferences needed? |
| C | fixed R⁺ | [0, 0.7, 0.3] | is the task interaction needed? |
| D | fixed R⁺ | [0, 1, 0] | does the R stream alone accumulate the wrong way? |

The critic fits all three streams in every arm (loss_mask unchanged). Only
the actor surrogate weights change. The advantage scale is the std of
Σ ℓ_i A_i, so D's weight of 1 is the same step as 0.7.

Disclosed:
- **Controlled restart.** Env and sampler randomness is not in the
  checkpoint. Every branch uses run seed = seed + 10⁶. All four arms of a
  seed share it (common random numbers), and every env starts from a reset.
- **λ at u0 is not zero in every seed.** For 79101 at 450, λ_R⁺ = 3.68, so
  arm B still carries task pressure there.
- **One branch per arm per seed.** There is no repeat branch, so branch
  noise is bounded only by the three-checkpoint endpoint mean.

## Measurement

Each saved checkpoint, u0 included, is replayed with the FC-B protocol:
`f2a_bifurcation --traces`, reset seeds 910001 / 910002, steady window
33–128, the first 64 envs per reset seed, deterministic policy. The
conditions are C (0.5, 0.5), R⁺ (0.7, 0.3) and O⁺ (0.3, 0.7). Per
checkpoint k ∈ {0, 10, …, 50}:

- **raw Δ_R** = median window F_rate(R⁺) / median F_rate(C) − 1 (the FC-B
  metric);
- **matched Δ_R**: windows with window tl ≥ 0.40 from R⁺ and C are pooled
  into 8 quantile bins of window tl. In each bin with ≥ 5 windows per
  condition, take the ratio of median F_rate. Average these ratios weighted
  by the smaller count, then subtract 1. It needs ≥ 3 usable bins,
  otherwise it is missing. Synthetic check (in the script): with F_rate ∝
  tl², a pure tl shift of 0.2 gives |matched Δ_R| < 3 %, and a true +10 %
  gives +10 %;
- Δtl = tl(R⁺) − tl(C);
- S_shared = F_rate(C) and S_cond = F_rate(R⁺) − F_rate(C);
- Δ_O (O⁺ positive control).

A checkpoint is **viable** if replay tl(R⁺) ≥ 0.40 and tl(C) ≥ 0.40.

**Determinism check:** the k = 0 replay of every arm must equal FC-B's replay
of u0 (raw Δ_R difference reported). If it does not, the replay is broken
and nothing is read.

## Endpoint state per arm

The endpoint is k ∈ {30, 40, 50}, averaged over its viable checkpoints.
Fewer than 2 viable is **collapse**.

| state | condition |
|---|---|
| inverted | mean raw Δ_R ≥ +5 % **and** mean matched Δ_R ≥ +5 % |
| not inverted | both < +5 % |
| split | the two disagree (tracking-confounded) |
| unresolved | no matched value, or collapse |

Raw and matched Δ_R are read only as a pair.

## Pre-declared reading (per seed)

Walk D → C → B → A and stop at the first arm that is not "not inverted":

| first such arm | state | reading |
|---|---|---|
| D | inverted | the R stream's own accumulated learning is the problem |
| C | inverted | accumulated R/O interaction |
| B | inverted | task interaction |
| A | inverted | cross-preference / shared-parameter interference |
| any | split / collapse / unresolved | unresolved at that arm |
| none | — | historical drift not reproduced / unresolved |

If the named arm is not A and A is not inverted, the reading carries "arm A
did not reproduce the historical inversion". A source is named in the
verdict when ≥ 2/3 seeds give it. Otherwise the per-seed readings are
reported, with no majority imposed.

Descriptive only: the trajectories of S_shared and S_cond. "Shared
regularizer" is the pattern where F_rate(C) falls while S_cond stays near
zero. Also descriptive: Δ_O, λ and online tl per arm.

Case B from FC-B does not rule out path dependence from the high-λ phase.
Arms B and C bear on it but do not settle it.

Next-step mapping (not decided here): a named source leads to a targeted
formulation intervention. If no source is found, the next step is a direct
audit of the preference-conditioned representation and policy-family
separation.

Scripts: `train_v4c.py --resume <u0> --iterations u0+50 --save-every 10
[--fixed-w 0.7,0.3 --loss-arm full|pref|r]` (FB-2a flags otherwise). Also
`f2a_bifurcation.py --traces` per branch (u0 symlinked in as k = 0), and
`fcc_probe.py`.
Output: `runs/teacher_v4_fc-2026-09-29/fcc/seed<s>/<arm>/`, `fcc_probe.json`, `fcc.out`.
