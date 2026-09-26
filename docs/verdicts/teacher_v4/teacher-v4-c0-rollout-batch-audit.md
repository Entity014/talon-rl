# Teacher V4 — V4-C0 Rollout/Batch Contract Audit

Status: **AUDIT COMPLETE — recommendation: keep H=24, halve minibatch count, count budget in samples; trainer choice still open**
Date: 2026-09-26
Branch: `v4-a-teacher`

## Question

The canonical V4 env runs 2048 envs, not 4096 (see
[input-contract verdict](teacher-v4-a-b-input-contract-verdict.md)). Should
`num_steps_per_env` double to keep `N_env × H` equal, or stay the same?

Decision rule, agreed before the audit: double `H` only if it is purely a
collection length. If `H` changes temporal credit, bootstrap boundaries,
normalization, or PPO update geometry, keep `H` and restore the batch
contract elsewhere.

## Correction to the premise: which contract is "V3's"

The comparison baseline was assumed to be "4096 envs × H". The code shows
two different regimes in the V3 lineage:

| stage | where | N_env × H | samples/update | update recipe |
|---|---|---|---|---|
| **M0 root policy** | `runs/m0_1_seed0_2026-09-22/` (`model_299.pt`, root of `rv1_a_init` → V2 → Phase-1 → V3) | 4096 × 24 | 98,304 | stock `rsl_rl` A1 flat: 5 epochs × 4 minibatches, LR 1e-3 adaptive (desired KL 0.01), γ 0.99, λ 0.95, clip 0.2, 300 iterations = 29.49M samples |
| **Phase-1 / V2 / V3 fine-tunes** | e.g. `objective_set_g1_train.py`, `symmetric_training.py` | 8 × 32 | 256 | one full-batch step per update (the ratio is asserted to be ≈1), Adam 1e-3 constant, env reset every rollout, 30–75 updates, init from the previous stage |

The M0 sample count is confirmed by its artifact: `vector_rows = 29,491,200 = 300 × 4096 × 24`.

TeacherV4 has new shapes (a 16-D state bottleneck and an env encoder), so it
cannot start from the Phase-1 weights. **V4-C is therefore a from-scratch
run in the M0 regime.** The fine-tune regime (8 × 32) is not a comparable
budget for it. The batch question below is asked against M0: 4096 × 24,
5 × 4, adaptive KL.

## The five checks

### 1. Rollout buffer size = N_env × H

| option | N × H | samples/iteration |
|---|---|---|
| M0 | 4096 × 24 | 98,304 |
| A: keep H | 2048 × 24 | 49,152 |
| B: double H | 2048 × 48 | 98,304 |

### 2. GAE / return horizon — **H is not only a collection length**

In the `rsl_rl` runner and in MOPPO, rollouts continue across iterations,
so `H` does not truncate episodes. It does set where GAE stops and
bootstraps on `V(s_H)`. For a sample at step `t`, the bootstrap value
enters its advantage with weight `(γλ)^(H−t)`.

| γ, λ | effective horizon 1/(1−γλ) | H | mean bootstrap weight over the rollout | samples with bootstrap weight > 0.1 |
|---|---:|---:|---:|---:|
| 0.99, 0.95 | 16.8 | 24 | 0.508 | 100% |
| 0.99, 0.95 | 16.8 | 48 | 0.312 | 77% |
| 0.998, 0.95 | 19.3 | 24 | 0.549 | 100% |
| 0.998, 0.95 | 19.3 | 48 | 0.351 | 90% |

Doubling `H` cuts the average reliance on the critic's bootstrap by about 40%
(0.51 → 0.31) and moves the advantage estimator toward Monte Carlo. This
matters for V4 in particular. Each objective head `V_i` is bootstrapped
separately, and the lineage already has a documented critic-support
sensitivity (the 64-step support/ridge substrate). Changing `H` would change
both the architecture and the credit-assignment geometry at once.

**By the agreed rule, this alone argues for keeping H = 24.**

### 3. Minibatch construction

`rsl_rl` and MOPPO both fix the minibatch **count** (4), not the size.

| option | minibatch size | gradient steps/iteration (5 epochs) | gradient steps per 98,304 samples |
|---|---:|---:|---:|
| M0 (4096 × 24, 4 mb) | 24,576 | 20 | 20 |
| A, 4 mb | 12,288 | 20 | 40 |
| **A, 2 mb** | **24,576** | **10** | **20** |
| B (2048 × 48, 4 mb) | 24,576 | 20 | 20 |

Option A with the count unchanged doubles the number of gradient steps per
sample and halves the minibatch size, which means noisier gradients and a
noisier KL estimate. Option A with **2 minibatches** keeps the minibatch
size, the epochs per sample (5), and the gradient steps per sample, all
identical to M0.

### 4. Optimizer schedule

The `rsl_rl` adaptive LR reacts to the KL measured on each minibatch. It is
not tied to sample count or iteration index. Keeping the minibatch size at
24,576 (A with 2 minibatches, or B) keeps that KL estimate's noise level the
same as in M0. No schedule anneals by iteration count, and V3 used a
constant Adam 1e-3. The remaining effect of option A: the policy's trust
region is applied every 49k samples instead of every 98k. That means twice
as many PPO iterations per sample. Each iteration uses the same minibatch
geometry.

### 5. Logging / evaluation cadence

M0 checkpoints and the lineage's snapshots are keyed to the iteration or
update index (`model_299`, `SNAPS=(0,10,20,30)`). Under option A one
iteration holds half the samples. Every V4-C budget, checkpoint, and
comparison point must therefore be stated in **env samples**, with the
iteration index derived from it. For example, the M0 budget of 29.49M
samples is 600 iterations at 2048 × 24.

## Recommendation

**Option A': H = 24, num_mini_batches = 2, budget stated in samples.**

| property | vs M0 |
|---|---|
| GAE/bootstrap geometry (H, γ, λ) | identical |
| minibatch size | identical (24,576) |
| epochs per sample, gradient steps per sample | identical |
| KL estimate noise for adaptive LR | identical |
| total sample budget | identical, by construction (600 iterations ≈ 29.49M) |
| PPO iterations per sample | 2× (unavoidable with half the envs and the same H) |
| independent env lanes per batch | 2048 vs 4096 (the cost of real morphology DR) |

Option B would also match the batch and minibatch sizes, but it changes the
credit geometry that the agreed rule protects. Option A with 4 minibatches
changes the gradient noise and the steps per sample.

## Open before freezing

1. **Trainer choice.** The stock `rsl_rl` runner cannot host TeacherV4,
   which needs objective sets, per-objective query values, and an `e_t`
   input. The repo's MOPPO trainer has a different recipe (LR 3e-4, γ 0.998,
   4 × 4, B0 lineage). The numbers above assume the M0/`rsl_rl` recipe. If
   V4-C runs on MOPPO, its γ and LR also differ from M0 and must be pinned
   explicitly. The γ = 0.998 rows in the table above show that the
   H-sensitivity conclusion holds either way.
2. The action contract, which is still pending.
