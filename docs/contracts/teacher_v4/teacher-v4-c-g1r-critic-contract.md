# Teacher V4 — V4-C G1-R Revised Critic-Validation Contract

Status: **DRAFT — to be frozen before any G1-R evaluation runs**
Date: 2026-09-27

## Purpose

Re-evaluate critic validity for the same six frozen V4-C checkpoints with a
target that does not contain the critic's own bootstrap estimate. G1-R is a
follow-up characterization. It does not overwrite the preregistered result:

    V4-C G1 (teacher-v4-c-g1-contract.md) = FAIL 0/6, frozen.

## Unchanged from the G1 contract

Checkpoints (`model_300.pt` of the six runs), folds, seeds, objective sets,
preferences, suite seeds, evaluation states, the authority / semantic /
survival / continuum / center / permutation criteria and every threshold.
Those criteria are deterministic functions of the same rollouts, so G1-R
reuses their values from each run's `g1_evaluation.json`; the critic
rollouts below re-verify determinism (see Checks).

## Changed: the critic target only

Registered G1 target: truncated 32-step return inside each 32-step segment,
no bootstrap.

G1-R target, for every state s_t the G1 gate already scores (t = 0…63 of
each center/heavy rollout):

    G_t^MC256 = sum_{k=0}^{255} gamma^k r_{t+k},  gamma = 0.99

- Fixed 256-step horizon from each scored state, so the horizon does not
  shrink toward the end of the scored window. The rollout therefore runs
  64 + 255 = 319 steps from the same reset seed.
- Termination: the sum stops at the true episode end, with no bootstrap.
- Horizon end without termination: finite-horizon Monte Carlo truncation by
  design, with no bootstrap. The tail weight left out is gamma^256 ≈ 0.076.
- V is the same query value as in G1: the objective's own query under the
  rollout's own objective set, at s_t.

## Critic criterion (same form and thresholds as G1)

Per set: EV of V against G^MC256, computed per rollout × scored segment
(t = 0…31 and t = 32…63) × active objective, as in G1, then

    critic_valid_MC256 = mean EV > 0  and  negative-EV fraction <= 0.25

## G1-R pass logic (same as G1, critic criterion substituted)

- set pass = all stored G1 criteria except `critic_valid`, and `critic_valid_MC256`;
- m = 4 anchor pass = T/A/O endpoint semantics pass, `critic_valid_MC256`,
  endpoint survival ≥ 0.95;
- fold-seed pass = every held-out set passes and the anchor passes;
- G1-R supported only if both folds pass on all three seeds.

## Reported per set and objective (descriptive)

EV (registered pooling and pooled over all samples), bias mean(V − G),
Pearson correlation, target variance, prediction variance, fraction of
scored states whose 256-step window contains a termination, mean effective
horizon (steps until termination or 256).

## Interpretation, fixed in advance

- critic PASS under MC256 → the original G1 critic failure is attributed
  mainly to target mismatch;
- critic FAIL under MC256 → genuine critic generalization weakness remains;
- mixed, concentrated in S-containing sets → critic validity is objective-set
  dependent; reported as such, no blanket rescue.

No threshold, horizon, checkpoint or set changes after results are seen.

## Checks

- The first 64 steps of every critic rollout must reproduce the stored G1
  truncated-target EV (same states, same policy); a mismatch invalidates
  the run's G1-R result.
- Script, output file names and this contract's commit hash are recorded in
  each output.
