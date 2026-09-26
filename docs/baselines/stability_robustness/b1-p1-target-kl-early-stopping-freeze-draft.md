# B1-P1 — PPO + Target-KL Early Stopping

Status: **CLOSED — FAIL / PARTIAL MECHANISM EFFECT**

## Objective

B1-P1 is the first intervention for B0.1's update-to-update deterministic policy drift. It preserves the B0.1 reward, environment, architecture, fixed command, scheduled-fixed-std formula, rollout size, and deterministic monitor. The only formulation change is stopping PPO optimization epochs early when the policy distribution moves beyond a target KL.

## Frozen inheritance from B0.1

- scalar B0 reward and terminal semantics;
- `ActorCritic(..., reward_dim=1)` and `ScalarRolloutBuffer`;
- fixed `vx=0.5` command;
- scheduled std `0.82 → 0.10` unchanged;
- 4096 environments, 500 updates, seeds 0/1/2;
- deterministic monitor: `tanh(actor_mean)`, 64 independent monitor states × 500 steps;
- no preference/MOPPO/vector-reward state.

The 64 monitor states are evaluation-only. They are not used as a KL anchor or training batch.

## KL estimator

After each complete PPO actor epoch over the full rollout batch, evaluate the old and current **pre-tanh Gaussian** on the entire rollout observation tensor, using the scheduled standard deviation active for that update:

\[
\widehat{KL} = \frac{1}{N}\sum_i D_{KL}\left(\mathcal N(\mu_{old,i},\sigma_u^2)\;\|\;\mathcal N(\mu_{new,i},\sigma_u^2)\right).
\]

With fixed per-update diagonal std this is computed exactly from the two means; it must not use sampled actions, tanh-space actions, minibatch-final observations, or the independent monitor observations. This analytic KL is the sole source of truth for early stopping. Sampled-rollout `approx_kl` is logging/audit only and never controls stopping.

## Proposed stop contract

These values are proposed for design freeze and must appear in the effective-config manifest:

- `target_kl = 0.01`;
- `kl_stop_multiplier = 1.5`;
- hard stop threshold `0.015`;
- check after every complete actor PPO epoch over the full rollout batch;
- when threshold is exceeded, stop remaining PPO epochs for the current update;
- do not start another policy pass in that update;
- critic optimization completes its predeclared schedule even when actor epochs stop;
- advance `update_idx` and scheduled std exactly once as normal; never change the schedule in response to KL.

This is early stopping, not an adaptive KL penalty and not a TRPO step. No coefficient is tuned online.

## Required logging

Every update must record:

- `target_kl`, `kl_stop_threshold`, `kl_check_count`;
- KL value per checked epoch and `kl_stop_triggered`;
- completed policy epochs and skipped policy epochs;
- whether critic epochs were skipped;
- policy loss, value loss, reward mean, scheduled std;
- rollout tensor finiteness, ratio statistics, clip fraction, and gradient norm;
- checkpoint `update_idx`, effective config hash, and B1-P1 code hash.

The runner must emit `RUN_STARTED`, `RUN_DONE` or `ERROR` with traceback, and preserve checkpoint/resume semantics. A resumed run must retain the same target-KL configuration and update index.

## Required acceptance tests before freeze

- controlled mean shift matches the closed-form analytic KL;
- KL below `0.015` runs the next actor epoch;
- KL at or above `0.015` skips remaining actor epochs;
- critic parameters still update after actor early stop;
- scheduled std is unchanged by early stopping;
- resume preserves the B1-P1 config and update index;
- B1-P1 disabled reproduces B0.1 behavior and schedule.

## Design-freeze gate

Before training, verify and record hashes for B0 reward, environment wrapper, trainer, monitor, reset-state artifact, B1-P1 implementation, and effective config. Abort before `RUN_STARTED` if any frozen B0 hash or B1-P1 field mismatches.

The design is not frozen until the estimator direction, threshold, check frequency, stop scope, logging schema, and acceptance criteria are all locked in the manifest.

## Experiment-completion gate

After design freeze, run seeds 0/1/2 with the B0.1 budget. Evaluate only update-500 checkpoints with the unchanged deterministic gate:

- survival ≥ 90%;
- mean `vx` MAE ≤ 0.15 m/s;
- tilt p95 ≤ 15°;
- max tilt ≤ 30°;
- no numerical failures.

Update-250 is review-only and cannot replace the update-500 decision. If any seed fails, stop before B1-S or further coefficient changes and record the failure signature.
