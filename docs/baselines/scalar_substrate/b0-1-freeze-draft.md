# B0.1 freeze — flat deterministic locomotion

Status: **FROZEN — implementation validated; 3-seed training authorized**  
Parent: [B0 design](b0-minimal-deterministic-locomotion-baseline-design.md)

The immutable implementation/artifact manifest is
`artifacts/b0_1_freeze/FREEZE.json`. It records the code, reward, trainer,
wrapper, monitor, frozen-reset-state, and effective-config hashes. Any change
to a listed input requires a new freeze record; it is not a mid-run tuning
knob.

## Single question

With a minimal flat-nominal task and variance annealing, can PPO train a
deterministic `tanh(actor_mean)` policy that stably tracks `vx=0.5`?

## Fixed task

- Flat plane, nominal A1 morphology, nominal friction and motor gains.
- No payload, pushes, terrain curriculum, domain randomization, preference
  vector, command changes, or command sampling.
- Every episode uses `(vx, vy, wz) = (0.5, 0, 0)` from observation zero.
- Native nominal reset; deterministic monitor uses a frozen set of 64 saved
  reset states.
- Simulation `dt=0.01`, existing action representation and action scale.

## B0.1 reward

The per-step reward is:

`r = r_track + r_posture + r_height + r_fall`

| term | equation | coefficient |
|---|---|---:|
| tracking | `exp(-((vx - 0.5) / 0.25)^2)` | `+1.0` |
| posture | `-(roll^2 + pitch^2)` | `0.25` |
| height | `-(z - z_nominal)^2` | `2.0` |
| fall | terminal base-contact event | `-10.0` |

`z_nominal` is the robot default root height in the frozen environment config.
No alive bonus, directed-progress window, action penalty, impact trade-off,
energy objective, hip shaping, preference weighting, or reward normalization is
used in B0.1. The terminal fall penalty is applied once, on the termination
transition only.

Before freeze, reward sanity checks must verify: tracking is approximately 1.0
at `vx=0.5`; equal at `vx=0.25` and `0.75`; approximately `exp(-4)` at
`vx=0`; near zero for wrong-direction motion; fall is charged once; and
`z_nominal` is read from the effective environment with no hidden scaling or
normalization of posture or height terms.

## Single deployment-alignment mechanism

PPO, its optimizer, network architecture, action transform, rollout length,
and clipping rule remain unchanged. The only B0.1 formulation mechanism is an
explicit per-update Gaussian exploration standard-deviation schedule:

`std(u) = 0.82 - 0.72 * clamp((u - 100) / 300, 0, 1)`

where `u` is the zero-based update number. Thus std remains `0.82` through
update 100, decreases linearly to `0.10` at update 400, and remains `0.10`
through update 500. The scheduled std is the actual policy distribution std for
that update: rollout sampling, stored old log-probabilities, update-time new
log-probabilities, PPO ratios, entropy, and all log-probability calculations
use the same value. Checkpoints and logs persist the current scheduled std.
Here `u` is the index of the update about to collect its rollout; that same
value applies through rollout collection and every PPO minibatch for the update.
Any existing log-std ceiling or annealing mechanism is replaced, never composed,
by this schedule. Inference remains `tanh(actor_mean)` throughout. No
deterministic actions are mixed into PPO rollouts, no distillation term is
added, and no mean/action regularizer is used.

### Implementation contract

B0.1 introduces an explicit `scheduled_fixed_std` exploration mode. When it is
enabled, `log_std` remains a model field for checkpoint compatibility but is
excluded from the optimizer. For every update, the scheduler derives
`scheduled_log_std = log(std(u))`; `_pre_tanh_dist()` uses that value directly.
The existing `log_std_max` ceiling and its annealing state are disabled in this
mode. The same `_pre_tanh_dist()` path constructs distributions for rollout
sampling, old and new log-probabilities, PPO ratios, and entropy.

Checkpoint state records the exploration mode, update index, and current
scheduled standard deviation. Resume derives the schedule from the restored
update index; it must never restart at update zero. The existing non-B0 path
must retain its current behavior unchanged.

Before freeze, implementation sanity must verify: std is 0.82 at updates 0 and
100, follows the formula at 250, and is 0.10 at updates 400 and 500; every
distribution path has the same std; old/new log-probabilities agree when mean
and action are held fixed; entropy matches the scheduled Normal distribution;
the optimizer excludes `log_std`; resume preserves the correct schedule; and
the default non-B0 behavior is byte-identical.

Resume sanity includes an assertion that the restored update index derives the
same scheduled std recorded immediately before checkpointing, rather than only
checking that a stored field exists.

B0.1 is a capability formulation, not a variance-annealing ablation. Variance
annealing is part of the frozen formulation; no causal superiority claim is
made without a matched fixed-variance control.

## Deterministic monitor

At updates `0, 25, 50, ..., 500`, evaluate `tanh(actor_mean)` from the 64
frozen reset states for 500 steps (5 seconds), using the fixed B0.1 command.
The monitor runs in a separate environment instance and must not mutate the
training simulator, environment counters, training/command/reset RNG, actor
train/eval mode, observation or reward normalizer, or scheduled-std state. If
separation is impossible, all such state must be saved and restored exactly.
Monitor isolation sanity compares hashes of training RNG state, model
parameters and train/eval mode, environment state/counters, normalizer state,
and scheduled-std state before and after a monitor run; all must match.

Report per seed and pooled: survival through 500 steps, first-fall distribution,
forward displacement, mean absolute `vx` error, tilt p95 and max, height error,
and base-contact count. Save every monitor result; it is a learning curve, not
a checkpoint-selection mechanism.

## Stop/go and checkpoint rule

The only candidate checkpoint for terminal B0.1 assessment is update 500. No
seed is terminated individually at update 250. The early-review decision is
made only after all three seeds reach update 250: stop the complete experiment
only if every seed has deterministic survival below 25% and no seed improved
by at least 20 percentage points over its update-0 survival. Such a stop is a
declared failure, not permission to select an earlier checkpoint.

At update 500, B0.1 passes only if all three seeds simultaneously achieve:

- deterministic survival at 500 steps of at least 90% across the 64 states;
- mean absolute `vx` error at most 0.15 m/s among surviving trajectories;
- tilt p95 at most 15 degrees and maximum tilt at most 30 degrees;
- no numerical failures.

Failure leaves B0.2–B0.6 closed and requires a new formulation decision;
neither reward-coefficient nor annealing-schedule sweeps are implied.

## Authorized training budget and invariants

Use seeds `0, 1, 2`, 4096 environments, 500 updates, and a checkpoint at the
predeclared final update. Freeze the exact code/config hashes, simulator
version, reset-state artifact hash, monitor implementation hash, and reward
implementation hash before training. Any deviation changes the formulation and
requires a new freeze draft.

Run the isolated deterministic monitor at updates `0, 25, 50, ..., 500`.
The update-250 early-review rule and update-500-only final checkpoint rule in
this record are binding; monitor results are not permission to modify reward,
the std schedule, thresholds, or the budget mid-run.
