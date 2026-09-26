# G1 — Command Exposure / Deterministic Generalization Revision

Status: **FROZEN — implementation validated; retraining not yet started**  
Parent decision: `artifacts/r1_mismatch_audit/audit-report.md`  
Scope: training exposure only; R1 reward and the frozen final evaluator remain unchanged.

## Question

If training exposes the policy to the evaluator's command timing and command
history, does deterministic actor-mean execution improve without changing the
R1 reward, actor architecture, or PPO algorithm?

## Evidence this design addresses

The completed robustness audit found a large deterministic survival change when
the forced zero-settle/fixed-command condition was replaced by a training-style
command, while reset changes were smaller. Seed1 showed strong stochastic-versus-
mean sensitivity; seed2 showed strong first-observation and command-history
sensitivity. Terrain-only and physics-randomization-only probes did not rescue
the deterministic fixed-command behavior. These findings identify exposure as
the intervention target, not proof that reward causality is absent.

## Frozen invariants

G1 must preserve exactly:

- R1 reward equations, coefficients, gates, term names, and reward normalization;
- PPO/MOPPO hyperparameters, optimizer, network architecture, action scale,
  observation dimensions, stack sizes, simulation `dt=0.01`, and decimation;
- terrain, morphology, payload, friction, motor-gain, joint-limit and push
  distributions from the R1 training protocol;
- training seeds `0, 1, 2`, 500 updates, 4096 environments, and checkpoint format;
- the frozen Final Locomotion Evaluation protocol: 14,400 episodes, exact
  reset seeds/commands/preferences/thresholds, deterministic actor mean, and
  a new artifact directory.

Only command scheduling and command exposure change in G1.

## Exact exposure schedule

At every environment reset, sample one Bernoulli flag independently per lane
using `p_zero_hold = 0.25`. The random generator is the environment's existing
seeded generator; the flag and command are logged for replay.

For `zero_hold` lanes:

1. write command `(0, 0, 0)` before computing the first policy observation;
2. hold `(0, 0, 0)` for exactly 200 policy steps (2.0 s at `dt=0.01`);
3. sample one fixed command from the unchanged training ranges
   `vx∈[-0.3,1.0]`, `vy∈[-0.3,0.3]`, `wz∈[-0.5,0.5]`;
4. hold that command until the lane resets.

At the exact switch boundary, step 200 is the first step that uses the
post-hold command: write the sampled nonzero command before computing the
step-200 observation and before actor inference for that step. There is no
one-step command-observation lag at the switch.

When the command changes from zero to nonzero, reset the 50-step directed-
progress displacement window at the same boundary. Displacement accumulated
during the zero hold must not enter the post-switch directed-progress term.

If a lane terminates during the zero hold, its episode ends normally. The
automatic reset samples a new Bernoulli flag and new command for the new
episode; the lane never resumes the remaining hold steps from the old episode.

For the remaining `0.75` lanes, preserve the current R1 training behavior:
sample one command from those same ranges at reset, expose it in observation
zero, and hold it until reset. No command resampling is added within an
episode. The exact-zero probability in the current continuous command sampler
remains zero; G1 introduces exact zero only through the explicit 25% branch.

The 200-step duration is the frozen evaluator's settling duration, not a new
reward window. The post-hold command is sampled once and is not forced to
`-0.25`; evaluation remains unchanged.

G1 does not train a deterministic actor mean directly. PPO continues to use
its existing stochastic action sampling and unchanged loss. G1 tests whether
additional command-history exposure makes the resulting deterministic mean
more robust at evaluation time. If G1 fails, the result cannot by itself
separate insufficient exposure from a need for a dedicated actor-mean
robustness intervention; that question remains a later decision.

## Implementation checks before training

The implementation must pass read-only tests for: command written before
observation zero; zero hold lasting exactly 200 steps; no command resampling;
post-hold command written before step-200 observation/inference; directed-
progress window reset at the switch; termination during the hold creating a
fresh Bernoulli draw rather than resuming; per-lane reset isolation; seeded
reproducibility; unchanged reward/config dump outside the command scheduler;
and command/flag logging in the rollout record.
Run a short no-training simulator smoke with both branches and verify that the
first observation's command equals the command applied at step zero.

## Evaluation and decision rule

Retrain exactly three seeds after the implementation is frozen. Compare against
R1 using the unchanged final evaluator. Report deterministic mean survival,
first-fall timing, and all frozen quality gates; do not substitute stochastic
sampling for the evaluator. G1 is supported only if it improves the frozen
locomotion verdict and does not introduce a material regression in the quality
gates. If G1 fails, report whether early deterministic collapse remains and
then decide separately whether a reward or actor/training revision is warranted.

No coefficient sweep, reward edit, architecture change, PPO change, or broad
terrain/physics ablation is part of G1.
