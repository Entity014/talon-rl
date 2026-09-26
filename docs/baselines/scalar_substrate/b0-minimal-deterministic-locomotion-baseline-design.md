# B0 — Minimal Deterministic Locomotion Baseline

Status: **design draft — not frozen, not implemented, no training authorized**  
Rationale: [consolidated failure analysis](consolidated-locomotion-failure-analysis.md)  
Scope: a new formulation phase, not an R1/G1/D1 continuation.

## Research question

Can a policy learn a usable **deterministic actor-mean** locomotion behavior on
a flat, nominal task before preference conditioning, domain variation, or
robustness claims are introduced?

The deployment policy for every B0 gate is `tanh(actor_mean)`. Stochastic
rollouts may remain useful for exploration and diagnostics, but never satisfy
a B0 capability gate on their own.

## Minimal task

B0.1 uses only the following environment:

- flat plane terrain;
- nominal robot morphology and joint limits;
- no payload, pushes, friction variation, motor-gain variation, COM variation,
  terrain curriculum, or randomized robot geometry;
- native nominal reset, with a saved fixed set of evaluation initial states;
- one fixed command, `(vx, vy, wz) = (0.5, 0, 0)`;
- a single locomotion objective, without preference vectors or trade-off claims.

The precise reward equation, action representation, architecture, PPO settings,
and exploration/deployment formulation are design decisions that must be
specified and frozen before B0.1 training. They must be selected because they
produce a deterministic deployable policy, not merely high stochastic rollout
return.

## Deterministic evaluation gate

Before training, freeze a small fixed-state evaluation suite independent of
training lanes. At every `N` updates, run deterministic actor mean from the
same saved initial states for a fixed short horizon, without normalizer or
optimizer updates. Record survival, directed displacement, forward-velocity
tracking error, height, tilt, undesired contact, and first-fall step.

The B0.1 gate is passed only when all three seeds meet the predeclared minimum
survival and quality thresholds on this deterministic suite. The final values
of `N`, number of states, horizon, and thresholds must be written into the
freeze record; training lane age and stochastic return are sanity metrics only.

## Stop/go rule

The freeze record must define an early review update and a terminal update.

- At the early review, stop the run if deterministic survival remains near the
  untrained level and no seed shows sustained improvement in survival or
  first-fall timing.
- At the terminal update, advance only if the deterministic B0.1 gate passes
  for all seeds.
- A failure does not justify coefficient sweeps or checkpoint selection. It
  triggers a documented formulation decision before another run.

## Complexity ladder

Each stage requires the preceding stage's deterministic gate to pass before it
is designed, frozen, and trained.

| stage | added capability |
|---|---|
| B0.1 | flat, nominal, fixed `vx=0.5` |
| B0.2 | flat command range |
| B0.3 | zero-command holds and command transitions |
| B0.4 | terrain and domain randomization |
| B0.5 | preference conditioning and trade-off evaluation |
| B0.6 | payload, morphology, and held-out robustness tests |

No preference, terrain, payload, or morphology claim is evaluated before its
corresponding prerequisite deterministic gate is passed.

## Formulation question to resolve before freeze

Earlier evidence showed stochastic execution can sometimes survive when the
actor mean fails. B0 must therefore specify how stochastic exploration is
expected to yield a deployable deterministic mean. Candidate mechanisms may
include an exploration schedule, deterministic-policy evaluation during
optimization, deployment-aware objective terms, or distillation. None is
authorized by this document. The selected mechanism must be the single
formulation change tested by B0.1 and must include a causal rationale,
implementation checks, and an ablation plan.

## Required freeze contents

The B0 freeze record must include the exact environment config, reset-state
artifact and hash, command, reward/objective, actor and critic architecture,
PPO and exploration settings, deterministic evaluator implementation and
thresholds, seeds, training budget, checkpoint policy, early-stop rule, and
artifact hashes. It must also state every excluded complexity listed above.

## Decision boundary

This document authorizes design work only. Implementation, smoke testing,
training, and selection of a deployment-aware formulation require a separate
frozen B0.1 record.
