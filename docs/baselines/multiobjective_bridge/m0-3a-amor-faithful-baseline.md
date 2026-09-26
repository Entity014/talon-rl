# M0.3-A — AMOR-Faithful MOPPO Baseline

Status: **FROZEN — TRAINING AUTHORIZED**

FP-0C remains a validated fallback/control architecture, but its centered
conditioning is not the AMOR formulation. M0.3-A therefore tests the
literature-faithful early-scalarization baseline before introducing the thesis
late-weighting mechanism.

## Contract

- inherit the validated M0.1 environment, reward vector, reset, command,
  normalization, std, optimizer, checkpoint, and deterministic monitor;
- sample one preference `w ~ Dirichlet(1)` at episode reset and hold it for the
  episode; no preference sampling is used by the evaluator unless explicitly
  declared;
- concatenate the direct simplex `w` to the actor input;
- use a five-output vector critic and vector GAE/returns;
- scalarize advantages early: `A_scalar = A · w`;
- normalize `A_scalar` per PPO minibatch using a fixed epsilon;
- use the standard single clipped PPO surrogate on `A_scalar` (no
  per-objective clipping, D³PO late weighting, diversity term, or preference
  curriculum);
- use learned standard deviation and the reference adaptive-KL path;
- keep reward computation preference-independent.

## Gates

M0.3-A has two gates: (1) a production smoke proving preference/reset
plumbing, vector critic/GAE, scalarized-advantage normalization, standard PPO
clipping, checkpoint/resume, and deterministic explicit-`w_eval` monitoring;
then (2) a three-seed preservation and preference-response experiment.

Locomotion preservation is evaluated before trade-off claims. The fixed
uniform preference is one evaluation condition, while a predeclared set of
non-uniform preferences tests response only after preservation passes.

## Exclusions

No D³PO late weighting, FP-0C centered conditioning, ACAPS/SRM, RMA,
exteroception, terrain variation, reward sweep, or M0.2b state is included.
M0.3-A is a baseline reproduction, not the final thesis mechanism.
