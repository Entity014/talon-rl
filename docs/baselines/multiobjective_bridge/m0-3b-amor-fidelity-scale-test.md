# M0.3-B — AMOR Fidelity / Scale Test

Status: **FROZEN PENDING PRODUCTION SMOKE — TRAINING UNAUTHORIZED**

M0.3-A is closed as a failure under the thesis-scale contract. M0.3-B is a
single, predeclared fidelity test of whether that failure is explained by the
large capacity/training-scale gap to AMOR, rather than a new architecture
patch or a D³PO experiment.

## Frozen intent

- retain M0.1 stock task/reward/evaluator and AMOR early scalarization;
- direct per-episode `w ~ Dirichlet(1)` conditioning;
- vector critic/GAE, `A_scalar = A · w`, minibatch normalization, standard PPO clip;
- no late weighting, diversity, ACAPS/SRM, RMA, or reward changes;
- use AMOR-scale capacity (`4×1024`, ELU) and persisted actor/critic running
  observation normalization;
- use 8192 environments and 24 rollout steps per environment;
- predeclared checkpoints: updates `300`, `1000`, and `3000`;
- terminal budget is exactly `3000` updates; intermediate checkpoints are trend
  evidence only and cannot be selected as verdicts.

## Required pre-freeze decisions

The exact scale, normalization state/checkpoint schema, and deterministic
uniform-preservation gate are now recorded in the manifest. Production smoke
must still validate the implementation before authorization.

## Hard stop

If the scale-aligned run does not show clear deterministic locomotion
preservation by the terminal budget, close the AMOR/MOPPO implementation line
for this thesis. Do not respond by adding capacity, changing coefficients, or
opening D³PO.
