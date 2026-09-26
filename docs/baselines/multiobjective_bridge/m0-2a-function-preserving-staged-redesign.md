# M0.2a-FP — Function-Preserving Staged MOPPO Redesign

Status: **DESIGN DRAFT — TRAINING UNAUTHORIZED**

The update-100 decomposition was inconclusive because the matched scalar A0
control was not yet locomotion-mature. This design therefore stops retrospective
localization and introduces MOPPO structure from the validated M0.1 substrate
without an arbitrary early architectural jump.

## Principle

At the transition checkpoint, the deployed actor must initially implement the
M0.1 function on the fixed reference preference. New preference channels start
with zero influence. New value heads start from a scalar-consistent
initialization. Each subsequent mechanism is enabled only after a declared
short preservation check.

## Stages

### FP-0 — actor expansion, function preserved

- Load an M0.1 actor checkpoint.
- Expand the first actor layer from `obs` to `[obs,w]`.
- Copy all old observation columns and set the five preference columns to zero.
- Keep `w_ref` explicit; therefore actor output is identical to M0.1 at init.
- Keep the scalar critic and scalar PPO surrogate unchanged.
- Verify action and log-probability equality on a fixed probe batch before any
  optimizer step.

### FP-1 — vector critic warm-start

- Add five value heads while retaining the FP-0 actor.
- Initialize each value head so that the weighted aggregate reproduces the
  scalar critic at initialization (`V_k = V_scalar` under the declared
  reference aggregation).
- Keep scalarized advantage/PPO clipping during warm-up; vector returns and
  per-objective advantages are logged but not yet used for actor updates.
- Require finite values and preservation telemetry before proceeding.

### FP-2 — vector GAE and late weighting

- Enable per-objective GAE and the frozen `KΣw_kL_k` surrogate only after FP-1
  passes its preservation check.
- Keep `w_ref` fixed; preference diversity remains disabled.
- No reward, environment, command, std schedule, or monitor changes are made.

## Gates and controls

Each stage has a separate artifact and may stop without proceeding. The final
M0.2a preservation verdict still uses the existing deterministic model-300
gate, three seeds, and explicit `w_ref`. Probe tolerances for FP-0 are fixed
before training: action and log-probability absolute/relative error `1e-6`.

The staged path is not claimed to be equivalent to M0.1 after optimization;
its purpose is to ensure that any later divergence is attributable to a
declared stage rather than random first-layer initialization.

M0.2b remains blocked until the complete staged M0.2a preservation gate passes.
