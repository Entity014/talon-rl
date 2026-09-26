# Gate 0B — pipeline-aligned substrate authorization

Status: **PROTOCOL FROZEN; fresh rerun not yet started**

Gate 0B is a substrate authorization gate for the original thesis pipeline. It is not a final locomotion result and does not claim that the base policy is the thesis contribution.

## Fixed contract

- deployment action: deterministic `tanh(actor_mean)` only;
- environment: flat plane, nominal morphology, no preference, no RMA latent, no terrain/domain randomization;
- commands: three fixed forward cells `(vx, vy, wz) = (0.25,0,0), (0.50,0,0), (0.75,0,0)`;
- seeds: `0, 1, 2`;
- evaluation: a newly captured, training-independent reset suite of 64 states, reused unchanged for every seed and command;
- horizon: 500 simulator steps at `dt=0.01` (5 seconds);
- verdict checkpoint: update 500 only;
- no stochastic sampling, optimizer update, normalizer update, or privileged input during evaluation.

## Per seed-command acceptance

Every one of the nine seed×command cells must satisfy all conditions:

- survival rate `>= 0.80`;
- command-specific forward-velocity MAE `<= 0.25 m/s`;
- tilt p95 `<= 20°`;
- maximum tilt `<= 40°`;
- base-contact termination rate `<= 0.10`;
- finite metrics and valid completion artifacts.

These thresholds are prospective engineering requirements for a short substrate sanity horizon: they require repeatable forward motion and a meaningful fall margin, while remaining weaker than the B0 final-deployment gate. They were selected before the fresh rerun and are not derived from, or retrofitted to, B0/B1 outcomes.

## Aggregate decision

Gate 0B passes only if all nine cells pass. A failure authorizes an architecture-level baseline decision; it does not authorize threshold changes, checkpoint cherry-picking, or a return to MOPPO. A pass authorizes the original sequence:

`command transitions → domain variation → RMA adaptation → MOPPO/preference conditioning`.

## Required artifacts and provenance

Each seed must contain `RUN_STARTED`, `RUN_DONE` or `ERROR`, update-500 checkpoint, effective config, monitor artifacts for all three commands, and a copy of the Gate-0B manifest. The manifest hashes the code, reward, environment wrapper, deterministic evaluator, reset suite, and effective config. Old B0/B1 checkpoints are comparison evidence only and cannot satisfy this gate.
