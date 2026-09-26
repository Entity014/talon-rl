# L0 — literature-grounded locomotion baseline

Status: **L0-B frozen; production smoke passed; 3-seed training authorized**

## Reference selected

The installed Isaac Lab Unitree A1 velocity task provides a near-stock `rsl_rl` PPO reference. The flat runner inherits the A1 rough runner recipe and changes only the flat environment plus actor/critic widths:

- rollout: 24 steps per environment;
- actor and critic: `[128, 128, 128]`, ELU;
- learned initial action noise std: `1.0`;
- no actor/critic observation normalization;
- PPO epochs/minibatches: `5 / 4`;
- learning rate: `1e-3`, adaptive schedule, desired KL `0.01`;
- clip `0.2`, value coefficient `1.0`, entropy coefficient `0.01`;
- gamma `0.99`, lambda `0.95`, max gradient norm `1.0`.

The adaptive rule is the implementation in the installed `rsl_rl` package: KL above `2×desired_kl` divides LR by `1.5`, KL below `desired_kl/2` (and positive) multiplies LR by `1.5`, bounded to `[1e-5, 1e-2]`. This is a reference fact, not yet an authorization to copy every setting into the thesis runner.

## Comparison with B0/B1

| component | installed reference | B0/B1 custom path |
|---|---|---|
| actor/critic | 3×128 ELU | 2×64 ELU |
| action std | learned, init 1.0 | scheduled fixed `0.82→0.10` |
| rollout | 24 | 16 |
| PPO epochs/minibatches | 5 / 4 | 4 / full-batch custom loop |
| LR | `1e-3`, adaptive desired KL `.01` | `3e-4`, custom P2 adaptive actor LR |
| gamma/value/entropy | `.99 / 1.0 / .01` | `.998 / .5 / .001` |
| normalization | disabled in reference config | no normalizer in B0/B1 |
| command/reward | stock A1 velocity task | custom B0 scalar reward and command wrapper |
| evaluator | task-specific runner/play path | deterministic Gate-0B evaluator |

This matrix isolates the next causal question: can an established PPO/training recipe learn the same A1 task, before porting any MOPPO or RMA semantics? Reward/environment equivalence must be resolved explicitly; a reference algorithm on a different task is not a valid reproduction.

## L0-B execution status

The frozen manifest is [L0B_FREEZE.json](/home/xero/Master's%20Degree/Thesis/talon-rl/artifacts/l0b_freeze/L0B_FREEZE.json). A real-Isaac 4-env × 2-update smoke passed with learned standard deviation, 5 epochs × 4 minibatches, adaptive-KL learning-rate events, finite metrics, checkpoint save, and `RUN_STARTED → RUN_DONE`. P2/S1/R1 modules are not included.

## L0.0 decision boundary

L0.0 must first freeze the exact environment/reward mapping and reference PPO settings, then run a small production smoke. Only after that smoke passes may a fresh three-seed reference run be authorized and evaluated with the prospective Gate-0B contract. B0/B1 checkpoints remain comparison evidence only.
