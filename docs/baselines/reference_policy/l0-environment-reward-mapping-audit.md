# L0 environment/reward mapping audit

Status: **COMPLETE — training not authorized**

The installed Isaac Lab A1 flat velocity task and Gate-0B are not field-equivalent. The complete source-backed JSON record is [mapping.json](/home/xero/Master's%20Degree/Thesis/talon-rl/artifacts/l0_mapping_audit/mapping.json).

## Key findings

| Field | Stock Isaac Lab A1 flat | Gate-0B | Result |
|---|---|---|---|
| Policy observation | 45-D concatenation (relative joints, noisy proprioception, command, last action; flat height scan disabled) | 51-D custom vector (absolute joints, roll/pitch, binary foot contact, command, no corruption) | Not equivalent |
| Action | Joint-position target, default offset, scale 0.25 | Joint-position target, custom scale 0.15 | Not equivalent |
| Reset | Stock randomized x/y/yaw and A1 joint reset semantics | Frozen reset artifact plus nominalized reset path | Partial |
| Command | Manager-sampled velocity commands, held 10 s | Fixed cells 0.25/0.50/0.75, held per rollout, cycled by update | Not equivalent |
| Reward | Multi-term tracking/regularization/contact reward | Scalar B0 tracking + posture + height + terminal fall penalty | Not equivalent |
| Termination | Timeout + illegal base/trunk contact | Timeout + trunk/base contact, pre-reset terminal snapshot | Close, must verify sensor/timing |
| Horizon/rate | 20 s, physics dt .005, decimation 4 (control period .02) | 22 s, Gate-0B dt .01, decimation 1 (control period .01) | Not equivalent |
| Contact | History-3 all-body sensor and air-time/contact terms | Custom history-1 foot/trunk sensors; only trunk fall enters B0 reward | Not equivalent |
| Randomization | Stock training events remain unless explicitly disabled | Nominal plane/properties, no push/curriculum | Not equivalent |
| PPO | rsl_rl-style learned std, 24-step rollout, 5×4 updates, adaptive KL | Custom fixed scheduled std, 16-step rollout, custom update recipe | Not equivalent |

## L0.0 decision

Two experiments must not be conflated:

- **L0-A — stock task reproduction:** checks that the installed Isaac Lab + `rsl_rl` reference stack can solve its own A1 flat task.
- **L0-B — reference PPO on our task:** keeps Gate-0B environment/reward/reset/command/evaluator fixed and replaces only the PPO recipe with the declared stock-style reference recipe.

For the thesis causal question, **L0-B is the primary experiment**. L0-A is an optional stack sanity check, not evidence that Gate-0B is solved. Before either run, freeze the exact task mapping, command schedule, optimizer/std semantics, and evaluator hashes. No L0 training is authorized by this audit alone.
