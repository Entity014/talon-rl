# `b0`

15 scripts. One line each, taken from the file's own docstring — edit the docstring, not this file.

| file | lines | tags | description |
|---|---:|---|---|
| `authorize_gate0b.py` | 12 | — | **no docstring** |
| `b0_1_collapse_pair_audit.py` | 26 | class | Read-only same-state pair comparison for the two B0.1 collapse windows. |
| `b0_1_learning_curve_audit.py` | 122 | class | Did B0.1 ever learn deterministic locomotion, or never learn it at all? |
| `b0_1_stability_audit.py` | 126 | isaac | Read-only actor/policy-preservation audit over frozen B0.1 checkpoints. |
| `b0_1_stability_audit_offline.py` | 116 | class | Did the B0.1 collapses come with an abrupt jump in the actor's output? |
| `b0_1_verdict.py` | 44 | — | Aggregate the frozen B0.1 gate; this script never retrains or retunes. |
| `b0_env_smoke.py` | 30 | isaac | No-training smoke for the B0 scalar wrapper. |
| `b0_ppo_smoke.py` | 148 | isaac trains | End-to-end B0 PPO smoke: real B0Env, scalar rollout, and fixed-std PPO. |
| `b0_terminal_smoke.py` | 141 | isaac | Terminal-lifecycle smoke for the B0 scalar environment wrapper. |
| `evaluate_gate0b.py` | 61 | isaac | Deterministic Gate-0B evaluator over all three frozen command cells. |
| `freeze_b0_1.py` | 69 | — | Write B0.1's immutable formulation manifest from validated inputs. |
| `freeze_gate0b.py` | 12 | — | **no docstring** |
| `prepare_gate0b_reset_states.py` | 27 | isaac | **no docstring** |
| `train_gate0b.py` | 51 | isaac | Gate-0B training runner: cycles only the frozen forward command cells. |
| `verdict_gate0b.py` | 18 | — | **no docstring** |

## Still undescribed (4)

These have no module docstring, so there is nothing to put in the table above.

- `authorize_gate0b.py`
- `freeze_gate0b.py`
- `prepare_gate0b_reset_states.py`
- `verdict_gate0b.py`
