# `m0`

29 scripts. One line each, taken from the file's own docstring — edit the docstring, not this file.

| file | lines | tags | description |
|---|---:|---|---|
| `authorize_m0_1.py` | 12 | — | Authorize M0.1 confirmation only after production-path smoke. |
| `close_m0_1.py` | 22 | — | Close M0.1 only after the three-seed confirmation artifact passes. |
| `close_m0_2a_fail.py` | 9 | — | **no docstring** |
| `close_m0_2a_fp0.py` | 4 | — | **no docstring** |
| `convert_m0_1_to_fp0.py` | 15 | class | Convert an rsl_rl M0.1 checkpoint into the FP-0 ActorCritic schema. |
| `eval_m0_2a.py` | 36 | isaac | Deterministic fixed-w evaluation for M0.2a final checkpoints. |
| `eval_m0_2a_debug.py` | 36 | isaac | Single-env evaluator lifecycle probe; never substitutes for the gate. |
| `eval_m0_2a_decomp.py` | 27 | isaac | Common deterministic evaluator for A1/A2/A3 checkpoint-100 screens. |
| `eval_m0_3a.py` | 28 | isaac | Deterministic explicit-preference evaluation for M0.3-A checkpoints. |
| `eval_m0_3b.py` | 23 | isaac | Read-only deterministic evaluator for scale-aligned AMOR checkpoints. |
| `freeze_m0_1.py` | 15 | — | Freeze M0.1 after exact real-Isaac reward reconstruction smoke. |
| `freeze_m0_2a.py` | 37 | — | Freeze M0.2a only after algebraic tests and real-Isaac smoke pass. |
| `freeze_m0_2a_fp0.py` | 6 | — | **no docstring** |
| `m0_1_isaac_smoke.py` | 25 | isaac | Real Isaac smoke for exact stock reward-term vector reconstruction. |
| `m0_1_production_smoke.py` | 30 | isaac | Production-path M0.1 smoke: vector reward enters stock rsl_rl PPO. |
| `m0_2a_decomp_screen.py` | 62 | isaac trains | Short causal-localization screens A1/A2/A3 (not thesis verdicts). |
| `m0_2a_fp0_smoke.py` | 28 | isaac | Real-Isaac FP-0 smoke: function-preserving actor expansion only. |
| `m0_2a_fp0c_smoke.py` | 26 | isaac trains | **no docstring** |
| `m0_2a_production_smoke.py` | 84 | isaac trains | Real-Isaac M0.2a smoke: vector reward -> vector GAE/critic -> actor loss. |
| `m0_3a_production_smoke.py` | 73 | isaac trains | Small real-Isaac smoke for the AMOR early-scalarization path. |
| `record_m0_2a_fp0_smoke.py` | 8 | — | **no docstring** |
| `summarize_m0_1.py` | 78 | — | Summarize the completed M0.1 three-seed confirmation. |
| `summarize_m0_2a.py` | 17 | — | Summarize M0.2a machinery sanity without overstating preservation. |
| `train_m0_1.py` | 31 | isaac | M0.1 three-seed confirmation runner: stock rsl_rl + exact reward vector adapter. |
| `train_m0_2a.py` | 75 | isaac trains | M0.2a fixed-preference three-seed preservation runner. |
| `train_m0_3a.py` | 159 | isaac trains | Production AMOR early-scalarization runner (M0.3-A). |
| `verdict_m0_2a.py` | 17 | — | Apply the fixed-preference deterministic preservation verdict. |
| `verdict_m0_2a_fp0.py` | 5 | — | **no docstring** |
| `verdict_m0_3a.py` | 15 | — | Aggregate the predeclared M0.3-A preservation gate. |

## Still undescribed (6)

These have no module docstring, so there is nothing to put in the table above.

- `close_m0_2a_fail.py`
- `close_m0_2a_fp0.py`
- `freeze_m0_2a_fp0.py`
- `m0_2a_fp0c_smoke.py`
- `record_m0_2a_fp0_smoke.py`
- `verdict_m0_2a_fp0.py`
