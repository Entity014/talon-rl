# `v1a`

12 scripts. One line each, taken from the file's own docstring — edit the docstring, not this file.

| file | lines | tags | description |
|---|---:|---|---|
| `freeze_v1a_e0.py` | 114 | — | Freeze the prospective V1A-E0 evaluation protocol. |
| `freeze_v1a_e05.py` | 79 | — | Freeze the baseline-only V1A-E0.5 calibration protocol. |
| `freeze_v1a_e06.py` | 62 | — | Freeze the baseline-only V1A-E0.6 deployment-contract audit. |
| `freeze_v1a_e1r.py` | 68 | — | Freeze the corrected V1A-E1R measurement contract. |
| `freeze_v1a_e1r_cal.py` | 24 | — | **no docstring** |
| `train_v1a.py` | 121 | isaac | Authorized V1-A preservation runner: one seed, 300 scalar-PPO updates. |
| `v1a_e05_calibrate.py` | 115 | isaac | M0.1-only variability calibration for V1A-E0.5. |
| `v1a_e06_audit.py` | 141 | isaac | Baseline-only audit of stock rsl_rl deployment action semantics. |
| `v1a_e1r_calibrate.py` | 37 | isaac | M0.1-only corrected-path calibration; never loads V1-A. |
| `v1a_e1r_eval.py` | 76 | isaac | Corrected paired M0.1/V1-A evaluator using stock action clipping. |
| `v1a_isaac_smoke.py` | 148 | isaac | Short real-Isaac scalar-PPO V1-A integration smoke. |
| `v1a_rsl_equivalence.py` | 104 | class | Checkpoint-level M0.1 -> V1-A equivalence check in the Isaac runtime. |

## Still undescribed (1)

These have no module docstring, so there is nothing to put in the table above.

- `freeze_v1a_e1r_cal.py`
