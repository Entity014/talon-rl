# `l0`

15 scripts. One line each, taken from the file's own docstring — edit the docstring, not this file.

| file | lines | tags | description |
|---|---:|---|---|
| `authorize_l0b.py` | 30 | — | Authorize L0-B only after the real-Isaac lifecycle smoke passes. |
| `freeze_l0b.py` | 58 | — | Create the L0-B formulation manifest; authorization remains off. |
| `l0_c0_audit.py` | 113 | class | Why does L0-B seed 2 reach a good basin while seeds 0 and 1 do not? |
| `l0_mapping_audit.py` | 142 | — | Emit the frozen, source-backed L0 reference/task mapping audit. |
| `l0_reference_inventory.py` | 10 | — | **no docstring** |
| `l0a_screen_report.py` | 14 | — | Record the read-only result of the stock Isaac Lab L0-A screen. |
| `l0b_production_smoke.py` | 66 | isaac | Small real-Isaac L0-B lifecycle smoke; never authorizes full training. |
| `l0c1_analyze.py` | 34 | — | Compare L0-C1 snapshots and locate the earliest telemetry divergence. |
| `l0c2_audit.py` | 86 | class | What the C1 snapshots can and cannot say about lane-level geometry. |
| `l0c2j_audit.py` | 131 | class | Joint-wise saturation and lane-level covariance over the C2R arrays. |
| `l0c2r_analyze.py` | 26 | — | Summarize lane-indexed C2R replay arrays and saturation/failure links. |
| `l0c2r_replay.py` | 49 | isaac | Read-only lane-indexed replay of existing L0-B/C1 checkpoints. |
| `train_l0b.py` | 57 | isaac | Authorized L0-B full runner: reference PPO on the unchanged Gate-0B task. |
| `train_l0c1.py` | 52 | isaac | L0-C1: formulation-identical L0-B runner with telemetry only. |
| `verdict_l0b.py` | 22 | — | Apply the prospective Gate-0B thresholds to L0-B update-500 artifacts. |

## Still undescribed (1)

These have no module docstring, so there is nothing to put in the table above.

- `l0_reference_inventory.py`
