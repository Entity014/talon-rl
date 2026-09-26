# `r1`

10 scripts. One line each, taken from the file's own docstring — edit the docstring, not this file.

| file | lines | tags | description |
|---|---:|---|---|
| `r1_directed_progress_smoke.py` | 90 | isaac class | Isaac Sim smoke check for a1_env.py's directed-progress wiring (R1, 2026-09-20, artifacts/r1_freeze/FREEZE.md) -- the pure math is unit-tested in tests/test_directed_progress.py, but the plumbing (root_pos_w/yaw extraction, reset-hook ordering, terminal-frame double-call) only runs through real Isaac Lab, which test_a1_env.py can't do on this machine. Not a pytest -- prints per-step directed_progress so a human/this session can eyeball "no spike at reset or command change" before committing to a 3-seed retrain. |
| `r1_mismatch_audit.py` | 211 | isaac | Frozen-checkpoint diagnostic. Never updates policy, normalizers or rewards. |
| `r1_offline_audit.py` | 213 | — | R1 offline audit: analyze the 45-cell read-only replay to decide, per sub-reward, retain / gate / move / disable -- no simulator needed. |
| `r1_readonly_replay.py` | 156 | isaac class | Read-only R1 replay: record missing per-step reward inputs, never update. |
| `r1_replay_validate.py` | 111 | — | Validate that R1 replay traces reproduce the frozen v1 uniform cells. |
| `run_r1_mismatch_audit.py` | 29 | — | Run the bounded 3-checkpoint audit, with separate simulator processes. |
| `run_r1_replay_shards.py` | 27 | — | Run/resume all nine read-only R1 replay shards. |
| `run_r1_retrain_shards.py` | 60 | — | Run/resume the R1 3-seed retrain (artifacts/r1_freeze/FREEZE.md). |
| `run_r1_single_cell.py` | 41 | — | Direct one-cell runner with lifecycle markers and GPU diagnostics. |
| `summarize_r1_mismatch_audit.py` | 172 | — | Summarize saved mismatch traces, without simulator access. |
