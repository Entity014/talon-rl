# `final`

3 scripts. One line each, taken from the file's own docstring — edit the docstring, not this file.

| file | lines | tags | description |
|---|---:|---|---|
| `final_locomotion_score.py` | 165 | — | Score nine completed Final Locomotion Evaluation v1 shards. |
| `run_final_locomotion_shards.py` | 51 | — | Run/resume all nine GPU shards for Final Locomotion Evaluation v1. |
| `run_final_locomotion_shards_r1.py` | 52 | — | Run/resume all nine GPU shards for Final Locomotion Evaluation v1, rerun against the R1 checkpoints (artifacts/r1_freeze/FREEZE.md). Same protocol, same thresholds, same script (final_locomotion_eval.py) as the frozen v1 baseline run -- only the checkpoint set and output directory differ. See run_final_locomotion_shards.py (the original, left untouched, still points at the phase1_hipact_dt01_* baseline checkpoints). |
