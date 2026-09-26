# Teacher V4 — V4-C1 Trainability Screen Verdict

Status: **PASS (infrastructure/trainability only) — full V4-C run authorized pending the training-cardinality decision**
Date: 2026-09-27
Branch: `v4-a-teacher`

## Run

`train_v4c.py` (commit `822cbcb`) on `Isaac-Talon-A1-V4C-v0`, 4096 envs × 24
steps × 10 iterations (983,040 samples), seed 0, cardinalities {1,2,3,4},
frozen V4-C contracts. Output `runs/teacher_v4_c1_screen-2026-09-27/`
(`metrics.jsonl`, `model_5.pt`, `model_10.pt`). 1.7 s per iteration, so the
full 300-iteration run takes about 9 minutes.

No checkpoint was chosen by any semantic score. These numbers are gates, not
results.

## Gates

| gate | iteration 1 → 10 | verdict |
|---|---|---|
| all statistics finite, no stop gate | yes | pass |
| mean KL per minibatch vs desired 0.01 | 0.006 → 0.014 | pass |
| adaptive LR | 1e-3 → hits the 1e-2 cap early, settles 4.4e-3 | pass (rsl_rl behavior) |
| clip fraction | 0.07 → 0.20 | pass, watch |
| log-std range | [−0.008, 0.004] → [−0.070, 0.024] | pass |
| explained variance T / A / O / S | −1.4 / −3.4 / −1.7 / −5.0 → 0.18 / −1.3 / 0.70 / −1.7 | pass, improving; A and S still negative |
| preference authority, median ‖Δa‖ when only the set changes | 0.061 → 0.314 (V4-B init: 7.2e-4) | pass |
| plant authority, median ‖Δa‖ when only e_t changes | 0.022 → 0.049 | pass (only payload mass varies in V4-C) |
| termination fraction | 0.0000 | no survival collapse |

Per-step normalized reward moved in the right direction for O
(−0.036 → −0.026) and A (−0.048 → −0.039). T and S are flat over 10
iterations.

## Notes

- Termination 0 for 240 steps per env means no trunk contact yet under the
  initial high-noise policy; episodes are 1000 steps, so no time-outs either.
  Survival is only meaningful over the full run.
- Clip fraction near 0.2 by iteration 10 with the LR at the cap earlier: if
  it keeps rising, that is the first thing to look at in the full run.
