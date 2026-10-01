# Teacher V4 — FC-G gradient composition verdict

Status: **RESULT, 2026-10-01.** The [FC-G contract](../../contracts/teacher_v4/teacher-v4-fcg-gradient-composition-contract.md) was frozen at `bb4f974` before outcome branches; per-minibatch exposure logging and live preflight were committed at `34417e6`, also before outcome branches. Stage 1 used r = 11–13. Only the two open seed × `SPLIT-NORM-MATCHED` contrasts expanded to r = 14–15. No threshold, arm, or Stage 3 was added. All 26 registered branches trained and replayed. Output: `runs/teacher_v4_fc-2026-10-01/fcg/` (`fcg_stage1.json`, `expand.txt`, `fcg_stage2.json`, per-branch metrics, gradient audit, and replay traces).

The primary effect is `matched Δ_R(GLOBAL) − matched Δ_R(candidate)` in percentage points. Positive means lower harmful R separation under the candidate. Raw paired effects are reported alongside. Every one of the 16 registered pairs was valid: all branches had at least two viable endpoint checkpoints and a matched value. Each branch's u0 matched Δ_R replay was identical within seed (−12.66% for 79101, −4.715% for 79103).

| Seed | Contrast | Stage | Matched paired effects, pp | Median matched / raw, pp | Registered class |
|---|---|---:|---|---:|---|
| 79101 | GLOBAL − SPLIT | 1 | +0.87, −0.39, +0.21 | +0.21 / +7.08 | within practical band |
| 79103 | GLOBAL − SPLIT | 1 | +4.49, +3.22, +5.47 | +4.49 / +6.15 | within practical band |
| 79101 | GLOBAL − SPLIT-NORM-MATCHED | 2 | +6.59, −3.19, +5.02, +5.24, +12.26 | +5.24 / +8.27 | not resolved by registered budget |
| 79103 | GLOBAL − SPLIT-NORM-MATCHED | 2 | +8.10, +6.05, +8.07, −5.84, −1.16 | +6.05 / +3.32 | not resolved by registered budget |

The table rounds values; the probe applies strict thresholds to unrounded matched effects. Stage 1 `SPLIT-NORM-MATCHED` was open in both seeds, which triggered only those expansions. At Stage 2, 79101 had four positive pairs but median +5.24 pp, below the registered >+10 pp gate. 79103 had only three positive pairs and median +6.05 pp. Neither meets the replicated attenuation or worsening gate, and neither is within the practical band. Thus **relative task/non-task gradient composition under global clipping is not established as the replicated mechanism by FC-G**. The result is *not resolved by the registered budget*, not evidence of equivalence or proof that clipping has no effect.

The `GLOBAL − SPLIT` result is within the predefined practical band in both seeds. That class is descriptive, not a formal equivalence test. The dose diagnostic `SPLIT-NORM-MATCHED − SPLIT` had matched medians −4.82 pp (79101) and −2.84 pp (79103) over the three Stage 1 pairs; it was not a headline gate or Stage 2 expansion trigger.

## Integrity and optimizer interpretation

The treatment was exercised. Across branches, the mean fraction of PPO minibatches where `c_global < c_N` was approximately 1.00 in both seeds, and the mean fraction with `|c_T/c_N − 1| > 0.05` was at least 0.997. GLOBAL's mean `c_T/c_N` was 0.418 in 79101 and 0.317 in 79103. Its mean `cos(G0,G2)` was 0.987 and 0.980 respectively: a modest directional change, accumulated across updates. The complete per-minibatch distributions are in each branch's `gradient_minibatches.jsonl`.

Pre-Adam norm matching did not match Adam parameter-step norms. Mean per-minibatch actor parameter-delta norms for GLOBAL versus SPLIT-NORM-MATCHED were 0.0761 versus 0.0897 in 79101 and 0.0659 versus 0.0748 in 79103. This is an observed optimizer response under the continued trajectories, not a controlled per-minibatch comparison after policies diverge. It limits any claim of a pure composition effect even if an endpoint had passed. KL, entropy, and `log_std` were recorded and all branches passed the trainer's existing stop gates.

Viability and O preservation passed for every branch under the registered endpoint gates. **R semantics were restored in no branch:** none had both mean raw and matched Δ_R ≤ −5%. Median matched endpoint Δ_R was +12.75% (GLOBAL) versus +7.68% (SPLIT-NORM-MATCHED) in 79101, and +3.04% versus −0.34% in 79103. These endpoint medians are descriptive and do not replace paired votes.

FC-E's replicated task-pressure *loss-formulation* effect remains the strongest causal finding. FC-F did not localize a cross-seed parameter route. FC-G confirms numerical exposure to the clip-composition intervention but does not replicate endpoint attenuation at the registered 10 pp scale. The next study must be separately preregistered; FC-G has no Stage 3.
