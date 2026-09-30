# Teacher V4 — FC-E paired replication verdict

Status: **RESULT, 2026-09-30.** The sequential rule was frozen at `a1188ce`, before any FC-E branch was trained. All registered contrasts have reached a terminal class; no stage 3 is allowed.

Contract: [FC-E paired replication](../../contracts/teacher_v4/teacher-v4-fce-paired-replication-contract.md). Output: `runs/teacher_v4_fc-2026-09-30/fce/` (`fce_stage1.json`, `fce_stage2.json`, `fce.out`).

The endpoint is the paired difference in endpoint Δ_R, where Δ_R = F_R(R⁺)/F_R(C) − 1. The matched estimate is primary; raw is descriptive. Each endpoint averages viable checkpoints at k = 30, 40, 50. The already observed original branches (r = 0) are excluded from the repeat count. All 14 repeat pairs were valid (28 branches).

| Contrast (arm 1 − arm 2) | Role | Matched effects by repeat, pp | Median matched / raw, pp | Registered class |
|---|---|---|---:|---|
| task@79101 (A − D1) | primary | +29.7, +25.6, +25.6 | +25.6 / +30.7 | replicated, stage 1 |
| task@79103 (A − D1) | primary | +14.4, +5.8, +12.7 | +12.7 / +12.3 | replicated, stage 1 |
| region@79101 (D2p − D2) | primary | +22.6, +0.8, +6.4, +22.1, +15.0 | +15.0 / +21.1 | replicated, stage 2 |
| task@79102 (A − D1) | secondary | +16.3, +14.0, +19.4 | +16.3 / +18.3 | replicated, stage 1 |

The task contrast passes in **both prespecified primary seeds**; the secondary seed agrees. Thus task pressure in this mixed-preference continuation raises matched Δ_R relative to the otherwise paired task-free actor loss. The effect does not depend on one original branch. In seed 79101, R⁺-only task pressure also raises matched Δ_R relative to R-vertex-only task pressure. The region result is local to seed 79101; its five effects vary substantially, including a +0.8 pp pair.

These are effect-size conclusions, not universal binary inversion claims. In task@79103, only one A repeat has an `inverted` label even though all three paired effects are positive. The matched endpoint controls for tracking differences in the registered analysis; raw medians have the same direction. The contrast weakens an R-vertex-only explanation and supports R⁺ task pressure as a contributor in 79101. It does not establish that the R vertex is feasible or irrelevant in every seed. In the current PPO code, removing the task loss weight also changes the shared advantage scale; FC-E therefore identifies the effect of the task-pressure **loss formulation**, not an isolated task-gradient vector.

## Mechanism boundary and next step

Supported chain: mixed-preference continuation + task pressure → higher R⁺ rotation relative to C, measured by matched Δ_R. Earlier stages support the physical meaning of R and rule out the diagnosed critic failure. FC-E itself does **not** identify which actor parameters carry the task effect, nor demonstrate a particular representation-change pathway. Shared-state cross-talk and preference-family contamination remain hypotheses.

The next priority is parameter-path attribution of the **task-stream actor gradient**. Keep the preference-stream gradients, critic update, dual update, preference sampling, and endpoint protocol intact while routing only the task-stream gradient to selected actor parameter groups. See the [FC-F contract](../../contracts/teacher_v4/teacher-v4-fcf-parameter-path-contract.md), frozen before FC-F branches. FC-D3 R-vertex feasibility remains a separate formulation question, but is not a prerequisite for this attribution.
