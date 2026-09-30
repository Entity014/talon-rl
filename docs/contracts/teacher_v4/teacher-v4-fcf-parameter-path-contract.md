# Teacher V4 — FC-F parameter-path attribution contract

Status: **FROZEN 2026-09-30, before any FC-F branch was trained.** Motivated by the [FC-E verdict](../../verdicts/teacher_v4/teacher-v4-fce-paired-replication-verdict.md). Arm masks, primary seeds, endpoint, staged decision rule and training implementation are fixed here. The live-rollout preflight below is a launch gate; a failed check stops training rather than changing this contract after seeing outcomes.

## One causal question

Given the same rollout, per-stream advantages, and scalarization rule *within an update*, which actor parameter paths must receive the task-stream gradient for the mixed-preference continuation to raise matched Δ_R? Only the routing of the new task-stream actor gradient differs by arm. R and O actor gradients reach every actor parameter in every arm; critic, entropy, dual update, and preference sampling follow the same rules. No module is frozen.

On-policy training cannot hold an identical rollout across arms for all 50 iterations: routed updates change policies and later data. Paired branch seeds and an identical initial rollout reduce noise; each later update computes its streams by the **same formula** from that arm's own rollout. An exactly shared batch is useful for the gradient audit and first-update equivalence check, not as a claim that complete training trajectories share data.

## Exhaustive actor partition

| Group | `TeacherV4` parameters | Meaning |
|---|---|---|
| G1 | `state_trunk`, `env_encoder` | state and plant encoders |
| G2 | `policy_backbone` | shared policy feature layer |
| G3 | `family_w1_base`, `family_b1_base`, `family_w2_base`, `family_b2_base` | preference-invariant base policy θ₀ |
| G4 | `family_B_w1`, `family_B_b1`, `family_B_w2`, `family_B_b2` | preference-modulated residual bases Bₖ |
| G5 | `actor_set_encoder` including its objective embedding | preference set encoder |
| G6 | `family_hyper` | family coefficient network |
| G7 | `log_std` | action distribution scale |

The partition must be disjoint and cover `actor_parameters()` exactly. In particular, θ₀ (G3) and Bₖ (G4) have different architectural meanings and must never be pooled for the attribution analysis. Critic parameters are outside this partition.

## Candidate task-gradient routes

`g_R` and `g_O` always reach G1–G7. For the main screen, `g_T` reaches G7 in every arm, so stochasticity routing is held constant. A separate G7 contrast can follow if its contribution remains uncertain.

| Arm | Groups receiving `g_T` | Question isolated |
|---|---|---|
| FULL | G1–G7 | All-on reference; must reproduce A's actor update |
| NO-PREF-PATH | G1, G2, G3, G7 | Does the effect need Bₖ, preference encoder, or hypernetwork? |
| NO-BASES | G1, G2, G5, G6, G7 | Does the effect need θ₀ or Bₖ? |
| SHARED-FEATURES-ONLY | G1, G2, G7 | Can shared feature paths alone carry the effect, with shared base θ₀ blocked? |
| PREF-ONLY | G4, G5, G6, G7 | Can preference-conditioned paths alone carry it? |

The screen does not distinguish every member of a multi-group path. If NO-BASES changes the effect, separate **NO-θ₀** (all except G3) from **NO-Bₖ** (all except G4). If NO-PREF-PATH changes it, separate G4, G5, and G6 masks. A dedicated shared-path test can block G1 or G2 independently. Register any adaptive follow-up rule before looking at outcome data.

Use FC-E's u0 checkpoints, paired branch seeds, mixed-preference sampling, and matched Δ_R endpoint. Report raw Δ_R, per-arm tracking viability, Δtl, Δ_O, λ, entropy, and `log_std` alongside it. Do not decide from a single branch or binary inversion labels.

## Registered replication rule

The screen uses seeds **79101 and 79103 separately**; 79102 remains a secondary consistency check only if budget permits. Use fresh repeat indices **r = 6–10**, since FC-E already observed r = 1–5. For each seed and repeat, run one FULL reference and the four routed arms with the same `--branch-seed r`. For each routed arm, the primary paired effect is `matched Δ_R(FULL) − matched Δ_R(routed arm)` in percentage points. Positive values mean blocking that task-gradient route reduced the harmful separation change. Raw paired effects are reported alongside. A pair is invalid if either branch collapses or has no matched endpoint. The previously observed FC-E branches are contextual only and do not vote in FC-F.

Stage 1 uses r = 6, 7, 8. A contrast is **replicated attenuation** if all three pairs are valid, all effects are positive, and the median is > +10 pp. It is **replicated worsening** if all three are valid, all effects are negative, and the median is < −10 pp. It is **within practical band** if all three are valid, each effect has absolute value < 10 pp, and the absolute median is < 5 pp. Otherwise it is open.

Stage 2 adds r = 9, 10 only for open seed × routed-arm contrasts. The FULL branch for a seed/repeat is shared across its open contrasts. On all five pairs, classify **replicated attenuation** with at least four valid positive effects and median > +10 pp; **replicated worsening** with at least four valid negative effects and median < −10 pp; or **within practical band** with at least four valid effects inside (−10,+10) pp and absolute median < 5 pp. Everything else is **not resolved by the registered budget**. No stage 3. The practical-band class describes these observed effects; it is not a formal equivalence test or proof of no contribution.

Only a class reproduced in **both primary seeds** supports a cross-seed path reading. A class in one seed only is seed-dependent. If FULL fails its own viability gate in a repeat, all comparisons sharing it are invalid. The 10 pp threshold follows FC-E's registered effect scale; the 5 pp practical-band center keeps “preserved” distinct from a small systematic shift. The thresholds and transitions are implemented with self-checks in `fcf_probe.py`. Initial budget: 30 branches (5 arms × 2 seeds × 3 repeats); maximum screen budget: 50 branches if every contrast expands. Any G3/G4/G5/G6 split is a separately preregistered follow-up.

## Training and artifacts

Resume from `runs/teacher_v4_fb-2026-09-29/fb2a/seed79101/model_450.pt` or `seed79103/model_350.pt`, continue 50 iterations (to 500 or 400), and save every 10. All arms use `--objectives TAO --cardinalities 1,2 --num-envs 4096 --lagrange-tmin 0.52 --lagrange-lambda0 0.786 --lagrange-eta 0.15 --lagrange-cap 20 --loss-arm full` and the existing default PPO constants. Set `--seed`, `--resume`, `--iterations`, `--branch-seed r`, `--task-grad-route ARM`, and `--out runs/teacher_v4_fc-2026-09-30/fcf/seed<s>/ARM_r<r>/` for each branch. The trainer records the route, run seed, and clip summaries in its outputs. Replay checkpoints with the same `f2a_bifurcation.py --traces` protocol as FC-E, then run `fcf_probe.py --stage 1|2`. Output is `fcf_stage1.json`, `expand.txt`, and, when needed, `fcf_stage2.json`.

## Gradient construction and preflight

The current `objective_set_ppo.update` normalizes all advantage streams with a **single scale** based on weighted summed advantages. For every routed arm, compute GAE streams `A_T`, `A_R`, `A_O` from that arm's rollout before minibatching, then use the **FULL/A** loss weights, mask, centering, and common scale. Never recompute normalization after selecting a route. Decompose the existing per-stream clipped PPO surrogate into task, R, and O terms using the same minibatch and log-probability ratio, then form their parameter gradients separately. Apply a group mask to `g_T` only; add unmasked `g_R + g_O`, the unchanged entropy gradient, and the separately isolated critic gradient before the existing actor norm clip and optimizer step. Do not use `loss_w[:, 0] = 0` as a routing mechanism.

Compute the actor's **pre-clip global gradient norm** and realized clip coefficient for every PPO minibatch and arm: `c_clip = min(1, max_grad_norm / (norm + 1e-6))` for finite norm, matching `clip_grad_norm_` (here `max_grad_norm = 1`). Log their mean, minimum, and fraction clipped per iteration and over the branch. Routing can change the global norm, so post-clip R/O gradients can differ even when their pre-clip terms do not. The registered interpretation is **routing under the existing PPO optimizer**, not a pure vector-path effect before clipping or Adam.

Before any branch training, audit one shared rollout/minibatch and verify:

1. Every actor parameter belongs to exactly one of G1–G7; no critic parameter appears.
2. FULL's sum of per-stream gradients, including entropy, matches the existing A gradient and first parameter/optimizer update. Lock the tensor comparison at `atol=1e-6, rtol=1e-5` for pre-clip gradients, parameter deltas, and Adam `exp_avg` / `exp_avg_sq`; compare LR, KL, pre-clip norm, and clip coefficient at `atol=1e-6, rtol=1e-5`. Compare from cloned model, optimizer, batch, and RNG states, including the first complete PPO minibatch step. Any failed check blocks branch training.
3. For every mask, the **routed** task gradient norm is zero in blocked groups and equals the unmasked task gradient in allowed groups; R/O and entropy contributions stay unchanged across masks. Use a batch with nonzero task gradients where expected.
4. Value-loss gradients reach critic parameters only. Check finiteness, gradient clipping, KL/LR schedule, and resumed Adam state. An Adam momentum term from u0 may move a parameter even when the *new* task gradient is blocked; report this rather than interpreting total parameter motion as task-gradient flow. FC-F identifies routes of **newly generated task gradients after u0**; it does not erase historical task influence already stored in parameters or optimizer state at u0.

The intervention can identify a necessary or sufficient **route under the tested training context**; PPO clipping, global gradient norm clipping, Adam, and changed later rollouts can couple group effects. If blocking G4–G6 attenuates the effect relative to FULL, preference-conditioned parameters are implicated and private bases/adapters become a targeted intervention to test. If SHARED-FEATURES-ONLY falls within the practical band against FULL while FULL retains its FC-E-like endpoint, G1/G2 can carry the effect. If no route changes the effect, audit scaling and PPO interactions before claiming that parameter path is irrelevant. FC-D3 R-vertex feasibility remains a separate formulation question, not a prerequisite for this attribution.
