# Teacher V4 — FC-A R Credit / Gradient / Virtual-Step Audit (read-only)

Status: **DRAFT 2026-09-29. Not frozen. No FB-2a checkpoint audited.** Pipeline smoke-tested on a scratch 50-iteration checkpoint only.
Branch: `v4-c2-semantic-preservation`
Follows: [FC-0 verdict](../../verdicts/teacher_v4/teacher-v4-fc0-rotational-realization-audit-verdict.md). The measurement was checked and no defect found. The failure is **R semantic authority**: the policy responds to the preference, but R⁺ does not move toward lower rotation.

## Question

Where does the R signal get lost in the PPO update? Credit strength,
gradient interference with T or O, the credit direction itself, or drift
across updates?

## Data (training-like, FB-2a formulation)

FB-2a seeds 79101–79103, checkpoint 600. For each fixed preference
condition (R vertex (1, 0), R⁺ (.7, .3), C (.5, .5), O⁺ (.3, .7)): 1024 envs,
50 warm-up + 24 steps, stochastic policy, fixed audit seed. FB-2a streams
[T_lin, R, O] with their T3-B divisors. The checkpoint's own critic (query
ids T, R, O, conditioned on (R, O, w)), GAE, loss weights [λ_region, w_R,
w_O] with the checkpoint's saved λ, and the training advantage
normalization. Every statistic is reported over all samples and over
**locomoting envs** (rollout-mean track_lin ≥ 0.40).

Disclosed: each batch has one fixed w, whereas training batches mix
preferences. So the normalization scale is per condition here.

## Layers

1. **Credit mass:** raw advantage std per stream. Weighted contribution
   C_i = E|ℓ_i A_i| (normalized advantages) and shares. The preference-
   internal R share is s_R = C_R / (C_R + C_O).
2. **Gradient geometry** at ratio = 1: g_T, g_R, g_O, the per-stream actor
   gradients of the FB-2a surrogate. Norms, ‖g_R‖/‖g_T‖, ‖g_R‖/‖g_O‖,
   pairwise cosines, cos with the mixed gradient.
3. **Virtual steps** (R⁺ and C only). An in-memory copy of the actor moves
   along the R-objective ascent direction (−g_R), its opposite (+g_R), and
   the mixed ascent direction (−Σg). Each step is sized by bisection to a
   batch KL of 0.005, 0.01 and 0.02 (PPO's desired_kl is 0.01). Then a
   deterministic replay (reset seeds 910001 / 910002, steady window 33–128,
   surviving envs): F_rate (‖ω_xy‖²), F_osc, F_O, tl. Nothing is saved or
   trained. Isaac dynamics are not differentiable, so this is the only
   layer that checks whether the gradient moves *physical* rotation.

## Pre-declared reading (per seed, at R⁺; majority ≥ 2/3 for the verdict)

Change thresholds: a step "lowers" F_rate if F_rate falls ≥ 5 % against
the baseline at KL 0.01 and at 0.02. "Raises" means it rises ≥ 5 % at both.

| order | condition | class |
|---|---|---|
| 1 | s_R < 0.35 (under half its 0.7 preference weight), loco slice | **credit-strength problem** |
| 2 | R-ascent step lowers F_rate, mixed step does not lower it | **cancellation / multi-objective interaction** (interference flag if cos(g_R, g_O) or cos(g_R, g_T) ≤ −0.3) |
| 3 | R-ascent step raises F_rate | **R credit direction wrong** (next: advantage / critic / trajectory credit) |
| 4 | R-ascent and mixed steps both lower F_rate | **local update correct → long-horizon / across-update drift** |
| 5 | otherwise | unresolved |

The C condition, the R vertex, O⁺ and the all-samples slice are reported
descriptively. The R realization stays `ang_vel_xy_l2`. F_osc is logged
but no realization change is made (FC-0 gave no grounds).

Script: `fca_credit_audit.py --seed <s> --checkpoint …/fb2a/seed<s>/model_600.pt`.
Output: `runs/teacher_v4_fc-2026-09-29/fca/seed<s>/fca_credit_audit.json`.
