# Teacher V4 — FC-A R Credit / Gradient / Virtual-Step Audit (read-only)

Status: **FROZEN 2026-09-29 (r2), with `fca_credit_audit.py` and `fca_aggregate.py`, before any FB-2a checkpoint was audited.** Pipeline smoke-tested on a scratch 50-iteration checkpoint only.
Branch: `v4-c2-semantic-preservation`
Follows: [FC-0 verdict](../../verdicts/teacher_v4/teacher-v4-fc0-rotational-realization-audit-verdict.md). The measurement was checked and no defect found. The failure is **R semantic authority**: the policy responds to the preference, but R⁺ does not move toward lower rotation.

## Question

Where does the R signal get lost in the PPO update? Credit strength,
gradient interference with T or O, the credit direction itself, or drift
across updates?

## Data (training-like, FB-2a formulation)

FB-2a seeds 79101–79103.

- **Checkpoint 600: primary.** This is the endpoint semantic-authority
  failure. λ ≈ 0 in almost every region, so g_T ≈ 0 here. It tests R vs O
  and the local R direction.
- **Checkpoint 300: mechanism secondary** (layers 1–2 only, no vote). λ is
  still active, so it tests R vs task interaction. Without it, no R-vs-T
  claim is made.

For each fixed preference
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
   Also reported: ‖g_R,logσ‖ / ‖g_R‖.
3. **Virtual steps** (checkpoint 600; R⁺ and C). The directions come from the
   **locomoting-slice** gradients g^loco (primary). If there are fewer than
   1000 locomoting samples, the result is unresolved, with no fallback to
   all samples. The directions are R-objective ascent (−g_R), its opposite
   (+g_R), and mixed ascent (−Σg), with the **log_std component zeroed**.
   This is a physical probe of the controller *mean*: the deterministic
   replay never uses log_std, and log_std could otherwise use up the KL
   budget. The full gradient stays in layer 2. Each step is sized by
   bisection to a batch KL (which here is induced by the mean change only)
   of 0.005, 0.01 and 0.02. Then a deterministic replay (reset seeds 910001
   / 910002, steady window 33–128, surviving envs): F_rate (‖ω_xy‖²), F_osc,
   F_O, tl. Nothing is saved or trained.
   **Null check:** the baseline is replayed twice. The ±5 % semantic gate
   is usable only if the null repeat differs by < 5 % in F_rate. Otherwise
   the seed's virtual-step result is uninterpretable. The threshold is not
   revised after seeing the null.

## Pre-declared reading (per seed, checkpoint 600, R⁺, loco slice; majority ≥ 2/3)

"Lowers" means F_rate falls ≥ 5 % against the baseline at KL 0.01 and at
0.02. "Raises" means it rises ≥ 5 % at both.

Causal class, led by the virtual steps:

| condition | class |
|---|---|
| R-ascent step raises F_rate | **R credit direction wrong** (next: advantage / critic / trajectory credit) |
| R-ascent step lowers F_rate, mixed step does not | **local R direction correct; mixed cancellation / multi-objective interaction** |
| both lower F_rate | **local update correct; the problem is later / across updates** |
| otherwise | unresolved |

Independent flags, appended to the class and never overriding it:

- **credit-weak:** s_R < 0.35 (under half its 0.7 preference weight);
- **R/O interference:** cos(g_R, g_O) ≤ −0.3;
- **R/T interference:** cos(g_R, g_T) ≤ −0.3 (meaningful at checkpoint 300,
  where λ > 0).

The verdict reads, for example, "local R direction correct; mixed
cancellation, with credit-weak and R/O interference". A flag is appended
when it holds in ≥ 2/3 seeds.

The C condition, the R vertex, O⁺ and the all-samples slice are reported
descriptively. The R realization stays `ang_vel_xy_l2`. F_osc is logged
but no realization change is made (FC-0 gave no grounds).

Scripts: `fca_credit_audit.py --seed <s> --checkpoint …/model_600.pt` (primary) and `… model_300.pt --no-steps` (secondary); `fca_aggregate.py`.
Output: `runs/teacher_v4_fc-2026-09-29/fca/seed<s>/` and `seed<s>_300/`, aggregate `fca_aggregate.json`.
