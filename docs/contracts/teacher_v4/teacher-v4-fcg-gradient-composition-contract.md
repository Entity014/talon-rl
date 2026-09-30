# Teacher V4 — FC-G gradient-composition candidate

Status: **FROZEN 2026-10-01, before FC-G branch outcomes.** Live-rollout preflight is a launch gate; no branch may train until it passes. Follows the [FC-E](../../verdicts/teacher_v4/teacher-v4-fce-paired-replication-verdict.md) and [FC-F](../../verdicts/teacher_v4/teacher-v4-fcf-parameter-path-verdict.md) verdicts. No new actor architecture is introduced.

## Question and evidence boundary

Does the **single global actor-gradient clip** couple task and preference updates enough to raise matched Δ_R during mixed-preference training? FC-E replicated an effect of the task-pressure *loss formulation*. FC-F found seed-dependent routing responses and different global clip coefficients, but did not isolate clipping as the mechanism or rule out a network path. Gradient composition is the next hypothesis to test, not an established root cause. FC-F also did not isolate Adam or historical optimizer state, so their importance cannot yet be ranked low from that result.

## One intervention at a time

Keep the FC-F/FULL network, task-pressure loss weights, **single shared advantage normalization**, per-stream clipped PPO surrogate, preference sampling, critic/dual updates, entropy coefficient, Adam state, and adaptive KL rule. Compute the same actor gradients on each arm's minibatch:

- `g_T`: task-stream surrogate;
- `g_R + g_O`: preference-stream surrogate;
- `g_H`: entropy term;
- `g_N = g_R + g_O + g_H`: all non-task actor terms.

The entropy term must be accounted for explicitly. Treating `g_N` as the protected non-task block keeps its loss coefficient unchanged, although its **effective** step changes when clipping changes. Value loss remains on critic parameters only.

Let `C(g) = min(1, 1/(||g|| + 1e-6)) g` for the frozen actor max norm of 1. Candidate arms:

| Arm | Gradient supplied to the existing Adam actor optimizer | Purpose |
|---|---|---|
| GLOBAL (reference) | `G0 = C(g_T + g_N)` | Reproduce current FULL update exactly |
| SPLIT | `G1 = C(g_T) + C(g_N)` | Clip task and non-task streams independently |
| SPLIT-NORM-MATCHED | `G2 = G1 × ||G0|| / ||G1||` when `G1 ≠ 0`; zero otherwise | Hold the combined **pre-Adam** norm equal to GLOBAL while changing stream composition |

SPLIT can have a combined norm greater than 1. This is an intended treatment difference, not a silent violation of the old cap: record its norm, KL, adaptive LR, and optimizer step. SPLIT-NORM-MATCHED tests whether a benefit survives when that combined norm is matched to GLOBAL on the same minibatch. It does **not** guarantee equal Adam parameter-step norms, since Adam moments and coordinates matter. Do not add another global clip after SPLIT: that would change the registered intervention.

Separate task/preference **advantage normalization** is a different intervention and is excluded from this first clipping test. Likewise, no G1–G7 routing mask, parameter freeze, new adapter, or change to λ is part of these arms.

## Live-rollout preflight before training

On a shared live rollout/minibatch from each primary u0 checkpoint, log `||g_T||`, `||g_R||`, `||g_O||`, `||g_H||`, `||g_N||`, `||g_T+g_N||`, their task/non-task cosine, `c_global`, `c_T`, `c_N`, the three combined gradient norms, Adam parameter-delta norms, KL, and LR. Check stream gradients sum to the legacy pre-clip actor gradient. GLOBAL must match the legacy update **after** clipping and optimizer step, including parameter deltas, Adam `exp_avg`/`exp_avg_sq`, LR and KL. Check SPLIT's task and non-task coefficients separately and SPLIT-NORM-MATCHED's norm identity, including zero-gradient and strongly clipped cases. The critic path and dual update must match across arms on the same batch.

Also measure how often the live FULL update gives `c_global < c_N`, which directly quantifies task-associated suppression of the non-task gradient on that batch. A difference in branch-mean clip coefficients alone does not establish this within-update relation.

For every PPO minibatch in every arm log `c_T/c_N`, `cos(G0,G2)`, the fraction of minibatches with `|c_T/c_N − 1| > 0.05`, and the fraction with `c_global < c_N − 1e-6`. The trainer writes each row to `gradient_minibatches.jsonl` and iteration means/fractions to `metrics.jsonl`. The 5% coefficient difference is a numerical exposure threshold, not a hypothesis-test p value. Report means and the distribution across iterations; if coefficients are nearly equal and `cos(G0,G2)` nearly one, record that the intervention had little directional exposure. If both `G0` and `G2` are zero, define cosine as one; if exactly one is zero, define it as zero. These audit quantities are descriptive and never change the registered sequential decision rule.

## Registered outcome design

Use the same u0 checkpoints and 50-iteration mixed continuation as FC-F, with primary seeds 79101 and 79103. Use **fresh** paired branch indices r = **11, 12, 13** in stage 1; add **14, 15** only for open seed × contrast combinations. Branch seed is seed + 10⁶ + 1000r for all arms of a pair. Do not reuse FC-E/FC-F branches as votes. Initial budget is 18 branches (3 arms × 2 seeds × 3 repeats); maximum is 30 if both primary contrasts remain open everywhere. A common GLOBAL branch is shared by both contrasts at a given seed and repeat.

At the FC-C endpoint (mean of viable k = 30, 40, 50 checkpoints), define primary paired effects in percentage points: `E_split = 100[matched Δ_R(GLOBAL) − matched Δ_R(SPLIT)]` and `E_composition = 100[matched Δ_R(GLOBAL) − matched Δ_R(SPLIT-NORM-MATCHED)]`. Positive means attenuation. Report corresponding raw effects alongside. `E_dose = 100[matched Δ_R(SPLIT-NORM-MATCHED) − matched Δ_R(SPLIT)]` is a mechanism diagnostic, never a headline gate or an expansion trigger. A pair is invalid if either endpoint has fewer than two viable checkpoints or no matched value.

Stage 1: all three pairs must be valid. All three positive and median > +10 pp is **replicated attenuation**; all three negative and median < −10 pp is **replicated worsening**; each |effect| < 10 pp and |median| < 5 pp is **within practical band**. Otherwise **open**. Stage 2: among all five pairs, at least four valid positive and median of valid pairs > +10 pp is replicated attenuation; at least four valid negative and median < −10 pp is replicated worsening; at least four valid inside (−10,+10) pp and |median| < 5 pp is within practical band. Otherwise **not resolved by registered budget**. No stage 3. The practical band is descriptive, not a formal equivalence test. A cross-seed composition claim requires replicated attenuation of `E_composition` in both primary seeds; `E_split` is read separately.

An attenuation of paired matched Δ_R is a mechanism signal. Calling **R semantics restored** requires both mean raw and mean matched Δ_R ≤ −5% at the endpoint; this is stricter than FC-B's raw-only correct state and uses FC-C's paired reading to avoid tracking confounding. Viability reuses replay `tl(R⁺) ≥ 0.40` and `tl(C) ≥ 0.40` at ≥2 endpoint checkpoints. O preservation reuses FC-B's sign control, `Δ_O = F_O(O⁺)/F_O(C)−1 < 0`, at ≥2 viable endpoint checkpoints; FC-B registered no magnitude threshold, so FC-G adds none. KL, entropy and `log_std` are reported and must pass existing trainer stop gates; no post-hoc cutoff is introduced. Report raw Δ_R, Δtl, λ trajectories, clip coefficients and actual Adam parameter deltas descriptively. Analyze each seed separately before any cross-seed claim.

Train with `--gradient-composition ARM`, `--branch-seed r`, and FC-F's same checkpoint, TAO/FB-2 settings, PPO configuration, 50 iterations and checkpoint cadence. Set `--out runs/teacher_v4_fc-2026-10-01/fcg/seed<s>/ARM_r<r>/`. Replay with the FC-C/FC-E deterministic trace protocol and classify with `fcg_probe.py`. The preflight compares cloned model, optimizer, rollout minibatch and RNG states at `atol=1e-6, rtol=1e-5` for stream-sum gradient, GLOBAL first parameter delta, Adam moments, LR, KL and clip coefficient. Failed preflight blocks branch launch; it does not authorize changing arm definitions or thresholds after seeing outcomes.

If SPLIT and SPLIT-NORM-MATCHED both replicate improvement while viability and O control hold, relative gradient composition is implicated under this optimizer. If only SPLIT improves, increased total gradient dose is a live explanation. If neither improves, the test does not prove a network path; Adam history, other PPO interactions, or distributed parameter effects remain. A working FC-G treatment would justify testing the existing policy family without an architectural change, not claim that MoE or adapters are unnecessary in general.
