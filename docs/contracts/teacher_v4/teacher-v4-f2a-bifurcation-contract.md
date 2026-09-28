# Teacher V4 — F2-A Matched-Seed Bifurcation Contract (rare gentle gait)

Status: **DRAFT 2026-09-28. Not frozen. No F2 metric has been computed.** The script was smoke-tested only on an excluded-lineage run (V4-C3 G1-1 s73101).
Branch: `v4-c2-semantic-preservation`
Provenance: [F2-0 inventory](teacher-v4-f2-0-provenance-inventory.md)

## Question

Among code- and fold-matched V4-C runs, what observable historical or
checkpoint-replay differences precede or accompany the rare low-cost
locomotion capability of V4-C G1-2 s73102?

No causal claim, no PASS/FAIL. The output is a characterization.

## Runs

| tier | runs | role |
|---|---|---|
| target | V4-C G1-2 s73102 | — |
| primary matched | V4-C G1-2 s73101, s73103 | same code (`822cbcb`, pre-fix normalizer), same fold, same training support {3, 4} |
| secondary same-code | V4-C G1-3 s73101–73103 | same code, other fold (support {2, 4}); does the pattern hold across folds? |
| excluded | s73104–73108 (post-fix normalizer, F2-B only) and every cross-lineage run in F2-0 | never used in F2-A |

F2-A compares the target against **2 matched-fold controls**. The 3
secondary runs are reported separately.

## Evidence types

- `h_*` = `online_log` (`metrics.jsonl`, every iteration). A historical
  claim is allowed. "Onset" is used only here.
- `p_*` = `checkpoint_replay` (model_50 … model_300). Capability at that
  checkpoint only; no historical claim. Timestamps are `t_probe_*`.
  Advantage recomputed on the probe (`p_adv_*`) supports "the checkpoint
  already had a different credit geometry", never "training advantage
  diverged".

Not available: episode length (not logged), realized preferences (F2-0:
contract-level only), online gait or contacts.

## F2-A1 — historical trajectories (h_*)

Series: reward_per_step T, A, O (mean over sampled preferences);
explained_variance T, A, O; preference_authority; plant_authority; log_std
mean; entropy; kl; clip_frac; surrogate; value loss; termination_fraction.
h_t0 = mean of iterations 1–10, reported per run.

**Divergence onset:** the first iteration from which the target stays
outside [min, max] of the two primary controls for 10 consecutive
iterations. The same is reported against all five controls. The series are
raw, with no smoothing.

## F2-A2 — fixed-probe checkpoint replay (p_*)

The same protocol for every checkpoint and run, with no policy-conditioned
warm-up. Initial states come from env resets with seeds 910001 and 910002 ×
256 envs, so they are identical across checkpoints and runs. Each rollout
is 319 steps, deterministic (`act_inference`, K from the checkpoint).

Conditions: **C primary** (center), **T⁺ secondary** (`heavy_w(K, 0)`).

Per checkpoint and condition:

- gait and task, over steps 33–128, excluding envs that terminated before
  step 128: touchdown-step fraction td, weighted linear tracking tl, the F1
  class (same rules), R = [T, D, O] (F1 formula), ‖ω_xy‖, tilt, ‖q̈‖, ‖q̇‖,
  action rate;
- credit, on t < 64 with the MC256 target (G1-R `mc_targets`, no
  bootstrap): critic EV per objective T, A, O; advantage A = G − V per
  objective, its std, and its correlations corr(A_T, A_A) and corr(A_T, A_O).

## F2-A3 — objective-space geometry

Primary displays, per run and condition: absolute pT vs |pD| and pT vs
|pO| over checkpoints 50 → 300, with an arrow from p_t0 = checkpoint 50.
Also ΔR(u) = R(u) − R(50). Any ratio η = ΔT / (|ΔD| + ε) is shown only next
to ΔT and |ΔD|.

## F2-A4 — temporal ordering (target, per condition)

Properties per checkpoint:

- motion: class ≠ standing;
- contact: td ≥ 0.02;
- locomotion: class ∈ {partial, established};
- credit divergence: at least one credit metric (EV T / A / O, advantage std
  T / A / O, corr TA / TO) outside [min, max] of the two primary controls
  at that checkpoint.

For each property: t_probe_first (first checkpoint showing it) and
t_probe_persistent (first checkpoint from which every later checkpoint
shows it). Also C vs T⁺ locomotion order:

- T⁺ before C: capability existed before the center could use it;
- together: a more global basin transition;
- C first: a center-specific path.

## F2-A5 — characterization (pre-declared, primary = C)

Let u = t_probe_first(locomotion, C). A precursor must appear at or before
u − 50, because the capability arose somewhere in (u − 50, u].

| condition | label |
|---|---|
| no locomotion at C in any checkpoint | unresolved |
| u = 50 (locomotion at C already at the first checkpoint) | unresolved: before available resolution |
| exploration h-onset (log_std, entropy, kl, clip_frac) ≤ u − 50, and a credit precursor | mixed |
| exploration h-onset ≤ u − 50 only | exploration precursor |
| credit precursor only: EV or value-loss h-onset ≤ u − 50, or p credit divergence at a checkpoint < u | credit precursor |
| neither, and locomotion at C persistent from u | basin-entry-like |
| neither, and not persistent | unresolved: no precursor observed, locomotion not persistent |

Wording rule: "no precursor observed at available resolution", never "no
precursor existed".

## Deliverables

Replay: `runs/teacher_v4_f2a-2026-09-28/replay/<fold>_seed<seed>/f2a_replay.json`
(6 runs). Analysis: `runs/teacher_v4_f2a-2026-09-28/f2a.json`. Script:
`f2a_bifurcation.py` (`--mode replay`, `--mode analyze`). `R_shared`
intervention stays closed until F2-A is closed.
