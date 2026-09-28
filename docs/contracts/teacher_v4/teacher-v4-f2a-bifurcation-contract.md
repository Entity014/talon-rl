# Teacher V4 — F2-A Matched-Seed Bifurcation Contract (rare gentle gait)

Status: **FROZEN 2026-09-28 (r3) in the commit that sets this line, before any F2 replay or analysis. No F2 metric had been computed.** Smoke tests used only excluded-lineage runs (V4-C3 G1-1 s73101, s73103), including a synthetic end-to-end analyze.
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
outside [min, max] of the controls **on one side** (all below, or all above)
for 10 consecutive iterations. The series are raw, with no smoothing.

- `h_onset_matched`: envelope of the two primary controls (primary).
- `h_onset_samecode`: envelope of all five same-code controls. An onset
  under both is *cross-fold robust*. One under matched only is *matched-fold
  only*.

**Specificity null.** A two-run envelope is narrow, so any run can leave it.
The same detector runs on pseudo-targets: each of the five same-code
controls against every pair of the other four (5 × 6 = 30 configurations).

q_null = (1 + #{null onsets ≤ target onset}) / 31. A null `None` counts as
later than any finite onset, and ties count against the target.

| q_null | class | use |
|---|---|---|
| ≤ 0.20 (at most 5 of 30 at least as early) | specific | may drive the F2-A5 label |
| 0.20 < q < 0.50 | ambiguous | reported only |
| ≥ 0.50 | nonspecific | reported only |

**q_null is not a p-value.** The 30 configurations reuse the same control
runs and are not independent. It is an empirical rarity calibration of the
onset detector: how often an ordinary same-code run gives an onset this
early. The null bank deliberately includes cross-fold pairs, because it
calibrates the detector, not the G1-2 distribution. Inference on the target
still uses the G1-2 s73101 / s73103 envelope.

Specificity and cross-fold robustness are separate. A signal is
*cross-fold robust* if `h_onset_samecode` exists (the target also leaves the
five-control envelope). A specific but matched-fold-only signal may drive
the label, and the label then carries the qualifier "matched-fold only"
(or "partly cross-fold robust" / "cross-fold robust").

## F2-A2 — fixed-probe checkpoint replay (p_*)

The same protocol for every checkpoint and run, with no policy-conditioned
warm-up. Initial states come from env resets with seeds 910001 and 910002 ×
256 envs, so they are identical across checkpoints and runs. Each rollout
is 319 steps, deterministic (`act_inference`, K from the checkpoint). This
measures the mean-policy capability. Training stochasticity is covered by
h_log_std and h_entropy. A fixed-noise stochastic probe is a possible
extension only if F2-A ends unresolved.

Conditions: **C primary** (center), **T⁺ secondary** (`heavy_w(K, 0)`).

Per checkpoint and condition:

- gait and task, over steps 33–128, excluding envs that terminated before
  step 128: touchdown-step fraction td, weighted linear tracking tl, the F1
  class (same rules), R = [T, D, O] (F1 formula), ‖ω_xy‖, tilt, ‖q̈‖, ‖q̇‖,
  action rate;
- credit, with the MC256 target (G1-R `mc_targets`, no bootstrap): critic EV
  per objective T, A, O; advantage A = G − V per objective, its std, and its
  correlations corr(A_T, A_A) and corr(A_T, A_O). **Primary window t = 32–63**,
  after the reset transient. t = 0–31 is reported descriptively (`*_t0_32`).
- reset-state fingerprint: sha256 of policy obs, privileged obs, root state,
  joint pos and joint vel right after each reset, before the first action.
  `analyze` checks that each seed gives one fingerprint across all 6 runs,
  all checkpoints and both conditions. If it does, the claim is "same initial
  states". If not, the claim is downgraded to "same reset seed".

## F2-A3 — objective-space geometry

Primary displays, per run and condition: absolute pT vs |pD| and pT vs
|pO| over checkpoints 50 → 300, with an arrow from p_t0 = checkpoint 50.
Also ΔR(u) = R(u) − R(50). Any ratio η = ΔT / (|ΔD| + ε) is shown only next
to ΔT and |ΔD|.

## F2-A4 — temporal ordering (target, per condition)

Properties per checkpoint:

- activity (three separate properties): ‖q̇‖, action rate and ‖q̈‖ each
  above the max of the two primary controls at that checkpoint;
- contact: td ≥ 0.02;
- locomotion: class ∈ {partial, established};
- credit divergence: at least one primary-window credit metric (EV T / A /
  O, advantage std T / A / O, corr TA / TO) outside [min, max] of the two
  primary controls at that checkpoint. The same q_null rule applies to its
  first checkpoint (30 pseudo-target configurations, specific if q ≤ 0.20).
  Cross-fold robust if the target also diverges from the five-control
  envelope at some checkpoint.

The intended reading of the ordering: joint/action activity rises, then a
touchdown pattern appears, then translation.

For each property: t_probe_first (first checkpoint showing it) and
t_probe_persistent (first checkpoint from which every later checkpoint
shows it). Also C vs T⁺ locomotion order:

- T⁺ before C: capability existed before the center could use it;
- together: a more global basin transition;
- C first: a center-specific path.

## F2-A5 — characterization (pre-declared, primary = C)

Let u = t_probe_first(locomotion, C). The capability arose somewhere in
(u − 50, u]. Timing of a specific signal at time t:

- t ≤ u − 50: **precursor**;
- u − 50 < t ≤ u: **within-resolution accompaniment**;
- t > u: **follows** the capability.

Signal groups (h onsets use `h_onset_matched`):

- stochasticity: log_std, entropy;
- actor update: kl, clip_frac;
- credit: explained_variance T / A / O, value loss, and p credit divergence
  (at a checkpoint ≤ u − 50 = precursor, at u = accompaniment).

KL and clip fraction are actor-update dynamics, not exploration.

| condition | label |
|---|---|
| no locomotion at C in any checkpoint | unresolved |
| u = 50 | unresolved: before available resolution |
| actor-side (stochasticity or actor-update) and credit precursors | mixed |
| actor-side precursor only | actor-side precursor (subtype listed) |
| credit precursor only | credit precursor |
| no precursor, at least one accompaniment | unresolved: change accompanies acquisition within replay resolution |
| no precursor, no accompaniment, locomotion at C persistent from u | basin-entry-like |
| no precursor, no accompaniment, not persistent | unresolved: no precursor observed, locomotion not persistent |

Wording rule: "no precursor observed at available resolution", never "no
precursor existed". Cross-fold robustness (`h_onset_samecode`) is reported
for every signal and does not change the label.

## Deliverables

Replay: `runs/teacher_v4_f2a-2026-09-28/replay/<fold>_seed<seed>/f2a_replay.json`
(6 runs). Analysis: `runs/teacher_v4_f2a-2026-09-28/f2a.json`. Script:
`f2a_bifurcation.py` (`--mode replay`, `--mode analyze`). `R_shared`
intervention stays closed until F2-A is closed.
