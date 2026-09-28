# Teacher V4 — F3 Preference-Invariant Locomotion Substrate Screen

Status: **FROZEN 2026-09-28 (r2) in the commit that sets this line, before any F3 training run.**
Branch: `v4-c2-semantic-preservation`
Follows: [F2-A verdict](../../verdicts/teacher_v4/teacher-v4-f2a-bifurcation-verdict.md)

## Question

What helps V4 policies get from stepping in place to translation? F2-A found
the rare walker moved activity → contact → step-in-place → translation at
low D/O cost, while the controls stalled at the step stage.

This is a mechanism screen, not a fix. **Success does not validate a new
objective. It shows that the frozen T/D/O objective set may require a
preference-invariant locomotion substrate for reliable optimization.**

## Treatment

Frozen: objectives {T, D = A, O}, T3-B divisors, TeacherV4, the V4-C PPO
shell, the dead-critic gate, the current (post-`61a483d`) normalizer, the V4-C
env. Training support: V4-C3 fold G1-1, cardinalities {2, 3}, K = 3.
300 iterations, 4096 envs, checkpoints every 50.

Only the training reward changes:

    r_i = R_i + λ · Σ_{k ∈ arm} r_k          for every objective i
    ⇒ wᵀr = wᵀR + λ · Σ_k r_k               (active weights sum to one)

r_k are the stock-weighted step terms. λ = 1 / 1.71946 (the T divisor). This
puts the shared terms in the same T-normalized units as the full T
objective and keeps their stock relative magnitudes. Because R_shared is
preference-invariant, its ratio to the preference-weighted T contribution
grows as w_T falls: at C (w_T = 1/3) it is about 3× M0's ratio. That is a
consequence of preference invariance, not a scale error.
R_shared has no preference weight, is not an objective, and is never used
in evaluation. Tracking terms are never in R_shared, because that would
raise T's effective weight everywhere.

| arm | R_shared | H_help | H_collapse |
|---|---|---|---|
| B0 `none` | — | frozen control | — |
| B1 `linz` | lin_vel_z_l2 | a vertical-motion constraint turns stepping into controlled forward motion | rewards immobility (standing has v_z ≈ 0) |
| B2 `torque_acc` | dof_torques_l2 + dof_acc_l2 | regularization widens the low-cost gait basin | rewards immobility (standing has low torque and q̈) |
| B3 `air` | feet_air_time | a gait prior makes a repeatable stepping cycle | may revive a costly M0-style gait |
| B4 `all` | B1 + B2 + B3 | the roles act together (stock-like substrate) | — |

B4 never runs alone. It is interpreted only next to B1–B3.

**Implementation consequence (disclosed).** R_shared is added to every
objective's reward stream, not given its own value head. Each critic head
therefore learns R_i + λR_shared. Advantage centering is per objective over
its active samples, and all objectives share one scale (the std of
Σ w_i A_i). So V_T ≠ V(R_T): critic EV compares different targets across
arms and is **descriptive only** in F3. Behavioral replay measures the frozen
raw T, D, O without R_shared, so it is unaffected. A separate V_shared head
with A(w) = Σ w_i A_i + A_shared is a later refinement, only if F3 confirms
that R_shared is needed. Code:
`--shared` in `train_v4c.py`, `SHARED_ARMS` / `shared_vector` in
`talon_rl/rewards/objectives.py`, tested in `tests/rewards/test_objectives.py`.

## Seeds

s73101–73103 are selection data (s73102 is the positive exemplar), so they
are never used here.

- **Screen:** 74101, 74102, 74103, for every arm including B0 (15 runs).
- **Confirmation:** 74201, 74202, 74203, only for screen-positive arms, with
  a concurrent B0.

Both seed sets are frozen now.

## Measurement

Every checkpoint (50 … 300) of every run is replayed with the frozen F2-A
protocol (`f2a_bifurcation.py --mode replay`): the same reset suite (910001 /
910002 × 256 envs), deterministic, conditions C and T⁺, the F1 classes. The
F2-A reset-fingerprint check applies.

Per run and condition: t_contact (first td ≥ 0.02), t_step (first class ≥
step-in-place), t_translate (first class ∈ {partial, established}),
t_translate_persistent, and the final class, tl, td, R, and critic EV.

## Pre-declared reading (primary = C)

A run **locomotes at C** if t_translate_persistent(C) exists: translation at
C on at least two consecutive checkpoints through 300 (e.g. 250 and 300).
Translation at 300 only is reported as "late locomotion, persistence
unresolved" and does not count toward screen-positive.

| per arm (3 screen seeds) | reading |
|---|---|
| ≥ 2/3 locomote at C **and** B0 ≤ 1/3 | **screen-positive**, go to confirmation |
| 0/3 locomote at C **and** more T⁺-standing runs at 300 than B0 | **collapse** (H_collapse): the substrate rewards immobility |
| otherwise | no screen signal |

Confirmation: the same rule on 74201–74203 against the concurrent B0. An arm
is **confirmed** only if it is screen-positive on both seed sets.

**Guardrail (objective compatibility)**, for positive arms: at least 2 of
the locomoting runs must have final C cost R_D ≥ −0.23 and R_O ≥ −0.48 (half
of M0's cost). If they do not, the reading is "gait discovery improved,
objective-compatible gait not recovered".

## Dose audit (read-only, informational, before freezing)

`runs/teacher_v4_f3_dose_audit-2026-09-28.json`, from existing V4-C
substrate-attribution levels. λ was **not** tuned on it. There was no
numerical catastrophe, so λ = 1 / d_T stands. Score at C, wᵀR + λR_shared:

| behavior | none | linz | torque_acc | air | all |
|---|---|---|---|---|---|
| standing (G1-2 s73101 C) | 0.128 | 0.123 | 0.108 | 0.128 | 0.102 |
| step-in-place (G1-2 s73106 T⁺) | 0.111 | 0.078 | 0.089 | 0.104 | 0.049 |
| gentle gait (G1-2 s73102 C) | 0.217 | 0.068 | 0.176 | 0.203 | 0.014 |
| M0 | −0.089 | −0.097 | −0.153 | −0.104 | −0.176 |

Recorded before training: the observed gentle gait carries a large
lin_vel_z penalty (λ·r = −0.148, against −0.008 for M0). Under `linz` and
`all`, it scores below standing at C. `feet_air_time` is negative for every
observed V4 behavior (steps shorter than the stock 0.5 s threshold). This
predicts H_collapse pressure for B1 and B4 against the only observed gentle
gait. It is a static comparison of observed behaviors, not a result.

## Secondary (descriptive)

- T⁺ before C (t_translate order), per run.
- The progression t_contact → t_step → t_translate.
- Final objective vectors, touchdown and tracking.
- Online preference authority, the shared reward per step and critic EV
  (last 10 iterations; EV descriptive only, see above).
- F1-style J comparison of the new locomoting behaviors against the
  standing envelope.

## Out of scope

No critic repair, no exploration or entropy change, no objective or divisor
change. If every arm fails, an actor/exploration contract opens separately.

Aggregate: `f3_screen.py --stage screen|confirm --root runs/teacher_v4_f3-2026-09-28`.
Run layout: `runs/teacher_v4_f3-2026-09-28/<arm>_seed<seed>/` (training) and
`.../replay/f2a_replay.json`.
