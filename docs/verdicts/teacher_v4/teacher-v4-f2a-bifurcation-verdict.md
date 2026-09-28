# Teacher V4 — F2-A Bifurcation Verdict (rare gentle gait, V4-C G1-2 s73102)

Status: **FROZEN. Registered label: "actor-side precursor, matched-fold only". It rests on one specific signal, clip fraction (onset iteration 26, q_null 0.065), which is not cross-fold robust and precedes locomotion by ~270 iterations. Every other historical and credit signal is nonspecific or ambiguous. The replay shows a gradual low-cost path: joint activity higher from the first checkpoint, then contact (150), stepping in place (150–250), then translation (300). T⁺ reaches locomotion one checkpoint before C.**
Date: 2026-09-28
Branch: `v4-c2-semantic-preservation`
Contract: [teacher-v4-f2a-bifurcation-contract.md](../../contracts/teacher_v4/teacher-v4-f2a-bifurcation-contract.md) (r3, frozen at `4da5810`, before any replay)
Output: `runs/teacher_v4_f2a-2026-09-28/f2a.json`, replays in `replay/` (`f2a_bifurcation.py`)

F2-A does not establish why s73102 walks. It reports which precursor
class the available evidence fits best, at 50-iteration replay resolution,
with no online gait data and no realized-preference record (F2-0).

Initial states: the reset fingerprints matched for every run, checkpoint and
condition, so the replays start from the same initial states.

## 1. C locomotion capability (replay)

Target at C (class, tl, td, R = [T, D, O]):

| it | class | tl | td | R |
|---|---|---|---|---|
| 50 | standing | 0.27 | 0.001 | [0.29, −0.01, −0.15] |
| 100 | standing | 0.27 | 0.008 | [0.32, −0.01, −0.13] |
| 150 | step-in-place | 0.27 | 0.067 | [0.44, −0.02, −0.06] |
| 200 | step-in-place | 0.29 | 0.079 | [0.48, −0.02, −0.05] |
| 250 | step-in-place | 0.37 | 0.114 | [0.55, −0.04, −0.05] |
| 300 | partial | 0.48 | 0.135 | [0.64, −0.04, −0.05] |

t_probe_first(locomotion, C) = t_probe_persistent = 300. That is the last
checkpoint, so persistence cannot be checked beyond it.

None of the five controls' C ever leaves standing (td ≤ 0.012, tl 0.26–0.27
at every checkpoint).

## 2. T⁺ vs C order

T⁺ locomotion first at 250, C at 300: **T⁺ before C.** The capability
appears under T-heavy preference one checkpoint before the center can use
it. Among the controls, T⁺ reaches stepping in place in 4/5 runs (by 150–250)
but never translates (tl ≤ 0.31).

## 3. Specific historical signals (h_*, vs the matched envelope)

| group | signal | onset (matched / same-code) | q_null | class |
|---|---|---|---|---|
| actor update | clip_frac | 26 / — | 0.065 | **specific**, matched-fold only |
| actor update | kl | — / — | — | no onset |
| stochasticity | log_std, entropy | 26–27 / 29–30 | 0.77 | nonspecific |
| credit | EV T, A | 28, 12 / 28, 19 | 0.87, 0.58 | nonspecific |
| credit | EV O | 6 / 25 | 0.32 | ambiguous |
| credit | value loss | 11 / — | 0.48 | ambiguous |
| other | reward T, A, O; authority; surrogate; termination | 3–95 | 0.23–0.71 | ambiguous or nonspecific |

Most series leave the two-control envelope within the first 30 iterations.
The null shows ordinary same-code runs do so just as early. Only clip_frac
is rarer than that.

## 4. Specific replay signals (p_*)

- Activity (‖q̇‖, action rate, ‖q̈‖ above both matched controls): from
  checkpoint 50 at C, persistent. The target is already more active at the
  first checkpoint. This is before the available resolution, and activity
  has no null calibration.
- Contact (td ≥ 0.02): first and persistent at 150 (C and T⁺).
- p credit divergence: at 50, q_null 1.0, **nonspecific.** Every pseudo-target
  also diverges at the first checkpoint.

## 5. Timing (u = 300, window (250, 300])

The only specific signal, clip_frac, is a precursor by the rule (26 ≤ 250).
There are no specific accompaniments.

## 6. Robustness

clip_frac leaves the matched envelope but not the five-control envelope,
so the qualifier is "matched-fold only".

## Reading

- The registered label rests on a single actor-update signal. It is 274
  iterations before locomotion and not cross-fold robust. That is weak
  evidence for an actor-side mechanism, and it does not select one.
- **The replay ordering is the clearer finding:** activity (≤ 50) → contact
  (150) → stepping in place (150–250) → translation (T⁺ 250, C 300).
  Acquisition is gradual across checkpoints, not a jump between two
  checkpoints. The target moves through the same step-in-place stage the
  controls' T⁺ reach, and then goes on to translate.
- **Low-cost path (case B).** T rises 0.29 → 0.64 while D stays within
  −0.04 and O improves (−0.15 → −0.05). The gait is found directly in the
  low-D/O-cost region. It is not a costly gait refined later.
- The only early difference in replay is higher joint activity at the first
  checkpoint. It fits an early actor-side difference (clip_frac at 26 is
  consistent with it), but checkpoint 50 is the resolution floor.

## Implication for the next contract (not decided here)

The evidence does not point to a credit mechanism: every credit signal is
nonspecific or ambiguous. It is weakly consistent with an early actor-side
difference, and clearly shows a gradual stand → step → translate
progression that the controls stall at the step stage. That favors
interventions that help policies past the step-in-place stage toward
translation (gait shaping or R_shared) over critic repair. This is an
inference for choosing the next experiment, not an F2-A result.
