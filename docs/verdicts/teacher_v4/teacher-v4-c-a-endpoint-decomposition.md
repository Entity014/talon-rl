# Teacher V4 — V4-C A-Endpoint Decomposition (post-hoc, descriptive)

Status: **DESCRIPTIVE — no gate, no verdict change; classification rule chosen after the G1 results**
Date: 2026-09-27
Data: the six frozen `g1_evaluation.json` (iteration 300); nothing re-run.

## Question

In the m = 4 anchors where A fails (4/6), and in every other set containing
A, is the A-heavy endpoint wrong in direction, a proxy mismatch, or too weak?

## Method

For each A endpoint (A-heavy vs center, 4 suites): objective delta (A
reward, higher is better) and physical delta (mean ‖ω_xy‖, lower is
better), with the stored correct-fractions and survival. Post-hoc rule:

- A5 near-flat: |physical delta| < 0.005 rad/s;
- otherwise by the sign of the mean deltas:
  A1 both correct, A2 objective correct + physical wrong,
  A3 objective wrong + physical correct, A4 both wrong.

7 sets contain A per run → 42 endpoints.

## Result

| class | all 42 | failing A endpoints (26) |
|---|---:|---:|
| A1 both correct | 21 | 5 (correct on average, but correct in only 2/4 suites) |
| A2 objective ✓ physical ✗ | 4 | 4 |
| A3 objective ✗ physical ✓ | 0 | 0 |
| A4 both wrong (reversal) | 10 | 10 |
| A5 near-flat | 7 | 7 |

m = 4 anchors:

| run | class | physical delta (rad/s) | stored gate |
|---|---|---:|---|
| G1-2 s73101 | A1, suite-inconsistent (objective 2/4) | −0.018 | fail |
| G1-2 s73102 | A1 | −0.163 | pass |
| G1-2 s73103 | A2 | +0.007 | fail |
| G1-3 s73101 | A1 | −0.008 | pass |
| G1-3 s73102 | A2 | +0.014 | fail |
| G1-3 s73103 | A4 | +0.025 | fail |

Per run, class of the A endpoint in {TA, AO, AS, TAO, TAS, AOS, TAOS}:

    G1-2 s73101  1 5 5 1 1 5 1
    G1-2 s73102  1 5 1 1 1 1 1
    G1-2 s73103  1 4 1 1 1 5 2
    G1-3 s73101  1 2 4 1 4 1 1
    G1-3 s73102  1 4 1 5 5 2 2
    G1-3 s73103  4 4 1 4 4 4 4

## Three distinct patterns, not one mechanism

1. **Seed-level reversal.** G1-3 s73103 is A4 in 6 of 7 sets: making A heavy
   makes angular stability worse and lowers the A reward. That is a learning
   or credit failure for A in that seed, not a proxy problem. G1-2 s73102 is
   the opposite, clean A1 everywhere.
2. **A–O coupling.** The {A,O} set never yields A1 (A5 ×2, A4 ×3, A2 ×1), and
   its authority is the lowest of any set (pairwise retention 0.42–0.97).
   Angular velocity in roll and pitch is the rate of the tilt that O
   penalizes, so A-heavy and O-heavy ask for nearly the same behavior and
   the policy barely separates them.
3. **Weak or inconsistent effect.** The seven A5 endpoints and the five A1
   endpoints that are correct on average but only in 2/4 suites are
   small-effect cases, and the sign does not hold across suites.

Proxy mismatch (A2) is minor: 4/42 endpoints, 2 of them anchors, all small
(|physical delta| ≤ 0.014 rad/s). A3 never occurs. The A reward and the
‖ω_xy‖ metric agree in direction whenever the objective moves correctly and
the effect is not tiny.

Survival confounds four failures, all in G1-3 with S present ({A,S}
s73101 and s73103, {T,A,S} s73101 and s73103; A-heavy survival 0.12–0.62).

## Link to the critic result (G1-R)

The critic for A is invalid under MC256, with predictions varying 2× more
than true 256-step returns. Patterns 2 and 3 fit a weak, entangled A signal:
the true A return differs little between A-heavy and O-heavy (or center),
so the critic's A estimate varies with preference more than the return
does. Pattern 1 fits a seed whose A value and policy moved the wrong way
together. Neither link is tested yet.

## Candidate next checks (not run)

- A–O entanglement: correlation of A and O rewards per step, and
  ‖a(A-heavy) − a(O-heavy)‖ on the probe states, per run.
- G1-3 s73103 reversal: A-heavy vs center on checkpoints 50…250
  (diagnostics only), to see whether A was ever correct and then reversed.
- A advantage share: magnitude of the A advantage vs the others in the
  scalarized update.

## Follow-up: A–O entanglement audit (read-only, 2026-09-27)

Script `ao_entanglement.py` (commit `7dfa165`), output `ao_entanglement.json`
per run. m = 4 set, G1 m = 4 suite seeds, center + four heavy endpoints;
probe states with the physical-nominal e_t. Each A–O number is ranked among
the six objective pairs; no thresholds.

Medians over the six runs:

| pair | reward corr | action dist (probe) | action dist (rollout states) | V3 G0 action dist (probe) | outcome dist |
|---|---:|---:|---:|---:|---:|
| TA | 0.02 | 1.25 | 1.12 | 0.50 | 0.0042 |
| TO | 0.05 | 1.05 | 0.89 | 1.25 | 0.0036 |
| TS | −0.03 | 1.29 | 1.14 | 0.54 | 0.0050 |
| **AO** | **0.25** | 0.98 | 0.73 | 1.14 | 0.0025 |
| AS | 0.15 | **0.76** | **0.58** | **0.34** | **0.0022** |
| OS | −0.03 | 1.04 | 0.62 | 0.87 | 0.0040 |

A–O rank per run (reward corr by |value|, 6 = most correlated; distances,
1 = least separated):

| run | reward corr | action (probe / rollout) | outcome |
|---|---|---|---|
| G1-2 s73101 | 0.20, 5/6 | 3 / 3 | 1 |
| G1-2 s73102 | 0.18, 6/6 | 5 / 3 | 4 |
| G1-2 s73103 | 0.13, 5/6 | 3 / 3 | 1 |
| G1-3 s73101 | 0.32, 6/6 | 2 / 3 | 3 |
| G1-3 s73102 | 0.29, 6/6 | 2 / 2 | 2 |
| G1-3 s73103 | 0.56, 6/6 | 5 / 3 | 1 |

Reading, against the interpretation fixed beforehand:

- Reward overlap: A–O is the most correlated pair in 4/6 runs and 5th in the
  other two, but moderate in size (median 0.25; 0.56 only in G1-3 s73103).
- Behavioral separation: the policy does separate A-heavy from O-heavy
  actions. A–O is mid-ranked (2–5 of 6) and never the least separated pair.
  The least separated pair is A–S, in V4 and in the Phase-1 G0 model.
- Outcome separation: low. A–O is the least separated outcome pair in 3/6
  runs, and A–S has the lowest median.

So the case is "reward overlap exists, the policy disentangles the actions,
but the outcomes barely separate". The strong-support pattern (all three
low or high together) does not hold. **A–O entanglement is at most a partial
explanation.** A–S shows the same or lower separation on every axis.

In the reversal seed, G1-3 s73103, O-heavy reduces ‖ω_xy‖ below center
(0.288 vs 0.311 rad/s) while A-heavy raises it (0.336). There, the O
preference achieves A's physical goal better than the A preference does.
That points to a learning or credit failure specific to A in that seed,
which the temporal check (checkpoints 50…250) can locate.

## Follow-up: A checkpoint ladder (read-only, 2026-09-27)

Script `a_checkpoint_ladder.py` (commit `8c24942`), output
`a_checkpoint_ladder.json` in each run directory. m = 4 set, G1 m = 4 suite
seeds, checkpoints 50…300 (diagnostic only; the G1 contract fixes 300).
At iteration 300 the ladder reproduces the stored ΔJ_A of all three runs.
Runs: G1-3 s73103 (reversal), G1-2 s73102 (clean A), and G1-3 s73101
(same fold as s73103, A anchor passes), added to remove the fold confound.

"+" means A-heavy is better than center. ‖ω‖ is mean ‖ω_xy‖ (rad/s) for
center / A-heavy / O-heavy. Critic A is the MC256 EV (registered / pooled)
and the prediction-to-target variance ratio.

G1-3 s73103:

| it | ΔJ_A (suites ✓) | Δ‖ω‖ improvement (✓) | ‖ω‖ C / A / O | auth A–C | critic A EV | var× |
|---:|---|---|---|---:|---|---:|
| 50 | −0.00045 (0/4) | −0.034 (0/4) | 0.325 / 0.360 / 0.470 | 0.61 | −17.8 / −14.2 | 12.2 |
| 100 | −0.00068 (0/4) | −0.087 (0/4) | 0.323 / 0.411 / 0.414 | 1.06 | −63.9 / −12.9 | 9.5 |
| 150 | −0.00047 (0/4) | −0.063 (0/4) | 0.340 / 0.403 / 0.354 | 1.02 | −57.7 / −8.4 | 5.7 |
| 200 | −0.00029 (0/4) | −0.024 (0/4) | 0.330 / 0.354 / 0.329 | 1.06 | −26.5 / −2.9 | 2.6 |
| 250 | −0.00059 (0/4) | −0.048 (0/4) | 0.333 / 0.382 / 0.307 | 1.09 | −31.0 / −1.7 | 2.0 |
| 300 | −0.00039 (0/4) | −0.025 (0/4) | 0.311 / 0.336 / 0.288 | 1.12 | −12.5 / −2.7 | 3.7 |

G1-2 s73102:

| it | ΔJ_A (✓) | Δ‖ω‖ (✓) | ‖ω‖ C / A / O | auth | critic A EV | var× |
|---:|---|---|---|---:|---|---:|
| 50 | −0.00023 (0/4) | −0.010 (0/4) | 0.315 / 0.326 / 0.506 | 0.52 | −31.8 / −3.9 | 7.7 |
| 100 | +0.00002 (2/4) | +0.020 (4/4) | 0.330 / 0.311 / 0.356 | 0.88 | −28.7 / −14.2 | 13.6 |
| 150 | +0.00073 (4/4) | +0.117 (4/4) | 0.422 / 0.305 / 0.363 | 0.91 | −4.2 / −3.6 | 4.0 |
| 200 | +0.00066 (4/4) | +0.103 (4/4) | 0.427 / 0.324 / 0.396 | 0.87 | −5.9 / −1.5 | 2.7 |
| 250 | +0.00126 (4/4) | +0.175 (4/4) | 0.480 / 0.305 / 0.424 | 0.92 | −1.9 / +0.07 | 1.3 |
| 300 | +0.00113 (4/4) | +0.163 (4/4) | 0.495 / 0.332 / 0.419 | 0.86 | −1.8 / +0.29 | 1.3 |

G1-3 s73101:

| it | ΔJ_A (✓) | Δ‖ω‖ (✓) | ‖ω‖ C / A / O | auth | critic A EV | var× |
|---:|---|---|---|---:|---|---:|
| 50 | −0.00005 (1/4) | +0.001 (2/4) | 0.333 / 0.332 / 0.494 | 0.46 | −11.7 / −0.3 | 0.4 |
| 100 | −0.00052 (0/4) | −0.046 (0/4) | 0.391 / 0.437 / 0.370 | 0.57 | −114.9 / −1.7 | 1.4 |
| 150 | −0.00013 (1/4) | −0.017 (2/4) | 0.385 / 0.402 / 0.375 | 0.54 | −75.8 / −1.3 | 1.5 |
| 200 | +0.00011 (4/4) | +0.017 (3/4) | 0.371 / 0.354 / 0.365 | 0.53 | −50.7 / −1.2 | 3.2 |
| 250 | +0.00007 (3/4) | −0.000 (3/4) | 0.354 / 0.354 / 0.347 | 0.53 | −66.2 / −0.2 | 1.8 |
| 300 | +0.00015 (4/4) | +0.008 (3/4) | 0.336 / 0.329 / 0.317 | 0.54 | −9.2 / +0.2 | 1.9 |

Survival is 1.00 for A-heavy and center at every checkpoint in all three runs.

### Reading against the cases set beforehand

- **s73103 is case 2 (early basin):** A is wrong at every checkpoint from 50
  on, in 0/4 suites each time. It is never correct and then reversed. A
  authority stays high (0.6–1.1), so the policy responds to A, but in the
  wrong direction.
- **The seeds diverge between iterations 50 and 100.** At iteration 50 all
  three runs look alike: A is weak or wrong, and the A critic is invalid. By
  100, s73102 has flipped to correct (Δ‖ω‖ 4/4) and s73103 has moved
  further wrong (−0.087).
- **Behavior leads the critic (case 4 ordering).** In s73102, A behavior is
  correct from iteration 100–150, while the A critic is at its worst at
  iteration 100 (pooled EV −14.2, variance 13.6×). The critic becomes
  marginally valid only at 250–300. In s73103 the critic improves steadily
  (pooled −14 → −2.7) while the behavior never does. Critic validity neither
  precedes nor tracks A semantics, which argues against the critic being
  the cause.

### An unexpected pattern: the center moves, A-heavy does not

A-heavy's absolute ‖ω_xy‖ at iteration 300 is nearly the same in all three
runs (0.332 / 0.336 / 0.329 rad/s). What differs is the center policy: it
drifts to 0.495 in s73102 and stays at 0.311 in s73103. The A endpoint
"passes" when the rest of the policy family trades angular stability away
(presumably for tracking), and "fails" when it does not. There is no seed in
which A-heavy reaches a distinctly lower ‖ω_xy‖ than about 0.33 rad/s. In
s73103, O-heavy goes lower (0.288). This looks like a floor, or a weak A
gradient near it, rather than a sign error in A.

Not yet tested: whether about 0.3 rad/s is a physical floor for this gait
at the commanded speeds, or an A-specific learning limit (O-heavy reaching
0.288 suggests it is not a hard floor).

## Follow-up: A-credit audit (read-only, 2026-09-27)

Script `a_credit_audit.py` (commit `86b1f0a`), output `a_credit_audit.json`
per run. At checkpoints 50, 100 and 150: 1024 envs, 50 warm-up steps, then a
24-step batch collected as in `train_v4c.py` (fold cardinalities, the
checkpoint's own critic, training advantage normalization). At ratio 1 the
actor loss splits exactly into per-objective parts L_i; g_i = ∇ L_i on the
actor parameters.

| run | it | gradient share T / A / O / S | cos(g_i, g_mixed) T / A / O / S | cos(g_A, g_O) | cos(g_A, g_S) |
|---|---:|---|---|---:|---:|
| G1-3 s73103 (reversal) | 50 | .12 / **.22** / .23 / .43 | −.37 / **+.45** / −.17 / +.79 | −.56 | +.18 |
| | 100 | .19 / **.28** / .26 / .26 | +.40 / **+.55** / +.29 / +.24 | −.23 | +.01 |
| | 150 | .20 / **.21** / .28 / .31 | +.48 / **+.52** / +.23 / +.62 | −.34 | +.31 |
| G1-2 s73102 (clean A) | 50 | .09 / **.33** / .40 / .18 | +.27 / **+.11** / +.22 / +.47 | −.87 | +.45 |
| | 100 | .29 / **.20** / .26 / .24 | +.58 / **+.34** / +.25 / +.35 | −.41 | +.20 |
| | 150 | .32 / **.17** / .24 / .27 | +.53 / **+.36** / +.42 / +.54 | −.12 | +.41 |
| G1-3 s73101 (control) | 50 | .05 / **.23** / .63 / .08 | +.04 / **+.79** / +.95 / −.05 | +.58 | +.21 |
| | 100 | .02 / **.16** / .78 / .03 | −.06 / **+.71** / +.99 / −.02 | +.60 | −.08 |
| | 150 | .03 / **.34** / .58 / .04 | +.22 / **+.90** / +.97 / −.07 | +.78 | −.09 |

Weighted surrogate shares follow the same pattern (A: .21–.27 in s73103,
.19–.34 in s73102, .13–.25 in s73101).

Reading, against the cases set beforehand:

- **"A under-powered" is not supported.** A's gradient share is 0.16–0.34
  everywhere, around its fair share of 0.25, and it is not smaller in the
  reversal seed.
- **The A signal is as large and as aligned in s73103 as in s73102,** yet the
  outcomes are opposite. In the reversal seed, g_A is at least as aligned
  with the mixed update (+0.45 to +0.55) as in the clean seed (+0.11 to
  +0.36). This matches both "strong, aligned A gradient without closed-loop
  improvement" and "similar signal, different outcome → seed basin /
  trajectory geometry".
- **A–O gradient conflict exists, but it does not discriminate the seeds.**
  cos(g_A, g_O) is negative in both s73102 and s73103, and positive in the
  control. The control is dominated by O instead (gradient share
  0.58–0.78).

The open question this leaves: in s73103, A's gradient is strong and
aligned, but its direction does not improve true A. That gradient is built
from advantages of an A critic whose MC256 EV at these checkpoints is
between −14 and −8 (ladder above). If those A advantages do not correlate
with real A returns, the policy is pushed hard in a direction the critic
invented. A direct check is the correlation between GAE A-advantages and
MC256-based A-advantages on the same batch, per seed.
