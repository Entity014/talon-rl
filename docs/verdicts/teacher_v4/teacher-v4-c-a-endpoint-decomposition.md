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
