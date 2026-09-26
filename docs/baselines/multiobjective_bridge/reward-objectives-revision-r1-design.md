# Final Reward/Objectives Revision R1 — design

Status: design and offline-audit specification only. No reward code changed,
no retraining started, and no new PPO/gradient intervention opened.

R1 follows Final Locomotion Evaluation v1. That evaluation is frozen as a
failed baseline. R1 is deliberately limited to Progress, Balance's relationship
to functional locomotion, and the conceptual placement of `hip_activation`.
Efficiency and Impact formulas remain fixed unless later evidence directly
implicates them.

## Evidence boundary

The strongest case is baseline seed 1: 892/960 uniform trials survived, yet
surviving forward-command behavior had low displacement, poor command tracking,
and high tilt. Survival therefore cannot stand in for locomotion. Across all
uniform trials, only 526/2,880 met the movement criterion and 548/2,880 met the
tilt criterion. All four heavy-preference contrasts were NOT-DEMONSTRATED
because their locomotion prerequisites failed.

The current Progress kernel is

`exp(-(v_x - command_x)^2 / 0.5^2) + 0.04 * foot_air_time_reward`.

Before the air-time term, standing still receives 0.779 at a 0.25 m/s command,
0.368 at 0.50 m/s, and 0.105 at 0.75 m/s. A wrong-direction velocity of
-0.04 m/s still receives 0.312 at a 0.50 m/s command. Meanwhile Balance gives
an unconditional `alive_bonus=1` on each nonterminal step. These formulas make
the survived-but-stationary result structurally plausible; the conclusion does
not depend on reopening the gradient-tail investigation.

## R1 change table

| Current sub-reward | Observed failure | Proposed R1 change | Expected behavioral effect | Regression guardrail |
|---|---|---|---|---|
| Progress velocity kernel: `exp(-error²/std²)` | Large positive credit remains available at zero or wrong-direction velocity, especially for low commands | For nonzero commands, multiply tracking quality by a signed engagement gate: `clip(sign(c_x)*v_x / max(abs(c_x), eps), 0, 1)`. Retain the tracking kernel so overspeed remains penalized. Treat zero command separately with the existing zero-velocity tracking kernel | Standing and reverse motion receive zero forward-progress credit; correct direction receives increasing credit and exact tracking remains optimal | At `c_x=0`, stable standing must remain rewarded. At every nonzero test command: `R(correct tracking) > R(under-speed) > R(standing) >= R(wrong direction)`. No discontinuity or division by zero near zero command |
| Progress uses instantaneous body-frame `v_x` only | Instantaneous/fall velocity can look useful without net directed displacement; v1 found low or negative displacement | Keep per-step velocity as the trainable dense signal, but add a short-window directed-progress term computed from world displacement projected onto the command heading. Zero it on terminal-fall steps and the existing pre-fall leak window | Rewards sustained translation and makes circling/falling-forward less profitable than continuing along the commanded direction | Window state resets per lane; no reward crosses episode boundaries; stationary jitter has near-zero net credit; reverse command uses the correct sign; no privileged signal unavailable at deployment is introduced |
| Foot-air-time bonus is embedded in Progress | Leg motion can receive credit without proving useful translation | Retain only as a small gait-support regularizer after offline attribution; gate positive air-time credit by the same signed locomotion engagement signal | Swing behavior supports locomotion but cannot independently make stationary stepping score as progress | Removing/gating it must not eliminate stepping for functional-forward episodes; audit its magnitude separately from tracking and displacement |
| Balance `alive_bonus=1` is unconditional while alive | Seed 1 survives without functional locomotion; Balance-heavy can optimize survival rather than the thesis behavior | For nonzero commands, gate the positive alive bonus by a low-threshold locomotion-engagement signal. Keep fall, tilt, tilt-rate, vertical-motion, and height penalties independent so unsafe motion remains penalized. Preserve unconditional standing support for zero command | Balance becomes support for commanded locomotion rather than a stationary-survival objective | Never remove fall or posture penalties when progress is poor. Zero-command survival remains valid. A balance-heavy policy must still pass the same displacement/tracking gate and must not increase falls |
| `hip_activation` bonus lives inside Balance | It rewards bilateral joint activity and is conceptually gait support; activity can increase Balance reward without proving stability or translation | Remove it from Balance. Candidate placement is a gated gait-support component inside Progress, conditional on offline evidence that it separates functional forward motion from stationary/wrong-direction behavior. If it does not separate them, disable it in R1 | Preference semantics become clearer: Balance measures stability; gait activity only earns credit when coupled to locomotion | No reward for raw hip speed alone; verify action/torque saturation and mechanical power do not regress; preserve asymmetric-terrain freedom by avoiding a hard symmetry constraint |
| Balance tilt/height/`v_z` penalties and fall penalty | v1 shows tilt failures, but does not isolate these coefficients as the cause | Keep formulas and coefficients fixed in R1's first candidate. Re-evaluate after Progress/alive/hip restructuring | Isolates the evidence-backed change and avoids broad tuning | Same physical stability criteria, falls, `|v_z|`, height and saturation metrics as v1 |
| Efficiency and Impact objectives | Trade-offs were untestable because locomotion prerequisite failed; no evidence identifies these objectives as the primary bottleneck | No R1 formula or coefficient changes | Clean causal comparison and retained hardware-related penalties | Report their physical metrics exactly as in v1; reopen only with direct post-R1 evidence |

The table defines one coherent R1 candidate, not a threshold sweep. Constants
inside the engagement/window construction must be calibrated offline and then
frozen before retraining. Do not train several variants and select one using
the final evaluation outcomes.

## Offline reward audit before implementation

Classify the frozen v1 episodes without changing their labels:

1. **Functional forward:** survived, positive forward command, and passed the
   locked tracking and directed-displacement criteria.
2. **Survived but stationary/nonfunctional:** survived but failed directed
   displacement or tracking, with absolute directed displacement below the
   command-specific lower bound.
3. **Wrong-direction/falling:** negative directed progress under a positive
   command, or base-contact termination.

For each training seed and command, report class counts and distributions of
current tracking-kernel score plus each proposed R1 component. The minimum
ordering requirement is:

`functional forward > survived stationary > wrong-direction/falling`.

Require this ordering within every training seed at `vx=0.25`, `0.50`, and
`0.75`, not only after pooling. Also require that zero-command standing remains
the best zero-command behavior and that reverse-command scoring is sign-correct.
Use effect sizes and overlapping distributions, not only means.

The v1 episode CSVs contain displacement, tracking, fall, tilt, vertical
motion, contact, saturation, energy, and impact summaries. They do **not**
contain the per-step `foot_air_time_reward`, hip velocities, height series, or
the exact pre-fall reward sequence. Consequently they can audit the proposed
tracking/directed-progress ordering at episode level, but cannot exactly
recompute or attribute every current sub-reward. Do not fabricate those
missing values. Before deciding whether to retain/gate/disable air-time and hip
activation, run a read-only replay of the frozen checkpoints that records the
missing per-step inputs; this fills v1's trace auditability gap and is not a
training experiment.

## Acceptance before GPU retraining

R1 may be implemented only if the offline audit shows:

- standing and wrong-direction behavior lose the positive Progress credit
  responsible for the current ordering failure;
- functional-forward episodes rank above both failure classes per seed;
- zero-command and reverse-command controls remain correct;
- Balance never removes physical penalties when locomotion engagement is low;
- hip/air-time decisions are backed by replay data rather than their names;
- Efficiency and Impact remain unchanged.

Record the final equations, constants, code diff, and audit artifact hashes.
Then retrain exactly three seeds for 500 PPO updates and rerun the frozen v1
evaluation matrix and thresholds in a new versioned artifact directory.

## Decision boundary after retraining

- If R1 passes the unchanged locomotion and four preference gates, freeze the
  new locomotion family and proceed to terrain, payload, held-out morphology,
  and Main-vs-Baseline validation.
- If R1 fails, report the specific remaining behavioral failure. Do not reopen
  P99, clipping thresholds, or gradient-tail experiments, and do not broaden
  into untargeted Efficiency/Impact tuning without new evidence.
