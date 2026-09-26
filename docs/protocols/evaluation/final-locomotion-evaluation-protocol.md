# Final Locomotion Evaluation Protocol v1

Defined: 2026-09-20. Status: protocol specified; evaluation not run.

All numerical acceptance thresholds below are **project-defined evaluation
criteria**, not thresholds supplied by the proposal, published benchmarks,
or hardware safety limits. They define a modest straight-line locomotion
milestone. They were specified after earlier diagnostics were available,
so this is prospective final evaluation, not preregistration of the earlier
experiments. Freeze this document and the execution manifest before collecting
final results; do not relax criteria after seeing those results.

Experiment 3 remains closed, P99 remains rejected, and no reward edits,
retraining, threshold search, or Experiment 4 belong to this protocol.

## Scope and candidates

Use `runs/phase1_hipact_dt01_seed{0,1,2}_2026-09-20/checkpoints/checkpoint.pt`.
These are three independent training seeds. Do not replace a poor seed or
count the extra seed-0 baseline run as an independent seed. Evaluation reset
seeds add repeated trials, not additional training seeds. Three training seeds
meet the lower end of the proposal's 3–5-seed requirement; uncertainty remains
substantial at this sample size.

Each checkpoint records `t=12000`, where `t` is rollout environment steps.
With the recorded `num_steps=24` per PPO update, this is `12000/24 = 500`
training updates, not 12,000 training iterations.

This evaluates the current non-P99 candidate family on a level, rigid plane,
with nominal A1 morphology, no added payload, no pushes, and no observation
noise. Disable terrain/command curricula and randomization of physical
parameters. Retain the checkpoint's controller, action scaling, observations,
reward configuration, and trained inference inputs. Record actual nominal
physical values in the manifest. If this setup cannot be reconstructed, stop
with an invalid evaluation, not a policy failure.

Passing is permission to freeze this locomotion milestone and proceed to
thesis-scale validation, not evidence of passing proposal §3.4 Gate 1 in full.
Terrain, payload, held-out morphology, Main/Baseline A/B, hardware adaptation,
and sim-to-real remain separate validation requirements (§3.7).

## Evaluation matrix

Full Cartesian matrix, including the standing and reverse controls:

| Axis | Values |
|---|---|
| Training seed | 0, 1, 2 |
| Command `(vx, vy, yaw_rate)` | `(-0.25,0,0)`, `(0,0,0)`, `(0.25,0,0)`, `(0.50,0,0)`, `(0.75,0,0)`; m/s and rad/s |
| Preference | uniform, progress-heavy, efficiency-heavy, impact-heavy, balance-heavy |
| Evaluation reset seed | 1001, 1002, 1003 |
| Trials per evaluation seed per cell | 64 lanes, one episode per lane |
| Episode | 2 s settling at zero command, then 20 s at the specified command |

Total: 75 training-seed/command/preference cells, 192 trials per cell,
14,400 episodes. Locomotion acceptance uses the uniform cells. Preference
acceptance uses all heavy preferences at forward commands 0.25/0.50/0.75;
reverse and zero-command heavy-preference cells are reported as controls.

Preference order is `(progress, efficiency, impact, balance)`:

| Preference | Effective vector |
|---|---|
| uniform | `(0.25, 0.25, 0.25, 0.25)` |
| progress-heavy | `(0.55, 0.15, 0.15, 0.15)` |
| efficiency-heavy | `(0.15, 0.55, 0.10, 0.20)` |
| impact-heavy | `(0.15, 0.10, 0.55, 0.20)` |
| balance-heavy | `(0.15, 0.15, 0.10, 0.60)` |

Pass these through the existing preference floor function and assert the
result matches the table to numerical tolerance. Floors are progress 0.15,
impact 0.05, balance 0.15. Hold preference constant from reset through episode
end; do not resample it at an automatic reset. This tests static preference
conditioning, not safe within-episode switching or HLP scheduling.

## Execution and measurement conventions

Use deterministic actor inference with training disabled. Freeze observation
and reward normalization statistics; do not call PPO updates. Reuse identical
initial-state draws across commands/preferences within each evaluation seed.
Reset to the trained nominal pose with horizontal offsets sampled uniformly
in ±0.05 m and yaw in ±0.05 rad; preserve nominal height and zero initial
velocities. Save sampled states so pairing does not rely solely on RNG seeds.

Use Case 3's 0.01 s policy step only after verifying the effective simulator
step and decimation produce that interval. Collect 200 settling and 2,000
command steps. Extend the evaluation time limit to 22 s; keep physical failure
terminations unchanged and record their exact definitions. A shorter native
timeout is not successful completion. Disable automatic resets for scoring or
mask each lane permanently at its first termination, capturing terminal
signals before reset. Never splice multiple attempts into one episode.

A physical failure during settling counts as an unsuccessful trial. No failed
trial is replaced. A fall means any existing non-timeout physical-failure
termination; retain separate reason counts. Numerical failures are failures,
with their incidence separately reported; missing instrumentation makes the
relevant decision invalid rather than a pass.

Score kinematics during the full 20 s command interval (including acceleration).
Measure displacement along the initial horizontal heading in world coordinates,
relative to the position at command onset. Reverse progress is its negative.
Use body-frame vx for command MAE and world vertical velocity for `|vz|`.
Report lateral displacement and yaw drift independently to expose circling.

For each episode, compute time means, temporal percentiles, and maxima as
specified below, then aggregate episodes equally. Failed episodes remain in
success denominators; their observed physical statistics are reported in a
separate all-trials table. Never impute zero energy/impact after failure.
Kinematic/energy trade-offs are scored only on paired successful episodes,
alongside all-trials failure rates. No result can pass on these conditional
statistics unless the all-trials success gate passes too.

## Episode success and locomotion criteria

An episode succeeds only if it survives all 22 s and satisfies every applicable
row below during the command interval. Thresholds intentionally require actual
translation and controlled motion, not merely a high reward.

| Measure | Per-episode acceptance |
|---|---|
| vx tracking MAE | ≤ `max(0.05, 0.20*abs(vx_command))` m/s |
| Directed displacement, nonzero command | 0.70–1.30 times `abs(vx_command)*20` m |
| Standing, zero command | horizontal net displacement ≤ 0.20 m, horizontal path length ≤ 1.0 m |
| Lateral drift | maximum absolute displacement from initial heading line ≤ 0.50 m |
| Heading drift | maximum absolute wrapped yaw change ≤ 20 degrees |
| Roll/pitch | temporal p95 of `max(abs(roll),abs(pitch))` ≤ 15 degrees; maximum ≤ 30 degrees |
| Vertical motion | mean `abs(vz_world)` ≤ 0.15 m/s and temporal p95 ≤ 0.40 m/s |
| Contact/support | any undesired non-foot contact in ≤ 1% of steps; no all-feet-off interval > 0.20 s |
| Torque saturation | ≤ 5% of joint-step samples at ≥ 95% of each effective absolute torque limit |
| Action saturation | ≤ 5% of joint-step samples at ≥ 95% of the final action bound |

Foot contact is normal force > 1 N. Also report number of supporting feet,
duty factor per foot, slip speed during contact, and body height distributions.
Do not require maximum foot contact count: standing on four feet is not the
target gait. Use applied torque after actuator limiting; record requested
versus applied torque when available. Measure action saturation at the actual
bounded controller input, not an unbounded latent Gaussian action.

Each uniform preference / command / training-seed cell must have:

- Success fraction ≥ 90% (at least 173 of 192 trials).
- Fall fraction ≤ 5% (at most 9 of 192 trials).

All 15 uniform cells must pass. Report 95% Wilson intervals for success/fall
fractions, but the locked decision uses observed fractions, not an implied
95%-confidence guarantee. Also report falls per observed simulated minute,
including settling, and time-to-failure. Do not pool seeds to hide a failed cell.

## Energy and impact: physical measures

Report these for every cell independently of normalized reward:

- Mechanical absolute power `P = sum_j abs(tau_applied_j*qdot_j)` in W;
  mechanical energy `integral P dt` in J over the command interval.
  These are mechanical proxies, not electrical battery consumption.
- For nonzero commands, energy per directed metre and mechanical CoT
  `energy/(mass*g*directed_displacement)`. Standing or nonpositive progress
  has undefined CoT; never substitute zero.
- Per-episode temporal p95 and maximum of the maximum foot normal force,
  divided by body weight; touchdown peak force per foot over the first 0.10 s
  after contact onset; report undesired-contact forces separately.
- Foot normal impulse integrated over time, action-rate and joint-acceleration
  statistics as secondary measures. Mean support force alone is not impact.

Energy and force have no absolute hardware pass threshold in this milestone:
no justified electrical budget or structural force limit is available here.
Their acceptance criteria are the relative trade-off tests below. This does
not establish hardware suitability.

## Preference trade-off criteria

Compare each heavy preference against uniform on the same checkpoint,
command, and initial states. Each heavy-preference forward cell must first
pass the same success/fall gates and episode criteria as uniform. Thus lower
energy/impact achieved by standing, moving too slowly, or falling fails.

Use the intersection of successful paired trials to compute ratios of mean
per-episode metrics. Require at least 154 paired successful trials per cell;
report retained count and both arms' failures. Undefined denominators or
insufficient paired trials mean the trade-off is not demonstrated, not passed.

| Contrast vs uniform | Primary improvement required |
|---|---|
| progress-heavy | vx MAE decreases ≥ 10%, and ≥ 0.005 m/s in absolute terms |
| efficiency-heavy | mean mechanical power AND energy per directed metre decrease ≥ 10% |
| impact-heavy | mean episode p95 peak-foot force / body weight decreases ≥ 10% |
| balance-heavy | mean episode RMS `sqrt(mean(roll²+pitch²))` decreases ≥ 10% |

For balance-heavy, mean absolute vertical speed must additionally not increase
by more than 5%. For impact-heavy, the mean episode maximum foot force must
not increase by more than 5%. All contrasts retain the locomotion gates.

For each contrast, at least two of the three forward commands must meet the
improvement threshold in **each training seed**. At the remaining command,
the contrast's primary metric(s) must not worsen by more than 5%. Report all
four contrasts, including failures; success on one cannot stand in for another.
If the uniform metric is already near zero, retain the rule and label the
contrast as not demonstrated rather than relaxing the improvement threshold.

These are practical effect-size decisions, not a claim of statistical
significance. Report per-training-seed paired effects, ranges across training
seeds, and 95% paired bootstrap intervals within each seed (10,000 resamples,
RNG seed 20260920, stratified by evaluation reset seed). Lanes and time steps
must not be treated as independent training replicates. A thesis-wide
superiority claim still requires its own comparison and statistical analysis.

## Objective mapping audit (initial source-backed mapping)

Source: proposal §3.3/table 3.3 and current `talon_rl/config.py` /
`talon_rl/reward.py`; verify source hashes against evaluated artifacts.

| Proposal | Current implementation | Design rationale and audit limitation |
|---|---|---|
| Progress | Progress | vx-only tracking kernel plus air-time bonus and terminal-fall mask; audit against displacement, lateral/yaw drift, and falls |
| Clearance | Absent from active vector | Deferred until meaningful exteroception signal; not established as absorbed into another objective; no obstacle-negotiation claim here |
| Energy | Part of Efficiency | Current formula is `0.1*energy_reward + smoothness_reward`; efficiency is not a simple rename |
| Smoothness | Part of Efficiency | Shares a preference weight with energy; configuration comments cite correlated signals and reduced preference dimension; report constituent metrics separately |
| — | Balance added | Orientation, height, vertical motion, survival/fall terms and hip terms; inspect enabled coefficients and audit hip activation as gait regularization |

The comments record historical design rationale, not a new validation of that
rationale. Complete the audit after locomotion and preference results, mapping
each enabled sub-term to observed behavior and noting any reward loophole.
An unexplained mapping is a revision candidate, not proof that editing it will
fix training. No reward changes are authorized by this document.

## Artifacts, validity, and decision

Before launching, save checkpoint SHA256s, full effective configs, exact CLI,
code commit plus dirty diff/untracked evaluation sources, simulator/library
versions, device, random seeds, initial states, termination definitions,
physical/action limits, preference vectors, and this protocol's hash. Assert
observation/action/objective dimensions agree with checkpoint tensors rather
than trusting stale descriptive fields. Verify physical signals and terminal
capture with an instrumentation check; exclude that check from final results.
No final scoring until the manifest is complete. Unexpected setup changes
require a documented protocol version, not silent substitution.

Retain episode-level CSV/JSON, per-step physical traces, all cell summaries,
paired comparisons, and fixed representative videos: lane 0 of reset seed 1001
for each checkpoint/command/preference, plus first failed lane when present.
Record normalized rewards only as supplementary diagnostics.

Final report must contain locomotion PASS/FAIL, each preference contrast's
PASS/NOT-DEMONSTRATED result, the mapping audit, and any invalid cells.

- **Acceptable:** every uniform cell and all four trade-off contrasts pass,
  with no unresolved audit evidence that invalidates the measured behavior.
  Freeze all three checkpoint hashes and the reward definition as the
  evaluated family; proceed to thesis-scale validation.
- **Inadequate:** valid evaluation fails any required gate or trade-off claim.
  Report exactly which behavior failed. Open an evidence-backed final
  reward/objectives revision only where the audit supports it, then retrain
  and re-evaluate; do not infer a reward defect from failure alone.
- **Invalid/incomplete measurement:** repair the measurement or setup and run
  the same frozen protocol. This is not a third scientific outcome or grounds
  for reward tuning. No candidate is frozen while required cells are invalid.
