# MDP definition

Ported from `chapters/chapter3.tex` tables 3.1-3.3 in
[TALON-thesis](https://github.com/Entity014/TALON-thesis) — this file exists
so the rationale for each field lives next to the code that implements it,
not only in the LaTeX source.

## Observation (`config.ObservationSpaceCfg`)

| Field | Dim | Source | Why it's here |
|---|---|---|---|
| `joint_pos` | 12 | proprioceptive | RMA `x_t` component \cite{kumar2021} |
| `joint_vel` | 12 | proprioceptive | RMA `x_t` component |
| `roll_pitch` | 2 | IMU | RMA `x_t` component |
| `foot_contact` | 4 | proprioceptive, binarized | RMA `x_t` component |
| `prev_action` | 12 | feedback | closes the loop for the policy |
| `command` | 3 | external ($v_x, v_y, \omega_z$) | what "progress" is tracking against |
| `base_ang_vel` | 3 | IMU (gyro rate) | added 2026-09-18 — neither RMA's `x_t` nor this repo previously observed trunk angular velocity at all, only individual `joint_vel` and a static `roll_pitch` snapshot; no signal for how fast the trunk itself is rotating. `root_ang_vel_b` was already read in `reward.py`'s `v_actual` but never exposed to the policy. Confirmed missing by diffing against jaykorea/Isaac-RL-Two-wheel-Legged-Bot's own quadruped env, which observes it as standard practice |
| `projected_gravity` | 3 | IMU-derived (gravity direction in body frame) | added 2026-09-18 alongside `base_ang_vel` — standard complement to `roll_pitch` in legged-gym-style observations, a bounded unit vector with no Euler-angle singularity |
| `preference` ($w$) | 5 | Multi-Objective Module | appended in `training/moppo.py`, **not** by the env — see `envs/base_env.py` |

**Deliberately absent right now**: $\hat z_t, \sigma_t$ (Adaptation Module)
and the Exteroception embedding. Both are `[TBD]` in the thesis pending
ablation, and out of scope for the Multi-Objective Module prelim — adding
them later just means widening `ObservationSpaceCfg` and updating
`BaseTalonEnv`'s transition dict contract, not touching `moppo.py`.

## Action (`config.ActionSpaceCfg`)

$a_t \in \mathbb{R}^{12}$ — target joint angle, converted to torque downstream
by a PD controller: $\tau = K_p(a_t - q) + K_d(\dot a_t - \dot q)$
\cite{kumar2021}. This repo doesn't implement the PD conversion itself — a
real env (Isaac Lab or the physics-free dummy) owns that.

## Reward vector (`config.RewardVectorCfg`, `reward.py`)

Fixed order — **never** reorder `term_names` without also checking every
place that indexes by position (there currently isn't one; everything goes
through the name, but a raw index would silently break if this changes):

1. `progress` — exp-kernel velocity tracking (go-anywhere navigation)
2. `efficiency` — negative raw power (energy) plus action-rate,
   joint-acceleration, action-magnitude, and joint-speed penalty
   (smoothness/hardware wear; see "Grouped sub-penalties" below). Merged
   from separate `energy`/`smoothness` terms 2026-09-19 — measured 0.91
   correlation between their Episode_Reward curves on a live training run,
   not two orthogonal preference axes in practice; merging also eases the
   Dirichlet(1,...,1) coverage problem noted below (5 dims -> 4).
3. `impact` — continuous impact mitigation \cite{strauch2025crashcourse},
   penalizes peak contact force above a threshold, not just falls, plus
   foot-slip and undesired-contact sub-penalties (see "Grouped
   sub-penalties" below)
4. `balance` — negative squared trunk roll/pitch (a dense anti-fall signal
   before the contact-based fall termination fires), plus an `alive_bonus`
   and a vertical-bounce sub-penalty (see "Grouped sub-penalties" below)

All four are dimensions of the Phase-1 preference vector $w$. This follows
AMOR's use of smoothness and root orientation as separately weighted
objectives: the policy can negotiate tracking, wear, and stability rather
than hard-coding one simulator-specific trade-off. Fall termination and the
eventual hardware safety supervisor remain independent hard safety layers.

`progress`, `impact`, and `balance` each carry a permanent preference floor
(`RewardVectorCfg.progress_floor_eps`/`impact_floor_eps`/`balance_floor_eps`,
enforced by `preference.floor_clip_terms`) so none of them can be diluted to
near-zero by an unfavorable Dirichlet draw — `efficiency` deliberately does
not (see `preference.py`'s own docstring on the water-filling projection
this requires once 3+ terms are floored).

`clearance` is intentionally absent for now. Its current scripted obstacle
distance is not a meaningful training signal; putting a zero-valued channel
in $w$ would train policy contexts that cannot receive a corresponding reward.
It becomes a sixth objective only with the Exteroception Module.

### Grouped sub-penalties

2026-09-18: a fixed-$w$ physics validation of a checkpoint that scored
well under training-time (Dirichlet-favored) evaluation, but scored far
worse under a **uniform** $w$, found the policy survives by standing
nearly still (84% of joints saturated against `ACTION_CLIP`; tracking
ratio only 6.6% even under a forced 0.5 m/s command) rather than by
walking — a "safe but not useful" local optimum, rational given
`fall_penalty` makes any movement risk disproportionately costly relative
to the modest reward `progress`'s exp-kernel already gives near-zero
velocity. Comparing against a separate 10-term reference locomotion
reward vector surfaced 5 missing sub-penalties (lateral/rotation, action
magnitude, joint speed, vertical bounce, foot slip).

Reducing `fall_penalty` or widening `progress_std` were both ruled out
(the former was the exact fix that solved the original
mean\_episode\_length floor problem earlier the same day; the latter's
math showed standing still already earns non-trivial reward, so it has
limited leverage). Instead, 4 of the 5 missing terms were **folded into
the existing 5 objectives' own reward functions** rather than added as
new `term_names`/preference dimensions:

- `smoothness` += action magnitude (`-sum(action**2)`) and joint speed
  (`-0.01*sum(joint_vel**2)`)
- `balance` += vertical bounce (`-v_z**2`, body-frame vertical velocity)
- `impact` += foot slip (`-0.01*sum(contact * ||v_foot||^2)`, gated by a
  per-foot contact indicator so swing-phase velocity isn't penalized)

The 5th (lateral/rotation) was judged likely redundant with `progress`'s
existing combined tracking error (`v_command`'s vy/omega\_z components,
held at 0, already implicitly penalize lateral drift within the same
exp-kernel) and was not implemented separately.

Grouping instead of adding new $w$-dimensions is deliberate: a fixed-$w$
diagnostic the same day found uniform $w$ performing far worse than
Dirichlet-favored corners even at the current 5 dimensions — adding more
preference dimensions would only widen the simplex
$\mathrm{Dirichlet}(1,\ldots,1)$ has to cover, worsening a problem already
observed, not fixing it. See each added term's docstring in `reward.py`
for the full derivation.

**Follow-up, same day**: after ~3500 updates on the grouped reward above,
a repeat of the same forced-command physics probe found the tracking
ratio essentially UNCHANGED (6.7% vs. the original 6.6%) and action
saturation barely moved (81% vs. 84%) — the 4 grouped penalties alone did
not fix the underlying "stand still" optimum, only taxed it slightly
differently. Two more gaps were found and closed:

- **Crouching**: `v_z` only penalizes vertical *motion* — a lane that
  crouches low and then holds perfectly still pays nothing. `balance` +=
  height (`-(height-0.42)**2`, height measured above the lane's own spawn
  origin; 0.42 is `UNITREE_A1_CFG`'s own spawn height, not tuned). A video
  export of the checkpoint at this point visually confirmed the robot was
  crouched, not upright-and-stationary.
- **No positive incentive to step**: every grouped penalty above only
  taxes bad behavior — standing still pays each of them their minimum
  (often exactly 0), so none of them push the policy TOWARD walking, only
  away from specific bad variants of not-walking. `progress` +=
  feet-air-time (legged_gym/Rudin et al. 2022, `Sigma_feet
  (air_time_at_touchdown - 0.5)`, weight `2*dt=0.04` matching their own
  calibration since this task uses the same `dt=0.02`) — a foot that
  actually completes a proper swing phase before landing earns a real
  bonus a "do nothing" policy cannot collect. Grouped into `progress`
  (not smoothness/impact, where the other event-based/gait terms live)
  specifically so the bonus scales with $w_{progress}$: whenever the
  preference vector actually cares about locomoting, this positive nudge
  gets proportionally stronger too.

Also found missing in `MOPPOTrainer` itself, not the reward vector:
**policy/critic observations were never normalized** (only the reward
vector and the encoder's extrinsics input were, via `reward_norm`/
`extrinsics_norm`) — "What Matters in On-Policy Reinforcement Learning"
(Andrychowicz et al. 2021) found input normalization the single strongest
lever in their whole large-scale study. Added `obs_norm`
(`RunningMeanStd`, `center=True`) behind a single choke point,
`MOPPOTrainer.push_obs()`, that every caller (training's own rollout
collection, `play.py`, diagnostic scripts) must go through instead of
touching `self.stack.push` directly.

And **entropy was found pinned bit-identical at the annealed `log_std`
ceiling for 3000+ consecutive `update()` calls** — the fixed
`entropy_coef` multiplier's upward gradient pressure never once lost to
the policy gradient's own preference, the whole run. SAC-style automatic
entropy-coefficient tuning (Haarnoja et al. 2018) was tried in its place —
a learned `log_alpha`, adjusted via its own gradient step toward a target
entropy (the midpoint log_std between `LOG_STD_MIN` and
`log_std_max_anneal_final`, chosen because SAC's own target-entropy
heuristic, $-\dim(\mathcal{A})$, is calibrated for the tanh-corrected
action distribution's entropy while this codebase's `entropy()`
deliberately returns the much-larger-scale *pre-tanh* differential
entropy instead — see that method's docstring).

**Reverted the same day.** Resuming with it active showed entropy
declining *monotonically* every single iteration (11.03 at resume -> 10.29
by iteration 100 -> 7.14 by iteration 282, never leveling off), with
`mean_episode_len` and fall fraction tracking it in lockstep the whole way
(peak 70.6 at iteration 8 -> 27 by iteration 100 -> 22 by iteration 282;
fall fraction climbing to 0.98-0.99). This is a clean reproduction of the
exact "premature entropy collapse" failure `entropy_coef` itself was added
2026-09-17 to prevent: alpha cut exploration noise faster than the policy
could adapt to the same day's other new reward terms and `obs_norm`,
locking in confidently-wrong, falling behavior. "What Matters in On-Policy
RL" (Andrychowicz et al. 2021) independently found entropy-coefficient
tuning barely moved the needle on their own benchmarks — weak expected
upside going in, and now direct evidence of harm on this task. Reverted to
the fixed `entropy_coef` rather than re-tuned (this was the mechanism's
first clean failure, not the 3+ that would call for keeping-but-fixing
instead of reverting outright).

**2026-09-19 follow-up**: two further resumes into the grouped-reward
checkpoint both failed before a fix stuck. The first is the auto-entropy
collapse described above. The second (`entropy_coef` already reverted,
otherwise identical) showed no crash/NaN but `mean_episode_len` flat-to-
declining (8.3 -> 6.8 over 200+ iterations, fall fraction pinned at
0.98-1.00) — most likely explained by `penalty_curriculum_k` (RMA-style
ramp, MOPPOConfig.penalty_curriculum_init) having already reached ~1.0
from the ORIGINAL checkpoint's long training history: the two brand-new
sub-penalties (`height`, `foot_air_time_reward`) inherited that same
already-matured multiplier on resume instead of the gentle 0.03-start ramp
the original 5 terms got when THEY were first added, hitting a policy
that had never seen them at full strength from iteration 1 — the exact
"fresh policy, no cushioning" failure `penalty_curriculum_k` was built to
prevent in the first place. Not conclusively confirmed (could partly be
run-to-run stochasticity — this environment's own physics/terrain
randomization isn't perfectly reproducible run to run even at a fixed
seed), but training from scratch entirely (discarding the old checkpoint,
letting `penalty_curriculum_k` and the annealed `log_std` ceiling both
start fresh together) avoided the problem outright: `mean_episode_len`
reached ~99/200 within 52 updates, and a forced-command physics probe at
update ~1500 (curriculum k=0.97) found the velocity-tracking ratio at
24.2% — a genuine ~3.6x improvement over the pre-fix 6.6-6.7% baseline,
not just a training-time-reward illusion this time.

That same probe's exported video surfaced one more loophole: the policy
was dragging its CALF along the ground to move instead of stepping with
the foot. Neither `foot_vel` (only watches the foot body) nor the
peak-force term (built for transient landing shock, not sustained
low-force dragging) could see this — a calf touching the ground produces
contact force on a body neither the env nor the reward vector monitored
at all, a genuine zero-cost way to make `progress` without paying any
`impact` cost. Fixed by adding a dedicated contact sensor over `.*_calf`
bodies and a new `impact` sub-term, `undesired_contact_count` (legged_gym's
"Collisions" term, `-n_collision`) — deliberately no force threshold
(unlike `foot_vel`'s `contact_threshold`), since any calf-ground contact
at all is wrong regardless of magnitude. Weighted more assertively (0.2)
than the other grouped sub-penalties (0.01-scale) since this is closer to
a hard constraint than a soft preference.

A separate, NOT-yet-fixed gap surfaced from the same video: at the moment
of initial spawn/settle, the legs visibly cross through each other.
Confirmed via `isaaclab_assets`' own `UNITREE_A1_CFG` source
(`enabled_self_collisions=False`, upstream default shared by every
Unitree quadruped config, not something this repo set) — self-collision
is off, so legs pass through one another with no physics response at all.
Left disabled deliberately for now: self-collision checks are expensive
at 4096 parallel envs, and the crossing only shows up during the brief
chaotic settle right after spawn/reset, not in the learned gait itself
(unlike calf-dragging, a stable, reinforced behavior, this isn't
something training reinforces). Worth revisiting nearer hardware
deployment, not a Phase-1 prelim priority.

Also deliberately NOT extended: height/pose tracking stays a single
scalar (trunk height above spawn only, no per-foot clearance or joint-
configuration terms) — a conscious choice to keep the reward vector
general rather than prescribing a specific standing posture, which would
trade away exactly the adaptability MOPPO's preference-conditioning exists
to preserve.

### Running per-objective normalization

chapter3.tex is explicit that every term needs "การปรับมาตรฐานแบบเคลื่อนที่ต่อ
วัตถุประสงค์ (running per-objective normalization) ... เพื่อป้องกันไม่ให้เทอมที่มี
ขนาดใหญ่ครอบงำเกรเดียนต์ของเทอมอื่น" — `RunningMeanStd`
(`scripts/rl/core/running_norm.py`, Welford batched update) tracks a running
mean/std per reward term and `MOPPOTrainer._collect_rollout` divides each
term by its running std **and subtracts its running mean** (`center=True`)
before GAE. Raw scale is still wildly uneven — `smoothness` around -300 vs.
`progress` around 0-1 — but the stored/normalized reward stays within the
normalizer's clip range (`±10` by default) regardless.

Centering was added 2026-09-18 (the class originally defaulted to
divide-by-std-only, to keep a bounded term's 0-boundary meaningful) after
tracing an uncentered term's effect through GAE: an uncentered sample
carries a constant additive bias (`mean/std`) that GAE then accumulates
unevenly across a rollout (more lookahead steps mid-episode, less near an
episode boundary or the edge of the `num_steps` window), leaving a
position-dependent bias in the advantage that
`normalize_per_objective`'s later batch-mean subtraction doesn't fully
remove. Measured on this reward vector: `energy`'s raw running mean/std
ratio alone was -1.06, comparable to the term's own std.

## Preference vector $w$ (`preference.py`)

$w \sim \mathrm{Dirichlet}(\alpha{=}1.0)$ once at each episode reset
(AMOR's Phase-1 protocol), then is floor-clipped
($w_{impact} \geq \varepsilon$, chapter3.tex §3.2.3 — impact mitigation must
never be fully zeroed out) and renormalized to sum to 1. It stays constant
until that lane's next reset.

A future HLP or manual hardware scheduler may emit $w_t$ during an episode;
only that deployment path applies `max_delta_per_step` as a rate limiter
before the same floor clip and the safety/OOD guards.
