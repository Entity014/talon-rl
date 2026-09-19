"""MDP configuration dataclasses.

These mirror tables 3.1 (Observation Space), 3.2 (Action Space), and 3.3
(Reward Vector) in chapters/chapter3.tex of the thesis. Dimensions for the
base proprioceptive state, action, and RMA-derived quantities are taken
directly from Kumar et al. 2021 (RMA) since this thesis's Adaptation Module
extends that architecture — see \\cite{kumar2021} in chapter3.tex.

Prelim-scope note: the Adaptation Module ($\\hat z_t, \\sigma_t$) and the
Exteroception embedding are NOT part of this prelim's observation — both are
marked [TBD] in the thesis until the full pipeline is built. The prelim
observation is base proprioception + previous action + command + preference
vector only, which is enough to test the Multi-Objective Module in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class ObservationSpaceCfg:
    """Prelim observation layout. Total dim = sum of the fields below."""

    joint_pos_dim: int = 12  # q  (RMA x_t component)
    joint_vel_dim: int = 12  # qdot (RMA x_t component)
    roll_pitch_dim: int = 2  # theta (RMA x_t component)
    foot_contact_dim: int = 4  # g, binarized (RMA x_t component)
    prev_action_dim: int = 12  # a_{t-1}, feedback
    command_dim: int = 3  # v_x, v_y, omega_z target
    preference_dim: int = 5  # w — one weight per reward-vector term (table 3.3)
    # Added 2026-09-18: neither RMA's own x_t nor this repo previously
    # observed trunk angular velocity at all, only the individual joints'
    # (joint_vel) and a static roll_pitch snapshot -- no signal anywhere for
    # how fast the trunk itself is rotating, i.e. how fast it's falling.
    # Confirmed missing by diffing against jaykorea/Isaac-RL-Two-wheel-
    # Legged-Bot (the reference repo moppo.py's docstring already cites for
    # its rollout-collection convention), whose own quadruped env
    # (wolf_env/velocity_env_cfg.py) observes both base_ang_vel and
    # projected_gravity as standard practice. root_ang_vel_b was already
    # read elsewhere in this codebase (reward.py's v_actual, for the yaw-
    # rate term) but never exposed to the policy's own observation.
    base_ang_vel_dim: int = 3  # root_ang_vel_b (IMU gyro rate) — how fast the trunk itself is rotating
    # projected_gravity_b: gravity direction in the body frame, a unit
    # vector — standard alternative/complement to roll_pitch in legged-gym-
    # style observations, bounded and singularity-free unlike raw Euler
    # angles (roll_pitch already stays, not redundant: projected_gravity
    # collapses yaw information roll_pitch's atan2/asin form doesn't have
    # anyway, but the two together are the common convention, not either
    # alone).
    projected_gravity_dim: int = 3

    @property
    def total_dim(self) -> int:
        return (
            self.joint_pos_dim
            + self.joint_vel_dim
            + self.roll_pitch_dim
            + self.foot_contact_dim
            + self.prev_action_dim
            + self.command_dim
            + self.preference_dim
            + self.base_ang_vel_dim
            + self.projected_gravity_dim
        )


@dataclass(frozen=True)
class ActionSpaceCfg:
    """a_t: target joint angle, PD-converted to torque downstream. R^12 per RMA."""

    dim: int = 12


@dataclass(frozen=True)
class RewardVectorCfg:
    """Phase-1 reward objectives, in the fixed order used everywhere.

    `active` controls which terms are actually summed in `reward.compute_reward_vector`
    — kept here (not hardcoded) so a future full-scope run can add terms without
    touching the MOPPO/preference code.

    Clearance is deliberately not present in Phase 1: until exteroception
    produces a meaningful obstacle signal, sampling a preference weight for a
    zero-valued objective would create policy contexts with no learning signal.
    Add it back as a sixth objective only together with that signal.

    `energy` and `smoothness` merged into a single `efficiency` term
    2026-09-19: measured 0.91 correlation between their Episode_Reward
    curves across a live training run (energy_reward penalizes torque*vel,
    smoothness_reward's action_magnitude/joint_speed sub-penalties penalize
    largely the same raw motion-intensity signal from a different angle) --
    not two orthogonal preference axes in practice. Reducing 5 preference
    dimensions to 4 also directly eases the Dirichlet(1,...,1) coverage
    problem `preference.floor_clip_terms`'s own docstring and the
    smoothness_reward docstring both flag ("already struggles at 5
    dimensions"): fewer dims sharing the simplex means every term,
    including the newly-floored `progress`, gets diluted less often by an
    unfavorable draw. See `efficiency_reward` in reward.py -- it's the sum
    of the same `energy_reward`/`smoothness_reward` functions, not a new
    formula, so nothing about either term's own tuning changed, only that
    they're no longer independently weighted."""

    term_names: tuple[str, ...] = ("progress", "efficiency", "impact", "balance")
    active: tuple[bool, ...] = field(default_factory=lambda: (True, True, True, True))

    # Reward-shaping constants (rough starting points, not tuned — expect to
    # retune once running on the real terrain curriculum, not the dummy env).
    progress_std: float = 0.5  # exp-kernel std for velocity tracking
    impact_floor_eps: float = 0.05  # w_impact >= eps, per chapter3.tex §3.2.3
    balance_floor_eps: float = 0.15  # keep anti-fall learning signal present during every preference episode
    # Added 2026-09-19: unlike impact/balance, progress previously had NO
    # floor -- an unfavorable Dirichlet(1,...,1) draw could dilute it to
    # near-zero in any episode, with nothing preventing it. A same-day
    # preference-curriculum experiment (anneal alpha toward progress early,
    # back to uniform later) tried to work around this and made things
    # WORSE, not better: training-time progress reward climbed during the
    # anneal but reversed the moment alpha returned to flat uniform
    # Dirichlet(1) -- the curriculum's protection was temporary and faded,
    # re-exposing the exact same starvation it was meant to fix (see
    # talon-thesis/03_Daily_Notes/2026-09-19.md for the full ablation:
    # curriculum underperformed a plain-uniform baseline on both survival
    # and tracking ratio under repeated-trial eval). A permanent floor,
    # same mechanism as impact/balance, doesn't have a "fade" failure mode
    # -- this is the untried next step that motivated it.
    progress_floor_eps: float = 0.15  # w_progress >= eps, permanent (not a curriculum) -- see above
    # 5.0 (was) is dwarfed by what surviving would have earned: progress_reward
    # alone averages ~0.75/step across every run so far regardless of episode
    # length, so gamma=0.99 discounted over the ~200-step horizon is worth
    # ballpark 0.75*(1-0.99**200)/(1-0.99) =~ 65 in foregone reward -- a fall
    # at step 13 barely dents that. Bumped 5x as a first experiment (not
    # re-tuned/validated) after mean_episode_length peaked at iter ~15-20 then
    # regressed back to its ~13-step floor over the rest of training in three
    # consecutive runs (2026-09-17) despite curriculum/easy-start fixes ruling
    # out terrain difficulty as the cause.
    fall_penalty: float = 25.0  # terminal balance penalty for non-timeout falls
    # Flat reward for every step not yet fallen, added to balance_reward —
    # AMOR's constant survival bonus c_alive / classic Gym alive_bonus.
    # Bumped 0.3->1.0 (2026-09-19, first experiment, not re-swept): Reda et
    # al. 2020 ("Learning to Locomote: Understanding How Environment Design
    # Matters for Deep RL", Sec. 9) names "falling-forward" as the exact
    # failure mode a too-small survival bonus produces -- a live training
    # run (phase1_allfixes2) settled into a stable regime (15+ iterations,
    # not noise) with Episode_Reward/progress ~0.3 and episode length ~10,
    # matching a policy that gains transient forward velocity by toppling
    # rather than walking. The original 0.3 was deliberately kept modest
    # relative to the roll_pitch^2 penalty (steady-state balance reward
    # -0.3 to -0.5) precisely so upright-but-imperfect posture stayed
    # distinguishable from truly-upright -- but that same modesty is what
    # Reda et al. warns leaves falling-forward as the more profitable
    # local minimum. fall_penalty already got an analogous untuned 5x bump
    # (5.0->25.0, 2026-09-17) for the same class of instability.
    alive_bonus: float = 1.0
    # Coefficient on balance_reward's -sum(roll_pitch^2) term. Default 1.0
    # is the old unscaled behavior -- found 2026-09-19 (per-step trajectory
    # trace) it gives too little dynamic range at realistic tilt angles
    # (0.05 rad -> 0.98 rad -> 0.15 rad -> 0.9775, a difference of 0.02
    # against alive_bonus's flat +1.0) to distinguish normal walking pitch
    # from pre-fall pitch. Exposed for a k_theta sweep (5, 10) before
    # touching any other term -- see talon-thesis/03_Daily_Notes/
    # 2026-09-19.md for the sweep's own reasoning.
    balance_tilt_coef: float = 1.0
    # Coefficient on balance_reward's -sum(roll_pitch_rate^2) term. Default
    # 0.0 = disabled (old behavior). Calibrated against the same
    # 438-fall-event distribution as balance_tilt_coef above:
    # |pitch_rate| separates pre-fall from normal states ~3.2-3.6x
    # (p50/p90), a sharper signal than |pitch|'s own ~2x -- candidates
    # 0.004/0.008 (p90(|pitch_rate|)=5.037 rad/s at pre-fall -> k*5.037^2
    # ~= 0.10/0.20).
    balance_tilt_rate_coef: float = 0.0

    @property
    def dim(self) -> int:
        return len(self.term_names)


@dataclass(frozen=True)
class PreferenceCfg:
    """Preference settings for Phase 1 and the later HLP deployment stage.

    Phase-1 MOPPO samples one Dirichlet vector at each episode reset and holds
    it fixed for that episode, matching AMOR. `max_delta_per_step` is reserved
    for a future HLP/manual scheduler that changes w while the robot runs.
    """

    dirichlet_alpha: float = 1.0
    max_delta_per_step: float = 0.05  # rate-limiter cap on ||w_t - w_{t-1}||

    # Preference-vector curriculum (2026-09-19) -- OFF by default (both
    # fields below leave every existing default-constructed PreferenceCfg,
    # and therefore every test/dummy-env run, byte-identical to before this
    # was added). Found necessary via a real fromscratch_v2 run: 20000
    # updates of uniform Dirichlet(1,...,1) from step 0 plateaued around
    # ~10k updates with near-zero velocity-tracking under a forced command
    # (see runs/phase1_fromscratch_v2_2026-09-19's own validation) --
    # `progress` has no floor (unlike impact/balance, see
    # preference.floor_clip_terms) so it gets diluted whenever a Dirichlet
    # draw happens to undersample it, and a policy trained from scratch
    # under that diluted signal never reliably learns to track a velocity
    # command before the rest of training moves on to other w regions. This
    # curriculum front-loads training on a `curriculum_alpha_start` that
    # over-weights balance (must stand before anything else matters, see
    # reward.py's balance_reward docstring on the ~12-14/200-step floor
    # every run hit before balance existed) then progress (the actual
    # locomotion objective), and linearly anneals every term's alpha back
    # to the flat `dirichlet_alpha` (the real deployment-time distribution
    # this repo's thesis contract requires covering) over
    # `curriculum_updates` — see preference.curriculum_alpha().
    curriculum_alpha_start: tuple[float, ...] | None = None  # per RewardVectorCfg.term_names order; None disables the curriculum
    curriculum_updates: int = 0  # anneal to dirichlet_alpha over this many trainer._t steps; 0 disables the curriculum


@dataclass(frozen=True)
class ObservationStackCfg:
    """num_policy_stacks / num_critic_stacks, after Flamingo
    (jaykorea/Isaac-RL-Two-wheel-Legged-Bot) — see obs_stack.py. Defaults to 1/1
    (no stacking) so existing single-timestep code paths are unaffected."""

    num_policy_stacks: int = 1
    num_critic_stacks: int = 1


@dataclass(frozen=True)
class ExtrinsicsCfg:
    """Phase 1 privileged extrinsics e_t (chapter3.tex §3.2.1, RMA's Env
    Factor Encoder input). Dimensions are placeholders — not tuned,
    chapter3.tex marks the true dimensionality [TBD] pending ablation (its
    7-factor set differs from RMA's original 17)."""

    payload_mass_dim: int = 1
    payload_com_offset_dim: int = 3
    friction_dim: int = 1
    motor_power_scale_dim: int = 2  # Kp (stiffness) + Kd (damping) -- RMA randomizes/observes both, not just Kp
    leg_length_scale_dim: int = 1
    joint_range_scale_dim: int = 1
    terrain_height_dim: int = 1

    payload_treatment: Literal["explicit_observed_rewarded", "noise_only"] = "explicit_observed_rewarded"

    adaptation_latent_dim: int = 8  # z_t width — RMA's original default, [TBD] pending ablation

    @property
    def payload_dim(self) -> int:
        return self.payload_mass_dim + self.payload_com_offset_dim

    @property
    def dim(self) -> int:
        """Total e_t width — shrinks under noise_only (payload excluded
        entirely, not just unrewarded — see Pipeline_Summary.md §3.10)."""
        non_payload = (
            self.friction_dim + self.motor_power_scale_dim + self.leg_length_scale_dim
            + self.joint_range_scale_dim + self.terrain_height_dim
        )
        if self.payload_treatment == "noise_only":
            return non_payload
        return non_payload + self.payload_dim
