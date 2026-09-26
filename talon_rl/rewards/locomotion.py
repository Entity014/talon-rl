"""The reward vector from chapter3.tex table 3.3 (tab:reward-vector), plus one
addition (`balance`, 2026-09-17) not yet in that table -- see its docstring
below for why. Batched (N, ...) -> (N,) throughout — this is the one place
the reward formulas are written (talon_rl/tasks/locomotion/a1_env/'s
manager-based env calls compute_reward_vector directly rather than
reimplementing any of this in torch — see that module's docstring).

Each function takes batched arrays describing the current transition and
returns a per-lane (N,) float32 array; `compute_reward_vector` stacks them
into a fixed-order (N, dim) array matching RewardVectorCfg.term_names.

None of these terms know about the preference vector w — arbitration between
them happens downstream (see training/moppo.py), consistent with chapter3.tex
treating w as something applied to the reward *vector*, not baked into any
single term.
"""

from __future__ import annotations

import numpy as np

from ..config import RewardVectorCfg


def signed_engagement(v_actual_x: np.ndarray, v_command_x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """R1's directional gate (frozen 2026-09-20, artifacts/r1_freeze/FREEZE.md):
    "is this lane actually moving toward what was commanded, and how much of
    it". 1.0 for a zero command (nothing to engage with -- a stationary lane
    isn't disengaging from anything). Otherwise `clip(sign(c_x)*v_x/|c_x|, 0,
    1)`: 0 while moving the wrong way or standing still, ramping to 1 once
    v_x reaches the commanded magnitude in the commanded direction, capped
    there (overshoot isn't extra credit). Shared by progress_reward (gates
    hip_activation) and balance_reward (gates alive_bonus) so both use the
    exact same notion of "engaged", not two independently-drifting ones."""
    v_actual_x = np.asarray(v_actual_x, dtype=np.float32)
    v_command_x = np.asarray(v_command_x, dtype=np.float32)
    zero_cmd = np.abs(v_command_x) < eps
    denom = np.where(zero_cmd, 1.0, np.abs(v_command_x))
    engaged = np.clip(np.sign(v_command_x) * v_actual_x / denom, 0.0, 1.0)
    return np.where(zero_cmd, 1.0, engaged).astype(np.float32)


def progress_reward(
    v_actual: np.ndarray, v_command: np.ndarray, std: float,
    terminal_fall: np.ndarray | None = None,
    hip_qdot_L: np.ndarray | None = None, hip_qdot_R: np.ndarray | None = None, hip_activation_coef: float = 0.0,
    directed_progress: np.ndarray | None = None, directed_coef: float = 0.0,
    engagement_eps: float = 1e-6,
) -> np.ndarray:
    """Go-anywhere navigation: engagement-gated exp-kernel velocity tracking,
    plus (R1, 2026-09-20) a directed-progress bonus and hip-activation
    shaping, both gated the same way. (N, 3), (N, 3) -> (N,). See
    artifacts/r1_freeze/FREEZE.md for the frozen equations/constants this
    implements -- do not retune any of them from training results, only from
    a proven implementation bug.

    `signed_engagement` gate (R1): the pre-R1 kernel rewarded standing still
    at v_x=0 under command 0 (correctly, e=1 there) but also gave partial
    credit for moving in the WRONG direction whenever the tracking error
    happened to be small by coincidence. Gating the whole tracking term by
    `e` removes that: wrong-direction motion now scores exactly the
    engagement floor (0), not "whatever the exp-kernel happened to compute".

    `terminal_fall` (2026-09-19, zeroes this step's reward entirely where
    True): `v_actual` is `root_lin_vel_b`, BODY-frame velocity (a1_env.py)
    -- on the exact step a lane topples over, its body-frame forward
    velocity spikes from the fall/rotation itself, which can happen to
    align with v_command and score a high tracking reward for a step that
    was falling, not walking. Found via a live training run
    (`phase1_allfixes_2026-09-19`) showing Episode_Reward/progress rising
    while episode length collapsed and termination hit 100% fall -- the
    opposite of real locomotion improving. Same "no credit on the step you
    stopped" principle balance_reward's alive_bonus already applies to
    fall_penalty (see its own docstring); short episodes made this worse
    since a same-size spike dominates a larger fraction of a short
    episode's averaged reward than a long one's. Applied to the WHOLE R1
    sum (tracking + hip + directed-progress), not just tracking -- all
    three are locomotion-quality signals that shouldn't pay out on a
    falling step.

    `hip_qdot_L`/`hip_qdot_R`/`hip_activation_coef` (moved here from
    balance_reward, R1 freeze): `+hip_activation_coef * e *
    min(|hip_qdot_L|, |hip_qdot_R|)`, same formula as before (see
    balance_reward's docstring, kept there, for the original diagnosis --
    a structurally frozen hip contributing to falls) but now gated by `e`
    and grouped with progress instead of balance. The R1 offline audit
    (artifacts/r1_offline_audit/audit-report.md) found this term's
    magnitude scales with commanded speed (0.12 @ vx=0.25 -> 0.21 @
    vx=0.5), i.e. it functions as a gait-support signal, not a
    speed-independent balance regularizer -- moving it here makes the
    grouping match what the data shows it actually is. Gating by `e`
    closes the loophole where a frozen-Balance version paid this out even
    while standing still or moving the wrong way.

    `directed_progress`/`directed_coef` (R1 freeze): `directed_progress` is
    a precomputed, already-clipped [0, 1] per-lane ratio -- world
    displacement toward the command, projected onto heading at command
    onset, over a 0.5s/50-step window, `clip(sign(c_x)*dx_heading /
    (|c_x|*T_w+eps), 0, 1)` -- see FREEZE.md for the exact formula. This
    function stays a pure stateless (N,)->(N,) transform per its own
    module docstring, so it does NOT compute that ratio itself; the
    STATEFUL displacement-buffer bookkeeping (origin position/heading,
    reset on command change) lives wherever foot_air_time_reward's
    equally-stateful bookkeeping lives (a1_env.py, see
    IsaacLabTalonEnv._compute_foot_air_time_reward for the pattern) and
    is passed in already-computed, same as that term was. Optional/None
    default (0 contribution) so callers without real position tracking
    (DummyTalonEnv, unit tests) don't need a placeholder.

    Removed from R1 (2026-09-20, see FREEZE.md): the pre-R1
    `foot_air_time_reward` bonus. The R1 offline audit found it
    contributing to 1-3% of steps and net NEGATIVE on average there (~1000x
    smaller than tracking) -- dead weight at best, mis-signed at worst.
    Disabled outright rather than re-derived; re-enabling it is a future
    decision, not something this freeze made."""
    # Forward-velocity (x) error only (2026-09-19, was sum over all 3 axes:
    # v_x, v_y, yaw-rate). Found via a reward/evaluation-metric mismatch --
    # play.py's PhysicsValidator tracking_ratio only ever measured v_x, but
    # this kernel was also scored on v_y/yaw-rate error, which the
    # evaluation never cared about and (under a forced straight-ahead
    # v_command=[0.5,0,0]) the policy had no reason to zero out on its own.
    # At std=0.5, the resulting kernel was forgiving enough that standing
    # still (v_x=0, err=0.25) scored 0.368 and drifting backward (v_x=-0.04,
    # err~0.29) scored 0.312 -- both close to a policy that actually tracks
    # (1.0), while tracking_ratio scored those same three cases 0%, -8%,
    # 100% respectively: barely any daylight in the training signal for
    # behavior the evaluation considered night-and-day different. Restricting
    # to v_x alone removes the v_y/yaw-rate dilution (not the std-magnitude
    # issue, which is separate -- see RewardVectorCfg.progress_std). Tradeoff
    # noted, not fixed here: v_y/yaw-rate tracking now has NO reward
    # incentive at all when v_command samples a nonzero lateral/turn
    # component (mdp/events.py's randomize_velocity_command can do this) --
    # acceptable for now since Phase 1's own scope is forward-velocity
    # tracking (chapter3.tex), not general omnidirectional command-following.
    v_actual_x, v_command_x = v_actual[..., 0], v_command[..., 0]
    e = signed_engagement(v_actual_x, v_command_x, engagement_eps)
    err = (v_actual_x - v_command_x) ** 2
    reward = e * np.exp(-err / (std**2))
    if hip_qdot_L is not None and hip_qdot_R is not None:
        reward = reward + hip_activation_coef * e * np.minimum(
            np.abs(hip_qdot_L.astype(np.float32)), np.abs(hip_qdot_R.astype(np.float32))
        )
    if directed_progress is not None:
        reward = reward + directed_coef * directed_progress.astype(np.float32)
    if terminal_fall is not None:
        reward = reward * (1.0 - np.asarray(terminal_fall, dtype=np.float32))
    return reward.astype(np.float32)


def clearance_reward(obstacle_dist: np.ndarray, safe_dist: float = 0.5) -> np.ndarray:
    """Autonomous obstacle negotiation — scripted signal until the Exteroception
    Module exists (out of scope here). (N,) -> (N,)."""
    return np.clip(obstacle_dist / safe_dist, 0.0, 1.0).astype(np.float32)


def energy_reward(joint_torque: np.ndarray, joint_vel: np.ndarray) -> np.ndarray:
    """Raw power per step, negated. (N, 12), (N, 12) -> (N,)."""
    power = np.sum(np.abs(joint_torque * joint_vel), axis=-1)
    return (-power).astype(np.float32)


def impact_reward(
    foot_contact_force: np.ndarray,
    threshold: float = 50.0,
    foot_vel: np.ndarray | None = None,
    contact_threshold: float = 1.0,
    undesired_contact_count: np.ndarray | None = None,
) -> np.ndarray:
    """Continuous impact mitigation — penalize peak landing force, not just
    falls. Threshold is an arbitrary prelim placeholder; retune against real
    A1 contact-force ranges before any hardware test. (N, 4) -> (N,).

    `foot_vel` (added 2026-09-18, `-0.01*sum(contact * ||v_foot||^2)`, (N, 4,
    3) -> (N,)): foot-slip penalty, grouped into `impact` rather than its own
    preference dimension (same "Grouped sub-penalties" reasoning as
    smoothness's/balance's additions the same day, see docs/methods/general/mdp.md) since
    slip and landing-impact force are both, in spirit, about foot-ground
    contact quality. Matches the reference term `-||diag(g^t)·v_f^t||^2`:
    `contact` (`foot_contact_force > contact_threshold`) stands in for the
    reference's binary contact indicator g^t -- a foot only pays the slip
    penalty while actually touching the ground, not while swinging through
    the air where nonzero velocity is expected and desired. Optional
    (defaults to None, no contribution) so callers/tests without real foot
    kinematics (e.g. DummyTalonEnv) don't need a placeholder value. Same
    0.01 weight as smoothness's joint_speed/acc terms (both raw
    velocity-scale physical quantities, not action-scale ones).

    `undesired_contact_count` (added 2026-09-19, `-0.5*count`, (N,) ->
    already a scalar count per lane, no axis to sum over): legged_gym's
    "Collisions" term (Rudin et al. 2022, `-n_collision`), grouped into
    `impact` since it's the same "ground contact quality" family as
    foot_slip above. Found necessary via a fixed-command video export: the
    trained policy dragged its CALF along the ground to move instead of
    stepping, a zero-cost loophole neither foot_vel (only watches the
    FOOT) nor the peak-force term above (built for transient landing
    shock, not sustained low-force dragging) could see or penalize. No
    force threshold here deliberately -- unlike a foot, a calf touching
    anything at all is undesired regardless of magnitude, so the env
    passes a raw contact COUNT already thresholded at the sensor level
    (see a1_env.py's `undesired_contact_count` field). Weight raised
    0.2->0.5 (2026-09-19, second pass): repeated-trial validate on a
    `mean_reg_coef=0.01` checkpoint still showed calf contact on 91.7% of
    steps under the original 0.2 -- deliberately still a penalty weight,
    not a per-foot clearance reward (docs/methods/general/mdp.md documents that omission
    as intentional, to keep the reward vector posture-agnostic for
    MOPPO's preference-negotiation rather than prescribing a specific
    gait/clearance trajectory); this only makes ANY calf contact cost
    more, it doesn't reward any particular leg trajectory."""
    if foot_contact_force.shape[-1] == 0:
        return np.zeros(foot_contact_force.shape[0], dtype=np.float32)
    peak = np.max(np.abs(foot_contact_force), axis=-1)
    reward = -np.maximum(0.0, peak - threshold) / threshold
    if foot_vel is not None:
        contact = (foot_contact_force > contact_threshold).astype(np.float32)
        foot_speed_sq = np.sum(foot_vel**2, axis=-1)
        reward = reward - 0.01 * np.sum(contact * foot_speed_sq, axis=-1)
    if undesired_contact_count is not None:
        reward = reward - 0.5 * undesired_contact_count.astype(np.float32)
    return reward.astype(np.float32)


def smoothness_reward(
    action: np.ndarray, prev_action: np.ndarray, joint_acc: np.ndarray, joint_vel: np.ndarray
) -> np.ndarray:
    """Action-rate, acceleration, raw action-magnitude, and (2026-09-18)
    joint-speed objective, grouped together rather than as their own
    preference dimensions -- see docs/methods/general/mdp.md's "Grouped sub-penalties"
    note for why. (N, 12) each -> (N,).

    It is preference-conditioned in Phase 1, as in AMOR's smoothness
    objective, so the policy can negotiate tracking versus hardware wear.

    action_magnitude (`-sum(action**2)`) added 2026-09-18: found via a
    ground-truth physics probe that the trained policy saturates ~84% of
    joints against ACTION_CLIP almost every step, and via a forced-command
    test (v_command pinned to 0.5 m/s the whole rollout) that it does this
    while barely moving at all (tracking ratio 6.6%) -- action_rate alone
    only penalizes CHANGING the action sharply, not holding it at an
    extreme value steadily, so a policy that locks onto one saturated
    action and rarely changes it pays almost nothing here despite being
    maximally aggressive on every joint.

    joint_vel (`-0.01*sum(joint_vel**2)`) added the same day, alongside
    action_magnitude: a separately-named reference reward vector this
    task was compared against (a 10-term quadruped locomotion reward) has
    both "Action Magnitude" and "Joint Speed" as distinct terms -- raw
    joint angular velocity is penalized independently of the torque
    driving it (unlike `energy`, which is torque*velocity and can stay
    small even at high joint speed if torque happens to be low at that
    instant). Same 0.01 weight as `acc` above (both are raw
    velocity/acceleration-scale physical quantities, not action-scale
    ones like the other two terms here).

    No separate w-dimension for either addition (same reasoning): adding
    new preference dimensions widens the simplex Dirichlet(1,...,1) has to
    cover, which chapter3.tex's own MO-negotiation already struggles with
    at 5 dimensions (see the fixed-w diagnostic in
    runs/phase1_longrun5_2026-09-18/validation/ finding uniform w
    performed far worse than either extreme corner)."""
    action_rate = np.sum((action - prev_action) ** 2, axis=-1)
    acc = np.sum(joint_acc**2, axis=-1)
    action_magnitude = np.sum(action**2, axis=-1)
    joint_speed = np.sum(joint_vel**2, axis=-1)
    return (-(action_rate + 0.01 * acc + action_magnitude + 0.01 * joint_speed)).astype(np.float32)


def balance_reward(
    roll_pitch: np.ndarray, terminal_fall: np.ndarray | None = None, fall_penalty: float = 0.0,
    alive_bonus: float = 0.0, v_z: np.ndarray | None = None, height: np.ndarray | None = None,
    target_height: float = 0.42, tilt_coef: float = 1.0,
    roll_pitch_rate: np.ndarray | None = None, tilt_rate_coef: float = 0.0,
    height_coef: float = 1.0,
    hip_q_L: np.ndarray | None = None, hip_q_R: np.ndarray | None = None, hip_sym_coef: float = 0.0,
    v_actual: np.ndarray | None = None, v_command: np.ndarray | None = None, alive_gate_threshold: float = 0.20,
) -> np.ndarray:
    """Penalizes trunk tilt directly -- a dense, per-step gradient against
    falling. Not in chapter3.tex's original table 3.3; added because none of
    the other 5 terms give any signal toward staying upright before it's
    already fallen (only `base_contact` termination, after the fact) --
    diagnosed after 4 independent training runs (action scale, terrain
    curriculum, Kp/Kd randomization, network width) each failed to move
    mean_episode_length off its ~12-14/200-step floor. RMA's own reward set
    has the equivalent term (#8, Orientation: -||theta_roll,pitch||^2) for
    exactly this reason; AMOR's root-orientation term is reference-tracking
    (needs mocap) and doesn't port to this reference-free task.

    `tilt_coef` (added 2026-09-19, default 1.0 = old unscaled behavior):
    found via a per-step trajectory trace that `-sum(roll_pitch**2)` has too
    little dynamic range to distinguish a normal walking pitch (~0.05 rad)
    from a pre-fall one (~0.15 rad) -- 0.0025 vs 0.0225, a difference of
    only 0.02 against `alive_bonus`'s flat +1.0 baseline, so balance_reward
    sat at ~0.98-1.0 almost regardless of tilt right up until the terminal
    fall_penalty fired. This is NOT alive_bonus mathematically cancelling
    the tilt gradient (a constant can't do that) -- it's that the tilt
    penalty's own coefficient (implicitly 1) is too small for its
    quadratic-in-radians scale to matter next to a bonus of that
    magnitude. Exposed as a preference-orthogonal scale knob to calibrate
    against, not a fixed multiplier chosen a priori. Calibrated 2026-09-19
    against a 438-fall-event distribution (prefall_window_analysis.py):
    pre-fall p90(|pitch|)=0.374 rad vs normal p90=0.168 -- 3.5 solves
    `k * 0.374**2 ~= 0.49`, a penalty magnitude judged meaningful without
    dominating the raw (pre-normalize_per_objective) return.

    `tilt_rate_coef`/`roll_pitch_rate` (added 2026-09-19, `-tilt_rate_coef *
    sum(roll_pitch_rate**2)`): the SAME 438-fall-event calibration found
    |pitch_rate| separates pre-fall from normal-walking states far more
    sharply than |pitch| itself -- p50 0.394->1.262 rad/s (~3.2x) and p90
    1.379->5.037 rad/s (~3.6x), versus pitch's own ~2x at both percentiles.
    `roll_pitch_rate` is body-frame angular velocity's x/y components
    (`robot.data.root_ang_vel_b[:, 0:2]`, a1_env.py) -- real physics, not a
    finite-difference approximation of roll_pitch across steps. Optional
    (defaults to None with tilt_rate_coef unused, i.e. old behavior),
    same reasoning as v_z/height above (DummyTalonEnv has no angular
    velocity to report).

    `alive_bonus` (added 2026-09-18): a flat positive reward every step the
    lane hasn't fallen, matching AMOR's constant survival bonus c_alive and
    the classic Gym alive_bonus (Hopper/Walker2d). Found necessary because
    per-step reward (Episode_Reward/progress, /smoothness) kept improving
    across a 20000-update run while mean_episode_length stayed flat at its
    ~6-8 floor the whole time -- the policy was getting better at whatever
    it experienced without any of that requiring surviving longer, since
    every other term here is a per-step rate, not tied to episode duration.
    This term is the only one that pays out purely for elapsed time alive,
    independent of how well any other objective is being tracked. (N, 2) ->
    (N,).

    `v_actual`/`v_command`/`alive_gate_threshold` (R1 freeze, 2026-09-20,
    artifacts/r1_freeze/FREEZE.md): gates `alive_bonus` by
    `clip(signed_engagement(v_actual_x, v_command_x) / alive_gate_threshold,
    0, 1)`. Final Locomotion Evaluation v1 (artifacts/final_locomotion_eval_v1/
    FROZEN.md) found seed1 surviving nearly every episode while barely
    locomoting -- the unconditional alive_bonus was paying full survival
    credit for standing still under a nonzero command, exactly the
    "falling-forward" alternative's opposite loophole. Threshold 0.20 is a
    LOW bar (engagement, not quality) deliberately: full credit once a lane
    reaches 20% of commanded directed velocity, not "walks well". A zero
    command reaches gate=1.0 automatically (signed_engagement returns 1.0
    there, and 1.0/0.20 clips back to 1.0) so standing still under a
    genuine stop command still gets full unconditional survival credit,
    same as before. Optional (defaults to gate=1.0, i.e. old unconditional
    behavior) when v_actual/v_command aren't provided -- callers without
    command context (DummyTalonEnv, existing unit tests) keep the pre-R1
    behavior rather than silently losing their alive_bonus. Penalties
    below (tilt/vz/height/fall) are NOT gated -- a lane tilting, bouncing,
    crouching, or falling is penalized whether or not it's engaging the
    command; only the POSITIVE survival credit is conditional.

    `v_z` (added 2026-09-18, `-v_z**2`, (N,) -> already a scalar per lane,
    root_lin_vel_b's own z-component, no axis to sum over): vertical
    body-frame velocity,
    penalizing bouncing/bobbing -- grouped into balance rather than its own
    preference dimension (same "Grouped sub-penalties" reasoning as
    smoothness's action_magnitude/joint_speed additions the same day, see
    docs/methods/general/mdp.md) since vertical bounce and roll/pitch tilt are both, in
    spirit, about staying physically stable/upright. Optional (defaults to
    None, no contribution) so callers/tests that don't have it (e.g.
    DummyTalonEnv, which has no real vertical dynamics to report) don't
    need a placeholder value.

    `height` (added 2026-09-18, `-(height-target_height)**2`): found via a
    forced-command physics probe that a checkpoint trained with every OTHER
    grouped sub-penalty active (action_magnitude, joint_speed, v_z,
    foot_slip) still crouched and stood nearly still (6.7% velocity-
    tracking ratio, 81% action saturation, both essentially unchanged from
    before those additions) -- v_z only penalizes vertical MOTION, so a
    lane that crouches low and then holds perfectly still pays nothing
    (v_z=0 the whole time). This closes that loophole directly.
    `target_height=0.42` is UNITREE_A1_CFG's own spawn height
    (isaaclab_assets/robots/unitree.py) -- the pose the robot is already
    built to stand at, not a swept/tuned value. Optional, same reasoning as
    v_z above (DummyTalonEnv has no terrain to measure height above).

    `height_coef` (added 2026-09-20, default 1.0 = old unscaled behavior):
    found via balance_decomposition.py that checkpoints A/B/D (tilt_coef
    swept 1.0/3.5/2.0, tilt_rate_coef 0/0.004/0) ALL settled around
    height~0.21-0.25 vs target_height=0.42 regardless of tilt strength --
    crouching is a height-reward loophole orthogonal to the tilt sweep, not
    caused by it. At a typical ~0.20 height error, `-(0.20)**2 = -0.04` is
    tiny next to fall_penalty=-25 on termination and alive_bonus=+1 for
    surviving in the crouched pose -- same "coefficient too small for its
    quadratic scale to register" issue tilt_coef fixed for pitch, now
    applied to height. Exposed as its own scale knob rather than folded
    into a larger constant, same pattern as tilt_coef/tilt_rate_coef.

    `hip_activation_coef` (moved to progress_reward, R1 freeze
    2026-09-20 -- see that function's docstring): originally added here
    2026-09-20 (Experiment 2A.4-5) after gait_joint_trace.py diagnosed a
    checkpoint with a structurally frozen hip (near-constant from spawn,
    not fall-induced -- hip_asymmetry_analysis.py's early-vs-late split
    showed asymmetry SHRINKING toward each fall) while the other hip did
    essentially all the work. hip_symmetry_intervention.py found forcing
    the frozen hip to mirror the active one (eval-only, no retraining)
    reduced falls/improved v_z/pitch in the clearest-effect seed --
    directional evidence, not a fully explained mechanism (subsequent
    activity-ratio/phase-correlation analyses found no robust cross-seed
    signal, see gait_activity_ratio.py/hip_functional_correlation.py).
    The R1 offline audit (artifacts/r1_offline_audit/audit-report.md)
    then found this term's magnitude scales with commanded speed -- a
    gait-support signal, not a balance regularizer -- which is why R1
    moved it to progress_reward (gated by signed_engagement there) rather
    than keeping it here unconditional.

    `hip_sym_coef`/`hip_q_L`/`hip_q_R` (added 2026-09-20, `-hip_sym_coef
    * (hip_q_L + hip_q_R)**2`): a bilateral MIRROR-symmetry penalty,
    included as a diagnostic baseline to compare hip_activation_coef
    against, not because it's expected to be the better choice --
    UNITREE_A1_CFG's own default standing pose (FL_hip=+0.1, FR_hip=
    -0.1) already encodes hip_q_L = -hip_q_R at the symmetric stance,
    so `hip_q_L + hip_q_R` is exactly 0 there and grows with any L/R
    mirror-asymmetry (real joint position, not target). Disabled by
    default (0.0, unchanged by R1) -- forcing literal bilateral symmetry
    would remove exactly the adaptability (e.g. to a tilted ramp) the
    hip_activation alternative is meant to preserve; kept available for a
    controlled comparison, not because it's the intended fix."""
    e = (
        signed_engagement(v_actual[..., 0], v_command[..., 0])
        if v_actual is not None and v_command is not None else 1.0
    )
    alive_gate = np.clip(np.asarray(e, dtype=np.float32) / alive_gate_threshold, 0.0, 1.0)
    reward = -tilt_coef * np.sum(roll_pitch**2, axis=-1) + alive_bonus * alive_gate
    if roll_pitch_rate is not None:
        reward = reward - tilt_rate_coef * np.sum(roll_pitch_rate.astype(np.float32) ** 2, axis=-1)
    if v_z is not None:
        reward = reward - v_z.astype(np.float32) ** 2
    if height is not None:
        reward = reward - height_coef * (height.astype(np.float32) - target_height) ** 2
    if hip_q_L is not None and hip_q_R is not None:
        reward = reward - hip_sym_coef * (hip_q_L.astype(np.float32) + hip_q_R.astype(np.float32)) ** 2
    if terminal_fall is not None:
        reward = reward - np.asarray(terminal_fall, dtype=np.float32) * fall_penalty
    return reward.astype(np.float32)


_ENERGY_COEF = 0.1  # see docstring below -- without this, energy silently swallows smoothness inside the merge


def efficiency_reward(
    joint_torque: np.ndarray, joint_vel: np.ndarray,
    action: np.ndarray, prev_action: np.ndarray, joint_acc: np.ndarray,
) -> np.ndarray:
    """`energy` and `smoothness` merged into one preference dimension
    (2026-09-19) -- see RewardVectorCfg's own docstring for why (0.91
    measured correlation between their Episode_Reward curves, plus easing
    the 5-dimension Dirichlet coverage problem).

    Originally an unweighted sum ("no new tuning"). Found 2026-09-19 via
    PhysicsValidator's energy-vs-smoothness breakdown (added the same day)
    on a trained checkpoint: mean|energy|=951.9 vs mean|smoothness|=98.6,
    a 9.66x raw-scale gap -- energy (raw torque*vel power, no coefficient
    of its own) was silently dominating the merged term's variance, so
    RunningMeanStd's per-term normalization (which normalizes `efficiency`
    as one merged scalar, not its two inputs separately) effectively
    tracked energy alone. `action_magnitude`, the sub-term inside
    `smoothness` added specifically to fix action saturation, was being
    diluted to near-zero effective gradient inside the merge as a result.
    `_ENERGY_COEF=0.1` brings energy down to smoothness's own raw scale
    (matches the existing 0.01-weight convention smoothness_reward already
    uses on its own acc/joint_speed sub-terms for the same reason) --
    chosen as a fixed constant, not adaptive per-sub-term normalization,
    to stay consistent with how every other grouped multi-scale term in
    this file (e.g. impact_reward's peak-force/foot-slip/contact-count
    mix) is already balanced: hand-tuned constants, not stateful
    normalizers, since reward.py is pure/stateless throughout. (N, 12)
    each -> (N,)."""
    return _ENERGY_COEF * energy_reward(joint_torque, joint_vel) + smoothness_reward(
        action, prev_action, joint_acc, joint_vel
    )


_TERM_FUNCS = {
    "progress": lambda t, cfg: progress_reward(
        t["v_actual"], t["v_command"], cfg.progress_std, t.get("terminal_fall"),
        hip_qdot_L=t.get("hip_qdot_L"), hip_qdot_R=t.get("hip_qdot_R"),
        hip_activation_coef=cfg.progress_hip_activation_coef,
        directed_progress=t.get("directed_progress"), directed_coef=cfg.progress_directed_coef,
    ),
    "clearance": lambda t, cfg: clearance_reward(t["obstacle_dist"]),
    "impact": lambda t, cfg: impact_reward(
        t["foot_contact_force"], foot_vel=t.get("foot_vel"),
        undesired_contact_count=t.get("undesired_contact_count"),
    ),
    "efficiency": lambda t, cfg: efficiency_reward(
        t["joint_torque"], t["joint_vel"], t["action"], t["prev_action"], t["joint_acc"]
    ),
    "balance": lambda t, cfg: balance_reward(
        t["roll_pitch"], t.get("terminal_fall"), cfg.fall_penalty, cfg.alive_bonus,
        t.get("v_z"), t.get("height"), target_height=cfg.target_height, tilt_coef=cfg.balance_tilt_coef,
        roll_pitch_rate=t.get("roll_pitch_rate"), tilt_rate_coef=cfg.balance_tilt_rate_coef,
        height_coef=cfg.balance_height_coef,
        hip_q_L=t.get("hip_q_L"), hip_q_R=t.get("hip_q_R"), hip_sym_coef=cfg.balance_hip_sym_coef,
        v_actual=t.get("v_actual"), v_command=t.get("v_command"), alive_gate_threshold=cfg.balance_alive_gate_threshold,
    ),
}


def compute_reward_vector(transition: dict, cfg: RewardVectorCfg) -> np.ndarray:
    """Returns r as an (N, dim) array in cfg.term_names order. Inactive terms are 0."""
    n = transition["obs"].shape[0]
    values = []
    for name, active in zip(cfg.term_names, cfg.active):
        values.append(_TERM_FUNCS[name](transition, cfg) if active else np.zeros(n, dtype=np.float32))
    return np.stack(values, axis=-1).astype(np.float32)
