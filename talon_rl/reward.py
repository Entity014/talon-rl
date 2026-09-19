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

from .config import RewardVectorCfg


def progress_reward(
    v_actual: np.ndarray, v_command: np.ndarray, std: float,
    foot_air_time_reward: np.ndarray | None = None,
) -> np.ndarray:
    """Go-anywhere navigation: exp-kernel velocity tracking. (N, 3), (N, 3) -> (N,).

    `foot_air_time_reward` (added 2026-09-18, legged_gym/Rudin et al. 2022,
    "Learning to Walk in Minutes"): Sigma_feet (air_time_at_touchdown - 0.5),
    computed statefully in a1_env.py (see
    IsaacLabTalonEnv._compute_foot_air_time_reward) and passed through here
    already-summed. Added after a forced-command physics probe found a
    checkpoint trained with every OTHER grouped sub-penalty active
    (action_magnitude, joint_speed, v_z, foot_slip, height) still stood
    nearly still under a real forward command (6.7% velocity-tracking
    ratio) -- none of those sub-penalties give any POSITIVE incentive to
    actually take a step; they only tax bad behavior, and standing
    perfectly still pays every one of them their minimum (often exactly
    zero). This term is different in kind: a foot that swings for close to
    0.5s before landing earns a small BONUS, one a "do nothing" policy
    cannot collect (a foot that never leaves the ground never triggers
    `first_contact`, so it scores exactly 0 here too -- same as too-short
    shuffling steps, which score negative). Grouped into `progress` (not
    smoothness/balance like the other additions) since its purpose is
    specifically "is the robot actually locomoting", the same question
    progress's own tracking term asks. 0.04 weight (2*dt, dt=0.02) matches
    legged_gym's own calibration for this exact term -- their step size is
    the same 0.02s, so the scale carries over directly, unlike the other
    grouped terms' weights which were reasoned from scratch."""
    err = np.sum((v_actual - v_command) ** 2, axis=-1)
    reward = np.exp(-err / (std**2))
    if foot_air_time_reward is not None:
        reward = reward + 0.04 * foot_air_time_reward.astype(np.float32)
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
    smoothness's/balance's additions the same day, see docs/mdp.md) since
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

    `undesired_contact_count` (added 2026-09-19, `-0.2*count`, (N,) ->
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
    (see a1_env.py's `undesired_contact_count` field). 0.2 weight is a
    stronger deterrent than the other grouped sub-penalties (0.01-scale)
    since this should be closer to a hard constraint than a soft
    preference -- untuned, like every other weight introduced this
    session."""
    if foot_contact_force.shape[-1] == 0:
        return np.zeros(foot_contact_force.shape[0], dtype=np.float32)
    peak = np.max(np.abs(foot_contact_force), axis=-1)
    reward = -np.maximum(0.0, peak - threshold) / threshold
    if foot_vel is not None:
        contact = (foot_contact_force > contact_threshold).astype(np.float32)
        foot_speed_sq = np.sum(foot_vel**2, axis=-1)
        reward = reward - 0.01 * np.sum(contact * foot_speed_sq, axis=-1)
    if undesired_contact_count is not None:
        reward = reward - 0.2 * undesired_contact_count.astype(np.float32)
    return reward.astype(np.float32)


def smoothness_reward(
    action: np.ndarray, prev_action: np.ndarray, joint_acc: np.ndarray, joint_vel: np.ndarray
) -> np.ndarray:
    """Action-rate, acceleration, raw action-magnitude, and (2026-09-18)
    joint-speed objective, grouped together rather than as their own
    preference dimensions -- see docs/mdp.md's "Grouped sub-penalties"
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
    target_height: float = 0.42,
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

    `v_z` (added 2026-09-18, `-v_z**2`, (N,) -> already a scalar per lane,
    root_lin_vel_b's own z-component, no axis to sum over): vertical
    body-frame velocity,
    penalizing bouncing/bobbing -- grouped into balance rather than its own
    preference dimension (same "Grouped sub-penalties" reasoning as
    smoothness's action_magnitude/joint_speed additions the same day, see
    docs/mdp.md) since vertical bounce and roll/pitch tilt are both, in
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
    v_z above (DummyTalonEnv has no terrain to measure height above)."""
    reward = -np.sum(roll_pitch**2, axis=-1) + alive_bonus
    if v_z is not None:
        reward = reward - v_z.astype(np.float32) ** 2
    if height is not None:
        reward = reward - (height.astype(np.float32) - target_height) ** 2
    if terminal_fall is not None:
        reward = reward - np.asarray(terminal_fall, dtype=np.float32) * fall_penalty
    return reward.astype(np.float32)


_TERM_FUNCS = {
    "progress": lambda t, cfg: progress_reward(
        t["v_actual"], t["v_command"], cfg.progress_std, t.get("foot_air_time_reward")
    ),
    "clearance": lambda t, cfg: clearance_reward(t["obstacle_dist"]),
    "energy": lambda t, cfg: energy_reward(t["joint_torque"], t["joint_vel"]),
    "impact": lambda t, cfg: impact_reward(
        t["foot_contact_force"], foot_vel=t.get("foot_vel"),
        undesired_contact_count=t.get("undesired_contact_count"),
    ),
    "smoothness": lambda t, cfg: smoothness_reward(t["action"], t["prev_action"], t["joint_acc"], t["joint_vel"]),
    "balance": lambda t, cfg: balance_reward(
        t["roll_pitch"], t.get("terminal_fall"), cfg.fall_penalty, cfg.alive_bonus,
        t.get("v_z"), t.get("height"),
    ),
}


def compute_reward_vector(transition: dict, cfg: RewardVectorCfg) -> np.ndarray:
    """Returns r as an (N, dim) array in cfg.term_names order. Inactive terms are 0."""
    n = transition["obs"].shape[0]
    values = []
    for name, active in zip(cfg.term_names, cfg.active):
        values.append(_TERM_FUNCS[name](transition, cfg) if active else np.zeros(n, dtype=np.float32))
    return np.stack(values, axis=-1).astype(np.float32)
