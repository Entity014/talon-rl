import numpy as np

from talon_rl.config import RewardVectorCfg
from talon_rl.reward import (
    balance_reward,
    clearance_reward,
    compute_reward_vector,
    efficiency_reward,
    energy_reward,
    impact_reward,
    progress_reward,
    smoothness_reward,
)


def test_progress_reward_is_max_at_zero_error():
    v = np.tile(np.array([0.5, 0.0, 0.0]), (3, 1))
    r_same = progress_reward(v, v, std=0.5)
    r_diff = progress_reward(v, v * 2, std=0.5)
    assert r_same.shape == (3,)
    assert np.allclose(r_same, 1.0)
    assert np.all(r_diff < 1.0)


def test_progress_reward_gives_positive_bonus_for_a_proper_touchdown():
    """Found 2026-09-18: every OTHER grouped sub-penalty this session taxes
    bad behavior, but none give a POSITIVE incentive to actually take a
    step -- a "stand still" policy pays every one of them their minimum.
    feet_air_time_reward is different in kind: a foot airborne for close to
    0.5s before landing (a proper stride, legged_gym/Rudin et al. 2022)
    should beat a policy with the same tracking error but no such bonus."""
    v = np.tile(np.array([0.5, 0.0, 0.0]), (2, 1))
    r_no_bonus = progress_reward(v, v, std=0.5)
    r_with_bonus = progress_reward(v, v, std=0.5, foot_air_time_reward=np.array([2.0, 0.0]))
    assert r_with_bonus[0] > r_no_bonus[0]  # proper stride: bonus applied
    assert r_with_bonus[1] == r_no_bonus[1]  # no touchdown event this step: unchanged


def test_progress_reward_zeroes_out_on_terminal_fall_step():
    """Found 2026-09-19 via a live training run (phase1_allfixes):
    Episode_Reward/progress rose while episode length collapsed and
    termination hit 100% fall. v_actual is root_lin_vel_b (BODY-frame,
    a1_env.py) -- a lane toppling forward spikes its body-frame forward
    velocity from the fall/rotation itself, which can align with
    v_command and score a high tracking reward on the exact step it was
    falling, not walking. Same "no credit on the step you stopped"
    principle as balance_reward's fall_penalty/alive_bonus."""
    v = np.tile(np.array([0.5, 0.0, 0.0]), (2, 1))
    terminal_fall = np.array([True, False])

    reward = progress_reward(v, v, std=0.5, terminal_fall=terminal_fall)

    assert reward[0] == 0.0  # fell this step: tracking spike zeroed out
    assert reward[1] == 1.0  # didn't fall: normal tracking reward stands


def test_clearance_reward_clips_to_unit_interval():
    dist = np.array([10.0, 0.0])
    r = clearance_reward(dist, safe_dist=0.5)
    assert r.shape == (2,)
    assert r[0] == 1.0
    assert r[1] == 0.0


def test_energy_reward_is_nonpositive():
    torque = np.array([[1.0, -2.0, 0.5], [0.0, 0.0, 0.0]])
    vel = np.ones((2, 3))
    r = energy_reward(torque, vel)
    assert r.shape == (2,)
    assert r[0] <= 0.0
    assert r[1] == 0.0


def test_impact_reward_penalizes_only_above_threshold():
    forces = np.array([[10.0, 10.0, 10.0, 10.0], [200.0, 0.0, 0.0, 0.0]])
    r = impact_reward(forces, threshold=50.0)
    assert r.shape == (2,)
    assert r[0] == 0.0
    assert r[1] < 0.0


def test_impact_reward_penalizes_foot_slip_only_while_in_contact():
    """Found 2026-09-18: a foot sliding while swinging through the air is
    normal and shouldn't be punished -- only slip WHILE planted (contact
    force above contact_threshold) counts, matching the reference term's
    binary contact-indicator gate."""
    forces = np.array([[10.0, 0.0, 0.0, 0.0], [10.0, 0.0, 0.0, 0.0]])
    vel_still = np.zeros((2, 4, 3))
    vel_slipping = np.zeros((2, 4, 3))
    vel_slipping[:, 0, 0] = 1.0  # foot 0 (in contact) moving sideways -> slip
    vel_swinging = np.zeros((2, 4, 3))
    vel_swinging[:, 1, 0] = 1.0  # foot 1 (NOT in contact, force=0) moving -> no slip penalty

    r_still = impact_reward(forces, foot_vel=vel_still)
    r_slipping = impact_reward(forces, foot_vel=vel_slipping)
    r_swinging = impact_reward(forces, foot_vel=vel_swinging)

    assert np.all(r_slipping < r_still)
    assert np.allclose(r_swinging, r_still)


def test_impact_reward_penalizes_undesired_contact_regardless_of_force_magnitude():
    """Found 2026-09-19: a fixed-command video export showed the policy
    dragging its CALF along the ground to move -- a zero-cost loophole,
    since foot_slip only watches the foot and the peak-force term is built
    for transient landing shocks, not sustained LOW-force dragging. Unlike
    those two, undesired_contact_count must penalize ANY nonzero contact,
    not just contact above some threshold -- a calf touching anything at
    all is wrong regardless of how gently."""
    forces = np.zeros((2, 4))
    r_no_contact = impact_reward(forces, undesired_contact_count=np.array([0.0, 0.0]))
    r_one_contact = impact_reward(forces, undesired_contact_count=np.array([1.0, 0.0]))
    assert r_no_contact[0] == 0.0
    assert r_one_contact[0] < 0.0
    assert r_one_contact[1] == r_no_contact[1]  # lane 1 unaffected


def test_smoothness_reward_penalizes_action_change():
    a = np.zeros((2, 12))
    b = np.ones((2, 12))
    acc = np.zeros((2, 12))
    vel = np.zeros((2, 12))
    r_same = smoothness_reward(a, a, acc, vel)
    r_diff = smoothness_reward(b, a, acc, vel)
    assert r_same.shape == (2,)
    assert np.allclose(r_same, 0.0)
    assert np.all(r_diff < 0.0)


def test_smoothness_reward_penalizes_joint_speed():
    """Found 2026-09-18: energy (torque*velocity) can stay small at high
    joint speed if torque happens to be low at that instant -- joint_vel
    must be penalized directly, independent of energy, matching the
    reference reward vector's separate "Joint Speed" term."""
    a = np.zeros((2, 12))
    acc = np.zeros((2, 12))
    vel_still = np.zeros((2, 12))
    vel_fast = np.full((2, 12), 5.0)
    r_still = smoothness_reward(a, a, acc, vel_still)
    r_fast = smoothness_reward(a, a, acc, vel_fast)
    assert np.all(r_fast < r_still)


def test_smoothness_reward_penalizes_a_large_action_held_steady():
    """Found 2026-09-18: action_rate alone only penalizes CHANGING the
    action -- a policy that locks onto one saturated (near ACTION_CLIP)
    action and barely changes it pays almost nothing under the old
    formula, matching what a physics validation rollout found (84% of
    joints saturated) and a forced-command test found (v_command pinned to
    0.5 m/s, tracking ratio only 6.6% -- the policy stood mostly still
    while holding near-maximal actions). A LARGE but UNCHANGING action
    must now cost more than a small unchanging one, which action_rate
    alone (zero for any unchanging action, regardless of its magnitude)
    could never express."""
    small = np.full((2, 12), 0.1)
    large = np.full((2, 12), 2.9)  # near ACTION_CLIP=3.0
    acc = np.zeros((2, 12))
    vel = np.zeros((2, 12))
    r_small_steady = smoothness_reward(small, small, acc, vel)  # unchanged -> action_rate=0
    r_large_steady = smoothness_reward(large, large, acc, vel)  # unchanged -> action_rate=0
    assert np.all(r_large_steady < r_small_steady), "a large steady action must cost more than a small steady one"


def test_efficiency_reward_is_energy_plus_smoothness():
    """energy and smoothness merged into one preference dimension
    2026-09-19 (0.91 measured correlation, eases the 5-dim Dirichlet
    coverage problem). Found the same day via PhysicsValidator's
    energy-vs-smoothness breakdown: unweighted, energy (raw torque*vel
    power) dominated smoothness by 9.66x raw scale on a trained
    checkpoint, drowning out action_magnitude (added specifically to fix
    action saturation) inside the merge -- so efficiency_reward applies
    energy_reward's own 0.1 coefficient (_ENERGY_COEF) before summing,
    matching the 0.01-weight convention smoothness_reward already uses on
    its own acc/joint_speed sub-terms for the identical reason."""
    torque = np.full((2, 12), 3.0)
    vel = np.full((2, 12), 1.0)
    action = np.full((2, 12), 0.5)
    prev_action = np.zeros((2, 12))
    acc = np.zeros((2, 12))

    r_energy = energy_reward(torque, vel)
    r_smoothness = smoothness_reward(action, prev_action, acc, vel)
    r_efficiency = efficiency_reward(torque, vel, action, prev_action, acc)

    assert np.allclose(r_efficiency, 0.1 * r_energy + r_smoothness)


def test_balance_reward_is_max_at_zero_tilt():
    # comment says which failure this prevents: a sign error here would
    # reward tipping over instead of penalizing it, silently undoing the
    # whole point of adding this term (see reward.py's docstring for why
    # it exists).
    flat = np.zeros((3, 2))
    tilted = np.tile(np.array([0.3, -0.2]), (3, 1))
    r_flat = balance_reward(flat)
    r_tilted = balance_reward(tilted)
    assert r_flat.shape == (3,)
    assert np.allclose(r_flat, 0.0)
    assert np.all(r_tilted < 0.0)


def test_balance_reward_penalizes_crouching_below_target_height():
    """Found 2026-09-18: v_z only penalizes vertical MOTION -- a lane that
    crouches low and then holds perfectly still pays nothing under v_z
    (matches a forced-command physics probe that found a checkpoint with
    every other grouped sub-penalty active still crouched and stood still,
    6.7% velocity-tracking ratio). height must be penalized directly,
    relative to target_height, regardless of whether the lane is moving."""
    flat = np.zeros((3, 2))
    at_target = np.full(3, 0.42)
    crouched = np.full(3, 0.20)
    r_at_target = balance_reward(flat, height=at_target, target_height=0.42)
    r_crouched = balance_reward(flat, height=crouched, target_height=0.42)
    assert np.allclose(r_at_target, 0.0)
    assert np.all(r_crouched < 0.0)


def test_balance_reward_penalizes_vertical_bounce():
    """Found 2026-09-18: none of the existing terms penalized bobbing --
    roll/pitch alone can be zero (perfectly flat) while the trunk bounces
    up and down every step. v_z is root_lin_vel_b's own z-component, so a
    sign error here (rewarding bounce instead of penalizing it) would
    silently encourage exactly the failure mode this term exists to stop."""
    flat = np.zeros((3, 2))
    r_still = balance_reward(flat, v_z=np.zeros(3))
    r_bouncing = balance_reward(flat, v_z=np.full(3, 0.5))
    assert np.allclose(r_still, 0.0)
    assert np.all(r_bouncing < 0.0)


def test_balance_reward_hip_activation_rewards_the_smaller_side():
    """hip_activation_coef must reward min(|hip_qdot_L|, |hip_qdot_R|) --
    the SMALLER side's own activity, not a symmetry comparison between
    them. A frozen-R/active-L lane (the exact pattern gait_joint_trace.py
    diagnosed 2026-09-20) must score near-zero bonus regardless of how
    large L's own activity is, since the reward exists to stop a hip
    collapsing to near-zero, not to reward total motion."""
    flat = np.zeros((2, 2))
    frozen_R = balance_reward(
        flat, hip_qdot_L=np.array([5.0, 5.0]), hip_qdot_R=np.array([0.01, 0.01]), hip_activation_coef=0.02,
    )
    both_active = balance_reward(
        flat, hip_qdot_L=np.array([5.0, 5.0]), hip_qdot_R=np.array([5.0, 5.0]), hip_activation_coef=0.02,
    )
    assert np.allclose(frozen_R, 0.02 * 0.01, atol=1e-6)  # bonus tracks the SMALLER side (R), not L
    assert np.allclose(both_active, 0.02 * 5.0)
    assert np.all(both_active > frozen_R)


def test_balance_reward_hip_sym_penalizes_deviation_from_mirror_symmetry():
    """UNITREE_A1_CFG's own default standing pose (FL_hip=+0.1, FR_hip=
    -0.1) means hip_q_L = -hip_q_R at the symmetric stance -- hip_sym_coef
    must penalize (hip_q_L + hip_q_R)^2, which is exactly 0 at that
    mirror-symmetric pose and grows for any L/R asymmetry, not
    (hip_q_L - hip_q_R)^2 (which would incorrectly treat the DEFAULT
    stance itself as maximally asymmetric)."""
    flat = np.zeros((2, 2))
    mirror_symmetric = balance_reward(
        flat, hip_q_L=np.array([0.1, 0.1]), hip_q_R=np.array([-0.1, -0.1]), hip_sym_coef=1.0,
    )
    asymmetric = balance_reward(
        flat, hip_q_L=np.array([0.1, 0.1]), hip_q_R=np.array([0.1, 0.1]), hip_sym_coef=1.0,
    )
    assert np.allclose(mirror_symmetric, 0.0)
    assert np.all(asymmetric < 0.0)


def test_balance_reward_penalizes_terminal_fall_but_not_timeout():
    roll_pitch = np.zeros((2, 2))
    terminal_fall = np.array([True, False])

    reward = balance_reward(roll_pitch, terminal_fall=terminal_fall, fall_penalty=5.0)

    assert reward[0] == -5.0
    assert reward[1] == 0.0


def test_alive_bonus_adds_flat_reward_only_while_not_fallen():
    """Found 2026-09-18: per-step reward (progress, smoothness) kept
    improving across a 20000-update run while mean_episode_length stayed
    flat, since none of the existing terms pay out purely for elapsed time
    alive. alive_bonus must add flat reward every non-terminal step, and
    must NOT survive the fall_penalty subtraction on the terminal step
    (AMOR's c_alive / Gym's alive_bonus convention: you don't get credit for
    "being alive" on the exact step you stopped being alive)."""
    roll_pitch = np.zeros((2, 2))
    terminal_fall = np.array([True, False])

    reward = balance_reward(roll_pitch, terminal_fall=terminal_fall, fall_penalty=5.0, alive_bonus=0.3)

    assert reward[1] == 0.3  # still alive this step: +alive_bonus, no penalty
    assert reward[0] == 0.3 - 5.0  # fell this step: alive_bonus still added, then fall_penalty subtracted


def test_compute_reward_vector_respects_active_mask_and_order():
    cfg = RewardVectorCfg(active=(True, False, True, False))
    n = 4
    transition = {
        "obs": np.zeros((n, 1)),  # only used by compute_reward_vector to infer N
        "v_actual": np.zeros((n, 3)), "v_command": np.zeros((n, 3)),
        "obstacle_dist": np.ones(n),
        "joint_torque": np.ones((n, 12)), "joint_vel": np.ones((n, 12)),
        "foot_contact_force": np.zeros((n, 4)),
        "action": np.zeros((n, 12)), "prev_action": np.zeros((n, 12)), "joint_acc": np.zeros((n, 12)),
        "roll_pitch": np.zeros((n, 2)),
    }
    r = compute_reward_vector(transition, cfg)
    assert r.shape == (n, 4)
    assert np.allclose(r[:, 1], 0.0)  # efficiency (energy+smoothness merged) masked off
    assert np.allclose(r[:, 3], 0.0)  # balance masked off
    assert np.all(r[:, 0] != 0.0)     # progress active
