import numpy as np
import pytest

from moppo.obs_stack import ObservationStack


def test_default_stack_of_one_is_passthrough():
    stack = ObservationStack(num_envs=2, obs_dim=4, num_policy_stacks=1, num_critic_stacks=1)
    obs = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
    stack.reset(obs)
    assert np.allclose(stack.policy_obs, obs)
    assert np.allclose(stack.critic_obs, obs)


def test_reset_zero_pads_history_then_pushes_current_frame():
    stack = ObservationStack(num_envs=1, obs_dim=2, num_policy_stacks=3, num_critic_stacks=1)
    obs = np.array([[5.0, 6.0]], dtype=np.float32)
    stack.reset(obs)
    assert np.allclose(stack.policy_obs, [[0, 0, 0, 0, 5, 6]])
    assert np.allclose(stack.critic_obs, obs)


def test_push_slides_the_window_per_lane():
    stack = ObservationStack(num_envs=2, obs_dim=1, num_policy_stacks=2, num_critic_stacks=2)
    stack.reset(np.array([[1.0], [10.0]], dtype=np.float32))
    stack.push(np.array([[2.0], [20.0]], dtype=np.float32), done_mask=np.zeros(2, dtype=bool))
    assert np.allclose(stack.policy_obs, [[1.0, 2.0], [10.0, 20.0]])
    stack.push(np.array([[3.0], [30.0]], dtype=np.float32), done_mask=np.zeros(2, dtype=bool))
    assert np.allclose(stack.policy_obs, [[2.0, 3.0], [20.0, 30.0]])


def test_push_with_done_mask_resets_only_that_lane():
    stack = ObservationStack(num_envs=2, obs_dim=1, num_policy_stacks=2, num_critic_stacks=2)
    stack.reset(np.array([[1.0], [10.0]], dtype=np.float32))
    stack.push(np.array([[2.0], [20.0]], dtype=np.float32), done_mask=np.zeros(2, dtype=bool))
    # lane 0 terminates and auto-resets with a fresh obs; lane 1 keeps sliding
    stack.push(np.array([[99.0], [30.0]], dtype=np.float32), done_mask=np.array([True, False]))
    assert np.allclose(stack.policy_obs[0], [0.0, 99.0])   # lane 0: history wiped, fresh frame
    assert np.allclose(stack.policy_obs[1], [20.0, 30.0])  # lane 1: unaffected, kept sliding


def test_policy_and_critic_stacks_can_differ():
    stack = ObservationStack(num_envs=3, obs_dim=1, num_policy_stacks=4, num_critic_stacks=1)
    assert stack.policy_obs_dim == 4
    assert stack.critic_obs_dim == 1


def test_invalid_stack_count_rejected():
    with pytest.raises(ValueError):
        ObservationStack(num_envs=1, obs_dim=1, num_policy_stacks=0)
