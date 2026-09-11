import numpy as np

from talon_rl.obs_stack import ObservationStack


def test_default_stack_of_one_is_passthrough():
    stack = ObservationStack(obs_dim=4, num_policy_stacks=1, num_critic_stacks=1)
    obs = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    stack.reset(obs)
    assert np.allclose(stack.policy_obs, obs)
    assert np.allclose(stack.critic_obs, obs)


def test_reset_zero_pads_history_then_pushes_current_frame():
    stack = ObservationStack(obs_dim=2, num_policy_stacks=3, num_critic_stacks=1)
    obs = np.array([5.0, 6.0], dtype=np.float32)
    stack.reset(obs)
    # 3-stack: [zeros, zeros, obs]
    assert np.allclose(stack.policy_obs, np.array([0, 0, 0, 0, 5, 6], dtype=np.float32))
    assert np.allclose(stack.critic_obs, obs)  # 1-stack critic is unaffected


def test_push_slides_the_window():
    stack = ObservationStack(obs_dim=1, num_policy_stacks=2, num_critic_stacks=2)
    stack.reset(np.array([1.0], dtype=np.float32))
    stack.push(np.array([2.0], dtype=np.float32))
    assert np.allclose(stack.policy_obs, [1.0, 2.0])
    stack.push(np.array([3.0], dtype=np.float32))
    assert np.allclose(stack.policy_obs, [2.0, 3.0])  # oldest frame drops off


def test_policy_and_critic_stacks_can_differ():
    stack = ObservationStack(obs_dim=1, num_policy_stacks=4, num_critic_stacks=1)
    assert stack.policy_obs_dim == 4
    assert stack.critic_obs_dim == 1


def test_invalid_stack_count_rejected():
    import pytest

    with pytest.raises(ValueError):
        ObservationStack(obs_dim=1, num_policy_stacks=0)
