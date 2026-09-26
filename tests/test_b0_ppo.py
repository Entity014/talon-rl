import numpy as np
import pytest

import torch
from rl.core.algorithms.scalar_ppo import B0PPOConfig, B0PPOTrainer, ScalarRolloutBuffer, scalar_gae
from rl.core.modules.actor_critic import ActorCritic


def test_scalar_gae_terminal_masks_only_its_lane():
    rewards = np.ones((2, 2), np.float32)
    values = np.zeros((3, 2), np.float32)
    dones = np.array([[True, False], [False, False]])
    adv = scalar_gae(rewards, values, dones, 1.0, 1.0)
    np.testing.assert_allclose(adv, [[1.0, 2.0], [1.0, 1.0]])


def test_scalar_gae_uses_final_bootstrap():
    rewards = np.zeros((1, 1), np.float32)
    values = np.array([[0.0], [3.0]], np.float32)
    dones = np.zeros((1, 1), bool)
    np.testing.assert_allclose(scalar_gae(rewards, values, dones, 0.5, 0.95), [[1.5]])


def test_b0_std_schedule_values():
    cfg = B0PPOConfig()
    np.testing.assert_allclose(
        [cfg.scheduled_std(u) for u in (0, 100, 250, 400, 500)],
        [0.82, 0.82, 0.46, 0.10, 0.10],
    )


def test_b0_optimizer_excludes_log_std_and_ratio_is_one():
    model = ActorCritic(3, 3, 2, 1, [4])
    trainer = B0PPOTrainer(model)
    trainer.begin_update()
    actor_obs = critic_obs = torch.zeros(5, 3)
    actions, old_logp = model.act(actor_obs)
    policy, value, entropy, ratio, clip = trainer.ppo_loss(
        actor_obs, critic_obs, actions, old_logp.detach(), torch.ones(5), torch.zeros(5)
    )
    assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-6)
    assert clip.item() == 0.0
    assert all(param is not model.log_std for group in trainer.optim.param_groups for param in group["params"])


def test_b0_checkpoint_resume_preserves_schedule(tmp_path):
    model = ActorCritic(3, 3, 2, 1, [4])
    trainer = B0PPOTrainer(model)
    trainer.update_idx = 250
    std = trainer.begin_update()
    path = tmp_path / "b0.pt"
    trainer.save(path)
    restored = B0PPOTrainer(ActorCritic(3, 3, 2, 1, [4]))
    restored.load(path)
    assert restored.update_idx == 250
    assert restored.model.exploration_mode == "scheduled_fixed_std"
    assert restored._std_for_update == pytest.approx(std)
    restored.finish_update()
    assert restored.begin_update() == pytest.approx(restored.cfg.scheduled_std(251))
    assert all(param is not restored.model.log_std for group in restored.optim.param_groups for param in group["params"])


def test_scalar_rollout_buffer_shapes_and_clear():
    b = ScalarRolloutBuffer(2, 3)
    for t in range(2):
        b.append(np.zeros((3, 4)), np.zeros((3, 2)), np.ones(3), np.arange(3),
                 np.array([False, t == 0, False]), np.zeros(3))
    b.finish(np.ones(3)); a=b.arrays(); flat=b.flatten()
    assert a["rewards"].shape == a["dones"].shape == a["values"].shape == (2, 3)
    assert a["final_value"].shape == (3,)
    assert flat["obs"].shape == (6, 4) and flat["actions"].shape == (6, 2)
    b.clear(); assert b.obs == [] and b.final_value is None
