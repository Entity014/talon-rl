"""Tests for scalar PPO and stability-preservation variants."""

# -----------------------------------------------------------------------------
# Former: test_b0_ppo.py
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# Former: test_b1_p1.py
# -----------------------------------------------------------------------------

import numpy as np
import torch
import pytest
from rl.core.algorithms.scalar_ppo import B0PPOConfig, B0PPOTrainer, analytic_gaussian_kl
from rl.core.modules.actor_critic import ActorCritic

def batch(n=8):
    torch.manual_seed(4)
    obs=torch.randn(n,3); model=ActorCritic(3,3,2,1,[4]); trainer=B0PPOTrainer(model, B0PPOConfig(b1_p1_enabled=True, actor_epochs=3), lr=1e-2)
    trainer.begin_update(); actions, old=model.act(obs); return trainer, model, obs, actions.detach(), old.detach()

def test_b1_closed_form_kl():
    old=torch.zeros(2,3); new=torch.full((2,3), .2)
    expected=torch.full((2,), 3*.2*.2/(2*.5*.5))
    assert torch.allclose(analytic_gaussian_kl(old,new,.5), expected)

def test_b1_low_kl_continues_epochs():
    trainer, model, obs, actions, old = batch(); trainer.cfg = B0PPOConfig(b1_p1_enabled=True, actor_epochs=3, target_kl=100.0)
    out=trainer.optimize_batch(obs,actions,old,torch.ones(8),torch.zeros(8))
    assert not out["actor_stopped"] and out["actor_epochs_completed"] == 3

def test_b1_high_kl_stops_actor_but_critic_updates():
    trainer, model, obs, actions, old = batch(); trainer.cfg = B0PPOConfig(b1_p1_enabled=True, actor_epochs=3, target_kl=1e-12)
    critic_before=model.critic_head.weight.detach().clone(); out=trainer.optimize_batch(obs,actions,old,torch.ones(8),torch.zeros(8))
    assert out["actor_stopped"] and out["actor_epochs_completed"] == 1 and out["critic_completed"]
    assert not torch.equal(critic_before, model.critic_head.weight)

def test_b1_std_unchanged_and_flag_off_single_epoch():
    trainer, model, obs, actions, old = batch(); std=trainer._std_for_update
    trainer.cfg = B0PPOConfig(b1_p1_enabled=False, actor_epochs=9)
    out=trainer.optimize_batch(obs,actions,old,torch.ones(8),torch.zeros(8))
    assert out["actor_epochs_completed"] == 1 and trainer._std_for_update == std

def test_b1_checkpoint_resume_preserves_config(tmp_path):
    trainer, model, *_ = batch(); trainer.update_idx = 7
    path=tmp_path/'b1.pt'; trainer.save(path)
    restored=B0PPOTrainer(ActorCritic(3,3,2,1,[4]), B0PPOConfig(b1_p1_enabled=True, actor_epochs=3))
    restored.load(path)
    assert restored.update_idx == 7 and restored.cfg.b1_p1_enabled and restored._std_for_update == pytest.approx(restored.cfg.scheduled_std(7))

def test_b1_p2_adaptive_lr_and_resume(tmp_path):
    cfg=B0PPOConfig(b1_p2_enabled=True, actor_epochs=2, kl_low=100.0, kl_high=200.0)
    model=ActorCritic(3,3,2,1,[4]); tr=B0PPOTrainer(model,cfg,lr=3e-4); tr.begin_update(); obs=torch.randn(8,3); act,old=model.act(obs)
    out=tr.optimize_batch(obs,act.detach(),old.detach(),torch.ones(8),torch.zeros(8))
    assert out['actor_stopped'] is False and out['actor_epochs_completed']==2 and out['actor_lr']>3e-4
    p=tmp_path/'p2.pt'; tr.save(p); restored=B0PPOTrainer(ActorCritic(3,3,2,1,[4]),cfg); restored.load(p)
    assert restored.actor_optim.param_groups[0]['lr']==pytest.approx(out['actor_lr'])

def test_b1_s1_time_mask_command_exclusion_and_resume(tmp_path):
    cfg=B0PPOConfig(b1_p2_enabled=True,b1_s1_enabled=True,actor_epochs=2,kl_low=-100,kl_high=100)
    m=ActorCritic(48,48,3,1,[8]); tr=B0PPOTrainer(m,cfg,lr=3e-4); tr.begin_update(); obs=torch.randn(4,3,48); obs[:,:,9:12]=torch.tensor([.5,0.,0.]); act,old=m.act(obs.reshape(-1,48)); dones=torch.zeros(4,3,dtype=torch.bool); dones[1,0]=True
    out=tr.optimize_batch(obs.reshape(-1,48),act.detach(),old.detach(),torch.ones(12),torch.zeros(12),rollout_obs=obs,rollout_dones=dones)
    assert out['spatial_loss']>=0 and out['temporal_loss']>=0 and out['actor_grad_norm']>0 and out['actor_stopped'] is False
    p=tmp_path/'s1.pt'; tr.save(p); restored=B0PPOTrainer(ActorCritic(48,48,3,1,[8]),cfg); restored.load(p); assert restored.cfg.b1_s1_enabled

def test_b1_r1_shared_trunk_reconstruction_actor_only_and_resume(tmp_path):
    cfg=B0PPOConfig(b1_p2_enabled=True,b1_r1_enabled=True,actor_epochs=1,kl_low=-100,kl_high=100)
    m=ActorCritic(5,5,2,1,[8], reconstruction_dim=3)
    tr=B0PPOTrainer(m,cfg,lr=3e-4); tr.begin_update()
    obs=torch.randn(12,5); act,old=m.act(obs); ro=obs.reshape(3,4,5)
    targets=torch.randn(3,4,3); critic_before=m.critic_head.weight.detach().clone()
    out=tr.optimize_batch(obs,act.detach(),old.detach(),torch.ones(12),torch.zeros(12),rollout_obs=ro,reconstruction_targets=targets)
    assert out['reconstruction_loss'] > 0 and out['reconstruction_grad_norm'] > 0
    assert out['actor_grad_norm'] > 0 and out['critic_completed']
    assert torch.equal(critic_before, m.critic_head.weight) is False
    p=tmp_path/'r1.pt'; tr.save(p)
    restored=B0PPOTrainer(ActorCritic(5,5,2,1,[8], reconstruction_dim=3),cfg); restored.load(p)
    assert restored.cfg.b1_r1_enabled and restored.update_idx == 0
