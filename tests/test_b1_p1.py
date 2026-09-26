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
    m=ActorCritic(51,51,3,1,[8]); tr=B0PPOTrainer(m,cfg,lr=3e-4); tr.begin_update(); obs=torch.randn(4,3,51); obs[:,:,42:45]=torch.tensor([.5,0.,0.]); act,old=m.act(obs.reshape(-1,51)); dones=torch.zeros(4,3,dtype=torch.bool); dones[1,0]=True
    out=tr.optimize_batch(obs.reshape(-1,51),act.detach(),old.detach(),torch.ones(12),torch.zeros(12),rollout_obs=obs,rollout_dones=dones)
    assert out['spatial_loss']>=0 and out['temporal_loss']>=0 and out['actor_grad_norm']>0 and out['actor_stopped'] is False
    p=tmp_path/'s1.pt'; tr.save(p); restored=B0PPOTrainer(ActorCritic(51,51,3,1,[8]),cfg); restored.load(p); assert restored.cfg.b1_s1_enabled

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
