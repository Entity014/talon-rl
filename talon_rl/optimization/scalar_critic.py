"""Pivot-P1 scalar critic, GAE, and PPO helpers."""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from ..models.foundations.three_objective import V1CSharedActorCritic

def _check(reward: Tensor, value: Tensor, next_value: Tensor, done: Tensor) -> None:
    if reward.shape[:2] != value.shape[:2] or next_value.shape != value.shape[1:] or done.shape != reward.shape[:2]:
        raise ValueError("reward/value/next_value/done shapes are incompatible")
    if not all(torch.isfinite(x).all() for x in (reward, value, next_value)):
        raise ValueError("GAE inputs must be finite")

def scalar_gae(reward: Tensor, value: Tensor, next_value: Tensor, done: Tensor, gamma: float = .99, lam: float = .95):
    if reward.ndim != 2 or value.ndim != 2 or next_value.ndim != 1:
        raise ValueError("scalar GAE expects reward/value [T,B] and next_value [B]")
    _check(reward, value, next_value, done)
    adv=torch.zeros_like(reward); last=torch.zeros_like(next_value)
    for t in range(reward.shape[0]-1,-1,-1):
        bootstrap=next_value if t==reward.shape[0]-1 else value[t+1]
        nonterminal=(~done[t]).to(reward.dtype)
        delta=reward[t]+gamma*bootstrap*nonterminal-value[t]
        last=delta+gamma*lam*nonterminal*last;adv[t]=last
    return adv, adv+value

def normalize_final_advantage(advantage: Tensor, eps: float = 1e-8) -> Tensor:
    if not torch.isfinite(advantage).all(): raise ValueError("advantage must be finite")
    return (advantage-advantage.mean())/(advantage.std(unbiased=False)+eps)

def scalar_ppo_loss(ratio: Tensor, advantage: Tensor, clip_eps: float = .2) -> Tensor:
    if ratio.ndim != 1 or advantage.shape != ratio.shape: raise ValueError("scalar PPO ratio/advantage shape mismatch")
    clipped=ratio.clamp(1-clip_eps,1+clip_eps)
    return -torch.minimum(ratio*advantage,clipped*advantage).mean()

def control_advantage(reward_vector: Tensor, weights: Tensor, value: Tensor, next_value: Tensor, done: Tensor):
    if reward_vector.ndim != 3 or weights.ndim != 3 or reward_vector.shape != weights.shape:
        raise ValueError("control reward/weights must be [T,B,3]")
    scalar_reward=(reward_vector*weights).sum(-1)
    raw,_=scalar_gae(scalar_reward,value,next_value,done)
    return normalize_final_advantage(raw), scalar_reward

def treatment_advantage(reward_vector: Tensor, weights: Tensor, value: Tensor, next_value: Tensor, done: Tensor):
    if reward_vector.ndim != 3 or weights.shape != reward_vector.shape or value.shape != reward_vector.shape:
        raise ValueError("treatment reward/weights/value must be [T,B,3]")
    raw_vec,_=vector_gae_local(reward_vector,value,next_value,done)
    raw=(raw_vec*weights).sum(-1)
    return normalize_final_advantage(raw), raw_vec

def vector_gae_local(reward: Tensor, value: Tensor, next_value: Tensor, done: Tensor, gamma: float = .99, lam: float = .95):
    if reward.ndim != 3 or value.ndim != 3 or next_value.ndim != 2:
        raise ValueError("vector GAE expects reward/value [T,B,O] and next_value [B,O]")
    _check(reward,value,next_value,done)
    adv=torch.zeros_like(reward);last=torch.zeros_like(next_value)
    for t in range(reward.shape[0]-1,-1,-1):
        bootstrap=next_value if t==reward.shape[0]-1 else value[t+1]
        nonterminal=(~done[t]).to(reward.dtype).unsqueeze(-1)
        delta=reward[t]+gamma*bootstrap*nonterminal-value[t]
        last=delta+gamma*lam*nonterminal*last;adv[t]=last
    return adv,adv+value


class P1ScalarCriticActorCritic(V1CSharedActorCritic):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims=None):
        super().__init__(obs_dim, action_dim, hidden_dims)
        self.critic_head = nn.Linear((hidden_dims or [128,128,128])[-1], 1)

def initialize_p1_scalar_from_rsl(model: P1ScalarCriticActorCritic, checkpoint, device='cpu') -> None:
    state=torch.load(checkpoint,map_location=device,weights_only=False);source=state.get('model_state_dict',state.get('model',state));target=model.state_dict()
    for i in (0,2,4):
        if i==0:
            target[f'actor_body.{i}.weight'].zero_();target[f'actor_body.{i}.weight'][:,:model.physical_obs_dim].copy_(source[f'actor.{i}.weight'])
            target[f'critic_body.{i}.weight'][:,:model.physical_obs_dim].copy_(source[f'critic.{i}.weight']);target[f'critic_body.{i}.weight'][:,model.physical_obs_dim:].zero_()
        else:
            target[f'actor_body.{i}.weight'].copy_(source[f'actor.{i}.weight']);target[f'critic_body.{i}.weight'].copy_(source[f'critic.{i}.weight'])
        target[f'actor_body.{i}.bias'].copy_(source[f'actor.{i}.bias']);target[f'critic_body.{i}.bias'].copy_(source[f'critic.{i}.bias'])
    target['actor_mean.weight'].copy_(source['actor.6.weight']);target['actor_mean.bias'].copy_(source['actor.6.bias']);target['critic_head.weight'].copy_(source['critic.6.weight']);target['critic_head.bias'].copy_(source['critic.6.bias']);target['log_std'].copy_(source['std'].log());model.load_state_dict(target)
