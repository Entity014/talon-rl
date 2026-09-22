"""Four-objective shared actor/critic primitives for T4 specialist generation."""
from __future__ import annotations
import torch
from torch import Tensor
from rl.core.modules.actor_critic import ActorCritic

OBJECTIVE_ORDER=("velocity_tracking","angular_stability","orientation_stability","control_smoothness")
NUM_OBJECTIVES=4

class T4SharedActorCritic(ActorCritic):
    # T5 consistency repair: the policy support exactly matches the action
    # applied to IsaacLab. No downstream external clamp is permitted.
    ACTION_CLIP = 1.0

    def __init__(self,obs_dim:int,action_dim:int,hidden_dims:list[int]|None=None):
        hidden_dims=hidden_dims or [128,128,128]
        super().__init__(obs_dim+NUM_OBJECTIVES,obs_dim+NUM_OBJECTIVES,action_dim,NUM_OBJECTIVES,hidden_dims)
        self.physical_obs_dim=obs_dim
    @staticmethod
    def _with_w(obs:Tensor,w:Tensor)->Tensor:
        if obs.ndim!=2 or w.ndim!=2 or obs.shape[0]!=w.shape[0] or w.shape[1]!=NUM_OBJECTIVES:
            raise ValueError("obs and w must be [batch,dim] and [batch,4]")
        if not torch.isfinite(w).all() or (w<0).any() or not torch.allclose(w.sum(-1),torch.ones(w.shape[0],device=w.device),atol=1e-6):
            raise ValueError("w must be finite and on the four-objective simplex")
        return torch.cat((obs,w),dim=-1)
    def act_with_preference(self,obs,w): return self.act(self._with_w(obs,w))
    def act_with_preference_latent(self,obs,w):
        actor_obs=self._with_w(obs,w)
        dist=self._pre_tanh_dist(actor_obs)
        u=dist.sample()
        action=torch.tanh(u)*self.ACTION_CLIP
        logp=(dist.log_prob(u)-self._log_det_jacobian(u)).sum(-1)
        return action,logp,u
    def act_inference_with_preference(self,obs,w): return self.act_inference(self._with_w(obs,w))
    def logp_with_preference(self,obs,w,action): return self.logp(self._with_w(obs,w),action)
    def logp_from_pre_tanh_with_preference(self,obs,w,u):
        dist=self._pre_tanh_dist(self._with_w(obs,w))
        return (dist.log_prob(u)-self._log_det_jacobian(u)).sum(-1)
    def value_with_preference(self,obs,w): return self.value(self._with_w(obs,w))

def initialize_from_rsl_m01(model:T4SharedActorCritic,checkpoint,device:str="cpu")->None:
    state=torch.load(checkpoint,map_location=device,weights_only=False)
    source=state.get("model_state_dict",state.get("model",state));target=model.state_dict()
    for i in (0,2,4):
        if i==0:
            target[f"actor_body.{i}.weight"].zero_()
            target[f"actor_body.{i}.weight"][:,:model.physical_obs_dim].copy_(source[f"actor.{i}.weight"])
            target[f"critic_body.{i}.weight"][:,:model.physical_obs_dim].copy_(source[f"critic.{i}.weight"])
            target[f"critic_body.{i}.weight"][:,model.physical_obs_dim:].zero_()
        else:
            target[f"actor_body.{i}.weight"].copy_(source[f"actor.{i}.weight"])
            target[f"critic_body.{i}.weight"].copy_(source[f"critic.{i}.weight"])
        target[f"actor_body.{i}.bias"].copy_(source[f"actor.{i}.bias"])
        target[f"critic_body.{i}.bias"].copy_(source[f"critic.{i}.bias"])
    target["actor_mean.weight"].copy_(source["actor.6.weight"])
    target["actor_mean.bias"].copy_(source["actor.6.bias"])
    target["critic_head.weight"].copy_(source["critic.6.weight"].expand(NUM_OBJECTIVES,-1))
    target["critic_head.bias"].copy_(source["critic.6.bias"].expand(NUM_OBJECTIVES))
    target["log_std"].copy_(source["std"].log())
    model.load_state_dict(target)

def vector_gae(reward:Tensor,value:Tensor,next_value:Tensor,done:Tensor,gamma:float=.99,lam:float=.95):
    if reward.ndim!=3 or reward.shape[-1]!=4 or value.shape!=reward.shape or next_value.shape!=reward.shape[1:] or done.shape!=reward.shape[:2]:
        raise ValueError("reward/value/next_value/done shape mismatch")
    adv=torch.zeros_like(reward);last=torch.zeros_like(next_value)
    for t in range(reward.shape[0]-1,-1,-1):
        bootstrap=next_value if t==reward.shape[0]-1 else value[t+1]
        nt=(~done[t]).to(reward.dtype).unsqueeze(-1)
        delta=reward[t]+gamma*bootstrap*nt-value[t]
        last=delta+gamma*lam*nt*last;adv[t]=last
    return adv,adv+value

def scalarized_late_weighted_ppo(ratio:Tensor,advantages:Tensor,w:Tensor,clip_eps:float=.2)->Tensor:
    if advantages.ndim!=2 or advantages.shape[1]!=4 or w.shape!=advantages.shape or ratio.shape!=advantages.shape[:1]:
        raise ValueError("ratio/advantages/w shape mismatch")
    clipped=ratio.clamp(1-clip_eps,1+clip_eps)
    po=torch.minimum(ratio[:,None]*advantages,clipped[:,None]*advantages)
    return -(4*(w*po).sum(-1)).mean()

def vector_value_loss(value:Tensor,returns:Tensor)->Tensor:
    if value.shape!=returns.shape or value.shape[-1]!=4: raise ValueError("value/returns shape mismatch")
    return (value-returns).pow(2).mean()
