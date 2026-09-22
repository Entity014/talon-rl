"""V2-R2 projected semantic-subspace actor/critic.

R2 replaces unconstrained actor FiLM with one additive hidden residual constrained
to a frozen, provenance-backed 2D basis [b_shared, b_BE].  Critic/manifold
semantics remain the V2 behavior-latent path.
"""
from __future__ import annotations
import torch
from torch import Tensor, nn
from torch.distributions import Normal
from rl.core.modules.actor_critic import ActorCritic
from talon_rl.v1c_actor_critic import NUM_OBJECTIVES

LATENT_DIM=2
COEFF_HIDDEN=16

class V2R2ProjectedActorCritic(ActorCritic):
    def __init__(
        self, obs_dim:int, action_dim:int, basis:Tensor,
        anchors:Tensor|None=None, hidden_dims:list[int]|None=None, *,
        projection_alpha:float=1.0,
    ):
        hidden_dims=hidden_dims or [128,128,128]
        if basis.shape != (2,hidden_dims[0]):
            raise ValueError(f"basis must be [2,{hidden_dims[0]}]")
        super().__init__(obs_dim,obs_dim+LATENT_DIM,action_dim,NUM_OBJECTIVES,hidden_dims)
        self.physical_obs_dim=obs_dim
        self.projection_alpha=float(projection_alpha)
        if not torch.isfinite(torch.tensor(self.projection_alpha)) or self.projection_alpha<0:
            raise ValueError("projection_alpha must be finite and non-negative")
        # Preserve V2 critic/manifold path.
        self.behavior_encoder=nn.Sequential(nn.Linear(NUM_OBJECTIVES,32),nn.ELU(),nn.Linear(32,LATENT_DIM))
        self.register_buffer("behavior_anchors",torch.zeros(NUM_OBJECTIVES,LATENT_DIM) if anchors is None else anchors.detach().clone(),persistent=True)
        # Projected actor path.
        self.actor_pre=nn.Linear(obs_dim,hidden_dims[0])
        self.coeff_net=nn.Sequential(nn.Linear(NUM_OBJECTIVES,COEFF_HIDDEN),nn.ELU(),nn.Linear(COEFF_HIDDEN,2))
        self.actor_rest=nn.Sequential(nn.Linear(hidden_dims[0],hidden_dims[1]),nn.ELU(),nn.Linear(hidden_dims[1],hidden_dims[2]),nn.ELU())
        self.actor_body=nn.Identity()
        b=basis.detach().clone()
        self.register_buffer("semantic_basis",b,persistent=True)
        self._init_identity_conditioning()

    def _init_identity_conditioning(self):
        nn.init.xavier_uniform_(self.behavior_encoder[0].weight);nn.init.zeros_(self.behavior_encoder[0].bias)
        nn.init.zeros_(self.behavior_encoder[2].weight);nn.init.zeros_(self.behavior_encoder[2].bias)
        nn.init.xavier_uniform_(self.coeff_net[0].weight);nn.init.zeros_(self.coeff_net[0].bias)
        nn.init.zeros_(self.coeff_net[2].weight);nn.init.zeros_(self.coeff_net[2].bias)

    @staticmethod
    def _validate_w(obs:Tensor,w:Tensor):
        if obs.ndim!=2 or w.ndim!=2 or obs.shape[0]!=w.shape[0] or w.shape[1]!=NUM_OBJECTIVES:
            raise ValueError("obs and w must be [batch,dim] and [batch,3]")
        if not torch.isfinite(w).all() or (w<0).any() or not torch.allclose(w.sum(-1),torch.ones(w.shape[0],device=w.device),atol=1e-6):
            raise ValueError("w must be finite and on the simplex")

    def behavior_z(self,w:Tensor)->Tensor:
        return self.behavior_encoder(w)

    def z_reference(self,w:Tensor)->Tensor:
        return w@self.behavior_anchors

    def manifold_loss(self,w:Tensor)->Tensor:
        return (self.behavior_z(w)-self.z_reference(w).detach()).pow(2).mean()

    def semantic_coefficients(self,w:Tensor)->Tensor:
        if w.ndim!=2 or w.shape[1]!=NUM_OBJECTIVES:
            raise ValueError("w must be [batch,3]")
        return torch.tanh(self.coeff_net(w))

    def projected_delta(self,w:Tensor)->Tensor:
        return self.projection_alpha*(self.semantic_coefficients(w)@self.semantic_basis)

    def actor_features(self,obs:Tensor,w:Tensor)->Tensor:
        self._validate_w(obs,w)
        h=torch.nn.functional.elu(self.actor_pre(obs))
        h=h+self.projected_delta(w)
        return self.actor_rest(h)

    def _dist(self,obs:Tensor,w:Tensor)->Normal:
        return Normal(self.actor_mean(self.actor_features(obs,w)),(self.log_std if self.exploration_mode=="learned" else self.scheduled_log_std).exp())

    def act_with_preference(self,obs:Tensor,w:Tensor):
        dist=self._dist(obs,w);u=dist.sample();return self._squash(dist,u)

    def act_inference_with_preference(self,obs:Tensor,w:Tensor)->Tensor:
        return torch.tanh(self._dist(obs,w).mean)*self.ACTION_CLIP

    def logp_with_preference(self,obs:Tensor,w:Tensor,action:Tensor)->Tensor:
        dist=self._dist(obs,w);u=torch.atanh((action/self.ACTION_CLIP).clamp(-1+self._ATANH_EPS,1-self._ATANH_EPS));_,logp=self._squash(dist,u);return logp

    def value_with_preference(self,obs:Tensor,w:Tensor)->Tensor:
        self._validate_w(obs,w)
        return self.critic_head(self.critic_body(torch.cat((obs,self.behavior_z(w)),dim=-1)))

def initialize_from_v1c(model:V2R2ProjectedActorCritic,source_state:dict,*,w_ref=(1/3,1/3,1/3))->None:
    w=torch.as_tensor(w_ref,dtype=model.actor_pre.weight.dtype,device=model.actor_pre.weight.device)
    with torch.no_grad():
        model.actor_pre.weight.copy_(source_state["actor_body.0.weight"][:,:model.physical_obs_dim])
        aw=source_state["actor_body.0.weight"].to(w);ab=source_state["actor_body.0.bias"].to(w)
        model.actor_pre.bias.copy_(ab+aw[:,model.physical_obs_dim:]@w)
        model.actor_rest[0].weight.copy_(source_state["actor_body.2.weight"]);model.actor_rest[0].bias.copy_(source_state["actor_body.2.bias"])
        model.actor_rest[2].weight.copy_(source_state["actor_body.4.weight"]);model.actor_rest[2].bias.copy_(source_state["actor_body.4.bias"])
        model.actor_mean.weight.copy_(source_state["actor_mean.weight"]);model.actor_mean.bias.copy_(source_state["actor_mean.bias"])
        model.critic_body[0].weight[:,:model.physical_obs_dim].copy_(source_state["critic_body.0.weight"][:,:model.physical_obs_dim])
        model.critic_body[0].weight[:,model.physical_obs_dim:].zero_()
        cw=source_state["critic_body.0.weight"].to(w);cb=source_state["critic_body.0.bias"].to(w)
        model.critic_body[0].bias.copy_(cb+cw[:,model.physical_obs_dim:]@w)
        model.critic_body[2].weight.copy_(source_state["critic_body.2.weight"]);model.critic_body[2].bias.copy_(source_state["critic_body.2.bias"])
        model.critic_body[4].weight.copy_(source_state["critic_body.4.weight"]);model.critic_body[4].bias.copy_(source_state["critic_body.4.bias"])
        model.critic_head.weight.copy_(source_state["critic_head.weight"]);model.critic_head.bias.copy_(source_state["critic_head.bias"])
        model.log_std.copy_(source_state["log_std"]);model.scheduled_log_std.copy_(source_state["scheduled_log_std"])
