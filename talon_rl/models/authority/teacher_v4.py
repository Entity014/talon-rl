"""V4 privileged teacher (docs/methods/architecture/teacher-architecture.md).

Actor:  h_t = StateTrunk(x_t)            48 -> 256 -> 128 -> 16
        z_t = EnvEncoder(e_t)            12 -> 256 -> 128 -> 8
        z_w = rho(sum_i w_i phi(E(o_i)))  DeepSets objective-set encoder
        u_t = policy([h_t, z_t]; theta(z_w)), a_t = tanh(u_t)
        The policy backbone 24 -> 256 is fixed; the last two layers
        (256 -> 128 -> 12) get a family residual theta_0 + sum_k c_k(z_w) B_k.
Critic: c_t = f_V(x_t, e_t, z_w), V_i = Q_V(c_t, q_i).

Authority isolation: the objective set reaches the actor only through the
family coefficients, and the critic owns a separate objective embedding and
set encoder, so value-loss gradients never move actor parameters.
Zero-weight entries contribute nothing to z_w, so they act as padding.
"""
from __future__ import annotations
import torch
from torch import Tensor, nn
from torch.nn import functional as F

OBJECTIVE_ORDER=("T","A","O","S")


def validate_objective_set(ids:Tensor,weights:Tensor,num_objectives:int)->None:
    if ids.ndim!=2 or ids.dtype!=torch.long:
        raise ValueError("objective ids must be a long tensor [B,M]")
    if weights.shape!=ids.shape:
        raise ValueError("weights must have the same shape [B,M] as ids")
    if ((ids<0)|(ids>=num_objectives)).any():
        raise ValueError(f"objective ids must be in [0,{num_objectives})")
    if not torch.isfinite(weights).all() or (weights<0).any():
        raise ValueError("weights must be finite and non-negative")
    if not torch.allclose(weights.sum(-1),torch.ones_like(weights[:,0]),atol=1e-6):
        raise ValueError("objective-set weights must sum to one")


class ObjectiveSetEncoder(nn.Module):
    """z_w = rho(sum_i w_i phi(E(o_i))). E is exposed as the query embedding."""

    def __init__(self,num_objectives:int,embed_dim:int,hidden:int,out_dim:int):
        super().__init__()
        self.num_objectives=num_objectives
        self.embedding=nn.Embedding(num_objectives,embed_dim)
        self.phi=nn.Sequential(nn.Linear(embed_dim,hidden),nn.ELU(),nn.Linear(hidden,hidden))
        self.rho=nn.Sequential(nn.ELU(),nn.Linear(hidden,hidden),nn.ELU(),nn.Linear(hidden,out_dim))

    def forward(self,ids:Tensor,weights:Tensor)->Tensor:
        validate_objective_set(ids,weights,self.num_objectives)
        pooled=torch.einsum("bm,bmh->bh",weights,self.phi(self.embedding(ids)))
        return self.rho(pooled)


class TeacherV4(nn.Module):
    ACTION_CLIP=1.0
    H_DIM=16
    Z_ENV_DIM=8
    Z_W_DIM=16
    EMBED_DIM=16
    SET_HIDDEN=64
    NUM_BASES=8
    COEFF_HIDDEN=32
    POLICY_HIDDEN=(256,128)
    CRITIC_HIDDEN=128
    QUERY_HIDDEN=64

    def __init__(self,obs_dim:int=48,env_dim:int=12,action_dim:int=12,num_objectives:int=4):
        super().__init__()
        self.obs_dim=obs_dim;self.env_dim=env_dim;self.action_dim=action_dim
        p1,p2=self.POLICY_HIDDEN

        # ----- actor -----
        self.state_trunk=nn.Sequential(
            nn.Linear(obs_dim,256),nn.LayerNorm(256),nn.ELU(),
            nn.Linear(256,128),nn.ELU(),
            nn.Linear(128,self.H_DIM),nn.ELU())
        self.env_encoder=nn.Sequential(
            nn.Linear(env_dim,256),nn.ELU(),
            nn.Linear(256,128),nn.ELU(),
            nn.Linear(128,self.Z_ENV_DIM))
        self.actor_set_encoder=ObjectiveSetEncoder(num_objectives,self.EMBED_DIM,self.SET_HIDDEN,self.Z_W_DIM)
        self.family_hyper=nn.Sequential(
            nn.Linear(self.Z_W_DIM,self.COEFF_HIDDEN),nn.ELU(),
            nn.Linear(self.COEFF_HIDDEN,self.NUM_BASES))
        self.policy_backbone=nn.Sequential(nn.Linear(self.H_DIM+self.Z_ENV_DIM,p1),nn.ELU())
        self.family_w1_base=nn.Parameter(torch.empty(p2,p1))
        self.family_b1_base=nn.Parameter(torch.zeros(p2))
        self.family_w2_base=nn.Parameter(torch.empty(action_dim,p2))
        self.family_b2_base=nn.Parameter(torch.zeros(action_dim))
        self.family_B_w1=nn.Parameter(torch.empty(self.NUM_BASES,p2,p1))
        self.family_B_b1=nn.Parameter(torch.empty(self.NUM_BASES,p2))
        self.family_B_w2=nn.Parameter(torch.empty(self.NUM_BASES,action_dim,p2))
        self.family_B_b2=nn.Parameter(torch.empty(self.NUM_BASES,action_dim))
        self.log_std=nn.Parameter(torch.zeros(action_dim))

        # ----- critic -----
        self.critic_set_encoder=ObjectiveSetEncoder(num_objectives,self.EMBED_DIM,self.SET_HIDDEN,self.Z_W_DIM)
        self.critic_body=nn.Sequential(
            nn.Linear(obs_dim+env_dim+self.Z_W_DIM,256),nn.ELU(),
            nn.Linear(256,256),nn.ELU(),
            nn.Linear(256,self.CRITIC_HIDDEN),nn.ELU())
        self.query_head=nn.Sequential(
            nn.Linear(self.CRITIC_HIDDEN+self.EMBED_DIM,self.QUERY_HIDDEN),nn.ELU(),
            nn.Linear(self.QUERY_HIDDEN,1))

        # Same family init as the Phase-1 authority-isolated actor.
        nn.init.orthogonal_(self.family_w1_base);self.family_w1_base.data.mul_(0.25)
        nn.init.orthogonal_(self.family_w2_base);self.family_w2_base.data.mul_(0.10)
        nn.init.normal_(self.family_B_w1,0,0.01)
        nn.init.normal_(self.family_B_w2,0,0.01)
        nn.init.normal_(self.family_B_b1,0,0.005)
        nn.init.normal_(self.family_B_b2,0,0.005)

    # ----- actor -----
    def family_coefficients(self,ids:Tensor,weights:Tensor)->Tensor:
        return self.family_hyper(self.actor_set_encoder(ids,weights))

    def pre_tanh_mean(self,obs:Tensor,env:Tensor,ids:Tensor,weights:Tensor)->Tensor:
        x=self.policy_backbone(torch.cat((self.state_trunk(obs),self.env_encoder(env)),dim=-1))
        c=self.family_coefficients(ids,weights)
        # Equal to using W_base + sum_k c_k B_k without materializing per-sample weights.
        z0=F.linear(x,self.family_w1_base,self.family_b1_base)
        zd=torch.einsum("bk,bko->bo",c,torch.einsum("bi,koi->bko",x,self.family_B_w1))+c@self.family_B_b1
        q=F.elu(z0+zd)
        y0=F.linear(q,self.family_w2_base,self.family_b2_base)
        yd=torch.einsum("bk,bko->bo",c,torch.einsum("bi,koi->bko",q,self.family_B_w2))+c@self.family_B_b2
        return y0+yd

    @staticmethod
    def _log_det_jacobian(u:Tensor)->Tensor:
        return 2.0*(torch.log(torch.tensor(2.0,device=u.device,dtype=u.dtype))-u-F.softplus(-2.0*u))

    def _dist(self,obs,env,ids,weights):
        mean=self.pre_tanh_mean(obs,env,ids,weights)
        return torch.distributions.Normal(mean,self.log_std.exp().expand_as(mean))

    def act_inference(self,obs:Tensor,env:Tensor,ids:Tensor,weights:Tensor)->Tensor:
        return torch.tanh(self.pre_tanh_mean(obs,env,ids,weights))*self.ACTION_CLIP

    def act(self,obs:Tensor,env:Tensor,ids:Tensor,weights:Tensor):
        """Returns (action, log_prob, pre-tanh sample u)."""
        dist=self._dist(obs,env,ids,weights)
        u=dist.sample()
        lp=(dist.log_prob(u)-self._log_det_jacobian(u)).sum(-1)
        return torch.tanh(u)*self.ACTION_CLIP,lp,u

    def logp_from_pre_tanh(self,obs:Tensor,env:Tensor,ids:Tensor,weights:Tensor,u:Tensor)->Tensor:
        dist=self._dist(obs,env,ids,weights)
        return (dist.log_prob(u)-self._log_det_jacobian(u)).sum(-1)

    # ----- critic -----
    def critic_features(self,obs:Tensor,env:Tensor,ids:Tensor,weights:Tensor)->Tensor:
        return self.critic_body(torch.cat((obs,env,self.critic_set_encoder(ids,weights)),dim=-1))

    def query_values(self,obs:Tensor,env:Tensor,ids:Tensor,weights:Tensor,query_ids:Tensor|None=None)->Tensor:
        """V[b,q] for each queried objective; queries default to the active set."""
        if query_ids is None:query_ids=ids
        if query_ids.ndim!=2 or query_ids.shape[0]!=obs.shape[0]:
            raise ValueError("query_ids must be [B,Q]")
        c=self.critic_features(obs,env,ids,weights)
        q=self.critic_set_encoder.embedding(query_ids)
        cq=torch.cat((c.unsqueeze(1).expand(-1,q.shape[1],-1),q),dim=-1)
        return self.query_head(cq).squeeze(-1)

    def actor_parameters(self):
        return [p for n,p in self.named_parameters() if not n.startswith(("critic_","query_head"))]

    def critic_parameters(self):
        return [p for n,p in self.named_parameters() if n.startswith(("critic_","query_head"))]
