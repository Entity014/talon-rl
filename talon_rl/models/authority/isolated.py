"""Authority-isolated actor/critic model family."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from ..foundations.four_objective import T4SharedActorCritic

class AuthorityIsolatedActor(nn.Module):
    ACTION_CLIP=1.0
    NUM_OBJECTIVES=4
    NUM_BASES=8
    COEFF_HIDDEN=32
    HIDDEN=128

    def __init__(self,obs_dim:int=48,action_dim:int=12):
        super().__init__()
        self.obs_dim=obs_dim;self.action_dim=action_dim
        self.state_trunk=nn.Sequential(
            nn.Linear(obs_dim,128),nn.ELU(),
            nn.Linear(128,128),nn.ELU(),
            nn.Linear(128,128),nn.ELU(),
        )
        self.state_action_head=nn.Linear(128,action_dim)

        self.family_hyper=nn.Sequential(
            nn.Linear(self.NUM_OBJECTIVES,self.COEFF_HIDDEN),
            nn.ELU(),
            nn.Linear(self.COEFF_HIDDEN,self.NUM_BASES),
        )
        self.family_w1_base=nn.Parameter(torch.empty(self.HIDDEN,128))
        self.family_b1_base=nn.Parameter(torch.zeros(self.HIDDEN))
        self.family_w2_base=nn.Parameter(torch.empty(action_dim,self.HIDDEN))
        self.family_b2_base=nn.Parameter(torch.zeros(action_dim))

        self.family_B_w1=nn.Parameter(torch.empty(self.NUM_BASES,self.HIDDEN,128))
        self.family_B_b1=nn.Parameter(torch.empty(self.NUM_BASES,self.HIDDEN))
        self.family_B_w2=nn.Parameter(torch.empty(self.NUM_BASES,action_dim,self.HIDDEN))
        self.family_B_b2=nn.Parameter(torch.empty(self.NUM_BASES,action_dim))

        nn.init.orthogonal_(self.family_w1_base);self.family_w1_base.data.mul_(0.25)
        nn.init.orthogonal_(self.family_w2_base);self.family_w2_base.data.mul_(0.10)
        nn.init.normal_(self.family_B_w1,0,0.01)
        nn.init.normal_(self.family_B_w2,0,0.01)
        nn.init.normal_(self.family_B_b1,0,0.005)
        nn.init.normal_(self.family_B_b2,0,0.005)

    def initialize_state_path_from_v2b(self,state:dict)->None:
        with torch.no_grad():
            self.state_trunk[0].weight.copy_(state["actor_body.0.weight"][:,:self.obs_dim])
            self.state_trunk[0].bias.copy_(state["actor_body.0.bias"])
            self.state_trunk[2].weight.copy_(state["actor_body.2.weight"])
            self.state_trunk[2].bias.copy_(state["actor_body.2.bias"])
            self.state_trunk[4].weight.copy_(state["actor_body.4.weight"])
            self.state_trunk[4].bias.copy_(state["actor_body.4.bias"])
            self.state_action_head.weight.copy_(state["actor_mean.weight"][:,:128])
            self.state_action_head.bias.copy_(state["actor_mean.bias"])

    def coefficients(self,w:Tensor)->Tensor:
        return self.family_hyper(w)

    def generated(self,w:Tensor):
        c=self.coefficients(w)
        dw1=torch.einsum("bk,khf->bhf",c,self.family_B_w1)
        db1=torch.einsum("bk,kh->bh",c,self.family_B_b1)
        dw2=torch.einsum("bk,kah->bah",c,self.family_B_w2)
        db2=torch.einsum("bk,ka->ba",c,self.family_B_b2)
        return dw1,db1,dw2,db2

    @staticmethod
    def _blinear(x:Tensor,W:Tensor,b:Tensor)->Tensor:
        return torch.einsum("bi,boi->bo",x,W)+b

    def pre_tanh_mean(self,obs:Tensor,w:Tensor)->Tensor:
        h=self.state_trunk(obs)
        base=self.state_action_head(h)
        c=self.coefficients(w)
        # Algebraically identical to using W_base + sum_k c_k B_k, without
        # materializing one full weight matrix per batch element.
        z0=F.linear(h,self.family_w1_base,self.family_b1_base)
        z_basis=torch.einsum("bi,koi->bko",h,self.family_B_w1)
        z_delta=torch.einsum("bk,bko->bo",c,z_basis)+torch.einsum("bk,ko->bo",c,self.family_B_b1)
        z=F.elu(z0+z_delta)
        y0=F.linear(z,self.family_w2_base,self.family_b2_base)
        y_basis=torch.einsum("bi,koi->bko",z,self.family_B_w2)
        y_delta=torch.einsum("bk,bko->bo",c,y_basis)+torch.einsum("bk,ko->bo",c,self.family_B_b2)
        return base+y0+y_delta

    def act(self,obs:Tensor,w:Tensor)->Tensor:
        return torch.tanh(self.pre_tanh_mean(obs,w))*self.ACTION_CLIP


class AuthorityIsolatedActorCritic(T4SharedActorCritic):
    NUM_BASES=8
    COEFF_HIDDEN=32
    FAMILY_HIDDEN=128

    def __init__(self,obs_dim:int,action_dim:int,hidden_dims:list[int]|None=None):
        hidden_dims=hidden_dims or [128,128,128]
        super().__init__(obs_dim,action_dim,hidden_dims)
        # Replace inherited actor with a state-only trunk. The critic stays untouched.
        self.actor_body=nn.Sequential(
            nn.Linear(obs_dim,128),nn.ELU(),
            nn.Linear(128,128),nn.ELU(),
            nn.Linear(128,128),nn.ELU(),
        )
        self.actor_mean=nn.Linear(128,action_dim)
        self.family_hyper=nn.Sequential(
            nn.Linear(4,self.COEFF_HIDDEN),nn.ELU(),
            nn.Linear(self.COEFF_HIDDEN,self.NUM_BASES),
        )
        self.family_w1_base=nn.Parameter(torch.empty(128,128))
        self.family_b1_base=nn.Parameter(torch.zeros(128))
        self.family_w2_base=nn.Parameter(torch.empty(action_dim,128))
        self.family_b2_base=nn.Parameter(torch.zeros(action_dim))
        self.family_B_w1=nn.Parameter(torch.empty(self.NUM_BASES,128,128))
        self.family_B_b1=nn.Parameter(torch.empty(self.NUM_BASES,128))
        self.family_B_w2=nn.Parameter(torch.empty(self.NUM_BASES,action_dim,128))
        self.family_B_b2=nn.Parameter(torch.empty(self.NUM_BASES,action_dim))
        nn.init.orthogonal_(self.family_w1_base);self.family_w1_base.data.mul_(0.25)
        nn.init.orthogonal_(self.family_w2_base);self.family_w2_base.data.mul_(0.10)
        nn.init.normal_(self.family_B_w1,0,0.01)
        nn.init.normal_(self.family_B_w2,0,0.01)
        nn.init.normal_(self.family_B_b1,0,0.005)
        nn.init.normal_(self.family_B_b2,0,0.005)

    def family_coefficients(self,w:Tensor)->Tensor:
        return self.family_hyper(w)

    def generated_parameter_vector(self,w:Tensor)->Tensor:
        c=self.family_coefficients(w)
        dw1=torch.einsum("bk,koi->boi",c,self.family_B_w1)
        db1=torch.einsum("bk,ko->bo",c,self.family_B_b1)
        dw2=torch.einsum("bk,koi->boi",c,self.family_B_w2)
        db2=torch.einsum("bk,ko->bo",c,self.family_B_b2)
        return torch.cat((dw1.flatten(1),db1,dw2.flatten(1),db2),dim=1)
    def _actor_mean_with_preference(self,obs:Tensor,w:Tensor)->Tensor:
        h=self.actor_body(obs)
        base=self.actor_mean(h)
        c=self.family_coefficients(w)
        z0=F.linear(h,self.family_w1_base,self.family_b1_base)
        zb=torch.einsum("bi,koi->bko",h,self.family_B_w1)
        zd=torch.einsum("bk,bko->bo",c,zb)+torch.einsum("bk,ko->bo",c,self.family_B_b1)
        z=F.elu(z0+zd)
        y0=F.linear(z,self.family_w2_base,self.family_b2_base)
        yb=torch.einsum("bi,koi->bko",z,self.family_B_w2)
        yd=torch.einsum("bk,bko->bo",c,yb)+torch.einsum("bk,ko->bo",c,self.family_B_b2)
        return base+y0+yd

    def _pre_tanh_dist_with_preference(self,obs:Tensor,w:Tensor):
        mean=self._actor_mean_with_preference(obs,w)
        std=self.log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean,std)

    def act_inference_with_preference(self,obs:Tensor,w:Tensor):
        return torch.tanh(self._actor_mean_with_preference(obs,w))*self.ACTION_CLIP

    def act_with_preference_latent(self,obs:Tensor,w:Tensor):
        dist=self._pre_tanh_dist_with_preference(obs,w)
        u=dist.sample();a=torch.tanh(u)*self.ACTION_CLIP
        lp=(dist.log_prob(u)-self._log_det_jacobian(u)).sum(-1)
        return a,lp,u
    def logp_from_pre_tanh_with_preference(self,obs:Tensor,w:Tensor,u:Tensor):
        dist=self._pre_tanh_dist_with_preference(obs,w)
        return (dist.log_prob(u)-self._log_det_jacobian(u)).sum(-1)

    def logp_with_preference(self,obs:Tensor,w:Tensor,action:Tensor):
        eps=1e-6
        scaled=(action/self.ACTION_CLIP).clamp(-1+eps,1-eps)
        return self.logp_from_pre_tanh_with_preference(obs,w,torch.atanh(scaled))


def initialize_from_transfer(model:AuthorityIsolatedActorCritic,student_state:dict,v2b_state:dict)->None:
    """Load transferred actor and frozen V2-B critic/log_std."""
    ss=student_state.get("model",student_state)
    vs=v2b_state.get("model",v2b_state)
    target=model.state_dict()
    # Actor mapping from transfer student.
    amap={
      "state_trunk.0.weight":"actor_body.0.weight","state_trunk.0.bias":"actor_body.0.bias",
      "state_trunk.2.weight":"actor_body.2.weight","state_trunk.2.bias":"actor_body.2.bias",
      "state_trunk.4.weight":"actor_body.4.weight","state_trunk.4.bias":"actor_body.4.bias",
      "state_action_head.weight":"actor_mean.weight","state_action_head.bias":"actor_mean.bias",
    }
    for sk,tk in amap.items(): target[tk].copy_(ss[sk])
    for k in list(target):
        if k.startswith("family_") and k in ss and target[k].shape==ss[k].shape:
            target[k].copy_(ss[k])
    # Critic path and stochastic scale remain exactly V2-B.
    for k in list(target):
        if (k.startswith("critic_") or k=="log_std") and k in vs and target[k].shape==vs[k].shape:
            target[k].copy_(vs[k])
    model.load_state_dict(target)


class AuthorityIsolatedWideCritic(AuthorityIsolatedActorCritic):
    def __init__(self,obs_dim:int,action_dim:int):
        super().__init__(obs_dim,action_dim)
        self.critic_body=nn.Sequential(
            nn.Linear(obs_dim+4,256),nn.ELU(),
            nn.Linear(256,256),nn.ELU(),
            nn.Linear(256,128),nn.ELU(),
        )
        self.critic_head=nn.Linear(128,4)

def initialize_actor_exact_from_ai_h1(model:AuthorityIsolatedWideCritic,state:dict)->None:
    src=state.get("model",state)
    tgt=model.state_dict()
    for k in list(tgt):
        if k.startswith("actor_") or k.startswith("family_") or k=="log_std":
            if k in src and tgt[k].shape==src[k].shape:
                tgt[k].copy_(src[k])
    model.load_state_dict(tgt)
