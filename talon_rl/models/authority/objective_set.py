"""Objective-set generalized authority-isolated actor/critic (Phase-2 G0).

G0 keeps the known T/A/O/S vocabulary but removes fixed ordering/cardinality
from the public actor/critic interface. One-hot compatibility tokens make the
weighted set context exactly equal to the legacy 4D preference vector.
"""
from __future__ import annotations
import torch
from torch import Tensor, nn
from torch.nn import functional as F

OBJECTIVE_ORDER=("T","A","O","S")
TOKEN_DIM=4

def canonical_tokens(device=None,dtype=torch.float32)->Tensor:
    return torch.eye(TOKEN_DIM,device=device,dtype=dtype)

def validate_set(tokens:Tensor,weights:Tensor)->None:
    if tokens.ndim!=3 or tokens.shape[-1]!=TOKEN_DIM:
        raise ValueError("tokens must have shape [B,M,4]")
    if weights.ndim!=2 or weights.shape!=tokens.shape[:2]:
        raise ValueError("weights must have shape [B,M]")
    if not torch.isfinite(tokens).all() or not torch.isfinite(weights).all():
        raise ValueError("objective set must be finite")
    if (weights<0).any():
        raise ValueError("weights must be non-negative")
    if not torch.allclose(weights.sum(-1),torch.ones(weights.shape[0],device=weights.device,dtype=weights.dtype),atol=1e-6):
        raise ValueError("active-set weights must sum to one")

def aggregate_objective_set(tokens:Tensor,weights:Tensor)->Tensor:
    validate_set(tokens,weights)
    return torch.einsum("bm,bmd->bd",weights,tokens)

def set_from_dense_preference(w:Tensor):
    if w.ndim!=2 or w.shape[-1]!=TOKEN_DIM:
        raise ValueError("dense preference must be [B,4]")
    tok=canonical_tokens(w.device,w.dtype).unsqueeze(0).expand(w.shape[0],-1,-1)
    return tok,w
class ObjectiveSetAuthorityIsolatedWideCritic(nn.Module):
    ACTION_CLIP=1.0
    NUM_BASES=8
    COEFF_HIDDEN=32
    FAMILY_HIDDEN=128

    def __init__(self,obs_dim:int,action_dim:int):
        super().__init__()
        self.obs_dim=obs_dim;self.action_dim=action_dim
        self.actor_body=nn.Sequential(
            nn.Linear(obs_dim,128),nn.ELU(),
            nn.Linear(128,128),nn.ELU(),
            nn.Linear(128,128),nn.ELU())
        self.actor_mean=nn.Linear(128,action_dim)
        self.family_hyper=nn.Sequential(
            nn.Linear(TOKEN_DIM,self.COEFF_HIDDEN),nn.ELU(),
            nn.Linear(self.COEFF_HIDDEN,self.NUM_BASES))
        self.family_w1_base=nn.Parameter(torch.empty(128,128))
        self.family_b1_base=nn.Parameter(torch.zeros(128))
        self.family_w2_base=nn.Parameter(torch.empty(action_dim,128))
        self.family_b2_base=nn.Parameter(torch.zeros(action_dim))
        self.family_B_w1=nn.Parameter(torch.empty(self.NUM_BASES,128,128))
        self.family_B_b1=nn.Parameter(torch.empty(self.NUM_BASES,128))
        self.family_B_w2=nn.Parameter(torch.empty(self.NUM_BASES,action_dim,128))
        self.family_B_b2=nn.Parameter(torch.empty(self.NUM_BASES,action_dim))
        self.log_std=nn.Parameter(torch.zeros(action_dim))

        self.critic_body=nn.Sequential(
            nn.Linear(obs_dim+TOKEN_DIM,256),nn.ELU(),
            nn.Linear(256,256),nn.ELU(),
            nn.Linear(256,128),nn.ELU())
        # Token-conditioned value basis. Public API returns scalar per queried
        # token rather than a fixed 4-vector.
        self.value_basis_weight=nn.Parameter(torch.empty(TOKEN_DIM,128))
        self.value_basis_bias=nn.Parameter(torch.empty(TOKEN_DIM))
        nn.init.orthogonal_(self.family_w1_base);self.family_w1_base.data.mul_(0.25)
        nn.init.orthogonal_(self.family_w2_base);self.family_w2_base.data.mul_(0.10)
        nn.init.normal_(self.family_B_w1,0,0.01)
        nn.init.normal_(self.family_B_w2,0,0.01)
        nn.init.normal_(self.family_B_b1,0,0.005)
        nn.init.normal_(self.family_B_b2,0,0.005)
        nn.init.normal_(self.value_basis_weight,0,0.01)
        nn.init.zeros_(self.value_basis_bias)
    @staticmethod
    def _log_det_jacobian(u:Tensor)->Tensor:
        return 2.0*(torch.log(torch.tensor(2.0,device=u.device,dtype=u.dtype))-u-F.softplus(-2.0*u))

    def objective_context(self,tokens:Tensor,weights:Tensor)->Tensor:
        return aggregate_objective_set(tokens,weights)

    def family_coefficients_from_set(self,tokens:Tensor,weights:Tensor)->Tensor:
        return self.family_hyper(self.objective_context(tokens,weights))

    def _actor_mean_from_context(self,obs:Tensor,z:Tensor)->Tensor:
        h=self.actor_body(obs);base=self.actor_mean(h);c=self.family_hyper(z)
        z0=F.linear(h,self.family_w1_base,self.family_b1_base)
        zb=torch.einsum("bi,koi->bko",h,self.family_B_w1)
        zd=torch.einsum("bk,bko->bo",c,zb)+torch.einsum("bk,ko->bo",c,self.family_B_b1)
        q=F.elu(z0+zd)
        y0=F.linear(q,self.family_w2_base,self.family_b2_base)
        yb=torch.einsum("bi,koi->bko",q,self.family_B_w2)
        yd=torch.einsum("bk,bko->bo",c,yb)+torch.einsum("bk,ko->bo",c,self.family_B_b2)
        return base+y0+yd

    def pre_tanh_mean_from_set(self,obs:Tensor,tokens:Tensor,weights:Tensor)->Tensor:
        return self._actor_mean_from_context(obs,self.objective_context(tokens,weights))

    def act_inference_from_set(self,obs:Tensor,tokens:Tensor,weights:Tensor)->Tensor:
        return torch.tanh(self.pre_tanh_mean_from_set(obs,tokens,weights))*self.ACTION_CLIP

    def _dist_from_set(self,obs:Tensor,tokens:Tensor,weights:Tensor):
        mean=self.pre_tanh_mean_from_set(obs,tokens,weights)
        return torch.distributions.Normal(mean,self.log_std.exp().expand_as(mean))
    def act_with_set_latent(self,obs:Tensor,tokens:Tensor,weights:Tensor):
        dist=self._dist_from_set(obs,tokens,weights)
        u=dist.sample();a=torch.tanh(u)*self.ACTION_CLIP
        lp=(dist.log_prob(u)-self._log_det_jacobian(u)).sum(-1)
        return a,lp,u

    def logp_from_pre_tanh_from_set(self,obs:Tensor,tokens:Tensor,weights:Tensor,u:Tensor)->Tensor:
        dist=self._dist_from_set(obs,tokens,weights)
        return (dist.log_prob(u)-self._log_det_jacobian(u)).sum(-1)

    def critic_features_from_set(self,obs:Tensor,tokens:Tensor,weights:Tensor)->Tensor:
        z=self.objective_context(tokens,weights)
        return self.critic_body(torch.cat((obs,z),dim=-1))

    def query_values_from_set(self,obs:Tensor,tokens:Tensor,weights:Tensor,query_tokens:Tensor|None=None)->Tensor:
        validate_set(tokens,weights)
        if query_tokens is None:query_tokens=tokens
        if query_tokens.ndim!=3 or query_tokens.shape[0]!=obs.shape[0] or query_tokens.shape[-1]!=TOKEN_DIM:
            raise ValueError("query_tokens must be [B,Q,4]")
        f=self.critic_features_from_set(obs,tokens,weights)
        # q_w[b,q,h] = sum_d token[b,q,d] * basis[d,h]
        qw=torch.einsum("bqd,dh->bqh",query_tokens,self.value_basis_weight)
        qb=torch.einsum("bqd,d->bq",query_tokens,self.value_basis_bias)
        return torch.einsum("bh,bqh->bq",f,qw)+qb

    # Dense-preference compatibility helpers used only by G0 parity/audits.
    def act_inference_with_preference(self,obs:Tensor,w:Tensor)->Tensor:
        t,x=set_from_dense_preference(w);return self.act_inference_from_set(obs,t,x)
    def value_with_preference(self,obs:Tensor,w:Tensor)->Tensor:
        t,x=set_from_dense_preference(w);return self.query_values_from_set(obs,t,x,t)
    def logp_from_pre_tanh_with_preference(self,obs:Tensor,w:Tensor,u:Tensor)->Tensor:
        t,x=set_from_dense_preference(w);return self.logp_from_pre_tanh_from_set(obs,t,x,u)
def initialize_exact_from_phase1(model:ObjectiveSetAuthorityIsolatedWideCritic,state:dict)->None:
    """Exact G0 migration from AuthorityIsolatedWideCritic state."""
    src=state.get("model",state)
    dst=model.state_dict()
    direct=[
        "actor_body.0.weight","actor_body.0.bias","actor_body.2.weight","actor_body.2.bias",
        "actor_body.4.weight","actor_body.4.bias","actor_mean.weight","actor_mean.bias",
        "family_hyper.0.weight","family_hyper.0.bias","family_hyper.2.weight","family_hyper.2.bias",
        "family_w1_base","family_b1_base","family_w2_base","family_b2_base",
        "family_B_w1","family_B_b1","family_B_w2","family_B_b2","log_std",
        "critic_body.0.weight","critic_body.0.bias","critic_body.2.weight","critic_body.2.bias",
        "critic_body.4.weight","critic_body.4.bias"]
    for k in direct:
        if k not in src:raise KeyError(f"missing Phase-1 parameter {k}")
        dst[k].copy_(src[k])
    dst["value_basis_weight"].copy_(src["critic_head.weight"])
    dst["value_basis_bias"].copy_(src["critic_head.bias"])
    model.load_state_dict(dst)
