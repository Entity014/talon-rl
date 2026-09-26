"""V2-PF: one-model explicit continuous policy family.

Preference w generates full-rank parameter deltas for a substantial residual
policy block.  The residual is written as

    f(x; theta0 + delta_theta(w)) - f(x; theta0)

so zero generated deltas give exact V2-B function preservation while retaining
non-zero first-order authority of the generated-parameter path.
"""
from __future__ import annotations
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from talon_rl.models.conditioning.single_site_film import V2BSingleSiteFiLMActorCritic
from talon_rl.models.foundations.preference_embedding import NUM_OBJECTIVES

class V2PFPolicyFamilyActorCritic(V2BSingleSiteFiLMActorCritic):
    FAMILY_HIDDEN = 128
    NUM_BASES = 8
    COEFF_HIDDEN = 32

    def __init__(self, obs_dim:int, action_dim:int, hidden_dims:list[int]|None=None):
        hidden_dims = hidden_dims or [128,128,128]
        super().__init__(obs_dim, action_dim, hidden_dims)
        feat_dim = self.actor_mean.in_features  # V2-B 128 + embed 16 = 144
        self.FAMILY_FEATURE_DIM = feat_dim
        self.FAMILY_ACTION_DIM = action_dim

        # Base family block.  This block does not change the V2-B function by
        # itself because we subtract its own base output exactly.
        self.family_w1_base = nn.Parameter(torch.empty(self.FAMILY_HIDDEN, feat_dim))
        self.family_b1_base = nn.Parameter(torch.zeros(self.FAMILY_HIDDEN))
        self.family_w2_base = nn.Parameter(torch.empty(action_dim, self.FAMILY_HIDDEN))
        self.family_b2_base = nn.Parameter(torch.zeros(action_dim))
        nn.init.orthogonal_(self.family_w1_base)
        nn.init.orthogonal_(self.family_w2_base)
        with torch.no_grad():
            self.family_w1_base.mul_(0.25)
            self.family_w2_base.mul_(0.10)

        # Preference -> latent coordinates on policy-parameter manifold.
        self.family_hyper = nn.Sequential(
            nn.Linear(NUM_OBJECTIVES, self.COEFF_HIDDEN),
            nn.ELU(),
            nn.Linear(self.COEFF_HIDDEN, self.NUM_BASES),
        )

        # Full-rank, semantically unlabeled learned bases covering both family
        # layers and biases.  This is intentionally much more substantial than
        # V2-H's low-rank action-head delta.
        self.family_B_w1 = nn.Parameter(torch.empty(self.NUM_BASES, self.FAMILY_HIDDEN, feat_dim))
        self.family_B_b1 = nn.Parameter(torch.empty(self.NUM_BASES, self.FAMILY_HIDDEN))
        self.family_B_w2 = nn.Parameter(torch.empty(self.NUM_BASES, action_dim, self.FAMILY_HIDDEN))
        self.family_B_b2 = nn.Parameter(torch.empty(self.NUM_BASES, action_dim))
        for B in (self.family_B_w1,self.family_B_w2):
            nn.init.normal_(B,mean=0.0,std=0.01)
        for B in (self.family_B_b1,self.family_B_b2):
            nn.init.normal_(B,mean=0.0,std=0.005)

        # Exact identity initialization through zero manifold coordinates.
        with torch.no_grad():
            self.family_hyper[-1].weight.zero_()
            self.family_hyper[-1].bias.zero_()

    def family_coefficients(self,w:Tensor)->Tensor:
        return self.family_hyper(w)

    def generated_family_deltas(self,w:Tensor):
        c=self.family_coefficients(w)
        dw1=torch.einsum('bk,khf->bhf',c,self.family_B_w1)
        db1=torch.einsum('bk,kh->bh',c,self.family_B_b1)
        dw2=torch.einsum('bk,kah->bah',c,self.family_B_w2)
        db2=torch.einsum('bk,ka->ba',c,self.family_B_b2)
        return dw1,db1,dw2,db2

    @staticmethod
    def _batched_linear(x:Tensor,W:Tensor,b:Tensor)->Tensor:
        return torch.einsum('bo,bho->bh',x,W)+b

    def _family_base(self,x:Tensor)->Tensor:
        h=F.elu(F.linear(x,self.family_w1_base,self.family_b1_base))
        return F.linear(h,self.family_w2_base,self.family_b2_base)

    def _family_conditional(self,x:Tensor,w:Tensor)->Tensor:
        dw1,db1,dw2,db2=self.generated_family_deltas(w)
        W1=self.family_w1_base.unsqueeze(0)+dw1
        b1=self.family_b1_base.unsqueeze(0)+db1
        W2=self.family_w2_base.unsqueeze(0)+dw2
        b2=self.family_b2_base.unsqueeze(0)+db2
        h=F.elu(self._batched_linear(x,W1,b1))
        return self._batched_linear(h,W2,b2)

    def generated_parameter_vector(self,w:Tensor)->Tensor:
        ds=self.generated_family_deltas(w)
        return torch.cat([x.reshape(len(w),-1) for x in ds],dim=1)

    def _conditional_actor_mean(self,obs:Tensor,w:Tensor)->Tensor:
        x=self._actor_features_v2a(obs,w)
        base=self.actor_mean(x)
        residual=self._family_conditional(x,w)-self._family_base(x)
        return base+residual

    def _pre_tanh_dist_with_preference(self,obs:Tensor,w:Tensor):
        mean=self._conditional_actor_mean(obs,w)
        std=self.log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean,std)

    def act_inference_with_preference(self,obs:Tensor,w:Tensor):
        return torch.tanh(self._conditional_actor_mean(obs,w))*self.ACTION_CLIP

    def act_with_preference_latent(self,obs:Tensor,w:Tensor):
        dist=self._pre_tanh_dist_with_preference(obs,w);u=dist.sample();a=torch.tanh(u)*self.ACTION_CLIP
        lp=(dist.log_prob(u)-self._log_det_jacobian(u)).sum(-1)
        return a,lp,u

    def logp_from_pre_tanh_with_preference(self,obs:Tensor,w:Tensor,u:Tensor):
        dist=self._pre_tanh_dist_with_preference(obs,w)
        return (dist.log_prob(u)-self._log_det_jacobian(u)).sum(-1)


def initialize_from_v2b(model:V2PFPolicyFamilyActorCritic,v2b_state:dict)->None:
    target=model.state_dict();source=v2b_state.get('model',v2b_state.get('model_state_dict',v2b_state))
    for k in list(target):
        if k in source and target[k].shape==source[k].shape:target[k].copy_(source[k])
    target['family_hyper.2.weight'].zero_();target['family_hyper.2.bias'].zero_()
    model.load_state_dict(target)
