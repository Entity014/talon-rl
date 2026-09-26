"""V2-A minimal preference-embedding actor.

RV1 direct conditioning is retained exactly.  V2-A adds only a small learned
preference embedding concatenated to the final actor hidden feature.  The
additional actor-head columns are zero-initialized, so initialization is
function-preserving with respect to a frozen RV1 checkpoint.
"""
from __future__ import annotations
import torch
from torch import Tensor, nn
from talon_rl.models.foundations.four_objective import T4SharedActorCritic, NUM_OBJECTIVES

class V2AMinimalEmbeddingActorCritic(T4SharedActorCritic):
    EMBED_DIM = 16

    def __init__(self, obs_dim:int, action_dim:int, hidden_dims:list[int]|None=None):
        hidden_dims = hidden_dims or [128,128,128]
        super().__init__(obs_dim, action_dim, hidden_dims)
        last_hidden = hidden_dims[-1]
        # Small preference representation branch.
        self.preference_embedding = nn.Sequential(
            nn.Linear(NUM_OBJECTIVES, self.EMBED_DIM),
            nn.ELU(),
        )
        # Replace only actor readout to admit the embedding.
        old_head = self.actor_mean
        self.actor_mean = nn.Linear(last_hidden + self.EMBED_DIM, action_dim)
        with torch.no_grad():
            self.actor_mean.weight.zero_()
            self.actor_mean.bias.copy_(old_head.bias)
            self.actor_mean.weight[:, :last_hidden].copy_(old_head.weight)

    def _actor_hidden(self, obs:Tensor, w:Tensor)->Tensor:
        return self.actor_body(self._with_w(obs,w))

    def _actor_features_v2a(self, obs:Tensor, w:Tensor)->Tensor:
        h = self._actor_hidden(obs,w)
        e = self.preference_embedding(w)
        return torch.cat((h,e),dim=-1)

    def _pre_tanh_dist_with_preference(self, obs:Tensor, w:Tensor):
        mean = self.actor_mean(self._actor_features_v2a(obs,w))
        std = self.log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean,std)

    def act_inference_with_preference(self,obs,w):
        mean=self.actor_mean(self._actor_features_v2a(obs,w))
        return torch.tanh(mean)*self.ACTION_CLIP

    def act_with_preference_latent(self,obs,w):
        dist=self._pre_tanh_dist_with_preference(obs,w)
        u=dist.sample()
        action=torch.tanh(u)*self.ACTION_CLIP
        logp=(dist.log_prob(u)-self._log_det_jacobian(u)).sum(-1)
        return action,logp,u

    def logp_from_pre_tanh_with_preference(self,obs,w,u):
        dist=self._pre_tanh_dist_with_preference(obs,w)
        return (dist.log_prob(u)-self._log_det_jacobian(u)).sum(-1)

    def logp_with_preference(self,obs,w,action):
        eps=1e-6
        scaled=(action/self.ACTION_CLIP).clamp(-1+eps,1-eps)
        u=torch.atanh(scaled)
        return self.logp_from_pre_tanh_with_preference(obs,w,u)

    # Critic path is intentionally inherited unchanged from RV1.

def initialize_from_rv1(model:V2AMinimalEmbeddingActorCritic, rv1_state:dict)->None:
    """Copy frozen RV1 parameters; zero the new embedding contribution exactly."""
    target=model.state_dict()
    source=rv1_state.get("model",rv1_state.get("model_state_dict",rv1_state))

    # Copy every unchanged tensor first.
    for k in list(target):
        if k in source and target[k].shape==source[k].shape:
            target[k].copy_(source[k])

    # Actor head expanded from [A,H] to [A,H+E].
    target["actor_mean.weight"].zero_()
    target["actor_mean.weight"][:,:source["actor_mean.weight"].shape[1]].copy_(source["actor_mean.weight"])
    target["actor_mean.bias"].copy_(source["actor_mean.bias"])

    # New embedding may have arbitrary internal features but exact zero authority.
    # Zero the readout columns that connect embedding -> action.
    target["actor_mean.weight"][:,source["actor_mean.weight"].shape[1]:].zero_()

    model.load_state_dict(target)
