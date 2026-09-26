"""V2-B: V2-A embedding plus one identity-initialized FiLM site."""
from __future__ import annotations
import torch
from torch import Tensor, nn
from talon_rl.models.foundations.preference_embedding import V2AMinimalEmbeddingActorCritic, NUM_OBJECTIVES

class V2BSingleSiteFiLMActorCritic(V2AMinimalEmbeddingActorCritic):
    def __init__(self, obs_dim:int, action_dim:int, hidden_dims:list[int]|None=None):
        hidden_dims = hidden_dims or [128,128,128]
        super().__init__(obs_dim, action_dim, hidden_dims)
        self.FILM_DIM = hidden_dims[-1]
        self.preference_film = nn.Linear(NUM_OBJECTIVES, 2*self.FILM_DIM)
        with torch.no_grad():
            self.preference_film.weight.zero_()
            self.preference_film.bias.zero_()

    def _film_hidden(self, obs:Tensor, w:Tensor)->Tensor:
        h=self._actor_hidden(obs,w)
        gb=self.preference_film(w)
        gamma,beta=torch.chunk(gb,2,dim=-1)
        return (1.0+gamma)*h+beta

    def _actor_features_v2a(self, obs:Tensor, w:Tensor)->Tensor:
        # Preserve V2-A embedding path; add only one FiLM site on final hidden feature.
        h=self._film_hidden(obs,w)
        e=self.preference_embedding(w)
        return torch.cat((h,e),dim=-1)

def initialize_from_v2a(model:V2BSingleSiteFiLMActorCritic, v2a_state:dict)->None:
    """Copy V2-A exactly and keep the new FiLM treatment at identity."""
    target=model.state_dict()
    source=v2a_state.get("model",v2a_state.get("model_state_dict",v2a_state))
    for k in list(target):
        if k in source and target[k].shape==source[k].shape:
            target[k].copy_(source[k])
    target["preference_film.weight"].zero_()
    target["preference_film.bias"].zero_()
    model.load_state_dict(target)
