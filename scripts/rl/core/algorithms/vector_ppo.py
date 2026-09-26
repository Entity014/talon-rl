"""Vector/preference-conditioned PPO primitives and function-preserving variants."""
from __future__ import annotations

import torch
from torch import Tensor

from ..modules.actor_critic import ActorCritic

NUM_OBJECTIVES = 5
REFERENCE_W = (0.2,) * NUM_OBJECTIVES
W_REF = REFERENCE_W

class M02aActorCritic(ActorCritic):
    """Actor-critic whose physical observation is augmented by explicit w."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list[int] | None = None):
        hidden_dims = hidden_dims or [128, 128, 128]
        super().__init__(obs_dim + NUM_OBJECTIVES, obs_dim + NUM_OBJECTIVES,
                         action_dim, NUM_OBJECTIVES, hidden_dims)
        self.physical_obs_dim = obs_dim

    @staticmethod
    def _with_w(obs: Tensor, w: Tensor) -> Tensor:
        if obs.ndim != 2 or w.ndim != 2 or obs.shape[0] != w.shape[0] or w.shape[1] != NUM_OBJECTIVES:
            raise ValueError("obs and w must be [batch,dim] and [batch,5]")
        return torch.cat((obs, w), dim=-1)

    def act_with_preference(self, obs: Tensor, w: Tensor):
        return self.act(self._with_w(obs, w))

    def act_inference_with_preference(self, obs: Tensor, w: Tensor):
        return self.act_inference(self._with_w(obs, w))

    def logp_with_preference(self, obs: Tensor, w: Tensor, action: Tensor):
        return self.logp(self._with_w(obs, w), action)

    def value_with_preference(self, obs: Tensor, w: Tensor):
        return self.value(self._with_w(obs, w))


def vector_gae(reward: Tensor, value: Tensor, next_value: Tensor,
               done: Tensor, gamma: float = 0.99, lam: float = 0.95):
    """GAE over `[time,batch,objectives]` rewards/values."""
    if reward.ndim != 3 or value.shape != reward.shape:
        raise ValueError("reward and value must have shape [T,B,K]")
    if next_value.shape != reward.shape[1:] or done.shape != reward.shape[:2]:
        raise ValueError("next_value/done shape mismatch")
    adv = torch.zeros_like(reward)
    last = torch.zeros_like(next_value)
    for t in range(reward.shape[0] - 1, -1, -1):
        bootstrap = next_value if t == reward.shape[0] - 1 else value[t + 1]
        nonterminal = (~done[t]).to(reward.dtype).unsqueeze(-1)
        delta = reward[t] + gamma * bootstrap * nonterminal - value[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    return adv, adv + value


def late_weighted_actor_loss(ratio: Tensor, advantages: Tensor, w: Tensor,
                             clip_eps: float = 0.2, scale_k: int = NUM_OBJECTIVES) -> Tensor:
    """`K * sum_k(w_k L_k)` with clipping independently per objective."""
    if advantages.ndim != 2 or advantages.shape[1] != NUM_OBJECTIVES:
        raise ValueError("advantages must be [batch,5]")
    if w.shape != advantages.shape or ratio.shape != advantages.shape[:1]:
        raise ValueError("ratio/w shape mismatch")
    clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps)
    per_objective = torch.minimum(ratio[:, None] * advantages,
                                  clipped[:, None] * advantages)
    return -(scale_k * (w * per_objective).sum(-1)).mean()


def vector_value_loss(value: Tensor, returns: Tensor) -> Tensor:
    """Declared M0.2a critic scale: mean over objective-wise MSE."""
    if value.shape != returns.shape or value.shape[-1] != NUM_OBJECTIVES:
        raise ValueError("value/returns must have shape [...,5]")
    return (value - returns).pow(2).mean()


def adaptive_kl_lr(optimizer: torch.optim.Optimizer, analytic_kl: Tensor,
                   desired_kl: float = 0.01, low: float = 0.005,
                   high: float = 0.02, factor: float = 1.5,
                   min_lr: float = 1e-5, max_lr: float = 1e-2) -> str:
    """Reference-style KL LR event; returns ``up``, ``down`` or ``hold``."""
    kl = float(analytic_kl.detach().mean())
    event = "hold"
    if kl > high:
        event, multiplier = "down", 1.0 / factor
    elif kl < low:
        event, multiplier = "up", factor
    else:
        multiplier = 1.0
    for group in optimizer.param_groups:
        group["lr"] = min(max(float(group["lr"]) * multiplier, min_lr), max_lr)
    return event


# Function-preserving expansion helpers.
def make_fp0_actor(obs_dim: int, action_dim: int, hidden_dims: list[int] | None = None):
    """Return scalar baseline and expanded actor/critic pair."""
    hidden_dims = hidden_dims or [128, 128, 128]
    base = ActorCritic(obs_dim, obs_dim, action_dim, 1, hidden_dims)
    expanded = ActorCritic(obs_dim + NUM_OBJECTIVES, obs_dim, action_dim, 1, hidden_dims)
    state = expanded.state_dict(); source = base.state_dict()
    for key, value in source.items():
        if key == "actor_body.0.weight":
            state[key].zero_(); state[key][:, :obs_dim].copy_(value)
        elif key in state and state[key].shape == value.shape:
            state[key].copy_(value)
    expanded.load_state_dict(state)
    return base, expanded

def fp0_probe(base: ActorCritic, expanded: ActorCritic, obs: Tensor, w: Tensor):
    """Compare deterministic action and log-probability on identical inputs."""
    expanded_obs = torch.cat((obs, w), dim=-1)
    with torch.no_grad():
        base_action = base.act_inference(obs)
        expanded_action = expanded.act_inference(expanded_obs)
        # A fixed probe action avoids stochastic sampling while testing logp.
        probe_action = base_action.detach()
        base_logp = base.logp(obs, probe_action)
        expanded_logp = expanded.logp(expanded_obs, probe_action)
    return base_action, expanded_action, base_logp, expanded_logp


# Reference-centered preference-conditioning variant.
class CenteredPreferenceActor(ActorCritic):
    def __init__(self,obs_dim:int,action_dim:int,hidden_dims:list[int]|None=None):
        super().__init__(obs_dim+NUM_OBJECTIVES,obs_dim,action_dim,1,hidden_dims or [128,128,128]);self.physical_obs_dim=obs_dim
    def centered(self,obs:Tensor,w:Tensor)->Tensor:
        if w.shape != (obs.shape[0],NUM_OBJECTIVES): raise ValueError('w shape mismatch')
        return torch.cat((obs,w-torch.tensor(W_REF,device=obs.device,dtype=obs.dtype)),dim=-1)
    def act_centered(self,obs,w): return self.act(self.centered(obs,w))
    def inference_centered(self,obs,w): return self.act_inference(self.centered(obs,w))
    def logp_centered(self,obs,w,action): return self.logp(self.centered(obs,w),action)
def initialize_centered_from_scalar(base:ActorCritic,obs_dim:int,action_dim:int,hidden_dims:list[int]|None=None):
    m=CenteredPreferenceActor(obs_dim,action_dim,hidden_dims);s=m.state_dict();b=base.state_dict()
    for k,v in b.items():
        if k in s and s[k].shape==v.shape:s[k].copy_(v)
    s['actor_body.0.weight'][:,:obs_dim].copy_(b['actor_body.0.weight']);s['actor_body.0.weight'][:,obs_dim:].zero_();m.load_state_dict(s);return m
