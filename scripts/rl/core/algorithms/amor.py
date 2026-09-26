"""AMOR algorithm family: faithful early scalarization and scale-aligned variant."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from ..modules.actor_critic import ActorCritic

NUM_OBJECTIVES = 5

class AmorActorCritic(ActorCritic):
    """Actor receives physical observation concatenated with direct simplex w."""

    def __init__(self, obs_dim: int, action_dim: int,
                 hidden_dims: list[int] | None = None):
        hidden_dims = hidden_dims or [128, 128, 128]
        super().__init__(obs_dim + NUM_OBJECTIVES, obs_dim + NUM_OBJECTIVES,
                         action_dim, NUM_OBJECTIVES, hidden_dims)
        self.physical_obs_dim = obs_dim

    @staticmethod
    def with_preference(obs: Tensor, w: Tensor) -> Tensor:
        if obs.ndim != 2 or w.ndim != 2 or obs.shape[0] != w.shape[0]:
            raise ValueError("obs and w must be [batch,dim] with matching batch")
        if w.shape[1] != NUM_OBJECTIVES:
            raise ValueError("w must have five objective weights")
        return torch.cat((obs, w), dim=-1)

    def act_with_preference(self, obs: Tensor, w: Tensor):
        return self.act(self.with_preference(obs, w))

    def act_inference_with_preference(self, obs: Tensor, w: Tensor):
        return self.act_inference(self.with_preference(obs, w))

    def logp_with_preference(self, obs: Tensor, w: Tensor, action: Tensor):
        return self.logp(self.with_preference(obs, w), action)

    def value_with_preference(self, obs: Tensor, w: Tensor):
        return self.value(self.with_preference(obs, w))


def sample_episode_preferences(batch: int, *, alpha: float | Tensor = 1.0,
                              device=None, dtype=torch.float32) -> Tensor:
    """Sample simplex preferences; caller broadcasts one row per episode."""
    concentration = torch.as_tensor(alpha, device=device, dtype=dtype)
    if concentration.ndim == 0:
        concentration = concentration.expand(NUM_OBJECTIVES)
    if concentration.shape != (NUM_OBJECTIVES,) or not torch.all(concentration > 0):
        raise ValueError("alpha must be positive scalar or shape [5]")
    w = torch.distributions.Dirichlet(concentration).sample((batch,))
    return w / w.sum(-1, keepdim=True)


def scalarize_advantages(advantages: Tensor, w: Tensor) -> Tensor:
    if advantages.ndim != 2 or advantages.shape[1] != NUM_OBJECTIVES:
        raise ValueError("advantages must be [batch,5]")
    if w.shape != advantages.shape:
        raise ValueError("w must match advantages shape")
    return (advantages * w).sum(-1)


def normalize_scalar_advantages(advantages: Tensor, eps: float = 1e-8) -> Tensor:
    if advantages.ndim != 1 or not torch.isfinite(advantages).all():
        raise ValueError("advantages must be finite [batch]")
    if eps <= 0:
        raise ValueError("eps must be positive")
    return (advantages - advantages.mean()) / torch.sqrt(advantages.var(unbiased=False) + eps)


def standard_clipped_actor_loss(ratio: Tensor, scalar_advantages: Tensor,
                                clip_eps: float = 0.2) -> Tensor:
    """Ordinary PPO clipping applied once after early scalarization."""
    if ratio.ndim != 1 or scalar_advantages.shape != ratio.shape:
        raise ValueError("ratio and scalar_advantages must both be [batch]")
    if clip_eps <= 0:
        raise ValueError("clip_eps must be positive")
    clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps)
    return -torch.minimum(ratio * scalar_advantages,
                          clipped * scalar_advantages).mean()


def vector_gae(reward: Tensor, value: Tensor, next_value: Tensor,
               done: Tensor, gamma: float = 0.99, lam: float = 0.95):
    """Per-objective GAE for [time,batch,5] rollouts."""
    if reward.ndim != 3 or value.shape != reward.shape:
        raise ValueError("reward and value must be [T,B,5]")
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


def vector_value_loss(value: Tensor, returns: Tensor) -> Tensor:
    if value.shape != returns.shape or value.shape[-1] != NUM_OBJECTIVES:
        raise ValueError("value/returns must have shape [...,5]")
    return (value - returns).pow(2).mean()


# Scale-aligned AMOR variant.
class RunningObservationNorm(nn.Module):
    """Checkpointable running mean/std; ``update`` is disabled when frozen."""
    def __init__(self, dim: int, clip: float = 10.0, eps: float = 1e-8):
        super().__init__(); self.clip = float(clip); self.eps = float(eps)
        self.register_buffer("mean", torch.zeros(dim)); self.register_buffer("var", torch.ones(dim)); self.register_buffer("count", torch.zeros(()))
        self.frozen = False
    @torch.no_grad()
    def update(self, x: Tensor) -> None:
        if self.frozen: return
        x = x.detach().reshape(-1, self.mean.numel()).float(); n = torch.tensor(float(x.shape[0]), device=x.device)
        if x.shape[0] == 0: return
        bm, bv = x.mean(0), x.var(0, unbiased=False); total = self.count + n
        delta = bm - self.mean; new_mean = self.mean + delta * n / total
        m2 = self.var * self.count + bv * n + delta.square() * self.count * n / total
        self.mean.copy_(new_mean); self.var.copy_(m2 / total); self.count.copy_(total)
    def normalize(self, x: Tensor) -> Tensor:
        return ((x - self.mean) / torch.sqrt(self.var + self.eps)).clamp(-self.clip, self.clip)
    def freeze(self): self.frozen = True
    def unfreeze(self): self.frozen = False
    def state_dict_extra(self): return {"frozen": self.frozen, "clip": self.clip, "eps": self.eps}
    def load_state_dict_extra(self, state): self.frozen = bool(state["frozen"]); self.clip = float(state["clip"]); self.eps = float(state["eps"])

class ScaleAlignedAmorActorCritic(AmorActorCritic):
    """AMOR actor/critic with independent actor and critic observation norms."""
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims=None):
        super().__init__(obs_dim, action_dim, hidden_dims or [1024] * 4)
        self.actor_obs_norm = RunningObservationNorm(obs_dim)
        self.critic_obs_norm = RunningObservationNorm(obs_dim)
    def _norm(self, obs: Tensor, norm: RunningObservationNorm) -> Tensor: return norm.normalize(obs)
    def act_with_preference(self, obs: Tensor, w: Tensor): return self.act(self.with_preference(self._norm(obs, self.actor_obs_norm), w))
    def act_inference_with_preference(self, obs: Tensor, w: Tensor): return self.act_inference(self.with_preference(self._norm(obs, self.actor_obs_norm), w))
    def logp_with_preference(self, obs: Tensor, w: Tensor, action: Tensor): return self.logp(self.with_preference(self._norm(obs, self.actor_obs_norm), w), action)
    def value_with_preference(self, obs: Tensor, w: Tensor): return self.value(self.with_preference(self._norm(obs, self.critic_obs_norm), w))
    def norm_state(self): return {"actor": self.actor_obs_norm.state_dict(), "critic": self.critic_obs_norm.state_dict(), "actor_extra": self.actor_obs_norm.state_dict_extra(), "critic_extra": self.critic_obs_norm.state_dict_extra()}
    def load_norm_state(self, state): self.actor_obs_norm.load_state_dict(state["actor"]); self.critic_obs_norm.load_state_dict(state["critic"]); self.actor_obs_norm.load_state_dict_extra(state["actor_extra"]); self.critic_obs_norm.load_state_dict_extra(state["critic_extra"])
