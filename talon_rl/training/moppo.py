"""Minimal preference-conditioned PPO (MOPPO), chapter3.tex §3.2.3 / fig 3.3.

Policy: pi(a | s, w) — observation is [s, w] concatenated (w appended by this
module, not by the env — see envs/base_env.py docstring).
Critic: vector critic V(s, w) -> R^5, one head per reward-vector term
(asymmetric actor-critic per AMOR \\cite{alegre2025}).
Policy-gradient advantage: scalarized as w . advantage_vector, i.e. the
preference vector arbitrates between objectives at the advantage level, not
by pre-summing the reward into a scalar before GAE.

CPU-only, single dummy env, small batch — this is the prelim smoke test, not
a throughput-tuned trainer. Swap in a vectorized Isaac Lab env + proper
multi-env rollout collection before trusting any result from this as "the"
Phase 1 training run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from ..config import ObservationSpaceCfg, PreferenceCfg, RewardVectorCfg
from ..envs.base_env import BaseTalonEnv
from ..preference import floor_clip, rate_limit, sample_preference_vector
from ..reward import compute_reward_vector


@dataclass
class MOPPOConfig:
    hidden_dim: int = 64
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    epochs_per_update: int = 4
    episodes_per_update: int = 8
    device: str = "cpu"


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, reward_dim: int, hidden_dim: int):
        super().__init__()
        # obs_dim here already includes the appended preference vector w
        self.actor_body = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
        )
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        self.critic_body = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
        )
        self.critic_head = nn.Linear(hidden_dim, reward_dim)  # V(s, c, w) -> R^reward_dim

    def act(self, obs_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.actor_mean(self.actor_body(obs_w))
        dist = Normal(mean, self.log_std.exp())
        action = dist.sample()
        logp = dist.log_prob(action).sum(-1)
        return action, logp

    def logp(self, obs_w: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        mean = self.actor_mean(self.actor_body(obs_w))
        dist = Normal(mean, self.log_std.exp())
        return dist.log_prob(action).sum(-1)

    def value(self, obs_w: torch.Tensor) -> torch.Tensor:
        return self.critic_head(self.critic_body(obs_w))


def _gae_per_objective(rewards: np.ndarray, values: np.ndarray, gamma: float, lam: float) -> np.ndarray:
    """rewards: (T, K), values: (T+1, K) -> advantages (T, K). K = reward_dim."""
    T, K = rewards.shape
    adv = np.zeros((T, K), dtype=np.float32)
    gae = np.zeros(K, dtype=np.float32)
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
    return adv


class MOPPOTrainer:
    def __init__(
        self,
        env: BaseTalonEnv,
        obs_cfg: ObservationSpaceCfg,
        reward_cfg: RewardVectorCfg,
        pref_cfg: PreferenceCfg,
        moppo_cfg: MOPPOConfig | None = None,
        seed: int = 0,
    ):
        self.env = env
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.pref_cfg = pref_cfg
        self.cfg = moppo_cfg or MOPPOConfig()
        self.rng = np.random.default_rng(seed)

        obs_w_dim = env.obs_dim + reward_cfg.dim
        self.model = ActorCritic(obs_w_dim, env.action_dim, reward_cfg.dim, self.cfg.hidden_dim)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)

    def _rollout_episode(self) -> dict:
        w_prev = sample_preference_vector(self.rng, self.reward_cfg, self.pref_cfg)
        w = floor_clip(w_prev, self.reward_cfg.term_names, self.reward_cfg.impact_floor_eps)

        transition = self.env.reset()
        obs_list, act_list, logp_list, rew_list, val_list = [], [], [], [], []
        done = False
        while not done:
            obs_w = np.concatenate([transition["obs"], w]).astype(np.float32)
            obs_w_t = torch.from_numpy(obs_w).unsqueeze(0)
            with torch.no_grad():
                action_t, logp_t = self.model.act(obs_w_t)
                value_t = self.model.value(obs_w_t)
            action = action_t.squeeze(0).numpy()

            transition, done = self.env.step(action)
            reward_vec = compute_reward_vector(transition, self.reward_cfg)

            obs_list.append(obs_w)
            act_list.append(action)
            logp_list.append(float(logp_t.item()))
            rew_list.append(reward_vec)
            val_list.append(value_t.squeeze(0).numpy())

            # w drifts slowly within-episode too, per the rate-limiter in fig 3.3
            w_target = sample_preference_vector(self.rng, self.reward_cfg, self.pref_cfg)
            w = floor_clip(
                rate_limit(w, w_target, self.pref_cfg.max_delta_per_step),
                self.reward_cfg.term_names,
                self.reward_cfg.impact_floor_eps,
            )

        final_obs_w = np.concatenate([transition["obs"], w]).astype(np.float32)
        with torch.no_grad():
            final_value = self.model.value(torch.from_numpy(final_obs_w).unsqueeze(0)).squeeze(0).numpy()

        return {
            "obs": np.stack(obs_list),
            "actions": np.stack(act_list),
            "logp": np.array(logp_list, dtype=np.float32),
            "rewards": np.stack(rew_list),
            "values": np.stack(val_list),
            "final_value": final_value,
            "w_final": w,
        }

    def update(self) -> dict:
        """Collects `episodes_per_update` episodes, then runs PPO for `epochs_per_update` epochs."""
        episodes = [self._rollout_episode() for _ in range(self.cfg.episodes_per_update)]

        all_obs, all_actions, all_logp_old, all_adv, all_returns = [], [], [], [], []
        for ep in episodes:
            values = np.concatenate([ep["values"], ep["final_value"][None, :]], axis=0)
            adv = _gae_per_objective(ep["rewards"], values, self.cfg.gamma, self.cfg.gae_lambda)
            returns = adv + ep["values"]
            w = ep["obs"][0, -self.reward_cfg.dim :]  # w is appended at the end of obs_w
            scalar_adv = adv @ w  # arbitration: w . advantage_vector

            all_obs.append(ep["obs"])
            all_actions.append(ep["actions"])
            all_logp_old.append(ep["logp"])
            all_adv.append(scalar_adv)
            all_returns.append(returns)

        obs_t = torch.from_numpy(np.concatenate(all_obs))
        actions_t = torch.from_numpy(np.concatenate(all_actions))
        logp_old_t = torch.from_numpy(np.concatenate(all_logp_old))
        adv_t = torch.from_numpy(np.concatenate(all_adv).astype(np.float32))
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        returns_t = torch.from_numpy(np.concatenate(all_returns).astype(np.float32))

        last_policy_loss = last_value_loss = 0.0
        for _ in range(self.cfg.epochs_per_update):
            logp_new = self.model.logp(obs_t, actions_t)
            ratio = torch.exp(logp_new - logp_old_t)
            clipped = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps)
            policy_loss = -torch.min(ratio * adv_t, clipped * adv_t).mean()

            values_pred = self.model.value(obs_t)
            value_loss = nn.functional.mse_loss(values_pred, returns_t)

            loss = policy_loss + 0.5 * value_loss
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            last_policy_loss, last_value_loss = float(policy_loss.item()), float(value_loss.item())

        mean_reward_vec = np.concatenate([ep["rewards"] for ep in episodes]).mean(axis=0)
        return {
            "policy_loss": last_policy_loss,
            "value_loss": last_value_loss,
            "mean_reward_vec": mean_reward_vec,
            "mean_episode_len": float(np.mean([len(ep["rewards"]) for ep in episodes])),
        }
