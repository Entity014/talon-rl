"""Minimal preference-conditioned PPO (MOPPO), chapter3.tex §3.2.3 / fig 3.3.

Policy: pi(a | s, w) — observation is [s, w] concatenated (w appended by this
module, not by the env — see envs/base_env.py docstring). The actor and
critic can see *different* amounts of temporal history via
`ObservationStackCfg` (num_policy_stacks / num_critic_stacks, after Flamingo
— jaykorea/Isaac-RL-Two-wheel-Legged-Bot — see obs_stack.py).

Critic: vector critic V(s, w) -> R^5, one head per reward-vector term
(asymmetric actor-critic per AMOR \\cite{alegre2025}).
Policy-gradient advantage: scalarized as w . advantage_vector, i.e. the
preference vector arbitrates between objectives at the advantage level, not
by pre-summing the reward into a scalar before GAE.

Rollout collection is fixed-horizon + auto-reset (Isaac Lab/rsl_rl/sb3-
standard, 2026-09-14) — N env lanes step in lockstep for `num_steps` per
update() call, any lane that terminates auto-resets internally and keeps
contributing to the same buffer, and the rollout is PERSISTENT: env.reset()
happens once (in __init__), and each update() call collects the next
num_steps timesteps continuing wherever the previous call left off (not a
fresh episode-based rollout every call, unlike the pre-2026-09-14 version).
GAE uses the standard done-masked recursion so value bootstrapping never
crosses an episode boundary within a lane.

The rollout collection here (including the per-step preference-vector
resampling) is MOPPO-specific, not a generic on-policy concern — a second
algorithm might not resample w every step at all — so it stays part of this
class rather than being pulled into a shared runner. Only the pieces
genuinely reusable across algorithms (the network shape, GAE math) live in
scripts/rl/modules/ and scripts/rl/storage/.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from talon_rl.config import ObservationSpaceCfg, ObservationStackCfg, PreferenceCfg, RewardVectorCfg
from talon_rl.envs.base_env import BaseTalonEnv
from talon_rl.reward import compute_reward_vector

from ..modules.actor_critic import ActorCritic
from ..obs_stack import ObservationStack
from ..preference import floor_clip, rate_limit, sample_preference_vector
from ..storage.rollout_storage import gae_per_objective


@dataclass
class MOPPOConfig:
    hidden_dim: int = 64
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    epochs_per_update: int = 4
    num_steps: int = 24  # rollout length per update() call, across all N lanes
    device: str = "cpu"


class MOPPOTrainer:
    def __init__(
        self,
        env: BaseTalonEnv,
        obs_cfg: ObservationSpaceCfg,
        reward_cfg: RewardVectorCfg,
        pref_cfg: PreferenceCfg,
        moppo_cfg: MOPPOConfig | None = None,
        stack_cfg: ObservationStackCfg | None = None,
        seed: int = 0,
    ):
        self.env = env
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.pref_cfg = pref_cfg
        self.cfg = moppo_cfg or MOPPOConfig()
        self.stack_cfg = stack_cfg or ObservationStackCfg()
        self.rng = np.random.default_rng(seed)
        self.n = env.num_envs

        self.stack = ObservationStack(
            self.n, env.obs_dim, self.stack_cfg.num_policy_stacks, self.stack_cfg.num_critic_stacks
        )
        actor_obs_w_dim = self.stack.policy_obs_dim + reward_cfg.dim
        critic_obs_w_dim = self.stack.critic_obs_dim + reward_cfg.dim
        self.model = ActorCritic(actor_obs_w_dim, critic_obs_w_dim, env.action_dim, reward_cfg.dim, self.cfg.hidden_dim)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)

        # Persistent rollout state — set up once here, advanced by update(),
        # never reset mid-training (auto-reset happens per-lane inside
        # env.step() itself).
        transition = self.env.reset()
        self.stack.reset(transition["obs"])
        self.w = sample_preference_vector(self.rng, self.reward_cfg, self.pref_cfg, self.n)
        self.w = floor_clip(self.w, self.reward_cfg.term_names, self.reward_cfg.impact_floor_eps)
        self._prev_done = np.zeros(self.n, dtype=bool)
        self._t = 0
        # Persistent per-lane steps-since-last-reset counter, for
        # mean_episode_len — must be instance state (not reset per update()
        # call) because the true episode horizon (e.g. 200) is much longer
        # than a single rollout window (num_steps, default 24), so a window-
        # local measurement can never see a full episode.
        self._lane_step_count = np.zeros(self.n, dtype=np.int64)

    def _collect_rollout(self) -> dict:
        actor_obs_list, critic_obs_list, act_list, logp_list, rew_list, val_list, done_list = (
            [], [], [], [], [], [], []
        )
        finished_lengths = []  # lane episode lengths completed within this window

        for _ in range(self.cfg.num_steps):
            # Lanes that just auto-reset (done from the PREVIOUS step) get a
            # fresh w sample (new episode -> new preference sample, fig 3.3);
            # still-running lanes rate-limit toward a freshly resampled
            # target, exactly as the pre-2026-09-14 per-episode version did.
            w_target = sample_preference_vector(self.rng, self.reward_cfg, self.pref_cfg, self.n)
            w_rate_limited = rate_limit(self.w, w_target, self.pref_cfg.max_delta_per_step)
            self.w = np.where(self._prev_done[:, None], w_target, w_rate_limited)
            self.w = floor_clip(self.w, self.reward_cfg.term_names, self.reward_cfg.impact_floor_eps)

            actor_obs_w = np.concatenate([self.stack.policy_obs, self.w], axis=-1).astype(np.float32)
            critic_obs_w = np.concatenate([self.stack.critic_obs, self.w], axis=-1).astype(np.float32)
            with torch.no_grad():
                action_t, logp_t = self.model.act(torch.from_numpy(actor_obs_w))
                value_t = self.model.value(torch.from_numpy(critic_obs_w))
            action = action_t.numpy()

            transition, done = self.env.step(action)
            self.stack.push(transition["obs"], done_mask=done)
            reward_vec = compute_reward_vector(transition, self.reward_cfg)

            self._lane_step_count += 1
            if done.any():
                finished_lengths.extend(self._lane_step_count[done].tolist())
                self._lane_step_count[done] = 0

            actor_obs_list.append(actor_obs_w)
            critic_obs_list.append(critic_obs_w)
            act_list.append(action)
            logp_list.append(logp_t.numpy())
            rew_list.append(reward_vec)
            val_list.append(value_t.numpy())
            done_list.append(done)

            self._prev_done = done
            self._t += 1

        with torch.no_grad():
            final_critic_obs_w = np.concatenate([self.stack.critic_obs, self.w], axis=-1).astype(np.float32)
            final_value = self.model.value(torch.from_numpy(final_critic_obs_w)).numpy()

        return {
            "actor_obs": np.stack(actor_obs_list),      # (T, N, actor_obs_w_dim)
            "critic_obs": np.stack(critic_obs_list),    # (T, N, critic_obs_w_dim)
            "actions": np.stack(act_list),               # (T, N, action_dim)
            "logp": np.stack(logp_list),                 # (T, N)
            "rewards": np.stack(rew_list),                # (T, N, K)
            "values": np.stack(val_list),                 # (T, N, K)
            "dones": np.stack(done_list),                  # (T, N)
            "final_value": final_value,                     # (N, K)
            "finished_lengths": finished_lengths,           # episode lengths completed this window
        }

    def update(self) -> dict:
        """Collects the next `num_steps` timesteps (continuing the persistent
        rollout), then runs PPO for `epochs_per_update` epochs."""
        r = self._collect_rollout()

        values_with_final = np.concatenate([r["values"], r["final_value"][None]], axis=0)  # (T+1, N, K)
        adv = gae_per_objective(r["rewards"], values_with_final, r["dones"], self.cfg.gamma, self.cfg.gae_lambda)
        returns = adv + r["values"]

        T, N = r["dones"].shape
        # w . advantage_vector per (t, n) — the preference vector arbitrates
        # between objectives at the advantage level (not a pre-summed scalar
        # reward before GAE). w is the last reward_cfg.dim columns of the
        # stored actor_obs (see the w-concatenation in _collect_rollout).
        w_used = r["actor_obs"][:, :, -self.reward_cfg.dim :]
        scalar_adv = np.einsum("tnk,tnk->tn", adv, w_used).reshape(T * N)

        actor_obs_t = torch.from_numpy(r["actor_obs"].reshape(T * N, -1))
        critic_obs_t = torch.from_numpy(r["critic_obs"].reshape(T * N, -1))
        actions_t = torch.from_numpy(r["actions"].reshape(T * N, -1))
        logp_old_t = torch.from_numpy(r["logp"].reshape(T * N))
        adv_t = torch.from_numpy(scalar_adv.astype(np.float32))
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        returns_t = torch.from_numpy(returns.reshape(T * N, -1).astype(np.float32))

        last_policy_loss = last_value_loss = 0.0
        for _ in range(self.cfg.epochs_per_update):
            logp_new = self.model.logp(actor_obs_t, actions_t)
            ratio = torch.exp(logp_new - logp_old_t)
            clipped = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps)
            policy_loss = -torch.min(ratio * adv_t, clipped * adv_t).mean()

            values_pred = self.model.value(critic_obs_t)
            value_loss = nn.functional.mse_loss(values_pred, returns_t)

            loss = policy_loss + 0.5 * value_loss
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            last_policy_loss, last_value_loss = float(policy_loss.item()), float(value_loss.item())

        mean_reward_vec = r["rewards"].reshape(T * N, -1).mean(axis=0)
        # Mean episode length, from the persistent per-lane step counter
        # (self._lane_step_count) rather than anything window-local — the
        # true episode horizon (e.g. 200) is much longer than num_steps
        # (default 24), so a window-local measurement could never report
        # the real value (see _lane_step_count's docstring in __init__).
        # Prefer lengths of episodes that actually finished this window; if
        # none finished (common — most num_steps-long windows contain no
        # boundary), fall back to the current in-progress counter averaged
        # across all lanes as a reasonable proxy of how far into their
        # episodes the lanes currently are.
        finished_lengths = r["finished_lengths"]
        if finished_lengths:
            mean_episode_len = float(np.mean(finished_lengths))
        else:
            mean_episode_len = float(self._lane_step_count.mean())

        return {
            "policy_loss": last_policy_loss,
            "value_loss": last_value_loss,
            "mean_reward_vec": mean_reward_vec,
            "mean_episode_len": mean_episode_len,
        }
