"""Minimal preference-conditioned PPO (MOPPO), chapter3.tex §3.2.3 / fig 3.3.

Policy: pi(a | s, w) — observation is [s, w] concatenated (w appended by this
module, not by the env — see envs/base_env.py docstring). The actor and
critic can see *different* amounts of temporal history via
`ObservationStackCfg` (num_policy_stacks / num_critic_stacks, after Flamingo
— jaykorea/Isaac-RL-Two-wheel-Legged-Bot — see obs_stack.py).

Critic: vector critic V(s, w) -> R^5, one head per reward-vector term
(asymmetric actor-critic per AMOR \\cite{alegre2025}).
Policy loss: D3PO's Late-Stage Weighting (Ambadkar et al. 2026,
arXiv:2602.07764) — the reward vector's per-objective advantage is kept
separate through GAE and the PPO clip, and only scalarized by w *after*
clipping (scripts/rl/core/losses.py's d3po_actor_loss), not AMOR's early
scalarization (clip(ratio, w . advantage)), which loses signal whenever
objectives conflict. Paired with a diversity regularizer (same module)
penalizing the policy for behaving too similarly under different w's,
using a second preference vector w' routed through the same floor_clip
pipeline as the real w (_sample_diversity_w).

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
scripts/rl/core/modules/ and scripts/rl/core/storage/.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from talon_rl.config import ExtrinsicsCfg, ObservationSpaceCfg, ObservationStackCfg, PreferenceCfg, RewardVectorCfg
from talon_rl.envs.base_env import BaseTalonEnv
from talon_rl.reward import compute_reward_vector

from ..losses import d3po_actor_loss, diversity_regularizer_loss, normalize_per_objective
from ..modules.actor_critic import ActorCritic
from ..modules.env_factor_encoder import EnvFactorEncoder
from ..obs_stack import ObservationStack
from ..preference import floor_clip, rate_limit, sample_preference_vector
from ..running_norm import RunningMeanStd
from ..storage.rollout_storage import gae_per_objective


@dataclass
class MOPPOConfig:
    hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])  # matches jaykorea/Isaac-RL-Two-wheel-Legged-Bot's humanoid config and TienKung-Lab's exported policy shape
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    epochs_per_update: int = 4
    num_steps: int = 24  # rollout length per update() call, across all N lanes
    device: str = "cpu"
    # D3PO's diversity regularizer (Ambadkar et al. 2026) — untuned starting
    # points, not swept against this repo's reward vector (paper sweeps
    # diversity_lambda in {0.01, 0.1, 0.5, 1.0} and diversity_alpha in
    # {0, 0.1, 1, 10} on MO-Gymnasium only, not a legged-locomotion task).
    diversity_lambda: float = 0.1
    diversity_alpha: float = 1.0


class MOPPOTrainer:
    def __init__(
        self,
        env: BaseTalonEnv,
        obs_cfg: ObservationSpaceCfg,
        reward_cfg: RewardVectorCfg,
        pref_cfg: PreferenceCfg,
        moppo_cfg: MOPPOConfig | None = None,
        stack_cfg: ObservationStackCfg | None = None,
        extrinsics_cfg: ExtrinsicsCfg | None = None,
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
        self.extrinsics_cfg = extrinsics_cfg
        self.encoder = EnvFactorEncoder(extrinsics_cfg.dim, extrinsics_cfg.adaptation_latent_dim) if extrinsics_cfg else None
        actor_obs_w_dim = self.stack.policy_obs_dim + reward_cfg.dim
        critic_obs_w_dim = self.stack.critic_obs_dim + reward_cfg.dim
        if extrinsics_cfg:
            actor_obs_w_dim += extrinsics_cfg.adaptation_latent_dim
            critic_obs_w_dim += extrinsics_cfg.adaptation_latent_dim
        self.model = ActorCritic(actor_obs_w_dim, critic_obs_w_dim, env.action_dim, reward_cfg.dim, self.cfg.hidden_dims)
        params = list(self.model.parameters()) + (list(self.encoder.parameters()) if self.encoder else [])
        self.optim = torch.optim.Adam(params, lr=self.cfg.lr)
        # Running per-objective reward normalization (chapter3.tex §3.2.3) —
        # without it, smoothness dominates progress by ~1000x in raw scale
        # (see CLAUDE.md / docs/mdp.md's "known gap" note this closes).
        self.reward_norm = RunningMeanStd(reward_cfg.dim)
        # Extrinsics channels span a ~1000x range (CoM offset ~±0.05 vs.
        # actuator stiffness ~44-66) — fed raw into EnvFactorEncoder's first
        # Linear layer, the large-magnitude channels dominate the small ones'
        # gradient contribution for a long time. Same fix as reward_norm
        # above, applied to the encoder's input instead of the reward vector.
        self.extrinsics_norm = RunningMeanStd(extrinsics_cfg.dim) if extrinsics_cfg else None

        # Persistent rollout state — set up once here, advanced by update(),
        # never reset mid-training (auto-reset happens per-lane inside
        # env.step() itself).
        transition = self.env.reset()
        self.stack.reset(transition["obs"])
        self._last_extrinsics = transition.get("extrinsics") if self.encoder else None
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

    def _actor_obs(self) -> np.ndarray:
        """Actor observation, [policy_obs, z_t, w] — z_t (when an encoder is
        active) is inserted BEFORE w, not appended after it, so that w stays
        the trailing reward_cfg.dim columns of actor_obs exactly as update()
        assumes (w_used slicing, the diversity regularizer's w' swap — see
        those call sites' comments)."""
        parts = [self.stack.policy_obs]
        if self.encoder:
            e_t = self.extrinsics_norm.normalize(self._last_extrinsics)
            z_t = self.encoder(torch.from_numpy(e_t).float()).detach().numpy()
            parts.append(z_t)
        parts.append(self.w)
        return np.concatenate(parts, axis=-1).astype(np.float32)

    def _critic_obs(self) -> np.ndarray:
        """Critic observation, [critic_obs, z_t, w] — same ordering rationale
        as _actor_obs()."""
        parts = [self.stack.critic_obs]
        if self.encoder:
            e_t = self.extrinsics_norm.normalize(self._last_extrinsics)
            z_t = self.encoder(torch.from_numpy(e_t).float()).detach().numpy()
            parts.append(z_t)
        parts.append(self.w)
        return np.concatenate(parts, axis=-1).astype(np.float32)

    def _collect_rollout(self) -> dict:
        actor_obs_list, critic_obs_list, act_list, logp_list, rew_list, val_list, done_list = (
            [], [], [], [], [], [], []
        )
        extrinsics_list = []  # (T, N, extrinsics_dim) — the e_t used to build this step's z_t, for update()'s live re-encode
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

            if self.encoder:
                # Update running stats before normalize()-ing this step's
                # extrinsics into _actor_obs()/_critic_obs() — same
                # update-then-normalize cadence as reward_norm below, so the
                # encoder always sees the freshest running scale.
                self.extrinsics_norm.update(self._last_extrinsics)
            actor_obs_w = self._actor_obs()
            critic_obs_w = self._critic_obs()
            if self.encoder:
                extrinsics_list.append(self._last_extrinsics)  # raw — update()'s epoch loop re-normalizes before re-encoding
            with torch.no_grad():
                action_t, logp_t = self.model.act(torch.from_numpy(actor_obs_w))
                value_t = self.model.value(torch.from_numpy(critic_obs_w))
            action = action_t.numpy()

            transition, done = self.env.step(action)
            self.stack.push(transition["obs"], done_mask=done)
            self._last_extrinsics = transition.get("extrinsics") if self.encoder else None
            reward_vec = compute_reward_vector(transition, self.reward_cfg)
            self.reward_norm.update(reward_vec)
            reward_vec = self.reward_norm.normalize(reward_vec)

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
            final_critic_obs_w = self._critic_obs()
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
            "extrinsics": np.stack(extrinsics_list) if self.encoder else None,  # (T, N, extrinsics_dim)
        }

    def update(self) -> dict:
        """Collects the next `num_steps` timesteps (continuing the persistent
        rollout), then runs PPO for `epochs_per_update` epochs."""
        r = self._collect_rollout()

        values_with_final = np.concatenate([r["values"], r["final_value"][None]], axis=0)  # (T+1, N, K)
        adv = gae_per_objective(r["rewards"], values_with_final, r["dones"], self.cfg.gamma, self.cfg.gae_lambda)
        returns = adv + r["values"]

        T, N = r["dones"].shape
        # w is the last reward_cfg.dim columns of the stored actor_obs (see
        # the w-concatenation in _collect_rollout). D3PO's Late-Stage
        # Weighting (arXiv:2602.07764) keeps the advantage per-objective
        # through the PPO clip instead of scalarizing with w first (AMOR's
        # early scalarization, which loses signal when objectives conflict
        # — see scripts/rl/core/losses.py) — normalized per objective, not
        # as one combined scalar stream.
        w_used = r["actor_obs"][:, :, -self.reward_cfg.dim :]
        adv_t = torch.from_numpy(normalize_per_objective(adv.reshape(T * N, -1)))
        w_t = torch.from_numpy(w_used.reshape(T * N, -1).astype(np.float32))

        actor_obs_t = torch.from_numpy(r["actor_obs"].reshape(T * N, -1))
        critic_obs_t = torch.from_numpy(r["critic_obs"].reshape(T * N, -1))
        actions_t = torch.from_numpy(r["actions"].reshape(T * N, -1))
        logp_old_t = torch.from_numpy(r["logp"].reshape(T * N))
        returns_t = torch.from_numpy(returns.reshape(T * N, -1).astype(np.float32))
        if self.encoder:
            # Normalize with the SAME running stats used during collection
            # (already updated for this rollout in _collect_rollout, above)
            # so the live re-encode sees the same input scale the frozen
            # snapshot did.
            extrinsics_normed = self.extrinsics_norm.normalize(r["extrinsics"].reshape(T * N, -1))
            extrinsics_t = torch.from_numpy(extrinsics_normed).float()
        else:
            extrinsics_t = None

        last_policy_loss = last_value_loss = 0.0
        for _ in range(self.cfg.epochs_per_update):
            # The stored actor_obs_t/critic_obs_t are numpy-frozen constants
            # (z_t baked in via a no_grad snapshot at collection time, see
            # _actor_obs()/_critic_obs()) — replaying them alone would make
            # self.encoder's optim membership a structural no-op, since a
            # numpy round-trip strips the autograd graph. Re-encode z_t live
            # from the stored extrinsics every epoch (encoder params change
            # each optim.step(), same reason logp_new/values_pred are also
            # recomputed fresh each epoch, not reused from collection time),
            # and splice it into the middle [obs, z_t, w] slice so gradients
            # actually reach self.encoder.parameters() through loss.backward().
            if self.encoder:
                z_t_live = self.encoder(extrinsics_t)
                p, lat = self.stack.policy_obs_dim, self.extrinsics_cfg.adaptation_latent_dim
                actor_obs_live = torch.cat([actor_obs_t[:, :p], z_t_live, actor_obs_t[:, p + lat :]], dim=-1)
                cp = self.stack.critic_obs_dim
                critic_obs_live = torch.cat([critic_obs_t[:, :cp], z_t_live, critic_obs_t[:, cp + lat :]], dim=-1)
            else:
                actor_obs_live, critic_obs_live = actor_obs_t, critic_obs_t

            logp_new = self.model.logp(actor_obs_live, actions_t)
            ratio = torch.exp(logp_new - logp_old_t)
            clip_loss = d3po_actor_loss(ratio, adv_t, w_t, self.cfg.clip_eps)

            # Diversity regularizer — resampled each epoch (cheap, avoids
            # overfitting to one w' draw). w' goes through the same
            # floor_clip pipeline as the real w (_sample_diversity_w).
            w_prime_t = torch.from_numpy(self._sample_diversity_w(T * N))
            actor_obs_prime_t = actor_obs_live.clone()
            actor_obs_prime_t[:, -self.reward_cfg.dim :] = w_prime_t
            mean_w = self.model.act_inference(actor_obs_live)
            mean_w_prime = self.model.act_inference(actor_obs_prime_t)
            diversity_loss = diversity_regularizer_loss(
                mean_w, mean_w_prime, w_t, w_prime_t, self.model.log_std.exp(), self.cfg.diversity_alpha
            )

            policy_loss = clip_loss + self.cfg.diversity_lambda * diversity_loss

            values_pred = self.model.value(critic_obs_live)
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

    def _sample_diversity_w(self, batch_size: int) -> np.ndarray:
        """A second preference vector w' for D3PO's diversity regularizer —
        routed through the SAME floor_clip pipeline as the real w (not a
        naive Dirichlet draw) or it would silently violate the
        w_impact >= eps invariant the rest of the system depends on."""
        w_prime = sample_preference_vector(self.rng, self.reward_cfg, self.pref_cfg, batch_size)
        return floor_clip(w_prime, self.reward_cfg.term_names, self.reward_cfg.impact_floor_eps)

    def act_inference(self) -> np.ndarray:
        """Deterministic action (no sampling) from the trainer's current
        internal state (self.stack.policy_obs + self.w) — for play.py, not
        training. Does not advance any state; call env.step() with the
        result and push the new obs onto self.stack yourself, same as
        _collect_rollout does."""
        actor_obs_w = self._actor_obs()
        with torch.no_grad():
            action_t = self.model.act_inference(torch.from_numpy(actor_obs_w))
        return action_t.numpy()

    def save(self, path: str) -> None:
        """Saves model + optimizer state (and the persistent step counter,
        for a resume to report a continuous update count) — not the env,
        rollout buffer, or preference-vector RNG state, which don't need to
        survive a resume the way training-loop progress does. The reward
        normalizer's running stats DO need to survive — reloading fresh
        stats would shock the reward scale the value function was trained
        against. Creates any missing parent directories (e.g. a fresh
        logs/<run>/ dir)."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optim": self.optim.state_dict(),
                "t": self._t,
                "reward_norm": self.reward_norm.state_dict(),
                "encoder": self.encoder.state_dict() if self.encoder else None,
                "extrinsics_norm": self.extrinsics_norm.state_dict() if self.extrinsics_norm else None,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Loads model + optimizer state saved by save(). The trainer must
        already be constructed with matching obs/action/reward dims (this
        does not reconstruct the model architecture, only its weights)."""
        checkpoint = torch.load(path, map_location=self.cfg.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optim.load_state_dict(checkpoint["optim"])
        self._t = checkpoint["t"]
        self.reward_norm.load_state_dict(checkpoint["reward_norm"])
        if self.encoder and checkpoint.get("encoder"):
            self.encoder.load_state_dict(checkpoint["encoder"])
        if self.extrinsics_norm and checkpoint.get("extrinsics_norm"):
            self.extrinsics_norm.load_state_dict(checkpoint["extrinsics_norm"])
