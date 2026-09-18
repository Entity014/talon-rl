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

The rollout collection here (including per-episode preference-vector
sampling) is MOPPO-specific, not a generic on-policy concern — a second
algorithm might use another conditioning scheme — so it stays part of this
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
from ..preference import floor_clip_terms, sample_preference_vector
from ..running_norm import RunningMeanStd
from ..storage.rollout_storage import gae_per_objective


@dataclass
class MOPPOConfig:
    hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])  # matches jaykorea/Isaac-RL-Two-wheel-Legged-Bot's humanoid config and TienKung-Lab's exported policy shape
    lr: float = 3e-4
    # 0.998, matching RMA's own Phase 1 PPO recipe (Kumar et al. 2021,
    # supplementary S1) exactly -- was 0.99. Effective horizon 1/(1-gamma):
    # 100 steps at 0.99, but this task's own episode horizon is 200 steps
    # (episode_length_s=4.0, a1_env_cfg.py) -- the value function couldn't
    # see the back half of an episode's consequences at all. 0.998 gives a
    # 500-step effective horizon, longer than the episode itself.
    gamma: float = 0.998
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    epochs_per_update: int = 4
    # RMA (Kumar et al. 2021, supplementary S1) splits each update's 80,000
    # transitions into 4 minibatches x 4 epochs -- this repo ran full-batch
    # (all T*N transitions, one gradient step per epoch) until 2026-09-18.
    # Full-batch isn't wrong, but it's a different gradient-noise/
    # conditioning regime than the reference recipe, untested against this
    # reward vector. Matching RMA's split removes that as a confound.
    num_minibatches: int = 4
    num_steps: int = 24  # rollout length per update() call, across all N lanes
    # Found missing entirely 2026-09-17 — this repo's PPO loss had no entropy
    # term at all (every reference implementation, OpenAI Baselines/SB3/
    # RSL-RL, has one). 0.01 matches RSL-RL/legged_gym's standard default,
    # the closest reference lineage for this task (moppo.py's own docstring
    # already cites rsl_rl for its rollout-collection convention).
    entropy_coef: float = 0.01
    # RMA (Kumar et al. 2021) ramps every non-tracking reward term by a
    # multiplier k_t: k_0 = 0.03, k_{t+1} = k_t ** penalty_curriculum_growth
    # (k in (0,1) raised to a power < 1 increases it — reaches k~=0.84 by
    # iteration ~1000 and k~=0.99 by ~2000 with growth=0.997, despite the
    # paper's own "smoothly over 15,000 iterations" framing). Their stated
    # purpose: prevent early-policy collapse where the robot stands still to
    # dodge cost penalties. Our failure mode looks different (falls fast
    # rather than freezing) but the general principle — full reward
    # magnitude from iteration 1 giving a fresh policy nothing but large,
    # possibly contradictory penalty gradients before it can stand at all —
    # is untested here, and phase1_longrun (2026-09-17, 20000 updates) found
    # mean_episode_len flat at its ~6-8 floor for the first ~12,000
    # iterations regardless of raw iteration count, so it's worth trying
    # before assuming more updates alone will help. Applied to every reward
    # term except "progress" (RMA's equivalent split: forward-velocity
    # tracking unramped, terms 3-10 including its own orientation/balance
    # term ramped) — see reward.py's compute_reward_vector call site.
    penalty_curriculum_init: float = 0.03
    penalty_curriculum_growth: float = 0.997
    device: str = "cpu"
    torch_compile: bool = False
    # D3PO's diversity regularizer (Ambadkar et al. 2026) — untuned starting
    # points, not swept against this repo's reward vector (paper sweeps
    # diversity_lambda in {0.01, 0.1, 0.5, 1.0} and diversity_alpha in
    # {0, 0.1, 1, 10} on MO-Gymnasium only, not a legged-locomotion task).
    # Disabled (lambda=0) 2026-09-18: the loss is
    # ((KL(mean_w, mean_w') - alpha*|w-w'|)^2).mean() — it actively PENALIZES
    # the policy whenever its action mean under different preference vectors
    # is too similar. For this task, "don't fall" is plausibly the same
    # optimal behavior regardless of w (fall termination is a hard failure
    # under every objective), so a nonzero diversity_lambda fights
    # convergence to that shared safe behavior before the policy has even
    # learned to survive once. Never validated in either direction on this
    # reward vector — kept at 0 as the untested default until Phase 1
    # reaches consistent survival, then reintroduce and sweep deliberately.
    diversity_lambda: float = 0.0
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
        self.device = torch.device(self.cfg.device)
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
        self.model.to(self.device)
        if self.encoder:
            self.encoder.to(self.device)
        if self.device.type == "cuda":
            torch.set_float32_matmul_precision("high")
        self._act = self.model.act
        self._value = self.model.value
        self._logp = self.model.logp
        self._act_inference = self.model.act_inference
        self._entropy = self.model.entropy
        if self.cfg.torch_compile and self.device.type == "cuda":
            # The stochastic action is copied to NumPy after every env step;
            # reduce-overhead's CUDA graphs reuse that output buffer and can
            # overwrite it before the copy completes. The default compiler
            # mode avoids that unsafe graph capture at this boundary.
            self._act = torch.compile(self._act, mode="default")
            self._value = torch.compile(self._value, mode="default")
            self._logp = torch.compile(self._logp, mode="default")
            self._act_inference = torch.compile(self._act_inference, mode="default")
            self._entropy = torch.compile(self._entropy, mode="default")
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
        self.w = self._clip_preference(self.w)
        self._prev_done = np.zeros(self.n, dtype=bool)
        self._t = 0
        # Persistent per-lane steps-since-last-reset counter, for
        # mean_episode_len — must be instance state (not reset per update()
        # call) because the true episode horizon (e.g. 200) is much longer
        # than a single rollout window (num_steps, default 24), so a window-
        # local measurement can never see a full episode.
        self._lane_step_count = np.zeros(self.n, dtype=np.int64)
        # RMA-style penalty curriculum state — see MOPPOConfig.penalty_curriculum_init's
        # comment. Ramped once per update() call, not per env step.
        self._penalty_k = self.cfg.penalty_curriculum_init
        self._progress_idx = self.reward_cfg.term_names.index("progress")

    def _actor_obs(self) -> np.ndarray:
        """Actor observation, [policy_obs, z_t, w] — z_t (when an encoder is
        active) is inserted BEFORE w, not appended after it, so that w stays
        the trailing reward_cfg.dim columns of actor_obs exactly as update()
        assumes (w_used slicing, the diversity regularizer's w' swap — see
        those call sites' comments)."""
        parts = [self.stack.policy_obs]
        if self.encoder:
            e_t = self.extrinsics_norm.normalize(self._last_extrinsics, center=True)
            z_t = self.encoder(torch.from_numpy(e_t).float().to(self.device)).detach().cpu().numpy()
            parts.append(z_t)
        parts.append(self.w)
        return np.concatenate(parts, axis=-1).astype(np.float32)

    def _critic_obs(self) -> np.ndarray:
        """Critic observation, [critic_obs, z_t, w] — same ordering rationale
        as _actor_obs()."""
        parts = [self.stack.critic_obs]
        if self.encoder:
            e_t = self.extrinsics_norm.normalize(self._last_extrinsics, center=True)
            z_t = self.encoder(torch.from_numpy(e_t).float().to(self.device)).detach().cpu().numpy()
            parts.append(z_t)
        parts.append(self.w)
        return np.concatenate(parts, axis=-1).astype(np.float32)

    def _collect_rollout(self) -> dict:
        actor_obs_list, critic_obs_list, act_list, logp_list, rew_list, val_list, terminal_list = (
            [], [], [], [], [], [], []
        )
        extrinsics_list = []  # (T, N, extrinsics_dim) — the e_t used to build this step's z_t, for update()'s live re-encode

        for _ in range(self.cfg.num_steps):
            # AMOR/Pipeline Phase 1 samples w once per episode. Lanes that
            # auto-reset on the PREVIOUS step receive a fresh sample here;
            # all surviving lanes keep exactly the same preference. A
            # rate-limited changing w_t belongs to the later HLP/manual
            # deployment scheduler, not teacher training.
            if self._prev_done.any():
                w_reset = sample_preference_vector(self.rng, self.reward_cfg, self.pref_cfg, self.n)
                w_reset = self._clip_preference(w_reset)
                self.w = np.where(self._prev_done[:, None], w_reset, self.w)

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
                action_t, logp_t = self._act(torch.from_numpy(actor_obs_w).to(self.device))
                value_t = self._value(torch.from_numpy(critic_obs_w).to(self.device))
            # ActorCritic.act() tanh-squashes internally, so `action` is
            # already bounded to +/-ACTION_CLIP — no separate env-side clip
            # needed (see that class's docstring for why an external clip,
            # applied inconsistently between logp and the env, was worse).
            action = action_t.cpu().numpy()

            transition, done = self.env.step(action)
            # `done` conflates real termination (a fall — no more reward was
            # possible) with time-out truncation (the episode was cut off
            # mid-stream — reward kept going, we just stopped watching).
            # GAE must only zero the value bootstrap on the former: masking
            # it on truncation too means an episode that survives to the
            # full horizon gets no more credit for it than one that fell,
            # which erases any training signal for "survive longer" once
            # episodes start approaching the horizon. IsaacLabTalonEnv
            # exposes the real-termination-only mask as `terminal_fall`
            # (a1_env.py); DummyTalonEnv has no early-termination concept at
            # all, so `done` is already correct there.
            terminal = transition.get("terminal_fall", done)
            self.stack.push(transition["obs"], done_mask=done)
            self._last_extrinsics = transition.get("extrinsics") if self.encoder else None
            # The observation returned for a done lane is already its reset
            # observation, but its reward must belong to the action that
            # ended the preceding episode. Envs provide that terminal-frame
            # view separately without disturbing the next-policy-state obs.
            reward_vec = compute_reward_vector(transition.get("reward_transition", transition), self.reward_cfg)
            # RMA-style penalty curriculum (see MOPPOConfig.penalty_curriculum_init) —
            # every term except progress is scaled by the current k, ramping
            # from near-zero toward full strength over training.
            reward_vec = reward_vec.copy()
            non_progress = np.arange(reward_vec.shape[-1]) != self._progress_idx
            reward_vec[:, non_progress] *= self._penalty_k
            self.reward_norm.update(reward_vec)
            reward_vec = self.reward_norm.normalize(reward_vec)

            self._lane_step_count += 1
            self._lane_step_count[done] = 0

            actor_obs_list.append(actor_obs_w)
            critic_obs_list.append(critic_obs_w)
            act_list.append(action)
            logp_list.append(logp_t.cpu().numpy())
            rew_list.append(reward_vec)
            val_list.append(value_t.cpu().numpy())
            terminal_list.append(terminal)

            self._prev_done = done
            self._t += 1

        with torch.no_grad():
            final_critic_obs_w = self._critic_obs()
            final_value = self._value(torch.from_numpy(final_critic_obs_w).to(self.device)).cpu().numpy()

        return {
            "actor_obs": np.stack(actor_obs_list),      # (T, N, actor_obs_w_dim)
            "critic_obs": np.stack(critic_obs_list),    # (T, N, critic_obs_w_dim)
            "actions": np.stack(act_list),               # (T, N, action_dim)
            "logp": np.stack(logp_list),                 # (T, N)
            "rewards": np.stack(rew_list),                # (T, N, K)
            "values": np.stack(val_list),                 # (T, N, K)
            "dones": np.stack(terminal_list),               # (T, N) — real termination only, see terminal above
            "final_value": final_value,                     # (N, K)
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
        adv_t = torch.from_numpy(normalize_per_objective(adv.reshape(T * N, -1))).to(self.device)
        w_t = torch.from_numpy(w_used.reshape(T * N, -1).astype(np.float32)).to(self.device)

        actor_obs_t = torch.from_numpy(r["actor_obs"].reshape(T * N, -1)).to(self.device)
        critic_obs_t = torch.from_numpy(r["critic_obs"].reshape(T * N, -1)).to(self.device)
        actions_t = torch.from_numpy(r["actions"].reshape(T * N, -1)).to(self.device)
        logp_old_t = torch.from_numpy(r["logp"].reshape(T * N)).to(self.device)
        returns_t = torch.from_numpy(returns.reshape(T * N, -1).astype(np.float32)).to(self.device)
        if self.encoder:
            # Normalize with the SAME running stats used during collection
            # (already updated for this rollout in _collect_rollout, above)
            # so the live re-encode sees the same input scale the frozen
            # snapshot did.
            extrinsics_normed = self.extrinsics_norm.normalize(r["extrinsics"].reshape(T * N, -1), center=True)
            extrinsics_t = torch.from_numpy(extrinsics_normed).float().to(self.device)
        else:
            extrinsics_t = None

        total = T * N
        minibatch_size = total // self.cfg.num_minibatches
        last_policy_loss = last_value_loss = last_entropy = 0.0
        for _ in range(self.cfg.epochs_per_update):
            # Reshuffled every epoch (standard PPO practice) so each of the
            # num_minibatches gradient steps this epoch sees a different
            # random split of the T*N transitions.
            perm = torch.randperm(total, device=self.device)
            for mb in range(self.cfg.num_minibatches):
                idx = perm[mb * minibatch_size : (mb + 1) * minibatch_size]
                actor_obs_mb, critic_obs_mb = actor_obs_t[idx], critic_obs_t[idx]
                actions_mb, logp_old_mb = actions_t[idx], logp_old_t[idx]
                returns_mb, adv_mb, w_mb = returns_t[idx], adv_t[idx], w_t[idx]
                extrinsics_mb = extrinsics_t[idx] if self.encoder else None

                # The stored actor_obs_mb/critic_obs_mb are numpy-frozen
                # constants (z_t baked in via a no_grad snapshot at
                # collection time, see _actor_obs()/_critic_obs()) —
                # replaying them alone would make self.encoder's optim
                # membership a structural no-op, since a numpy round-trip
                # strips the autograd graph. Re-encode z_t live from the
                # stored extrinsics every minibatch (encoder params change
                # each optim.step(), same reason logp_new/values_pred are
                # also recomputed fresh, not reused from collection time),
                # and splice it into the middle [obs, z_t, w] slice so
                # gradients actually reach self.encoder.parameters()
                # through loss.backward().
                if self.encoder:
                    z_t_live = self.encoder(extrinsics_mb)
                    p, lat = self.stack.policy_obs_dim, self.extrinsics_cfg.adaptation_latent_dim
                    actor_obs_live = torch.cat([actor_obs_mb[:, :p], z_t_live, actor_obs_mb[:, p + lat :]], dim=-1)
                    cp = self.stack.critic_obs_dim
                    critic_obs_live = torch.cat([critic_obs_mb[:, :cp], z_t_live, critic_obs_mb[:, cp + lat :]], dim=-1)
                else:
                    actor_obs_live, critic_obs_live = actor_obs_mb, critic_obs_mb

                logp_new = self._logp(actor_obs_live, actions_mb)
                ratio = torch.exp(logp_new - logp_old_mb)
                clip_loss = d3po_actor_loss(ratio, adv_mb, w_mb, self.cfg.clip_eps)

                # Diversity regularizer — resampled each minibatch (cheap,
                # avoids overfitting to one w' draw). w' goes through the
                # same floor_clip pipeline as the real w (_sample_diversity_w).
                w_prime_mb = torch.from_numpy(self._sample_diversity_w(idx.shape[0])).to(self.device)
                actor_obs_prime_mb = actor_obs_live.clone()
                actor_obs_prime_mb[:, -self.reward_cfg.dim :] = w_prime_mb
                mean_w = self._act_inference(actor_obs_live)
                mean_w_prime = self._act_inference(actor_obs_prime_mb)
                diversity_loss = diversity_regularizer_loss(
                    mean_w, mean_w_prime, w_mb, w_prime_mb, self.model.log_std.exp(), self.cfg.diversity_alpha
                )

                policy_loss = clip_loss + self.cfg.diversity_lambda * diversity_loss

                values_pred = self._value(critic_obs_live)
                value_loss = nn.functional.mse_loss(values_pred, returns_mb)

                # Entropy bonus — see ActorCritic.entropy()'s docstring for why
                # this was missing entirely before 2026-09-17 and what failure
                # mode that produced. Subtracted (maximize entropy = minimize
                # -entropy) same convention as every other PPO reference.
                entropy_bonus = self._entropy(actor_obs_live).mean()

                loss = policy_loss + 0.5 * value_loss - self.cfg.entropy_coef * entropy_bonus
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                last_policy_loss, last_value_loss = float(policy_loss.item()), float(value_loss.item())
                last_entropy = float(entropy_bonus.item())

        mean_reward_vec = r["rewards"].reshape(T * N, -1).mean(axis=0)
        # Mean episode length, from the persistent per-lane step counter
        # (self._lane_step_count) — NOT r["finished_lengths"] (found
        # 2026-09-17: that field only counts lanes that happened to
        # terminate within this specific num_steps=24 window; once the
        # policy gets good enough that most lanes survive past 24 steps,
        # the ones that DO finish within any given window are an
        # increasingly unrepresentative minority — the currently-worst
        # lanes only, since every genuinely-improving lane is still
        # mid-episode and invisible to this stat. That produced a
        # convincing-looking "mean episode length collapses over training"
        # curve across three separate runs even though the true per-step
        # behavior — Episode_Reward/balance, computed over every timestep in
        # the rollout, not just finished ones — stayed flat the whole time).
        # _lane_step_count.mean() has no such survivorship bias: every lane,
        # finished or still running, contributes its current age every step.
        mean_episode_len = float(self._lane_step_count.mean())

        # Advance the penalty curriculum for the NEXT update() call's rollout
        # — this call's _collect_rollout() already used the pre-advance k.
        penalty_k = self._penalty_k
        self._penalty_k = self._penalty_k ** self.cfg.penalty_curriculum_growth

        return {
            "policy_loss": last_policy_loss,
            "value_loss": last_value_loss,
            "entropy": last_entropy,
            "penalty_curriculum_k": penalty_k,
            "mean_reward_vec": mean_reward_vec,
            "mean_episode_len": mean_episode_len,
        }

    def _sample_diversity_w(self, batch_size: int) -> np.ndarray:
        """A second preference vector w' for D3PO's diversity regularizer —
        routed through the SAME floor_clip pipeline as the real w (not a
        naive Dirichlet draw) or it would silently violate the
        w_impact >= eps invariant the rest of the system depends on."""
        w_prime = sample_preference_vector(self.rng, self.reward_cfg, self.pref_cfg, batch_size)
        return self._clip_preference(w_prime)

    def _clip_preference(self, preference: np.ndarray) -> np.ndarray:
        return floor_clip_terms(
            preference,
            self.reward_cfg.term_names,
            {"impact": self.reward_cfg.impact_floor_eps, "balance": self.reward_cfg.balance_floor_eps},
        )

    def act_inference(self) -> np.ndarray:
        """Deterministic action (no sampling) from the trainer's current
        internal state (self.stack.policy_obs + self.w) — for play.py, not
        training. Does not advance any state; call env.step() with the
        result and push the new obs onto self.stack yourself, same as
        _collect_rollout does."""
        actor_obs_w = self._actor_obs()
        with torch.no_grad():
            action_t = self._act_inference(torch.from_numpy(actor_obs_w).to(self.device))
        return action_t.cpu().numpy()

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
                "penalty_curriculum_k": self._penalty_k,
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
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optim.load_state_dict(checkpoint["optim"])
        self._t = checkpoint["t"]
        self._penalty_k = checkpoint.get("penalty_curriculum_k", self._penalty_k)
        self.reward_norm.load_state_dict(checkpoint["reward_norm"])
        if self.encoder and checkpoint.get("encoder"):
            self.encoder.load_state_dict(checkpoint["encoder"])
        if self.extrinsics_norm and checkpoint.get("extrinsics_norm"):
            self.extrinsics_norm.load_state_dict(checkpoint["extrinsics_norm"])
