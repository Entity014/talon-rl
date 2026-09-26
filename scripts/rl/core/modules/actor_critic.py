"""Actor-critic network shape — reusable across on-policy algorithms, not
specific to MOPPO's preference-conditioned loss. See
scripts/rl/core/algorithms/moppo.py for how MOPPO builds and updates it.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def _mlp_body(in_dim: int, hidden_dims: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev_dim = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev_dim, h), nn.ELU()]
        prev_dim = h
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(
        self, actor_obs_dim: int, critic_obs_dim: int, action_dim: int, reward_dim: int, hidden_dims: list[int],
        reconstruction_dim: int = 0
    ):
        super().__init__()
        self.actor_body = _mlp_body(actor_obs_dim, hidden_dims)
        self.actor_mean = nn.Linear(hidden_dims[-1], action_dim)
        self.reconstruction_head = nn.Linear(hidden_dims[-1], reconstruction_dim) if reconstruction_dim > 0 else None
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        # Kept as model-level policy-distribution state so every caller of
        # _pre_tanh_dist() sees one source of truth. Default learned mode
        # deliberately preserves the historical self.log_std.exp() path.
        self.exploration_mode = "learned"
        self.register_buffer("scheduled_log_std", torch.zeros(action_dim), persistent=True)

        self.critic_body = _mlp_body(critic_obs_dim, hidden_dims)
        self.critic_head = nn.Linear(hidden_dims[-1], reward_dim)

    # Action bound before JointPositionActionCfg's scale=0.15 (a1_env_cfg.py)
    # turns it into a joint-position offset from default pose. Neither this
    # network nor that action term bounded actions at all originally — found
    # 2026-09-17: a standing-only rollout (v_command=0) showed the raw action
    # mean growing 0.5 -> 8.3 in magnitude over 5 steps and surviving 4.7x
    # *worse* than applying zero action on the identical env. Two cheaper
    # fixes were tried and both failed differently:
    #   - hard-clamping the sampled action before dist.log_prob() gave that
    #     dimension zero gradient past the bound -> value loss climbed
    #     27->54 then NaN within ~15 iterations once enough actions were
    #     saturating.
    #   - clamping only the env-applied action (leaving log_prob keyed to
    #     the unclamped sample) avoided NaN but trained *worse*
    #     (mean_episode_length ~6 vs ~13 baseline): PPO's gradient was
    #     computed against an action the robot never actually received, so
    #     credit assignment was wrong every time the clamp triggered.
    # tanh-squashing (standard SAC/PPO continuous-action bound) fixes both:
    # bounded everywhere with the Jacobian correction below, gradient is
    # smooth (no dead zone), and log_prob is always computed on the exact
    # value that gets applied -- see e.g. Haarnoja et al. 2018 appendix C.
    # torch.jit.script (exporter.py) needs these listed in __constants__ to
    # see them as compile-time constants inside a scripted method -- a
    # `typing.Final` annotation doesn't work here because this file uses
    # `from __future__ import annotations`, which makes annotations lazy
    # strings TorchScript can't introspect.
    __constants__ = ["ACTION_CLIP", "_ATANH_EPS", "LOG_STD_MIN", "LOG_STD_MAX"]
    ACTION_CLIP = 3.0
    _ATANH_EPS = 1e-6
    # Bounds on log_std, enforced OUTSIDE this forward pass entirely -- see
    # MOPPOTrainer.update()'s post-optimizer-step clamp
    # (`self.model.log_std.clamp_(LOG_STD_MIN, LOG_STD_MAX)` under
    # `torch.no_grad()`), not anything in this class. `_pre_tanh_dist`
    # below is deliberately a plain `self.log_std.exp()` with no
    # clamp/softplus/squash of any kind in the computational graph.
    #
    # History, why it ended up this simple: entropy_coef=0.01
    # (moppo.py) alone didn't stop log_std collapsing (found 2026-09-17,
    # phase1_longrun -- mean_episode_len briefly hit 70.45 then the policy
    # got stuck too deterministic to explore back to it). First fix was
    # `self.log_std.clamp(min=LOG_STD_MIN).exp()` INSIDE this forward pass
    # -- found 2026-09-18 (phase1_longrun5): entropy went perfectly flat at
    # the floor's exact value for 7000+ iterations, because `torch.clamp`'s
    # gradient is exactly zero outside the clamped range, so once log_std
    # drifted <= LOG_STD_MIN (Adam momentum alone carried it past the
    # boundary), BOTH the policy-loss and entropy-bonus gradients into
    # log_std became permanently 0 -- same anti-pattern as the
    # hard-clamp-before-logp action bug above. Replaced with a softplus
    # floor (differentiable everywhere) -- fixed that direction, but had no
    # ceiling, and a same-day performance-gated log_std mechanism
    # (MOPPOConfig.episode_len_baseline_decay -- blocks log_std from
    # DECREASING on any update() whose mean_episode_len underperforms its
    # own recent baseline) had nothing to push against once
    # mean_episode_len started declining: log_std climbed unbounded
    # (entropy hit +4.3 and rising, the mirror-image runaway). A second
    # softplus composed as a smooth ceiling was tried next and rejected
    # before ever training on it: LOG_STD_MAX-LOG_STD_MIN is only 1.6, well
    # inside softplus's curved (non-identity) region at beta=1, so the
    # composed function distorted values across the WHOLE usable range
    # (log_std=0 mapped to bounded=-0.605, not ~0); raising beta to sharpen
    # the transition just traded that for the original dead-gradient
    # problem back via float32 underflow (beta=8 gave exactly 0.0 gradient
    # by log_std=-10, same failure shape as the very first hard clamp).
    # Clamping the raw parameter's VALUE after each optimizer step, instead
    # of shaping a function inside the loss graph, sidesteps all of it: no
    # clamp/softplus ever appears in ANY backward pass, so std=log_std.exp()
    # has a real, well-defined, never-zero gradient at every value log_std
    # can ever actually hold, and both bounds are exact (no asymptotic
    # approximation to tune).
    LOG_STD_MIN = -1.6
    # 0.0 matches log_std's own init value (std=1.0) -- exploration is
    # capped at "as random as the untrained policy already was," a natural
    # reference point that needs no new magic number.
    LOG_STD_MAX = 0.0

    def _pre_tanh_dist(self, actor_obs_w: torch.Tensor) -> Normal:
        mean = self.actor_mean(self.actor_body(actor_obs_w))
        std = (self.log_std if self.exploration_mode == "learned" else self.scheduled_log_std).exp()
        return Normal(mean, std)

    def set_scheduled_fixed_std(self, std: float | torch.Tensor) -> None:
        """Enable a fixed pre-tanh Gaussian std for the current PPO update."""
        value = torch.as_tensor(std, device=self.log_std.device, dtype=self.log_std.dtype)
        if value.ndim == 0:
            value = value.expand_as(self.log_std)
        if value.shape != self.log_std.shape or not torch.isfinite(value).all() or not torch.all(value > 0):
            raise ValueError("scheduled std must be finite, positive, and action-dimension shaped")
        with torch.no_grad():
            self.scheduled_log_std.copy_(value.log())
        self.exploration_mode = "scheduled_fixed_std"

    def set_learned_std(self) -> None:
        """Restore the default learnable-log-std policy distribution."""
        self.exploration_mode = "learned"

    def _log_det_jacobian(self, u: torch.Tensor) -> torch.Tensor:
        """log|d(action)/d(u)| = log(ACTION_CLIP) + log(1 - tanh(u)^2).

        Found 2026-09-18: the naive direct form,
        `log(ACTION_CLIP*(1-tanh(u)**2) + _ATANH_EPS)`, has a reward-hacking
        loophole. As |u| grows, `1-tanh(u)**2` underflows toward the eps
        floor rather than continuing toward its true value of 0, so this
        term's output STOPS DECREASING and plateaus at a fixed constant
        (log(eps) ~= -13.8) once u is a few units past the tanh saturation
        point -- verified numerically: logp for a fixed std went from
        -0.57 (u=0.5) up to +14.1 (u=15) and stayed there for u=25, 50.
        Since `logp = dist.log_prob(u) - log_det_jacobian` and
        dist.log_prob(u) stays roughly constant whenever actor_mean tracks
        u (which PPO's gradient has every incentive to do), pushing u
        arbitrarily far from the origin was a free ~14-nat log-prob bonus
        with NO connection to actual reward -- PPO's ratio=exp(logp_new-
        logp_old) then rewards drifting deeper into saturation regardless
        of what the robot actually did, a gradient signal that dwarfs a
        typical policy_loss magnitude (~0.2) by roughly two orders of
        magnitude. This is very likely why actor_body's hidden-layer
        activations and actor_mean's raw output kept growing (activation
        max 5->10->15+, raw mean max up to 24) even with weight_decay=1e-4
        active -- weight_decay's pull can't compete with an artifact this
        large.

        Fixed with the standard numerically-stable SAC identity (Haarnoja
        et al. 2018, appendix C): log(1-tanh(u)^2) = 2*(log(2) - u -
        softplus(-2u)). This needs no epsilon and has no artificial floor
        -- it correctly continues toward -inf as |u| grows, so there's no
        free log-prob left to farm by saturating harder."""
        return math.log(self.ACTION_CLIP) + 2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))

    def _squash(self, dist: Normal, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """u: pre-tanh Gaussian sample/value -> (bounded action, its log_prob)."""
        action = torch.tanh(u) * self.ACTION_CLIP
        log_det_jacobian = self._log_det_jacobian(u)
        logp = (dist.log_prob(u) - log_det_jacobian).sum(-1)
        return action, logp

    def act(self, actor_obs_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self._pre_tanh_dist(actor_obs_w)
        u = dist.sample()
        return self._squash(dist, u)

    def act_inference(self, actor_obs_w: torch.Tensor) -> torch.Tensor:
        """Deterministic action (tanh of the Normal's mean, no sampling) —
        for play/deployment, not training. Same input always gives the same
        output, unlike act(). Also what gets exported for sim2sim/hardware:
        see scripts/rl/core/wrapper/exporter.py."""
        mean = self.actor_mean(self.actor_body(actor_obs_w))
        return torch.tanh(mean) * self.ACTION_CLIP

    def raw_mean(self, actor_obs_w: torch.Tensor) -> torch.Tensor:
        """The pre-tanh actor_mean output, unsquashed -- for MOPPOTrainer's
        mean-magnitude regularizer (MOPPOConfig.mean_reg_coef). Separate
        from act_inference() (which squashes) because the whole point of
        this term is to penalize the RAW magnitude before tanh hides it:
        found 2026-09-18 that actor_body's hidden-layer activations and
        actor_mean's raw output kept growing through training (activation
        max 5->10->15+, raw mean max up to 24) even with weight_decay=1e-4
        active on every parameter -- a generic weight penalty wasn't
        targeted at the actual symptom (the OUTPUT saturating), so an
        SAC-style direct penalty on mean.pow(2) was added instead (see
        moppo.py's update())."""
        return self.actor_mean(self.actor_body(actor_obs_w))

    def reconstruct(self, actor_obs_w: torch.Tensor) -> torch.Tensor:
        """Predict privileged training-only cues from the shared actor trunk."""
        if self.reconstruction_head is None:
            raise RuntimeError("reconstruction head is disabled")
        return self.reconstruction_head(self.actor_body(actor_obs_w))

    def entropy(self, actor_obs_w: torch.Tensor) -> torch.Tensor:
        """Entropy of the pre-tanh Gaussian (not the squashed distribution's
        exact entropy, which has no simple closed form) — for PPO's entropy
        bonus. Found 2026-09-17 missing entirely: this codebase's PPO loss
        was `policy_loss + 0.5 * value_loss`, no entropy term at all, no
        matter which reference implementation (OpenAI Baselines, SB3,
        RSL-RL) — the standard exploration incentive that keeps log_std from
        collapsing too early. Matches the failure shape seen in every
        training run this session: fast improvement for the first ~20-30
        iterations, then stuck (or declining) for the rest of a 2000-update
        run, consistent with premature entropy collapse rather than a
        converged optimum. The pre-tanh entropy still serves the same
        purpose (rewarding a wider pre-squash distribution), same
        convention SAC/TD3 implementations use when tanh-squashing."""
        dist = self._pre_tanh_dist(actor_obs_w)
        return dist.entropy().sum(-1)

    def logp(self, actor_obs_w: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Recomputes log_prob under the CURRENT params for an `action`
        returned by a past act() call (PPO's ratio needs both). Inverts the
        tanh+scale to recover u, clamping just inside +/-1 first since
        atanh(+/-1) is +/-inf -- action can land exactly on the boundary
        when the pre-tanh sample is large."""
        dist = self._pre_tanh_dist(actor_obs_w)
        normalized = (action / self.ACTION_CLIP).clamp(-1.0 + self._ATANH_EPS, 1.0 - self._ATANH_EPS)
        u = torch.atanh(normalized)
        log_det_jacobian = self._log_det_jacobian(u)
        return (dist.log_prob(u) - log_det_jacobian).sum(-1)

    def value(self, critic_obs_w: torch.Tensor) -> torch.Tensor:
        return self.critic_head(self.critic_body(critic_obs_w))
