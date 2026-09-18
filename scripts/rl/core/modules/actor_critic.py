"""Actor-critic network shape — reusable across on-policy algorithms, not
specific to MOPPO's preference-conditioned loss. See
scripts/rl/core/algorithms/moppo.py for how MOPPO builds and updates it.
"""

from __future__ import annotations

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
        self, actor_obs_dim: int, critic_obs_dim: int, action_dim: int, reward_dim: int, hidden_dims: list[int]
    ):
        super().__init__()
        self.actor_body = _mlp_body(actor_obs_dim, hidden_dims)
        self.actor_mean = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

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
    __constants__ = ["ACTION_CLIP", "_ATANH_EPS", "LOG_STD_MIN"]
    ACTION_CLIP = 3.0
    _ATANH_EPS = 1e-6
    # Floor under log_std, on top of the entropy bonus (moppo.py's
    # entropy_coef) -- found 2026-09-17 (phase1_longrun, 20000 updates):
    # entropy_coef=0.01 slowed collapse but didn't stop it. The run briefly
    # reached mean_episode_len=70.45 around iteration 14501, then log_std
    # kept shrinking (entropy ended at -14.92, implying std ~= 0.07) and the
    # policy got stuck too deterministic to explore back to that behavior
    # for the remaining 5500 iterations. -1.6 (std ~= 0.20 pre-tanh) still
    # lets log_std shrink substantially from init (0.0, std=1.0) but never
    # below a floor that still explores.
    #
    # First attempt used `self.log_std.clamp(min=LOG_STD_MIN).exp()` -- a
    # hard clamp. Found 2026-09-18 (phase1_longrun5): entropy went
    # perfectly flat at -2.1727 (== the floor's exact value) from iteration
    # ~2500 onward and never moved by so much as a float ULP for the next
    # 7000+ iterations, even though the entropy bonus should keep pushing
    # std back up whenever the policy gradient lets it. Checkpoint
    # inspection confirmed why: `torch.clamp`'s gradient is exactly zero
    # outside the clamped range, so once log_std drifts <= LOG_STD_MIN
    # (Adam momentum alone can carry it well past the boundary -- one dim
    # was found at -1.89), BOTH the policy-loss gradient and the
    # entropy-bonus gradient into log_std become 0 (entropy() reads the
    # same clamped std). The parameter is then permanently frozen: no
    # coefficient on the entropy bonus can matter once its gradient is
    # exactly zero. Identical failure shape to the hard-clamp-before-logp
    # bug documented above for actions -- same anti-pattern, different
    # parameter. Fixed with a softplus floor: always differentiable, still
    # asymptotes to LOG_STD_MIN from above, so the entropy bonus can pull
    # log_std back up even after it overshoots the floor.
    LOG_STD_MIN = -1.6

    def _pre_tanh_dist(self, actor_obs_w: torch.Tensor) -> Normal:
        mean = self.actor_mean(self.actor_body(actor_obs_w))
        std = (self.LOG_STD_MIN + F.softplus(self.log_std - self.LOG_STD_MIN)).exp()
        return Normal(mean, std)

    def _squash(self, dist: Normal, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """u: pre-tanh Gaussian sample/value -> (bounded action, its log_prob)."""
        action = torch.tanh(u) * self.ACTION_CLIP
        # d(action)/d(u) = ACTION_CLIP * (1 - tanh(u)^2); log|Jacobian| subtracted
        # per the standard change-of-variables correction for a squashed policy.
        log_det_jacobian = torch.log(self.ACTION_CLIP * (1 - torch.tanh(u) ** 2) + self._ATANH_EPS)
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
        log_det_jacobian = torch.log(self.ACTION_CLIP * (1 - normalized**2) + self._ATANH_EPS)
        return (dist.log_prob(u) - log_det_jacobian).sum(-1)

    def value(self, critic_obs_w: torch.Tensor) -> torch.Tensor:
        return self.critic_head(self.critic_body(critic_obs_w))
