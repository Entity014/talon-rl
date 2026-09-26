"""Policy export for deployment/sim2sim — what a MuJoCo validation harness
or real hardware would load (00_Proposal §3.4's Sim-to-Sim Validation).

Deliberately NOT a port of jaykorea/Isaac-RL-Two-wheel-Legged-Bot's
wrapper/exporter.py: that file wraps `actor_critic.actor` (a single
combined submodule), `actor_critic.is_recurrent`, and
`actor_critic.memory_a.rnn` — none of which exist on this repo's
`ActorCritic` (here the actor is `actor_body` + `actor_mean` as two
submodules, and there's no recurrent/LSTM support), so copying it verbatim
would fail immediately, not just be over-scoped. It also bundles an
unrelated PDF-report feature (`reportlab`) and an SRM/GRU exporter specific
to their own `srmppo` algorithm, neither applicable here. Only the
underlying technique — wrap the actor-only forward in a small nn.Module
and torch.jit.script it — carries over.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from ..modules.actor_critic import ActorCritic


class _PolicyExportWrapper(nn.Module):
    """Actor-only forward, deterministic (ActorCritic.act_inference) — no
    critic, no sampling. This is the scriptable unit that gets saved.

    Delegates to model.act_inference() rather than reimplementing the actor
    forward pass — found 2026-09-17: it used to call
    `actor_mean(actor_body(x))` directly, silently bypassing act_inference's
    tanh-squash bound (added the same day). Any future change to
    act_inference would have gone stale here the same way."""

    def __init__(self, model: ActorCritic):
        super().__init__()
        self.model = model

    def forward(self, actor_obs_w: torch.Tensor) -> torch.Tensor:
        return self.model.act_inference(actor_obs_w)


def export_policy_as_jit(model: ActorCritic, path: str) -> None:
    """Scripts model.act_inference's computation (actor_body + actor_mean,
    no sampling) and saves it as a TorchScript module at `path`. Creates
    any missing parent directories."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    wrapper = _PolicyExportWrapper(model)
    wrapper.eval()
    scripted = torch.jit.script(wrapper)
    scripted.save(path)
