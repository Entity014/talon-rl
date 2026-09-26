#!/usr/bin/env python3
"""How much preference authority a checkpoint keeps against a reference critic.

Two audits measure the same three things — sensitivity retention, tangent
retention and per-preference-pair edge retention — against the same reference
and the same support set, differing only in which checkpoints they look at and
what they report. The shared parts live here.
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import rl.experiments.shared.authority_isolated_h1_screen as h1
from rl.core.offline_audit import RUNS, OfflineAudit

PREFS = list(h1.ORDER)
PAIRS = list(itertools.combinations(PREFS, 2))
REFERENCE = "authority_isolated_coverage2_control-2026-09-25/model_20.pt"
SUPPORT = "authority_isolated_u20_authority_support-2026-09-25/u20_authority_support.npz"
OBS_DIM, ACT_DIM = 48, 12
ORIGINS, PHASES, PER_CELL = 5, 3, 25


def actions(m, x):
    """The policy's action at each support state, per preference."""
    out = {}
    with torch.no_grad():
        for lab in PREFS:
            w = torch.tensor(h1.PREFS[lab], device="cuda").repeat(len(x), 1)
            out[lab] = m.act_inference_with_preference(x, w)
    return out


def edge_retention(ref_actions, actions_):
    """Per preference-pair, how much of the reference's separation survives."""
    edges = {}
    for i, j in PAIRS:
        r = torch.linalg.vector_norm(ref_actions[i] - ref_actions[j], dim=1)
        z = torch.linalg.vector_norm(actions_[i] - actions_[j], dim=1)
        edges[f"{i}-{j}"] = float(torch.sqrt((z.pow(2).sum() + 1e-12) / (r.pow(2).sum() + 1e-12)))
    return edges


class AuthorityRetentionAudit(OfflineAudit):
    """Preference authority retained against the coverage2 reference critic."""

    schema: str = ""
    support_seed: int = 0

    def load_reference(self):
        from talon_rl.authority_isolated_wide_critic import AuthorityIsolatedWideCritic

        m = AuthorityIsolatedWideCritic(OBS_DIM, ACT_DIM).cuda()
        m.load_state_dict(torch.load(RUNS / REFERENCE, map_location="cuda",
                                     weights_only=False)["model"])
        m.eval()
        return m

    def load_policy(self, checkpoint):
        from talon_rl.authority_isolated_actor_critic import AuthorityIsolatedActorCritic

        m = AuthorityIsolatedActorCritic(OBS_DIM, ACT_DIM).cuda()
        m.load_state_dict(torch.load(RUNS / checkpoint, map_location="cuda",
                                     weights_only=False)["model"])
        m.eval()
        return m

    def support(self):
        """All support states, and a balanced sample of them for sensitivity."""
        d = np.load(RUNS / SUPPORT)
        x = torch.tensor(d["obs"], device="cuda")
        rng = np.random.default_rng(self.support_seed)
        ids = []
        for oi in range(ORIGINS):
            for pi in range(PHASES):
                z = np.flatnonzero((d["origin"] == oi) & (d["phase"] == pi))
                ids.extend(rng.choice(z, size=PER_CELL, replace=False).tolist())
        return x, x[np.asarray(ids)]

    @staticmethod
    def ratio(q, base, *keys):
        for k in keys[:-1]:
            q, base = q[k], base[k]
        return q[keys[-1]] / (base[keys[-1]] + 1e-12)
