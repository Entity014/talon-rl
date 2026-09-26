#!/usr/bin/env python3
"""Simplex-edge endpoint audit: authority kept by a narrow/wide pair at u10.

Two audits run this against different narrow/wide training pairs. They set the
pair and where the report goes; everything else is here.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import rl.experiments.shared.authority_isolated_h1_screen as h1
from rl.core.offline_audit import RUNS
from rl.experiments.shared.authority_retention import (
    PAIRS,
    REFERENCE,
    SUPPORT,
    AuthorityRetentionAudit,
    actions,
)

PHASES = ("initial", "early", "late")


class SimplexEdgeEndpointAudit(AuthorityRetentionAudit):
    """Authority kept by a narrow/wide training pair, against the u20 reference."""

    schema = "simplex_edge_endpoint_audit_v1"
    support_seed = 2609252701
    report = "endpoint_audit.json"
    narrow: tuple = ()   # (name, checkpoint) — the actor-critic arm
    wide: tuple = ()     # (name, checkpoint) — the wide-critic arm

    def models(self):
        m = {"u20": self.load_reference()}
        m[self.narrow[0]] = self.load_policy(self.narrow[1])
        m[self.wide[0]] = self.load_wide(self.wide[1])
        return m

    def load_wide(self, checkpoint):
        from talon_rl.authority_isolated_wide_critic import AuthorityIsolatedWideCritic

        m = AuthorityIsolatedWideCritic(48, 12).cuda()
        m.load_state_dict(torch.load(RUNS / checkpoint, map_location="cuda",
                                     weights_only=False)["model"])
        m.eval()
        return m

    def analyze(self):
        phase = np.load(RUNS / SUPPORT)["phase"]
        x, sample = self.support()
        m = self.models()
        base = h1.sensitivity(m["u20"], sample)
        acts = {k: actions(v, x) for k, v in m.items()}
        rep = {"schema": self.schema, "models": {}}
        for name in (self.narrow[0], self.wide[0]):
            q = h1.sensitivity(m[name], sample)
            pairret = self.ratio(q, base, "pairwise_action_distance", "mean")
            tanret = self.ratio(q, base, "tangent_jacobian_fro_mean")
            funret = self.ratio(q, base, "centered_functional_geometry", "specific_rms")
            edges, hh, hc = {}, [], []
            for i, j in PAIRS:
                rr = torch.linalg.vector_norm(acts["u20"][i] - acts["u20"][j], dim=1)
                qq = torch.linalg.vector_norm(acts[name][i] - acts[name][j], dim=1)
                er = float(torch.sqrt((qq.pow(2).sum() + 1e-12) / (rr.pow(2).sum() + 1e-12)))
                pr = {}
                for pi, pn in enumerate(PHASES):
                    ix = torch.tensor(np.flatnonzero(phase == pi), device="cuda")
                    pr[pn] = float(torch.sqrt((qq[ix].pow(2).sum() + 1e-12)
                                              / (rr[ix].pow(2).sum() + 1e-12)))
                edges[f"{i}-{j}"] = {"energy_retention": er, "phase": pr}
                (hc if "C" in (i, j) else hh).append(er)
            vals = [v["energy_retention"] for v in edges.values()]
            rep["models"][name] = {
                "pairwise_retention": pairret, "tangent_retention": tanret,
                "functional_rms_retention": funret,
                "parameter_rank": q["centered_parameter_geometry"]["effective_rank_5pct"],
                "functional_rank": q["centered_functional_geometry"]["effective_rank_5pct"],
                "heavy_heavy_mean": float(np.mean(hh)), "heavy_center_mean": float(np.mean(hc)),
                "min_edge": float(min(vals)),
                "min_edge_name": min(edges, key=lambda k: edges[k]["energy_retention"]),
                "edges": edges,
                "authority_gate": bool(pairret >= .9 and tanret >= .9),
                "edge_gate": bool(min(vals) >= .9)}
            self.print_model(name, rep["models"][name], edges)
        return rep

    def print_model(self, name, model, edges):
        print(name, json.dumps({k: v for k, v in model.items() if k != "edges"}, indent=2),
              flush=True)
        for e, v in edges.items():
            print(name, e, round(v["energy_retention"], 4), v["phase"], flush=True)
