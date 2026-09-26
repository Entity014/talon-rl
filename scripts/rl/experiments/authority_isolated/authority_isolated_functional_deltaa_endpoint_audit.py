#!/usr/bin/env python3
"""Does the functional-delta-a retain arm keep u20's preference geometry?

Measures the per-preference action offset from centre against the u20
reference, then sensitivity retention overall and per support phase.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import rl.experiments.shared.authority_isolated_h1_screen as h1
from rl.core.offline_audit import RUNS
from rl.experiments.shared.authority_retention import SUPPORT, AuthorityRetentionAudit

HEAVY = ("T", "A", "O", "S")
PHASES = ("initial", "early", "late")
ARMS = ("control", "retain")


class FunctionalDeltaAEndpointAudit(AuthorityRetentionAudit):
    """Preference geometry kept by the functional-delta-a arms at update 30."""

    run = "authority_isolated_functional_deltaa_endpoint_audit-2026-09-25"
    report = "endpoint_audit.json"
    schema = "functional_deltaa_endpoint_audit_v1"
    support_seed = 2609252501
    checkpoints = {
        "u20": "authority_isolated_coverage2_control-2026-09-25/model_20.pt",
        "control": "authority_isolated_functional_deltaa_control-2026-09-25/model_30.pt",
        "retain": "authority_isolated_functional_deltaa_retain-2026-09-25/model_30.pt"}

    def delta(self, m, x, lab):
        """Action offset of a preference from the centre preference."""
        n = len(x)
        w = torch.tensor(h1.PREFS[lab], device="cuda").repeat(n, 1)
        c = torch.tensor(h1.PREFS["C"], device="cuda").repeat(n, 1)
        with torch.no_grad():
            return m.act_inference_with_preference(x, w) - m.act_inference_with_preference(x, c)

    def analyze(self):
        d = np.load(RUNS / SUPPORT)
        x_all = torch.tensor(d["obs"], device="cuda")
        origin, phase = d["origin"], d["phase"]
        ms = {k: self.load_wide(v) for k, v in self.checkpoints.items()}

        rep = {"schema": self.schema, "models": {}}
        for name in ARMS:
            mse, rel = [], []
            for lab in HEAVY:
                r = self.delta(ms["u20"], x_all, lab)
                q = self.delta(ms[name], x_all, lab)
                mse.append(float(((q - r) ** 2).mean(1).mean().cpu()))
                rel.append(float((torch.linalg.vector_norm(q - r, dim=1)
                                  / (torch.linalg.vector_norm(r, dim=1) + 1e-8)).mean().cpu()))
            rep["models"][name] = {"deltaa_mse": float(np.mean(mse)),
                                   "deltaa_relative_error": float(np.mean(rel)),
                                   "by_axis_mse": dict(zip(HEAVY, mse))}

        # 384 states: up to 128 per phase, balanced over origins. The generator
        # is reused by the per-phase draw below, so the order of the two must
        # not change.
        rng = np.random.default_rng(self.support_seed)
        ids = []
        for pi in range(3):
            pools = []
            for oi in range(5):
                z = np.flatnonzero((origin == oi) & (phase == pi))
                pools.extend(rng.choice(z, size=25, replace=False).tolist())
            ids.extend(pools[:128] if len(pools) >= 128 else pools)
        x = x_all[np.asarray(ids[:384])]

        base = h1.sensitivity(ms["u20"], x)
        rep["u20_sensitivity"] = base
        for name in ARMS:
            q = h1.sensitivity(ms[name], x)
            m = rep["models"][name]
            m["sensitivity"] = q
            m["retention"] = {
                "pairwise": self.ratio(q, base, "pairwise_action_distance", "mean"),
                "tangent": self.ratio(q, base, "tangent_jacobian_fro_mean"),
                "functional_specific_rms": self.ratio(q, base, "centered_functional_geometry", "specific_rms"),
                "parameter_specific_energy": self.ratio(q, base, "centered_parameter_geometry", "specific_energy"),
                "parameter_rank": q["centered_parameter_geometry"]["effective_rank_5pct"],
                "functional_rank": q["centered_functional_geometry"]["effective_rank_5pct"]}
            m["authority_gate"] = bool(m["retention"]["pairwise"] >= .9
                                       and m["retention"]["tangent"] >= .9)

        rep["phase"] = {}
        for pi, pn in enumerate(PHASES):
            z = np.flatnonzero(phase == pi)
            xx = x_all[rng.choice(z, size=min(256, len(z)), replace=False)]
            b = h1.sensitivity(ms["u20"], xx)
            rep["phase"][pn] = {}
            for name in ARMS:
                q = h1.sensitivity(ms[name], xx)
                rep["phase"][pn][name] = {
                    "pairwise": self.ratio(q, b, "pairwise_action_distance", "mean"),
                    "tangent": self.ratio(q, b, "tangent_jacobian_fro_mean")}
        return rep

    def summarize(self, report):
        print(json.dumps({k: {"mse": v["deltaa_mse"], "relerr": v["deltaa_relative_error"],
                              "retention": v["retention"], "gate": v["authority_gate"]}
                          for k, v in report["models"].items()}, indent=2), flush=True)
        print("PHASE", json.dumps(report["phase"], indent=2), flush=True)


if __name__ == "__main__":
    FunctionalDeltaAEndpointAudit.main()
