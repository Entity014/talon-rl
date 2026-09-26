#!/usr/bin/env python3
"""Final H2 authority: what the semantic-path checkpoint keeps at update 25."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import rl.experiments.shared.authority_isolated_h1_screen as h1
from rl.experiments.shared.authority_retention import (
    AuthorityRetentionAudit,
    actions,
    edge_retention,
)


class FinalAuthorityAudit(AuthorityRetentionAudit):
    """Final H2 authority: what the semantic-path checkpoint keeps at update 25."""

    run = "authority_isolated_ai_h2_final_authority_audit-2026-09-25"
    report = "final_authority_audit.json"
    schema = "ai_h2_final_authority_audit_v1"
    support_seed = 2609252801
    checkpoint = "authority_isolated_ai_h2_semantic_path-2026-09-25/model_25.pt"

    def analyze(self):
        x, sample = self.support()
        ref = self.load_reference()
        m = self.load_policy(self.checkpoint)
        base = h1.sensitivity(ref, sample)
        q = h1.sensitivity(m, sample)
        pair = self.ratio(q, base, "pairwise_action_distance", "mean")
        tan = self.ratio(q, base, "tangent_jacobian_fro_mean")
        fun = self.ratio(q, base, "centered_functional_geometry", "specific_rms")
        edges = edge_retention(actions(ref, x), actions(m, x))
        heavy_heavy = [v for k, v in edges.items() if "C" not in k.split("-")]
        heavy_center = [v for k, v in edges.items() if "C" in k.split("-")]
        return {"schema": self.schema,
                "pairwise_retention": pair,
                "tangent_retention": tan,
                "functional_rms_retention": fun,
                "heavy_heavy_mean": float(np.mean(heavy_heavy)),
                "heavy_center_mean": float(np.mean(heavy_center)),
                "min_edge": float(min(edges.values())),
                "min_edge_name": min(edges, key=edges.get),
                "edges": edges,
                "parameter_rank": q["centered_parameter_geometry"]["effective_rank_5pct"],
                "functional_rank": q["centered_functional_geometry"]["effective_rank_5pct"],
                "authority_gate": bool(pair >= .9 and tan >= .9),
                "edge_gate": bool(min(edges.values()) >= .9)}

    def summarize(self, report):
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    FinalAuthorityAudit.main()
