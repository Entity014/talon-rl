#!/usr/bin/env python3
"""Early-window H2 authority: what the semantic path keeps at updates 5 and 10."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import rl.experiments.shared.authority_isolated_h1_screen as h1
from rl.experiments.shared.authority_retention import (
    AuthorityRetentionAudit,
    actions,
    edge_retention,
)

RUN = "authority_isolated_ai_h2_semantic_path-2026-09-25"


class EarlyWindowAuthorityAudit(AuthorityRetentionAudit):
    """Early-window H2 authority: what the semantic path keeps at updates 5 and 10."""

    run = "authority_isolated_ai_h2_early_window_compatibility-2026-09-25"
    report = "authority.json"
    schema = "ai_h2_early_window_authority_v1"
    support_seed = 2609252901
    snapshots = (5, 10)

    def analyze(self):
        x, sample = self.support()
        ref = self.load_reference()
        base = h1.sensitivity(ref, sample)
        ref_actions = actions(ref, x)
        rep = {"schema": self.schema, "snapshots": {}}
        for snap in self.snapshots:
            m = self.load_policy(f"{RUN}/model_{snap}.pt")
            q = h1.sensitivity(m, sample)
            pair = self.ratio(q, base, "pairwise_action_distance", "mean")
            tan = self.ratio(q, base, "tangent_jacobian_fro_mean")
            edges = edge_retention(ref_actions, actions(m, x))
            rep["snapshots"][str(snap)] = {
                "global_update": 30 + snap,
                "pairwise_retention": pair,
                "tangent_retention": tan,
                "min_edge": min(edges.values()),
                "min_edge_name": min(edges, key=edges.get),
                "edges": edges,
                "authority_gate": bool(pair >= .9 and tan >= .9),
                "edge_gate": bool(min(edges.values()) >= .9)}
        return rep

    def summarize(self, report):
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    EarlyWindowAuthorityAudit.main()
