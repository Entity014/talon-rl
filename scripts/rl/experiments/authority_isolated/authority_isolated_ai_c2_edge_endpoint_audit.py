#!/usr/bin/env python3
"""Simplex-edge endpoint authority for the edge-C2 narrow/wide pair."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rl.experiments.shared.simplex_edge_endpoint import SimplexEdgeEndpointAudit


class EdgeEndpointAudit(SimplexEdgeEndpointAudit):
    """Simplex-edge endpoint authority for the edge-C2 narrow/wide pair."""

    run = "authority_isolated_ai_c2_edge_endpoint_audit-2026-09-25"
    narrow = ("narrow", "authority_isolated_ai_c2_edgec2_narrow-2026-09-25/model_10.pt")
    wide = ("wide", "authority_isolated_ai_c2_edgec2_wide-2026-09-25/model_10.pt")


if __name__ == "__main__":
    EdgeEndpointAudit.main()
