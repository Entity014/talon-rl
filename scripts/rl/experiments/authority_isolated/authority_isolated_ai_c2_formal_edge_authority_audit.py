#!/usr/bin/env python3
"""Simplex-edge endpoint authority for the formal narrow/wide edge pair."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rl.experiments.shared.simplex_edge_endpoint import SimplexEdgeEndpointAudit


class FormalEdgeAuthorityAudit(SimplexEdgeEndpointAudit):
    """Simplex-edge endpoint authority for the formal narrow/wide edge pair."""

    run = "authority_isolated_ai_c2_formal_edge_authority_audit-2026-09-25"
    narrow = ("control", "authority_isolated_ai_c2_formal_edge_narrow-2026-09-25/model_10.pt")
    wide = ("edge", "authority_isolated_ai_c2_formal_edge_wide-2026-09-25/model_10.pt")


if __name__ == "__main__":
    FormalEdgeAuthorityAudit.main()
