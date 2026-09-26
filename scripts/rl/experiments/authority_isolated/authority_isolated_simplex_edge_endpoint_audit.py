#!/usr/bin/env python3
"""Simplex-edge endpoint authority for the functional-deltaa control versus the
simplex-edge retain arm, both wide-critic, at update 30."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rl.experiments.shared.simplex_edge_endpoint import SimplexEdgeEndpointAudit


class SimplexEdgeRetainAudit(SimplexEdgeEndpointAudit):
    """Functional-deltaa control versus the simplex-edge retain arm at update 30."""

    run = "authority_isolated_simplex_edge_endpoint_audit-2026-09-25"
    narrow = ("control", "authority_isolated_functional_deltaa_control-2026-09-25/model_30.pt")
    narrow_kind = "wide"
    wide = ("edge", "authority_isolated_simplex_edge_retain-2026-09-25/model_30.pt")


if __name__ == "__main__":
    SimplexEdgeRetainAudit.main()
