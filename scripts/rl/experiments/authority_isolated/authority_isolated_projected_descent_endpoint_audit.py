#!/usr/bin/env python3
"""Endpoint survival for the projected-tail-descent run at update 3."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rl.experiments.shared.projected_endpoint_audit import ProjectedEndpointAudit


class ProjectedDescentEndpointAudit(ProjectedEndpointAudit):
    """Endpoint survival for the projected-tail-descent run at update 3."""

    run = "authority_isolated_projected_tail_descent-2026-09-25"
    checkpoint = "model_3.pt"
    report = "projected_endpoint_audit.json"


if __name__ == "__main__":
    ProjectedDescentEndpointAudit.main()
