#!/usr/bin/env python3
"""Endpoint survival for the projected-tail-descent run at update 10."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rl.experiments.shared.projected_endpoint_audit import ProjectedEndpointAudit


class ProjectedDescentU10EndpointAudit(ProjectedEndpointAudit):
    """Endpoint survival for the projected-tail-descent run at update 10."""

    run = "authority_isolated_projected_tail_descent-2026-09-25"
    checkpoint = "model_10.pt"
    report = "projected_descent_u10_endpoint_audit.json"


if __name__ == "__main__":
    ProjectedDescentU10EndpointAudit.main()
