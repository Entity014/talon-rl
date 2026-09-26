#!/usr/bin/env python3
"""Endpoint survival for the tail-projected-gradient run."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rl.experiments.shared.projected_endpoint_audit import ProjectedEndpointAudit


class ProjectedGradientEndpointAudit(ProjectedEndpointAudit):
    """Endpoint survival for the tail-projected-gradient run."""

    run = "authority_isolated_tail_projected_gradient-2026-09-25"
    checkpoint = "model_3.pt"
    report = "projected_endpoint_audit.json"


if __name__ == "__main__":
    ProjectedGradientEndpointAudit.main()
