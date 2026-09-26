"""Keep the test suite out of the real runs/ directory.

`runs/` is gitignored and holds the only copy of every experiment artifact, so
a test that resolves a run directory for real can overwrite evidence. An
earlier version of these tests did exactly that. RUNS is redirected for every
test; a test that wants its own layout still patches it itself.
"""

import pytest


@pytest.fixture(autouse=True)
def _runs_under_tmp(tmp_path, monkeypatch):
    from rl.core.experiment_io import run_report

    monkeypatch.setattr(run_report, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(run_report, "ARTIFACTS", tmp_path / "artifacts")
