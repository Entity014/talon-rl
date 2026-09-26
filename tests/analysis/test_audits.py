"""Guards on scripts/analysis/audits.py.

These exist because the audits were consolidated from fifteen throwaway
scripts whose output was already cited in docs/*-verdict.md — a silent change
in a shared helper would change those numbers with nothing to catch it.
"""

import numpy as np
import pytest

from analysis.audits import AUDITS, Audit, RUNS, col, mean_abs_diff, top_k


def test_col_stacks_one_trace_field():
    trace = [{"action": [1.0, 2.0]}, {"action": [3.0, 5.0]}]
    assert col(trace, "action").tolist() == [[1.0, 2.0], [3.0, 5.0]]


def test_mean_abs_diff_is_windowed_and_per_coordinate():
    # the window is half-open [lo, hi) — the row at index 2 must not count,
    # which is what distinguishes it from the (lo, hi] some callers pass.
    a = np.array([[0.0, 0.0], [2.0, 0.0], [99.0, 99.0]])
    b = np.zeros((3, 2))
    assert mean_abs_diff(a, b, 0, 2).tolist() == [1.0, 0.0]


def test_top_k_is_biggest_first():
    d = np.array([0.1, 9.0, 3.0])
    assert [name for _, name, _ in top_k(d, ["a", "b", "c"], 2)] == ["b", "c"]


def test_every_audit_is_registered_with_a_distinct_name_and_report():
    subclasses = Audit.__subclasses__()
    assert len(AUDITS) == len(subclasses)
    assert len({c.report for c in subclasses}) == len(subclasses)
    for name, cls in AUDITS.items():
        assert name and cls.report


@pytest.mark.parametrize("name", sorted(AUDITS))
def test_report_path_resolves_under_runs(name):
    # runs/ is gitignored, so the file may be absent on a fresh clone — what
    # must hold is that the path is relative to this repo's runs/, never the
    # absolute /home/... paths the original tmp_ scripts hardcoded.
    path = AUDITS[name]().path
    assert path.is_relative_to(RUNS)
