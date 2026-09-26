"""Guards on scripts/rl/core/isaac_audit.py.

Isaac Sim is not importable in the test environment, so these cover the parts
that do not need it — the pathing and the --out contract. Those are what keep
an audit from overwriting a report in `runs/`, which is gitignored and holds
the only copy.
"""

import json

import pytest

from rl.core import isaac_audit, run_report
from rl.core.isaac_audit import IsaacAudit, obs_tensor


class _Demo(IsaacAudit):
    run = "demo-run"
    report = "demo_audit.json"

    def rollout(self, env, obs):
        rep = {"ok": True}
        self.write(rep)
        return rep


@pytest.fixture
def runs(tmp_path, monkeypatch):
    r = tmp_path / "runs"
    (r / "demo-run").mkdir(parents=True)
    monkeypatch.setattr(run_report, "RUNS", r)
    return r


def test_report_goes_to_the_run_directory_by_default(runs):
    _Demo().rollout(None, None)
    assert json.loads((runs / "demo-run" / "demo_audit.json").read_text()) == {"ok": True}


def test_out_leaves_the_run_directory_untouched(runs, tmp_path):
    # the point of --out: re-running an audit to compare it must not overwrite
    # the report it is being compared against
    out = tmp_path / "elsewhere"
    _Demo(out).rollout(None, None)
    assert not (runs / "demo-run" / "demo_audit.json").exists()
    assert (out / "demo_audit.json").exists()


def test_rollout_is_required():
    class Bare(IsaacAudit):
        run = "x"
    with pytest.raises(NotImplementedError):
        IsaacAudit.rollout(object.__new__(Bare), None, None)


def test_obs_tensor_takes_the_policy_group():
    import torch
    t = torch.zeros(2, 3)
    assert obs_tensor({"policy": t, "critic": torch.ones(2, 9)}) is t
    # an env that returns a bare array still has to come back as a tensor
    assert torch.is_tensor(obs_tensor([[0.0, 1.0]]))


def test_repo_root_is_on_sys_path():
    # talon_rl is imported from source in the Isaac environment, not installed
    import sys
    assert str(isaac_audit.REPO) in sys.path
    assert (isaac_audit.REPO / "talon_rl").is_dir()
