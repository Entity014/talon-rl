"""Guards on scripts/rl/core/freeze.py.

Freezing writes into `runs/`, which is gitignored and holds the only copy of
a closed experiment's provenance. These tests exist to catch a change that
would let a freeze touch the real run directory when it was told not to.
"""

import json

import pytest

from rl.core.freeze import Freeze


class _Demo(Freeze):
    run = "demo-run"
    schema = "demo_v1"
    status = "DEMO_CLOSED"
    artifacts = ("input.json", "synthesis.json")

    def body(self):
        return {"evidence": self.load("input.json")}

    def manifest_extra(self):
        return {"extra": "kept"}


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    import rl.core.freeze as mod
    runs = tmp_path / "runs"
    (runs / "demo-run").mkdir(parents=True)
    (runs / "demo-run" / "input.json").write_text('{"n": 1}')
    monkeypatch.setattr(mod, "RUNS", runs)
    return runs / "demo-run"


def test_freeze_writes_synthesis_and_manifest(run_dir):
    syn = _Demo().freeze()
    assert syn == {"schema": "demo_v1", "status": "DEMO_CLOSED", "evidence": {"n": 1}}
    manifest = json.loads((run_dir / "PROVENANCE_MANIFEST.json").read_text())
    assert manifest["status"] == "FROZEN_BY_HASH"
    assert set(manifest["artifacts"]) == {"input.json", "synthesis.json"}
    assert manifest["extra"] == "kept"


def test_out_leaves_the_run_directory_untouched(run_dir, tmp_path):
    # the whole point of --out: comparing a freeze must never overwrite the
    # record it is being compared against
    before = sorted(p.name for p in run_dir.iterdir())
    out = tmp_path / "elsewhere"
    _Demo(out).freeze()
    assert sorted(p.name for p in run_dir.iterdir()) == before
    assert (out / "synthesis.json").exists()


def test_manifest_hashes_the_synthesis_it_just_wrote(run_dir, tmp_path):
    import hashlib
    out = tmp_path / "elsewhere"
    _Demo(out).freeze()
    manifest = json.loads((out / "PROVENANCE_MANIFEST.json").read_text())
    expected = hashlib.sha256((out / "synthesis.json").read_bytes()).hexdigest()
    assert manifest["artifacts"]["synthesis.json"]["sha256"] == expected


def test_body_is_required():
    class Bare(Freeze):
        run = "x"
    with pytest.raises(NotImplementedError):
        Bare.body(object.__new__(Bare))
