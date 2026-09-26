"""Closing out an experiment: synthesis.json + PROVENANCE_MANIFEST.json.

Every `*_freeze.py` script writes the same two files into its run directory —
a synthesis keyed by schema and status, and a manifest hashing the artifacts
that synthesis rests on. Only the body of the synthesis and which extra
references the manifest carries differ, so those are what a subclass supplies.

Re-running a freeze overwrites a record that cannot be recovered: `runs/` is
gitignored, and the manifest pins sha256 values of source files that have
since changed. Pass `--out` to write somewhere else when the point is to
compare rather than to re-freeze.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RUNS = REPO / "runs"


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Freeze:
    run: str = ""                      # run directory name under runs/
    schema: str = ""
    status: str = ""
    artifacts: tuple[str, ...] = ()    # files in the run dir the manifest hashes

    def __init__(self, out: str | Path | None = None):
        self.out = Path(out) if out else self.dir
        self.out.mkdir(parents=True, exist_ok=True)

    @property
    def dir(self) -> Path:
        return RUNS / self.run

    def load(self, name: str, run: str | None = None) -> dict:
        """Read one JSON artifact, from this freeze's run dir unless told otherwise."""
        return json.loads(((RUNS / run if run else self.dir) / name).read_text())

    def sha(self, path) -> str:
        """sha256 of a path, resolved against the repo root when relative."""
        p = Path(path)
        return sha256(p if p.is_absolute() else REPO / p)

    # --- what a subclass fills in ---

    def body(self) -> dict:
        """Everything in synthesis.json except schema and status."""
        raise NotImplementedError

    def manifest_extra(self) -> dict:
        """Extra top-level keys for PROVENANCE_MANIFEST.json."""
        return {}

    def summary(self, syn: dict) -> dict:
        return {"status": syn["status"]}

    # --- shared ---

    def freeze(self) -> dict:
        syn = {"schema": self.schema, "status": self.status, **self.body()}
        (self.out / "synthesis.json").write_text(json.dumps(syn, indent=2) + "\n")
        manifest = {
            "status": "FROZEN_BY_HASH",
            # synthesis.json was just written to out; the rest stay where they were
            "artifacts": {f: {"sha256": sha256(self._artifact(f))} for f in self.artifacts},
            **self.manifest_extra(),
        }
        (self.out / "PROVENANCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(self.summary(syn), indent=2))
        return syn

    def _artifact(self, name: str) -> Path:
        p = self.out / name
        return p if p.exists() else self.dir / name

    @classmethod
    def main(cls) -> None:
        doc = (cls.__doc__ or "").strip().splitlines()
        ap = argparse.ArgumentParser(description=doc[0] if doc else f"freeze {cls.run}")
        ap.add_argument("--out", help="write the two files here instead of the run directory")
        cls(ap.parse_args().out).freeze()
