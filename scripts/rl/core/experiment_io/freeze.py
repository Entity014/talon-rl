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

import json
from pathlib import Path

from rl.core.experiment_io.run_report import REPO, RUNS, RunReport, sha256

__all__ = ["Freeze", "REPO", "RUNS", "sha256"]


class Freeze(RunReport):
    schema: str = ""
    status: str = ""
    artifacts: tuple[str, ...] = ()    # files in the run dir the manifest hashes
    report = "synthesis.json"

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
        self.write(syn)
        manifest = {
            "status": "FROZEN_BY_HASH",
            # synthesis.json was just written to out; the rest stay where they were
            "artifacts": {f: {"sha256": sha256(self._artifact(f))} for f in self.artifacts},
            **self.manifest_extra(),
        }
        self.write(manifest, "PROVENANCE_MANIFEST.json")
        print(json.dumps(self.summary(syn), indent=2))
        return syn

    def _artifact(self, name: str) -> Path:
        p = self.out / name
        return p if p.exists() else self.dir / name

    @classmethod
    def main(cls) -> None:
        cls(cls.parse_args().out).freeze()
