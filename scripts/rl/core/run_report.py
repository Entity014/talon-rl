"""Reading a run directory and writing a JSON artifact back into it.

Freezes, offline audits and Isaac rollout audits all do this, and all of them
need the same guarantee: `runs/` is gitignored, so an artifact written there is
the only copy. `--out` sends the write somewhere else, which is what makes a
script re-runnable for comparison against the record it would otherwise
overwrite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
# talon_rl is imported from source, not from site-packages, in the Isaac
# environment — the scripts used to do this for themselves
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
RUNS = REPO / "runs"
# a few audits write beside runs/ instead of into it
ARTIFACTS = REPO / "artifacts"


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RunReport:
    # None means runs/; the few audits that write beside it set ARTIFACTS.
    # Resolved on access, not at class definition, so it stays patchable.
    root: Path | None = None
    run: str = ""        # directory under root/
    report: str = ""     # artifact filename written into it
    sort_keys: bool = False   # a few audits serialise their JSON sorted

    def __init__(self, out: str | Path | None = None):
        self.out = Path(out) if out else self.dir
        self.out.mkdir(parents=True, exist_ok=True)

    @property
    def dir(self) -> Path:
        return (self.root or RUNS) / self.run

    def load(self, name: str, run: str | None = None):
        """Read one artifact by name, from this run's directory unless told otherwise."""
        p = (RUNS / run if run else self.dir) / name
        if p.suffix == ".npz":
            import numpy as np

            return np.load(p, allow_pickle=True)
        return json.loads(p.read_text())

    def sha(self, path) -> str:
        """sha256 of a path, resolved against the repo root when relative."""
        p = Path(path)
        return sha256(p if p.is_absolute() else REPO / p)

    def write(self, data: dict, name: str | None = None) -> None:
        text = json.dumps(data, indent=2, sort_keys=self.sort_keys) + "\n"
        (self.out / (name or self.report)).write_text(text)

    @classmethod
    def parse_args(cls, *extra):
        doc = (cls.__doc__ or "").strip().splitlines()
        ap = argparse.ArgumentParser(description=doc[0] if doc else f"report for {cls.run}")
        ap.add_argument("--out", help="write the artifact here instead of the run directory")
        for args, kwargs in extra:
            ap.add_argument(*args, **kwargs)
        return ap.parse_args()
