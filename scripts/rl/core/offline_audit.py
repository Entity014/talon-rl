"""Auditing a finished run from its artifacts, with no simulator.

These read JSON or npz out of one or more run directories, compute something
over it, and write a report back. They are cheap to re-run, which is exactly
why `--out` matters: without it, re-running one to check it overwrites the
record it was being checked against.
"""

from __future__ import annotations

from rl.core.run_report import ARTIFACTS, REPO, RUNS, RunReport

__all__ = ["OfflineAudit", "ARTIFACTS", "REPO", "RUNS"]


class OfflineAudit(RunReport):
    report: str = ""     # "" means this audit only prints

    # --- what a subclass fills in ---

    def analyze(self) -> dict:
        raise NotImplementedError

    def summarize(self, report: dict) -> None:
        """Print whatever the script used to print. Default: nothing."""

    # --- shared ---

    def execute(self) -> dict:
        report = self.analyze()
        if self.report:
            self.write(report)
        self.summarize(report)
        return report

    @classmethod
    def main(cls) -> None:
        cls(cls.parse_args().out).execute()
