#!/usr/bin/env python3
"""Did B0.1 ever learn deterministic locomotion, or never learn it at all?

Read-only over the frozen monitor artifacts; never opens a simulator. Plots
each acceptance metric against the training update for all three seeds and
marks the final-gate thresholds, so a transient pass is visible as such.
"""
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rl.core.offline_audit import ARTIFACTS, RUNS, OfflineAudit

UPDATES = list(range(0, 501, 25))
SEEDS = range(3)
GATE = {"survival": .9, "vx_mae": .15, "tilt_p95_deg": 15, "tilt_max_deg": 30}
METRICS = [("survival", "Survival"), ("first_fall_mean", "Mean first-fall step"),
           ("vx_mae", "vx MAE (m/s)"), ("displacement", "Mean displacement (m)"),
           ("tilt_p95_deg", "Tilt p95 (deg)"), ("tilt_max_deg", "Tilt max (deg)"),
           ("height_error", "Height error (m)"), ("std", "Scheduled std")]
INTERPRETATION = {
    "answer": "B0.1 did learn deterministic capability transiently; it did not simply fail to learn locomotion.",
    "evidence": "seed1 passes every final-gate metric at update 25; seed0 passes every final-gate metric at updates 400, 425, and 450, then collapses at 475 while std is already fixed at 0.10.",
    "caveat": "seed2 never satisfies all metrics simultaneously, so the instability is seed-sensitive rather than a uniformly solved locomotion task.",
    "next_decision_scope": "Investigate read-only policy/training-stability evidence before any new formulation; do not retune reward or std from this audit alone."}


def scheduled_std(u):
    """The std schedule B0.1 trained under: hold, then anneal over updates 100-400."""
    return .82 - .72 * min(max((u - 100) / 300, 0), 1)


class B01LearningCurveAudit(OfflineAudit):
    """B0.1 deterministic learning-curve audit over the frozen monitor artifacts."""

    root = ARTIFACTS
    run = "b0_1_learning_curve_audit"
    report = "summary.json"
    sort_keys = True
    schema = "b0_1_learning_curve_audit_v1"

    def row(self, seed, update):
        a = json.loads((RUNS / f"b0_1_seed{seed}_2026-09-21" / "monitor"
                        / f"u{update:03d}.json").read_text())["acceptance"]
        row = {"seed": seed, "update": update, "std": scheduled_std(update),
               "survival": a["survival_rate"],
               "first_fall_mean": sum(a["first_fall"]) / len(a["first_fall"]),
               "vx_mae": a["mean_abs_vx_error"],
               "displacement": a["mean_displacement"],
               "tilt_p95_deg": math.degrees(a["tilt_p95_rad"]),
               "tilt_max_deg": math.degrees(a["tilt_max_rad"]),
               "height_error": a["mean_height_error"]}
        row["gate_pass"] = (row["survival"] >= GATE["survival"]
                            and row["vx_mae"] <= GATE["vx_mae"]
                            and row["tilt_p95_deg"] <= GATE["tilt_p95_deg"]
                            and row["tilt_max_deg"] <= GATE["tilt_max_deg"])
        return row

    def analyze(self):
        self.rows = []
        summary = {"schema": self.schema, "read_only": True, "seeds": []}
        for seed in SEEDS:
            curve = [self.row(seed, u) for u in UPDATES]
            self.rows.extend(curve)
            passing = [r["update"] for r in curve if r["gate_pass"]]
            peak = max(curve, key=lambda r: (r["gate_pass"], r["survival"],
                                             -r["vx_mae"], -r["tilt_p95_deg"]))
            collapse = (max(passing) + 25) if passing and max(passing) < 500 else None
            summary["seeds"].append({
                "seed": seed, "full_gate_updates": passing, "best_update": peak["update"],
                "best": peak, "collapse_after_last_full_gate": collapse,
                "collapse_std_phase": ("post-anneal plateau" if collapse and collapse >= 425
                                       else "std hold/anneal" if collapse else None)})
        summary["interpretation"] = INTERPRETATION
        return summary

    def summarize(self, summary):
        with (self.out / "learning_curve.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        self.plot()
        lines = ["# B0.1 deterministic learning-curve audit", "",
                 "Read-only analysis of the frozen monitor artifacts.", "",
                 "## Finding", "", summary["interpretation"]["answer"], "",
                 summary["interpretation"]["evidence"], "",
                 summary["interpretation"]["caveat"], "",
                 "## Per-seed full-gate windows", ""]
        for item in summary["seeds"]:
            lines.append(f"- Seed {item['seed']}: {item['full_gate_updates'] or 'none'}; "
                         f"best update {item['best_update']}; collapse after last full gate: "
                         f"{item['collapse_after_last_full_gate']}.")
        (self.out / "report.md").write_text("\n".join(lines) + "\n")
        print(self.out)

    def plot(self):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(4, 2, figsize=(14, 14), sharex=True)
        for ax, (key, label) in zip(axes.flat, METRICS):
            for seed in SEEDS:
                r = [x for x in self.rows if x["seed"] == seed]
                ax.plot([x["update"] for x in r], [x[key] for x in r],
                        marker="o", label=f"seed {seed}")
            if key in GATE:
                ax.axhline(GATE[key], color="black", linestyle="--", linewidth=1)
            ax.set_title(label)
            ax.grid(alpha=.3)
            ax.legend(fontsize=8)
        for ax in axes[-1]:
            ax.set_xlabel("Training update")
        fig.tight_layout()
        fig.savefig(self.out / "learning_curve.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    B01LearningCurveAudit.main()
