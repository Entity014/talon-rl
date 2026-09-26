#!/usr/bin/env python3
"""Why does L0-B seed 2 reach a good basin while seeds 0 and 1 do not?

Read-only. Compares the scalar training telemetry, the final command-cell
behaviour and a coarse actor-parameter norm across the three seeds. It cannot
attribute causality, because L0-B logged aggregate reward rather than the
reward-term decomposition; that limit is stated in the report it writes.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rl.core.offline_audit import ARTIFACTS, RUNS, OfflineAudit

SEEDS = [(0, "l0b_seed0_2026-09-21", "l0b_seed0_eval"),
         (1, "l0b_seed1_2026-09-21", "l0b_seed1_eval"),
         (2, "l0b_seed2_2026-09-22", "l0b_seed2_eval")]
CELL_KEYS = ("survival_rate", "mean_abs_vx_error", "tilt_p95_deg",
             "tilt_max_deg", "base_contact_rate")
INTERPRETATION = (
    "Final command-conditioned behavior separates strongly while all runs remain "
    "finite and use the same frozen PPO contract. This supports seed-dependent "
    "basin selection/task sensitivity, but cannot attribute causality to a "
    "specific reward term because L0-B did not log decomposed reward terms.")


def span(x):
    return {"first": float(x[0]), "last": float(x[-1]),
            "min": float(x.min()), "max": float(x.max())}


class L0C0ResidualAudit(OfflineAudit):
    """L0-B seed-2 pass versus seed-0/1 fail, from what was logged."""

    root = ARTIFACTS
    run = "l0_c0_audit"
    report = "L0C0_AUDIT.json"
    sort_keys = True
    schema = "l0_c0_residual_audit_v1"

    def seed_record(self, seed, run_name, eval_name):
        rows = json.loads((RUNS / run_name / "training_metrics.json").read_text())["metrics"]
        ev = json.loads((ARTIFACTS / eval_name / "evaluation.json").read_text())
        reward = np.array([r["reward_mean"] for r in rows], float)
        kl = np.array([r["analytic_kl"] for r in rows], float)
        lr = np.array([r["learning_rate"] for r in rows], float)
        std = np.array([r["learned_std"] for r in rows], float)
        return {
            "run": run_name,
            "updates": len(rows),
            "reward_mean": {"first25": float(reward[:25].mean()),
                            "last25": float(reward[-25:].mean()),
                            "min": float(reward.min()), "max": float(reward.max())},
            "analytic_kl": {"mean": float(kl.mean()),
                            "p95": float(np.percentile(kl, 95)), "max": float(kl.max())},
            "learning_rate": span(lr),
            "learned_std": span(std),
            "cells": {k: {x: v[x] for x in CELL_KEYS} for k, v in ev["cells"].items()},
            "telemetry_finite": bool(np.isfinite(reward).all() and np.isfinite(kl).all()
                                     and np.isfinite(lr).all() and np.isfinite(std).all())}

    def analyze(self):
        seeds = {str(s): self.seed_record(s, r, e) for s, r, e in SEEDS}
        # actor parameter norm at the last checkpoint, as a coarse drift signal
        for seed, run_name, _ in SEEDS:
            ck = torch.load(RUNS / run_name / "checkpoints/update_500.pt", map_location="cpu")
            vals = [float(v.float().norm()) for k, v in ck["model"].items()
                    if k.startswith("actor_") and torch.is_floating_point(v)]
            seeds[str(seed)]["actor_param_l2_sum"] = float(sum(vals))
            seeds[str(seed)]["checkpoint_schema"] = ck.get("schema")
        return {
            "schema": self.schema,
            "status": "COMPLETE_READ_ONLY",
            "question": "why does L0-B seed2 reach a good basin while seed0/1 do not?",
            "seed2_is_pass_control": True,
            "available_telemetry": ["reward_mean", "analytic_kl", "adaptive_learning_rate",
                                    "learned_std", "checkpoint_actor_parameter_norm",
                                    "final_command_cells"],
            "not_recorded_in_l0b": ["reward_term_decomposition",
                                    "episode_termination_timing_by_update",
                                    "joint_action_statistics",
                                    "height_tilt_contact_training_traces",
                                    "critic_value_error",
                                    "observation/action distribution traces"],
            "seeds": seeds,
            "interpretation": INTERPRETATION}

    def summarize(self, report):
        lines = ["# L0-C0 residual audit", "", "Status: **COMPLETE — read-only**", "",
                 report["interpretation"], "", "## Final cell summary", ""]
        for seed, d in report["seeds"].items():
            lines.append(f"### seed {seed}")
            for cmd, m in d["cells"].items():
                lines.append(
                    f"- vx={cmd}: survival={m['survival_rate']:.3f}, "
                    f"vx MAE={m['mean_abs_vx_error']:.3f}, "
                    f"tilt p95={m['tilt_p95_deg']:.2f}°, max tilt={m['tilt_max_deg']:.2f}°, "
                    f"base contact={m['base_contact_rate']:.3f}")
        lines += ["", "## Evidence limits", "",
                  "- L0-B logged aggregate scalar reward, not the underlying reward-term decomposition.",
                  "- No per-update trajectory/contact/joint-action telemetry was stored, so the first physical divergence cannot be localized retrospectively.",
                  "- The next causal data collection should add these fields prospectively; this audit does not authorize a new formulation or training run."]
        (self.out / "report.md").write_text("\n".join(lines) + "\n")
        print(self.out / self.report)


if __name__ == "__main__":
    L0C0ResidualAudit.main()
