#!/usr/bin/env python3
"""Joint-wise saturation and lane-level covariance over the C2R arrays.

Read-only and descriptive: per joint it reports the signed action, how often
and how long it saturates, the gap between lanes that fell and lanes that did
not, how saturation co-varies with tilt/height/tracking/contact, and a 50-step
summary before each first fall. It decides nothing.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rl.core.offline_audit import ARTIFACTS, OfflineAudit

SEEDS = (0, 1, 2)
UPDATE = 500
SATURATION = 2.99
PRECURSOR_STEPS = 50


def quantiles(x):
    x = np.asarray(x, float)
    return {"p50": float(np.percentile(x, 50)), "p95": float(np.percentile(x, 95)),
            "max": float(np.max(x))}


def longest_streak(v):
    best = cur = 0
    for x in v:
        cur = cur + 1 if x else 0
        best = max(best, cur)
    return best


class L0C2JJointGeometryAudit(OfflineAudit):
    """Per-joint saturation geometry at update 500, across the three C2R seeds."""

    root = ARTIFACTS
    run = "l0c2j_audit"
    report = "L0C2J_AUDIT.json"
    sort_keys = True
    schema = "l0c2j_joint_geometry_audit_v1"

    def joint_metrics(self, a, sat, fail, j):
        duration = np.array([longest_streak(sat[:, lane, j]) for lane in range(sat.shape[1])])
        any_fail, any_surv = fail.any(), (~fail).any()
        return {
            "joint": j,
            "abs_action": quantiles(np.abs(a[:, :, j]).reshape(-1)),
            "signed_mean": float(a[:, :, j].mean()),
            "signed_p05": float(np.percentile(a[:, :, j], 5)),
            "signed_p95": float(np.percentile(a[:, :, j], 95)),
            "saturation_fraction": float(sat[:, :, j].mean()),
            "lane_max_streak_p95": float(np.percentile(duration, 95)),
            "failed_sat_fraction": float(sat[:, fail, j].mean()) if any_fail else None,
            "survived_sat_fraction": float(sat[:, ~fail, j].mean()) if any_surv else None,
            "failure_delta": (float(sat[:, fail, j].mean() - sat[:, ~fail, j].mean())
                              if any_fail and any_surv else None)}

    def seed(self, path):
        z = np.load(path)
        a, done = z["action"], z["done"].astype(bool)
        sat = np.abs(a) >= SATURATION
        roll, pitch, height, vx = z["roll"], z["pitch"], z["height"], z["vx"]
        contact = z["contact"].astype(bool)
        out = {}
        for ci, cmd in enumerate(z["commands"]):
            aa, ss = a[ci], sat[ci]
            fail = done[ci].any(0)
            tilt = np.maximum(np.abs(roll[ci]), np.abs(pitch[ci]))
            joints = [self.joint_metrics(aa, ss, fail, j) for j in range(aa.shape[-1])]
            co = ss.mean(-1)
            pair = np.corrcoef(np.stack([co.reshape(-1),
                                         np.abs(roll[ci]).reshape(-1),
                                         np.abs(pitch[ci]).reshape(-1),
                                         height[ci].reshape(-1),
                                         np.abs(vx[ci] - cmd[0]).reshape(-1),
                                         contact[ci].reshape(-1)]))
            pre = []
            for lane in range(ss.shape[1]):
                idx = np.flatnonzero(done[ci, :, lane])
                if not idx.size:
                    continue
                t = int(idx[0])
                lo = max(0, t - PRECURSOR_STEPS)
                pre.append({"lane": lane, "first_fall": t,
                            "sat_mean_last50": float(ss[lo:t, lane].mean()),
                            "tilt_start": float(np.degrees(tilt[lo, lane])),
                            "tilt_end": float(np.degrees(tilt[max(t - 1, lo), lane])),
                            "height_start": float(height[ci, lo, lane]),
                            "height_end": float(height[ci, max(t - 1, lo), lane])})
            out[str(float(cmd[0]))] = {"joint_metrics": joints,
                                       "co_saturation_correlation": pair.tolist(),
                                       "co_saturation_pattern_p95": quantiles(co.reshape(-1)),
                                       "failed_lanes": int(fail.sum()),
                                       "precursor_last50": pre}
        return out

    def analyze(self):
        return {"schema": self.schema, "status": "COMPLETE_READ_ONLY",
                "checkpoint_update": UPDATE,
                "seeds": {str(s): self.seed(ARTIFACTS / f"l0c2r_seed{s}" / f"update_{UPDATE:03d}.npz")
                          for s in SEEDS}}

    def summarize(self, report):
        lines = ["# L0-C2J joint-wise geometry audit", "",
                 "Status: **COMPLETE — read-only**", "",
                 "This audit uses raw C2R arrays at update 500. It reports signed per-joint "
                 "action, saturation fraction/streak, survivor/fall deltas, co-saturation "
                 "correlations, and 50-step pre-fall summaries.", ""]
        for s in SEEDS:
            lines.append(f"## seed {s}")
            for cmd, m in report["seeds"][str(s)].items():
                ranked = sorted((x for x in m["joint_metrics"] if x["failure_delta"] is not None),
                                key=lambda x: x["failure_delta"], reverse=True)[:4]
                lines.append(
                    f"- vx={cmd}: failed lanes={m['failed_lanes']}, "
                    f"co-sat p95={m['co_saturation_pattern_p95']['p95']:.3f}, "
                    "top failed-minus-survivor joints="
                    + ", ".join(f"j{x['joint']}:{x['failure_delta']:.3f}" for x in ranked))
        lines += ["", "No intervention decision is made by this descriptive audit."]
        (self.out / "report.md").write_text("\n".join(lines) + "\n")
        print(self.out / self.report)


if __name__ == "__main__":
    L0C2JJointGeometryAudit.main()
