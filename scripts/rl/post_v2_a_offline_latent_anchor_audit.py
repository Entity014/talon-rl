#!/usr/bin/env python3
"""V2-A offline latent-anchor and simplex-continuity audit.

This script intentionally does not import Isaac Lab, instantiate an actor, or
modify any checkpoint.  It uses only the frozen D1 specialist behavior report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LABELS = ("P", "B", "E")
PREFS = {
    "P": np.array([0.8, 0.1, 0.1], dtype=np.float64),
    "B": np.array([0.1, 0.8, 0.1], dtype=np.float64),
    "E": np.array([0.1, 0.1, 0.8], dtype=np.float64),
}
GRID = [
    [1 / 3, 1 / 3, 1 / 3],
    [0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8],
    [0.45, 0.45, 0.10], [0.45, 0.10, 0.45], [0.10, 0.45, 0.45],
    [0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6],
]


def descriptor(row: dict) -> np.ndarray:
    m = row["metrics_mean"]
    return np.asarray(
        list(row["objective_return_mean"])
        + [m["vx_error"], m["tilt_deg"], m["ang_vel_xy"],
           m["torque_norm"], m["action_rate"], m["survival"],
           row["action_norm_mean"]],
        dtype=np.float64,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=ROOT / "runs/post_v1_d1-2026-09-22/d1.json")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    source = json.loads(args.input.read_text())
    rows = source["behavior"]
    X = np.asarray([descriptor(r) for r in rows], dtype=np.float64)
    labels = np.asarray([r["specialist"] for r in rows])
    finite_input = bool(np.isfinite(X).all())
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale < 1e-12] = 1.0
    Xs = (X - mean) / scale
    centered = Xs - Xs.mean(axis=0)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2]
    Z = centered @ components.T
    anchors = {label: Z[labels == label].mean(axis=0) for label in LABELS}
    anchor_matrix = np.stack([anchors[label] for label in LABELS])
    pairwise = {}
    for i, a in enumerate(LABELS):
        for b in LABELS[i + 1:]:
            pairwise[f"{a}_vs_{b}"] = float(np.linalg.norm(anchors[a] - anchors[b]))
    max_pair = max(pairwise.values()) if pairwise else 0.0

    grid_rows = []
    for w in GRID:
        wv = np.asarray(w, dtype=np.float64)
        z = sum(wv[i] * anchors[label] for i, label in enumerate(LABELS))
        d = np.linalg.norm(anchor_matrix - z[None, :], axis=1)
        grid_rows.append({
            "w": w,
            "z_ref": z.tolist(),
            "finite": bool(np.isfinite(z).all()),
            "nearest_anchor": LABELS[int(np.argmin(d))],
            "nearest_anchor_distance": float(d.min()),
            "normalized_nearest_distance": float(d.min() / (max_pair + 1e-12)),
            "max_weight": float(max(wv)),
        })

    # Affine continuity audit over the frozen grid.  The ratio should be
    # stable and finite; a jump is impossible under the declared barycentric
    # map, but is still checked explicitly in the generated artifact.
    lipschitz = []
    jumps = 0
    for i, a in enumerate(GRID):
        za = np.asarray(grid_rows[i]["z_ref"])
        for j in range(i + 1, len(GRID)):
            b = GRID[j]
            dz = np.linalg.norm(za - np.asarray(grid_rows[j]["z_ref"]))
            dw = np.linalg.norm(np.asarray(a) - np.asarray(b))
            if dw > 1e-12:
                lipschitz.append(float(dz / dw))
                if not np.isfinite(dz / dw):
                    jumps += 1

    # Conservative structural gate: anchors must be finite and non-collapsed;
    # P must separate from the empirically close B/E pair.  B/E separation is
    # reported but is not required to be large because D1/D5A found it weak.
    p_separation = min(pairwise.get("P_vs_B", 0.0), pairwise.get("P_vs_E", 0.0))
    criteria = {
        "finite_descriptors_and_anchors": finite_input and bool(np.isfinite(anchor_matrix).all()),
        "anchor_rank_nonzero": bool(max_pair > 1e-6),
        "P_separates_from_B_E": bool(p_separation > 0.05),
        "simplex_mapping_finite": all(r["finite"] for r in grid_rows),
        "no_continuity_jump": jumps == 0,
        "latent_dimension_is_two": components.shape == (2, X.shape[1]),
        "intermediate_grid_not_collapsed": all(
            r["normalized_nearest_distance"] > 0.05
            for r in grid_rows if r["max_weight"] < 0.8
        ),
    }
    report = {
        "schema": "post_v2_a_offline_latent_anchor_audit_v1",
        "status": "PASS" if all(criteria.values()) else "FAIL",
        "measurement_only": True,
        "source": str(args.input),
        "descriptor_fields": ["progress_return", "balance_return", "efficiency_return", "vx_error", "tilt_deg", "ang_vel_xy", "torque_norm", "action_rate", "survival", "action_norm"],
        "d1_only_standardization": {"mean": mean.tolist(), "scale": scale.tolist()},
        "pca": {"components": components.tolist(), "singular_values": singular.tolist(), "explained_variance_ratio": ((singular ** 2) / max(float((singular ** 2).sum()), 1e-12)).tolist()},
        "anchors": {label: anchors[label].tolist() for label in LABELS},
        "anchor_pairwise_distance": pairwise,
        "grid": grid_rows,
        "continuity": {"max_lipschitz_ratio": float(max(lipschitz)), "min_lipschitz_ratio": float(min(lipschitz)), "jump_count": jumps, "pair_count": len(lipschitz)},
        "criteria": criteria,
        "interpretation": "This audit validates only the provenance and geometry of the proposed latent guide; it does not establish locomotion semantics because no V2 actor was evaluated.",
        "next_gate": "Authorize V2 implementation only if PASS; otherwise stop before actor training.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "criteria": criteria, "anchors": report["anchors"]}, indent=2))


if __name__ == "__main__":
    main()
