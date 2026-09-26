#!/usr/bin/env python3
"""Layer 1 of the state-distribution-shift diagnostic (2026-09-20 branch,
follow-up to multi_update_trace.py's bearing_cos wander/consistency result).
That result narrowed the open question to: is the actor mean's tangential
drift each update explained by (A) the policy actually visiting a different
region of state space update-to-update, or (B)/(C) something else (advantage
valuation, running-stat normalizer) acting on a roughly stable state
distribution? This script only tests A, by comparing consecutive rollouts'
raw physical-state distributions -- it does NOT touch advantage/value
(Layer 2, later) or claim anything about the normalizer.

Pure analysis, no sim/env/model dependency -- reads the per-update raw state
descriptor .npz files multi_update_trace.py's --save_descriptors writes
(vx, height, vz, pitch, pitch_rate, contact_count, action_norm, done,
term_time_out, term_base_contact; each (T, N) per update). Descriptors come
straight from `transition` inside the real rollout, never from actor_obs
(actor_obs is obs_norm-normalized -- using it here would confound any shift
found with running-stat drift, exactly the variable this branch has not
implicated yet).

For every consecutive update pair (k, k+1):
  - SMD (standardized mean difference) per field -- |mu_k - mu_k1| /
    sqrt((std_k^2 + std_k1^2)/2), computed on the RAW values (each rollout's
    own local mean/std, so already scale-normalized by construction; no
    global rescaling needed for this one).
  - occupancy_distance -- mean over fields of total-variation distance
    between k's and k+1's histograms, binned against a FIXED reference (see
    _global_bin_edges: quantile tertiles pooled across ALL 50 updates of
    the run, computed once -- never refit per-pair, so a shift in occupancy
    reflects the states moving, not the bins moving).
  - clean_occupancy_distance -- same, but states within
    --fall_exclude_window steps of a term_base_contact event are dropped
    first (both directions, same windowing multi_update_trace.py's probe-set
    selection uses). term_base_contact specifically, NOT `done` or
    term_time_out -- a timeout isn't a fall, and conflating them would
    contaminate the "clean" set with exactly what it's meant to exclude.

If --log is given (the run's stdout, e.g. seed0_50update.log),
cos(Delta_mu_k+1, Delta_mu_k) ("cos(dk,dk-1)" column) is parsed per update
and reported alongside the state-shift metrics -- update k+1's row is the
bearing for the (k, k+1) pair. Without --log, that column is NaN and the
table still reports state-shift alone.

    PYTHONPATH="$(pwd):$(pwd)/scripts" python scripts/analysis/run_diagnostics.py compare-consecutive-rollouts \\
        /tmp/.../layer1/seed0_50update.npz --log /tmp/.../layer1/seed0_50update.log \\
        --out_csv /tmp/.../layer1/seed0_pairs.csv
"""

from __future__ import annotations

import argparse
import re

import numpy as np

FIELDS = ("vx", "height", "vz", "pitch", "pitch_rate", "contact_count", "action_norm")
N_BINS = 3  # ponytail: quantile tertiles, not hand-tuned physical thresholds -- revisit if a
            # reviewer wants domain-semantic bins (e.g. vx stable/slow/forward at 0.15/0.35)
FALL_EXCLUDE_WINDOW = 5  # same convention as full_update_trace.py / multi_update_trace.py probe set


def _load_updates(npz_path: str) -> dict[int, dict[str, np.ndarray]]:
    d = np.load(npz_path)
    updates: dict[int, dict[str, np.ndarray]] = {}
    for key in d.files:
        k_str, field = key.split("__", 1)
        updates.setdefault(int(k_str), {})[field] = d[key]
    return updates


def _parse_bearing_log(log_path: str) -> dict[int, float]:
    """Parses multi_update_trace.py's printed table for the cos(dk,dk-1)
    column, keyed by update index -- that update's own row IS the bearing
    for the (update-1, update) pair."""
    bearing: dict[int, float] = {}
    row_re = re.compile(
        r"^\s*(\d+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+(nan|[-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*$"
    )
    with open(log_path) as f:
        for line in f:
            m = row_re.match(line)
            if m:
                k = int(m.group(1))
                bearing_str = m.group(5)
                bearing[k] = float("nan") if bearing_str == "nan" else float(bearing_str)
    return bearing


def _fall_exclude_mask(term_base_contact: np.ndarray, window: int) -> np.ndarray:
    """(T, N) bool mask, True = drop (within `window` steps of a fall, either direction) --
    identical windowing to full_update_trace.py's fall_exclude (that script's lines ~180-186)."""
    mask = term_base_contact.astype(bool).copy()
    for shift in range(1, window + 1):
        mask[:-shift] |= term_base_contact[shift:].astype(bool)
        mask[shift:] |= term_base_contact[:-shift].astype(bool)
    return mask


def _smd(a: np.ndarray, b: np.ndarray) -> float:
    mu_a, mu_b = a.mean(), b.mean()
    std_a, std_b = a.std(), b.std()
    denom = np.sqrt((std_a**2 + std_b**2) / 2.0)
    if denom < 1e-8:
        return 0.0
    return float(abs(mu_a - mu_b) / denom)


def _global_bin_edges(updates: dict[int, dict[str, np.ndarray]], field: str) -> np.ndarray:
    """Tertile edges pooled across every update's every lane/step for this field --
    the FIXED reference every pair's histogram is binned against (never refit per-pair)."""
    pooled = np.concatenate([u[field].ravel() for u in updates.values()])
    return np.quantile(pooled, [1 / 3, 2 / 3])


def _occupancy_distance(a: np.ndarray, b: np.ndarray, edges: np.ndarray) -> float:
    """Total-variation distance between a's and b's histograms over the fixed `edges` bins."""
    if a.size == 0 or b.size == 0:
        return float("nan")
    hist_a = np.histogram(a, bins=[-np.inf, *edges, np.inf])[0].astype(np.float64)
    hist_b = np.histogram(b, bins=[-np.inf, *edges, np.inf])[0].astype(np.float64)
    hist_a /= max(hist_a.sum(), 1e-8)
    hist_b /= max(hist_b.sum(), 1e-8)
    return float(0.5 * np.abs(hist_a - hist_b).sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("descriptors", help="Path to the --save_descriptors .npz from multi_update_trace.py")
    parser.add_argument("--log", default=None, help="Path to that run's stdout log, for bearing_cos -- omit to skip")
    parser.add_argument("--fall_exclude_window", type=int, default=FALL_EXCLUDE_WINDOW)
    parser.add_argument("--out_csv", default=None)
    args = parser.parse_args()

    updates = _load_updates(args.descriptors)
    ks = sorted(updates.keys())
    print(f"loaded {len(ks)} updates ({ks[0]}..{ks[-1]}) from {args.descriptors}")

    bearing = _parse_bearing_log(args.log) if args.log else {}
    if args.log:
        print(f"parsed bearing_cos for {len(bearing)} updates from {args.log}")

    bin_edges = {field: _global_bin_edges(updates, field) for field in FIELDS}

    rows = []
    header = f"{'k':>3}{'k+1':>5}{'bearing_cos':>13}" + "".join(f"{'SMD_'+f:>14.13}" for f in FIELDS) + f"{'occ_dist':>10}{'clean_occ':>11}"
    print(header)

    for k in ks[:-1]:
        u_k, u_k1 = updates[k], updates[k + 1]
        bearing_cos = bearing.get(k + 1, float("nan"))

        smds = {f: _smd(u_k[f].ravel(), u_k1[f].ravel()) for f in FIELDS}
        occ = float(np.mean([_occupancy_distance(u_k[f].ravel(), u_k1[f].ravel(), bin_edges[f]) for f in FIELDS]))

        clean_occ = float("nan")
        if "term_base_contact" in u_k and "term_base_contact" in u_k1:
            excl_k = _fall_exclude_mask(u_k["term_base_contact"], args.fall_exclude_window)
            excl_k1 = _fall_exclude_mask(u_k1["term_base_contact"], args.fall_exclude_window)
            clean_dists = [
                _occupancy_distance(u_k[f][~excl_k].ravel(), u_k1[f][~excl_k1].ravel(), bin_edges[f])
                for f in FIELDS
            ]
            if not all(np.isnan(clean_dists)):
                clean_occ = float(np.nanmean(clean_dists))

        rows.append({"k": k, "k1": k + 1, "bearing_cos": bearing_cos, **{f"SMD_{f}": smds[f] for f in FIELDS},
                      "occupancy_distance": occ, "clean_occupancy_distance": clean_occ})
        print(
            f"{k:>3}{k+1:>5}{bearing_cos:>13.5f}" + "".join(f"{smds[f]:>14.5f}" for f in FIELDS)
            + f"{occ:>10.4f}{clean_occ:>11.4f}"
        )

    mean_smd = np.array([np.mean([r[f"SMD_{f}"] for f in FIELDS]) for r in rows])
    abs_bearing = np.array([abs(r["bearing_cos"]) for r in rows])
    valid = ~np.isnan(abs_bearing)
    print(f"\n=== summary over {len(rows)} update pairs ===")
    print(f"mean SMD (all fields, all pairs): {mean_smd.mean():.4f}")
    print(f"mean occupancy_distance: {np.mean([r['occupancy_distance'] for r in rows]):.4f}")
    clean_vals = [r["clean_occupancy_distance"] for r in rows if not np.isnan(r["clean_occupancy_distance"])]
    if clean_vals:
        print(f"mean clean_occupancy_distance: {np.mean(clean_vals):.4f}")
    if valid.sum() >= 3:
        r_pearson = float(np.corrcoef(mean_smd[valid], abs_bearing[valid])[0, 1])
        print(f"corr(mean_SMD, |bearing_cos|) over {valid.sum()} pairs with a logged bearing: {r_pearson:.3f}")
        print("(not a claim of causation -- just what Layer 1's evidence-for-A read wants: does state shift")
        print(" track bearing change, per this branch's Case A/B/C read.)")
    else:
        print("fewer than 3 pairs have a logged bearing_cos -- pass --log to get the bearing correlation")

    if args.out_csv:
        import csv
        fieldnames = list(rows[0].keys())
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {len(rows)} rows -> {args.out_csv}")


def demo() -> None:
    """Self-check: synthetic 3-update trace where update 2 is a shifted copy of
    update 1 (known SMD/occupancy shift) and update 3 == update 2 (should read ~0)."""
    rng = np.random.default_rng(0)
    T, N = 24, 64
    updates = {}
    for k, shift in [(1, 0.0), (2, 2.0), (3, 2.0)]:
        updates[k] = {
            "vx": rng.normal(shift, 0.2, size=(T, N)).astype(np.float32),
            "height": rng.normal(0.3, 0.02, size=(T, N)).astype(np.float32),
            "vz": rng.normal(0.0, 0.05, size=(T, N)).astype(np.float32),
            "pitch": rng.normal(0.0, 0.05, size=(T, N)).astype(np.float32),
            "pitch_rate": rng.normal(0.0, 0.1, size=(T, N)).astype(np.float32),
            "contact_count": rng.integers(0, 5, size=(T, N)).astype(np.float32),
            "action_norm": rng.normal(6.0, 1.0, size=(T, N)).astype(np.float32),
            "term_base_contact": np.zeros((T, N), dtype=np.float32),
        }
    bin_edges = {field: _global_bin_edges(updates, field) for field in FIELDS}

    smd_1_2 = _smd(updates[1]["vx"].ravel(), updates[2]["vx"].ravel())
    smd_2_3 = _smd(updates[2]["vx"].ravel(), updates[3]["vx"].ravel())
    assert smd_1_2 > 5.0, f"expected a large SMD for a mean-shifted field, got {smd_1_2}"
    assert smd_2_3 < 0.2, f"expected ~0 SMD for identical distributions, got {smd_2_3}"

    occ_1_2 = _occupancy_distance(updates[1]["vx"].ravel(), updates[2]["vx"].ravel(), bin_edges["vx"])
    occ_2_3 = _occupancy_distance(updates[2]["vx"].ravel(), updates[3]["vx"].ravel(), bin_edges["vx"])
    assert occ_1_2 > occ_2_3, "shifted-distribution occupancy distance should exceed identical-distribution's"

    mask = _fall_exclude_mask(np.zeros((10, 2), dtype=np.float32), window=5)
    assert not mask.any(), "no falls in input -- exclude mask should be all-False"
    fall = np.zeros((10, 2), dtype=np.float32)
    fall[5, 0] = 1.0
    mask = _fall_exclude_mask(fall, window=2)
    assert mask[3:8, 0].all() and not mask[:3, 0].any() and not mask[8:, 0].any(), "window should be +/-2 around the fall step, lane 0 only"
    assert not mask[:, 1].any(), "lane 1 had no fall -- must stay unmasked"

    print("demo() OK: SMD/occupancy/fall-exclusion self-checks passed")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo()
    else:
        main()
