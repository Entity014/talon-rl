#!/usr/bin/env python3
"""Layer 2 of the state-distribution-shift diagnostic (2026-09-20 branch,
follow-up to compare_consecutive_rollouts.py's Layer 1 result: state shift
between consecutive rollouts is real but does NOT track bearing_cos change
-- Case C, ruling out the simple "policy moves -> sees new states -> gradient
follows" story). Layer 2 asks a different question: when two consecutive
rollouts DO put the policy in the same coarse physical-state region, does
that region get a different value/advantage assignment between k and k+1?
If comparable states get incomparable valuations, that's evidence FOR the
critic/advantage-geometry mechanism (Hypothesis B from the 2026-09-20
attractor-signature audit); if not, neither state-shift nor simple
valuation-shift explains the bearing diffusion.

Pure analysis, no sim/env/model dependency -- reads the same
--save_descriptors .npz multi_update_trace.py now writes once
trainer._collect_rollout is also wrapped (values, rewards, terminal_fall,
final_value, alongside Layer 1's raw state fields). advantage/return are
NOT stored -- recomputed here with the exact gae_per_objective() training
itself calls, from the raw ingredients, so there is no implementation
discrepancy with what update() actually did.

Comparable-state definition (locked 2026-09-20): a coarse state region over
z = [vx, height, vz, pitch, pitch_rate, contact_count] -- action_norm is
EXCLUDED deliberately (it's the policy's own output, not physical state;
gating on it would select for states where the new policy already acts
similarly to the old one, circular against the question this asks). Region
edges are fixed quantile tertiles/bisections pooled across the WHOLE run
(never refit per-pair), same principle as Layer 1's occupancy bins. A region
is only used for a pair if BOTH k and k+1 have >= --min_region_n samples in
it -- otherwise the comparison is noise, not signal.

Termination stratification (separate from the region grouping, locked after
Layer 1 found a timeout-rate confounder fall_exclude_window alone couldn't
see): every sample is exactly one of normal / timeout / fall / pre_fall.
Only `normal` samples enter the region-matched valuation comparison --
fall/timeout/pre_fall states have structurally different value by
definition (episode-ending or about to), comparing them across updates
would answer a different question than "does the SAME kind of ongoing
locomotion get valued differently". Their population fractions are reported
per update for visibility, not fed into the SMD comparison.

Never timestep-aligns k against k+1 directly -- trajectory phase doesn't
correspond across two different policies' rollouts. Only region-conditioned
aggregate comparison.

SMD_value / SMD_return / SMD_advantage reuse Layer 1's standardized-mean-
difference construction (|mean_k - mean_k+1| / pooled std) -- which is
already exactly "between-update shift" normalized by within-rollout
variability, so no separate baseline-comparison step is needed: SMD < ~1
means the shift is within within-rollout noise, SMD >> 1 means it isn't.

    PYTHONPATH="$(pwd):$(pwd)/scripts" python scripts/analysis/run_diagnostics.py compare-valuation \\
        /tmp/.../layer1/seed0_50update_v2.npz 0.7 0.1 0.1 0.1 \\
        --log /tmp/.../layer1/seed0_50update_v2.log --out_csv /tmp/.../layer1/seed0_valuation_pairs.csv
"""

from __future__ import annotations

import argparse
import re

import numpy as np

from rl.core.rollout.gae_functional import gae_per_objective

Z_FIELDS = ("vx", "height", "vz", "pitch", "pitch_rate", "contact_count")  # action_norm excluded -- see module docstring
PREFALL_WINDOW = 5  # same convention as Layer 1's fall_exclude_window


def _load_updates(npz_path: str) -> dict[int, dict[str, np.ndarray]]:
    d = np.load(npz_path)
    updates: dict[int, dict[str, np.ndarray]] = {}
    for key in d.files:
        k_str, field = key.split("__", 1)
        updates.setdefault(int(k_str), {})[field] = d[key]
    return updates


def _parse_bearing_log(log_path: str) -> dict[int, float]:
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


def _termination_categories(term_base_contact: np.ndarray, term_time_out: np.ndarray, window: int) -> dict[str, np.ndarray]:
    """Every (t,n) sample gets exactly one of these four masks."""
    fall = term_base_contact.astype(bool)
    timeout = term_time_out.astype(bool)
    pre_fall = np.zeros_like(fall)
    T = fall.shape[0]
    for t in range(T):
        cols = np.nonzero(fall[t])[0]
        if cols.size == 0:
            continue
        lo = max(0, t - window)
        pre_fall[lo:t, cols] = True
    pre_fall &= ~fall  # the fall step itself is `fall`, not `pre_fall`
    normal = ~(fall | timeout | pre_fall)
    return {"fall": fall, "timeout": timeout, "pre_fall": pre_fall, "normal": normal}


def _global_bin_edges(updates: dict[int, dict[str, np.ndarray]], field: str, n_bins: int) -> np.ndarray:
    pooled = np.concatenate([u[field].ravel() for u in updates.values()])
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    return np.quantile(pooled, qs)


def _region_ids(z: dict[str, np.ndarray], edges: dict[str, np.ndarray]) -> np.ndarray:
    """z[field] and edges[field] -> one int region id per flattened sample (digitize each
    field, then combine with a mixed-radix encoding -- same region id iff every field's bin matches)."""
    combined = np.zeros(next(iter(z.values())).size, dtype=np.int64)
    for field in Z_FIELDS:
        bins = np.digitize(z[field].ravel(), edges[field])  # 0..n_bins-1
        n_bins = len(edges[field]) + 1
        combined = combined * n_bins + bins
    return combined


def _smd(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    mu_a, mu_b = a.mean(), b.mean()
    std_a, std_b = a.std(), b.std()
    denom = np.sqrt((std_a**2 + std_b**2) / 2.0)
    if denom < 1e-8:
        return 0.0
    return float(abs(mu_a - mu_b) / denom)


def _compute_scalars(u: dict[str, np.ndarray], w: np.ndarray, gamma: float, gae_lambda: float) -> dict[str, np.ndarray]:
    """gae_per_objective + returns, scalarized by the run's fixed preference w
    (dot product) -- exactly the same function/formula update() uses internally,
    called here post-hoc (see module docstring)."""
    values_with_final = np.concatenate([u["values"], u["final_value"][None]], axis=0)
    adv = gae_per_objective(u["rewards"], values_with_final, u["terminal_fall"], gamma, gae_lambda)
    returns = adv + u["values"]
    return {
        "value_scalar": (u["values"] * w).sum(-1),
        "return_scalar": (returns * w).sum(-1),
        "adv_scalar": (adv * w).sum(-1),
        "adv_per_objective": adv,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("descriptors", help="Path to a combined Layer1+Layer2 --save_descriptors .npz from multi_update_trace.py")
    parser.add_argument("w", type=float, nargs=4, help="The fixed preference vector this run used (RewardVectorCfg.term_names order) -- must match the multi_update_trace.py invocation, used only to scalarize adv/value/return")
    parser.add_argument("--gamma", type=float, default=0.998, help="Must match MOPPOConfig.gamma used during the run (default matches MOPPOConfig's own default)")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="Must match MOPPOConfig.gae_lambda used during the run")
    parser.add_argument("--log", default=None)
    parser.add_argument("--n_bins", type=int, default=3, help="Quantile bins per z-field (region granularity)")
    parser.add_argument("--prefall_window", type=int, default=PREFALL_WINDOW)
    parser.add_argument("--min_region_n", type=int, default=20, help="Minimum normal-category samples a region needs in BOTH k and k+1 to be compared")
    parser.add_argument("--out_csv", default=None)
    args = parser.parse_args()

    w = np.array(args.w, dtype=np.float32)
    w = w / w.sum()

    updates = _load_updates(args.descriptors)
    ks = sorted(updates.keys())
    print(f"loaded {len(ks)} updates ({ks[0]}..{ks[-1]}) from {args.descriptors}")

    bearing = _parse_bearing_log(args.log) if args.log else {}
    if args.log:
        print(f"parsed bearing_cos for {len(bearing)} updates from {args.log}")

    bin_edges = {field: _global_bin_edges(updates, field, args.n_bins) for field in Z_FIELDS}

    per_update = {}
    for k in ks:
        u = updates[k]
        cats = _termination_categories(u["term_base_contact"], u["term_time_out"], args.prefall_window)
        scalars = _compute_scalars(u, w, args.gamma, args.gae_lambda)
        region_ids = _region_ids({f: u[f] for f in Z_FIELDS}, bin_edges)
        per_update[k] = {"cats": cats, "scalars": scalars, "region_ids": region_ids}

    print("\ntermination-category population fractions per update (normal/timeout/fall/pre_fall):")
    for k in ks:
        cats = per_update[k]["cats"]
        total = cats["normal"].size
        print(f"  upd {k:>3}: {cats['normal'].sum()/total:>6.3f} / {cats['timeout'].sum()/total:>6.3f} "
              f"/ {cats['fall'].sum()/total:>6.3f} / {cats['pre_fall'].sum()/total:>6.3f}")

    rows = []
    header = f"{'k':>3}{'k+1':>5}{'bearing_cos':>13}{'n_region':>10}{'SMD_value':>12}{'SMD_return':>12}{'SMD_adv':>12}{'Dfrac_pos':>11}"
    print(f"\n{header}")

    for k in ks[:-1]:
        pu_k, pu_k1 = per_update[k], per_update[k + 1]
        normal_k = pu_k["cats"]["normal"].ravel()
        normal_k1 = pu_k1["cats"]["normal"].ravel()
        rid_k, rid_k1 = pu_k["region_ids"][normal_k], pu_k1["region_ids"][normal_k1]

        val_k, val_k1 = pu_k["scalars"]["value_scalar"].ravel()[normal_k], pu_k1["scalars"]["value_scalar"].ravel()[normal_k1]
        ret_k, ret_k1 = pu_k["scalars"]["return_scalar"].ravel()[normal_k], pu_k1["scalars"]["return_scalar"].ravel()[normal_k1]
        adv_k, adv_k1 = pu_k["scalars"]["adv_scalar"].ravel()[normal_k], pu_k1["scalars"]["adv_scalar"].ravel()[normal_k1]

        shared_regions = np.intersect1d(np.unique(rid_k), np.unique(rid_k1))
        smd_v, smd_r, smd_a, dfrac_pos = [], [], [], []
        n_matched = 0
        for rid in shared_regions:
            mask_k, mask_k1 = rid_k == rid, rid_k1 == rid
            if mask_k.sum() < args.min_region_n or mask_k1.sum() < args.min_region_n:
                continue
            n_matched += 1
            smd_v.append(_smd(val_k[mask_k], val_k1[mask_k1]))
            smd_r.append(_smd(ret_k[mask_k], ret_k1[mask_k1]))
            smd_a.append(_smd(adv_k[mask_k], adv_k1[mask_k1]))
            frac_pos_k = float((adv_k[mask_k] > 0).mean())
            frac_pos_k1 = float((adv_k1[mask_k1] > 0).mean())
            dfrac_pos.append(abs(frac_pos_k1 - frac_pos_k))

        bearing_cos = bearing.get(k + 1, float("nan"))
        row = {
            "k": k, "k1": k + 1, "bearing_cos": bearing_cos, "n_region_matched": n_matched,
            "SMD_value": float(np.mean(smd_v)) if smd_v else float("nan"),
            "SMD_return": float(np.mean(smd_r)) if smd_r else float("nan"),
            "SMD_advantage": float(np.mean(smd_a)) if smd_a else float("nan"),
            "delta_frac_positive_adv": float(np.mean(dfrac_pos)) if dfrac_pos else float("nan"),
        }
        rows.append(row)
        print(
            f"{k:>3}{k+1:>5}{bearing_cos:>13.5f}{n_matched:>10}"
            f"{row['SMD_value']:>12.4f}{row['SMD_return']:>12.4f}{row['SMD_advantage']:>12.4f}{row['delta_frac_positive_adv']:>11.4f}"
        )

    smd_a_arr = np.array([r["SMD_advantage"] for r in rows])
    abs_bearing = np.array([abs(r["bearing_cos"]) for r in rows])
    valid = ~np.isnan(smd_a_arr) & ~np.isnan(abs_bearing)
    print(f"\n=== summary over {len(rows)} update pairs ===")
    print(f"mean n_region_matched: {np.mean([r['n_region_matched'] for r in rows]):.1f}")
    print(f"mean SMD_value: {np.nanmean([r['SMD_value'] for r in rows]):.4f}")
    print(f"mean SMD_return: {np.nanmean([r['SMD_return'] for r in rows]):.4f}")
    print(f"mean SMD_advantage: {np.nanmean([r['SMD_advantage'] for r in rows]):.4f}")
    if valid.sum() >= 3:
        r_pearson = float(np.corrcoef(smd_a_arr[valid], abs_bearing[valid])[0, 1])
        print(f"corr(SMD_advantage, |bearing_cos|) over {valid.sum()} pairs: {r_pearson:.3f}")
    else:
        print("fewer than 3 pairs have both a matched region and a logged bearing -- pass --log / lower --min_region_n")

    if args.out_csv:
        import csv
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {len(rows)} rows -> {args.out_csv}")


def demo() -> None:
    """Self-check: synthetic 2-update trace, same region occupancy both updates
    (so Layer 1 would read ~0 state shift), but update 2's value in that region is
    shifted by a known amount -- SMD_value must catch it even though states match."""
    rng = np.random.default_rng(0)
    T, N = 24, 64
    K = 4
    w = np.array([0.7, 0.1, 0.1, 0.1], dtype=np.float32)

    def make_update(value_shift: float) -> dict[str, np.ndarray]:
        return {
            "vx": rng.normal(0.3, 0.05, size=(T, N)).astype(np.float32),
            "height": rng.normal(0.3, 0.02, size=(T, N)).astype(np.float32),
            "vz": rng.normal(0.0, 0.02, size=(T, N)).astype(np.float32),
            "pitch": rng.normal(0.0, 0.02, size=(T, N)).astype(np.float32),
            "pitch_rate": rng.normal(0.0, 0.05, size=(T, N)).astype(np.float32),
            "contact_count": rng.integers(2, 4, size=(T, N)).astype(np.float32),
            "term_base_contact": np.zeros((T, N), dtype=np.float32),
            "term_time_out": np.zeros((T, N), dtype=np.float32),
            "terminal_fall": np.zeros((T, N), dtype=bool),
            "values": (value_shift + rng.normal(0.0, 0.05, size=(T, N, K))).astype(np.float32),
            "rewards": rng.normal(0.0, 0.1, size=(T, N, K)).astype(np.float32),
            "final_value": (value_shift + rng.normal(0.0, 0.05, size=(N, K))).astype(np.float32),
        }

    updates = {1: make_update(1.0), 2: make_update(1.0), 3: make_update(5.0)}
    bin_edges = {field: _global_bin_edges(updates, field, 3) for field in Z_FIELDS}

    cats = _termination_categories(updates[1]["term_base_contact"], updates[1]["term_time_out"], PREFALL_WINDOW)
    assert cats["normal"].all(), "no falls/timeouts in synthetic data -- every sample should be 'normal'"

    scalars_1 = _compute_scalars(updates[1], w, 0.998, 0.95)
    scalars_2 = _compute_scalars(updates[2], w, 0.998, 0.95)
    scalars_3 = _compute_scalars(updates[3], w, 0.998, 0.95)
    smd_same = _smd(scalars_1["value_scalar"].ravel(), scalars_2["value_scalar"].ravel())
    smd_shifted = _smd(scalars_1["value_scalar"].ravel(), scalars_3["value_scalar"].ravel())
    assert smd_same < 0.5, f"identical value fields should read ~0 SMD, got {smd_same}"
    assert smd_shifted > 5.0, f"a 4.0-unit value shift on a near-zero-variance field should read a large SMD, got {smd_shifted}"

    rid_1 = _region_ids({f: updates[1][f] for f in Z_FIELDS}, bin_edges)
    rid_3 = _region_ids({f: updates[3][f] for f in Z_FIELDS}, bin_edges)
    overlap = len(np.intersect1d(np.unique(rid_1), np.unique(rid_3))) / len(np.unique(rid_1))
    assert overlap > 0.5, f"same underlying state distribution -- most regions occupied by update 1 should still be occupied by update 3 (value differs, state doesn't), got {overlap:.2f} overlap"

    fall = np.zeros((10, 2), dtype=np.float32)
    fall[5, 0] = 1.0
    to = np.zeros((10, 2), dtype=np.float32)
    to[8, 1] = 1.0
    cats2 = _termination_categories(fall, to, window=2)
    assert cats2["fall"][5, 0] and not cats2["pre_fall"][5, 0], "the fall step itself must be `fall`, not `pre_fall`"
    assert cats2["pre_fall"][3:5, 0].all(), "steps strictly before the fall, within window, must be `pre_fall`"
    assert not cats2["pre_fall"][:3, 0].any(), "outside the window must not be `pre_fall`"
    assert cats2["timeout"][8, 1] and cats2["normal"][:, 1][cats2["normal"][:, 1]].size == 9, "lane 1's only non-normal step is its timeout"

    print("demo() OK: SMD-catches-value-shift, region-occupancy-match, termination-category self-checks passed")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo()
    else:
        main()
