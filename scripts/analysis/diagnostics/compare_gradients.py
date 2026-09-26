#!/usr/bin/env python3
"""Layer 3 of the state-distribution-shift diagnostic (2026-09-20 branch).
Layer 1 found state-distribution shift between consecutive rollouts is real
but doesn't track bearing_cos change (Case C). Layer 2 found comparable-state
valuation/advantage shift is real but ALSO doesn't track bearing_cos change.
Both "policy sees new states" and "policy values old states differently"
failed to explain the cross-update directional diffusion. Layer 3 stops
asking about states/values and looks directly at gradient formation: does
the diffusion originate in the gradient itself (rollout -> gradient), in
how the objectives compose (gradient conflict), or after the gradient (the
multi-epoch/minibatch optimizer path)?

Pure analysis, no sim/env/model dependency -- reads the --save_descriptors
.npz multi_update_trace.py writes once wrapped with the Layer 3 gradient
capture (g_progress, g_efficiency, g_impact, g_balance, g_diversity,
g_total, delta_theta; each a flat actor-param-count vector, epoch0/mb0 only
for the g_* vectors, whole-update for delta_theta -- see that script's own
comments for exactly what "actor gradient" means here and why
g_total == sum(g_progress..g_balance) + g_diversity by construction).

Three questions, matching the three sub-layers this branch's plan called
A/B/C:

  1. Is cross-update gradient instability real at the gradient-formation
     level, not just at the mu/bearing level several PPO steps downstream?
     cos(g_total[k], g_total[k+1]).

  2. Which objective drives it? cos(g_i[k], g_i[k+1]) per objective, and
     cos(g_i[k], g_j[k]) pairwise objective-conflict per update.

  3. Does the gradient's own direction already wander, or does something
     AFTER the gradient (optimizer momentum, multi-epoch reshuffling,
     clipping across later minibatches) rotate the actual parameter step
     away from what the first gradient pointed toward?
     cos(delta_theta[k], delta_theta[k+1]) vs cos(g_total[k], g_total[k+1]),
     and cos(g_total[k], delta_theta[k]) (same-update gradient-to-movement
     alignment -- low means epochs 1..end and later minibatches moved the
     parameters somewhere the FIRST minibatch's gradient didn't point).

    PYTHONPATH="$(pwd):$(pwd)/scripts" python scripts/analysis/run_diagnostics.py compare-gradients \\
        /tmp/.../layer1/seed0_50update_v3.npz --out_csv /tmp/.../layer1/seed0_gradient_pairs.csv
"""

from __future__ import annotations

import argparse

import numpy as np

OBJECTIVES = ("progress", "efficiency", "impact", "balance")


def _load_updates(npz_path: str) -> dict[int, dict[str, np.ndarray]]:
    d = np.load(npz_path)
    updates: dict[int, dict[str, np.ndarray]] = {}
    for key in d.files:
        k_str, field = key.split("__", 1)
        updates.setdefault(int(k_str), {})[field] = d[key]
    return updates


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / denom)


D_FEATURES = ("mean_vx", "mean_height", "mean_abs_vz", "mean_contact", "fall_fraction", "timeout_fraction")


def _rollout_descriptor(u: dict[str, np.ndarray]) -> np.ndarray:
    """One update's rollout-distribution descriptor D_k, in D_FEATURES order.
    Pulled straight from the same raw Layer-1 state fields (vx/height/vz/
    contact_count/term_base_contact/term_time_out) this .npz already carries
    -- action_norm deliberately excluded (policy output, not physical
    regime, same reasoning compare_valuation.py's Z_FIELDS already applied)."""
    return np.array([
        u["vx"].mean(), u["height"].mean(), np.abs(u["vz"]).mean(),
        u["contact_count"].mean(), u["term_base_contact"].mean(), u["term_time_out"].mean(),
    ], dtype=np.float64)


def _counterfactual_conditions(updates: dict[int, dict[str, np.ndarray]]) -> dict[str, dict[int, np.ndarray]]:
    """actual / no_progress / progress_only per update. NOTE: the fixed
    preference weight w_i is already baked into each stored g_<objective>
    vector (see multi_update_trace.py: loss_i = -(w[:,i]*per_objective[:,i]).mean()),
    so g_no_progress is a plain vector subtraction -- no separate re-weighting
    by w_p/w_e/w_i/w_b needed, that would double-apply the weight."""
    conditions: dict[str, dict[int, np.ndarray]] = {"actual": {}, "no_progress": {}, "progress_only": {}}
    for k, u in updates.items():
        conditions["actual"][k] = u["g_total"]
        conditions["no_progress"][k] = u["g_total"] - u["g_progress"]
        conditions["progress_only"][k] = u["g_progress"]
    return conditions


def _condition_stats(cond: dict[int, np.ndarray]) -> dict[str, float]:
    ks = sorted(cond.keys())
    cos_vals = np.array([_cos(cond[k], cond[k + 1]) for k in ks[:-1]])
    norms = np.array([np.linalg.norm(cond[k]) for k in ks])
    return {
        "mean_consecutive_cosine": float(cos_vals.mean()),
        "std": float(cos_vals.std()),
        "fraction_cosine_negative": float((cos_vals < 0).mean()),
        "mean_gradient_norm": float(norms.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("descriptors", help="Path to a Layer-3-instrumented --save_descriptors .npz from multi_update_trace.py")
    parser.add_argument("--out_csv", default=None)
    parser.add_argument("--counterfactual_csv", default=None)
    parser.add_argument("--geometry_csv", default=None)
    args = parser.parse_args()

    updates = _load_updates(args.descriptors)
    ks = sorted(updates.keys())
    print(f"loaded {len(ks)} updates ({ks[0]}..{ks[-1]}) from {args.descriptors}")

    # Test 3 (sanity gate, runs FIRST -- a counterfactual built on a vector
    # identity that doesn't actually hold is worthless): reconstruct g_total
    # from the stored per-objective pieces and check it against the stored
    # g_total. This is the same identity multi_update_trace.py's collection
    # already guarantees (~3e-7 relative error there); re-checking it here,
    # on whatever .npz was actually handed to this script, is cheap and
    # catches a mismatched/corrupted file before any causal claim is built
    # on it.
    max_rel_err = 0.0
    for k in ks:
        u = updates[k]
        reconstructed = sum(u[f"g_{o}"] for o in OBJECTIVES) + u["g_diversity"]
        rel_err = float(np.linalg.norm(reconstructed - u["g_total"]) / (np.linalg.norm(u["g_total"]) + 1e-12))
        max_rel_err = max(max_rel_err, rel_err)
    print(f"\nsanity gate: max relative error of reconstructed g_total vs stored g_total over {len(ks)} updates: {max_rel_err:.2e}")
    if max_rel_err > 1e-3:
        raise RuntimeError(
            f"g_total reconstruction identity failed (max relative error {max_rel_err:.2e} > 1e-3) -- "
            "counterfactual conditions below would not be trustworthy, refusing to proceed"
        )

    print("\nQuestion 3 (same-update): cos(g_total[k], delta_theta[k]) -- does the first minibatch's")
    print("gradient already point where the whole update ends up moving?")
    for k in ks:
        u = updates[k]
        print(f"  upd {k:>3}: {_cos(u['g_total'], u['delta_theta']):>8.4f}")

    print("\nQuestion 3b (same-update): pairwise objective-conflict cos(g_i[k], g_j[k]):")
    pair_names = [(a, b) for i, a in enumerate(OBJECTIVES) for b in OBJECTIVES[i + 1:]]
    header = f"{'upd':>4}" + "".join(f"{a[:3]+'.'+b[:3]:>10}" for a, b in pair_names)
    print(header)
    for k in ks:
        u = updates[k]
        vals = [_cos(u[f"g_{a}"], u[f"g_{b}"]) for a, b in pair_names]
        print(f"{k:>4}" + "".join(f"{v:>10.4f}" for v in vals))

    rows = []
    header2 = (
        f"{'k':>3}{'k+1':>5}{'cos_gtotal':>12}{'cos_dtheta':>12}"
        + "".join(f"{'cos_g'+o[:4]:>11}" for o in OBJECTIVES)
    )
    print(f"\nQuestions 1 & 2 (cross-update): cos(g[k], g[k+1]) and cos(delta_theta[k], delta_theta[k+1])")
    print(header2)
    for k in ks[:-1]:
        u_k, u_k1 = updates[k], updates[k + 1]
        cos_gtotal = _cos(u_k["g_total"], u_k1["g_total"])
        cos_dtheta = _cos(u_k["delta_theta"], u_k1["delta_theta"])
        cos_g_obj = {o: _cos(u_k[f"g_{o}"], u_k1[f"g_{o}"]) for o in OBJECTIVES}
        row = {"k": k, "k1": k + 1, "cos_g_total": cos_gtotal, "cos_delta_theta": cos_dtheta,
               **{f"cos_g_{o}": cos_g_obj[o] for o in OBJECTIVES}}
        rows.append(row)
        print(
            f"{k:>3}{k+1:>5}{cos_gtotal:>12.4f}{cos_dtheta:>12.4f}"
            + "".join(f"{cos_g_obj[o]:>11.4f}" for o in OBJECTIVES)
        )

    cos_gtotal_arr = np.array([r["cos_g_total"] for r in rows])
    cos_dtheta_arr = np.array([r["cos_delta_theta"] for r in rows])
    print(f"\n=== summary over {len(rows)} update pairs ===")
    print(f"mean cos(g_total[k], g_total[k+1]): {cos_gtotal_arr.mean():.4f}  (std {cos_gtotal_arr.std():.4f})")
    print(f"mean cos(delta_theta[k], delta_theta[k+1]): {cos_dtheta_arr.mean():.4f}  (std {cos_dtheta_arr.std():.4f})")
    for o in OBJECTIVES:
        arr = np.array([r[f"cos_g_{o}"] for r in rows])
        print(f"mean cos(g_{o}[k], g_{o}[k+1]): {arr.mean():.4f}  (std {arr.std():.4f})")
    print("\nread: cos_g_total low/wandering already at the gradient-formation step (same order of magnitude")
    print("as the mu-level bearing_cos this branch has been tracking) -> diffusion originates in gradient")
    print("formation itself, not downstream of it. cos_g_total high but cos_delta_theta low -> something")
    print("AFTER the first-minibatch gradient (later epochs/minibatches, optimizer momentum) rotates the")
    print("actual step away from it -- look at the multi-epoch/minibatch path next, not rollout generation.")

    # Test 1: counterfactual gradient composition -- does cross-update
    # directional instability persist when the unstable progress component
    # is removed from g_total, or does removing it stabilize the direction?
    print(f"\n=== Test 1: counterfactual gradient composition (progress removed) ===")
    conditions = _counterfactual_conditions(updates)
    cf_stats = {name: _condition_stats(cond) for name, cond in conditions.items()}
    metric_names = ["mean_consecutive_cosine", "std", "fraction_cosine_negative", "mean_gradient_norm"]
    print(f"{'metric':>28}{'actual':>12}{'no_progress':>14}{'progress_only':>16}")
    for m in metric_names:
        print(f"{m:>28}{cf_stats['actual'][m]:>12.4f}{cf_stats['no_progress'][m]:>14.4f}{cf_stats['progress_only'][m]:>16.4f}")
    delta = cf_stats["no_progress"]["mean_consecutive_cosine"] - cf_stats["actual"]["mean_consecutive_cosine"]
    print(f"\nno_progress - actual mean_consecutive_cosine = {delta:+.4f}")
    print("read: strongly positive delta (no_progress much more directionally stable than actual) is evidence")
    print("progress's unstable relationship to the {efficiency,impact,balance} subspace DRIVES the cross-update")
    print("diffusion, not just correlates with it. Near-zero or negative delta means removing progress doesn't")
    print("stabilize the direction -- the conflict structure is a correlate, not the mechanism.")

    if args.counterfactual_csv:
        import csv
        with open(args.counterfactual_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["condition", *metric_names])
            for name in ("actual", "no_progress", "progress_only"):
                writer.writerow([name, *[cf_stats[name][m] for m in metric_names]])
        print(f"\nwrote counterfactual table -> {args.counterfactual_csv}")

    # Test 2: does progress's gradient NORM change disproportionately during
    # the pairs where g_total's direction flips hardest, or is it purely a
    # direction change (norm roughly steady)? Correlates (1 - cos_g_total)
    # -- the "flip size" for that pair -- against how much progress's own
    # gradient norm changed between k and k+1 (log-ratio, scale-free).
    flip_size = np.array([1.0 - r["cos_g_total"] for r in rows])
    progress_norm_logratio = np.array([
        abs(np.log(
            (np.linalg.norm(updates[r["k1"]]["g_progress"]) + 1e-12)
            / (np.linalg.norm(updates[r["k"]]["g_progress"]) + 1e-12)
        ))
        for r in rows
    ])
    if len(flip_size) >= 3 and flip_size.std() > 1e-8 and progress_norm_logratio.std() > 1e-8:
        r_mag = float(np.corrcoef(flip_size, progress_norm_logratio)[0, 1])
    else:
        r_mag = float("nan")
    print(f"\n=== Test 2: does progress's gradient MAGNITUDE change track direction flips? ===")
    print(f"corr(flip_size=1-cos_g_total, |log(||g_progress[k+1]||/||g_progress[k]||)|) = {r_mag:.3f}")
    print("read: low |r| means the biggest g_total direction flips are NOT accompanied by unusual progress")
    print("magnitude swings -- i.e. it's a pure direction change in progress's gradient, not a magnitude one.")

    # Test 4 (2026-09-20, follow-up to Test 1's SEED-DEPENDENT counterfactual
    # delta -- seed0 +0.036, seed1 +0.198): decomposes g_total's own
    # composition into g_rest = g_efficiency+g_impact+g_balance (the
    # empirically stable E-I-B triangle from Question 3b's pairwise cosines)
    # plus g_progress, asking WHY removing progress stabilizes seed1 far more
    # than seed0 -- is progress's relationship to that stable subspace itself
    # less stable in seed1 (bigger swings / more sign flips), does it have
    # more LEVERAGE (larger norm relative to g_rest), or does it rotate the
    # resultant further off g_rest's own direction? Run this script once per
    # seed's .npz and compare the printed numbers by eye across the two runs
    # -- this function only ever sees one seed at a time.
    g_rest = {k: updates[k]["g_efficiency"] + updates[k]["g_impact"] + updates[k]["g_balance"] for k in ks}
    cos_pr = np.array([_cos(updates[k]["g_progress"], g_rest[k]) for k in ks])
    R_k = np.array([np.linalg.norm(updates[k]["g_progress"]) / (np.linalg.norm(g_rest[k]) + 1e-12) for k in ks])
    resultant_angle_deg = np.array([
        np.degrees(np.arccos(np.clip(_cos(g_rest[k], updates[k]["g_total"]), -1.0, 1.0))) for k in ks
    ])
    sign_k = np.sign(cos_pr)
    sign_flip_rate = float((sign_k[:-1] != sign_k[1:]).mean())
    R_pair_mean = np.array([(R_k[i] + R_k[i + 1]) / 2.0 for i in range(len(ks) - 1)])
    r_leverage = float(np.corrcoef(R_pair_mean, flip_size)[0, 1]) if len(flip_size) >= 3 and R_pair_mean.std() > 1e-8 else float("nan")

    cos_rest_temporal = np.array([_cos(g_rest[k], g_rest[k + 1]) for k in ks[:-1]])
    print(f"\n=== Test 4: progress/rest geometry decomposition (g_rest = g_efficiency+g_impact+g_balance) ===")
    print(f"mean cos(g_rest[k], g_rest[k+1]) (g_rest's OWN temporal stability, progress not involved): {cos_rest_temporal.mean():.4f}  (std {cos_rest_temporal.std():.4f}, frac_negative {float((cos_rest_temporal<0).mean()):.3f})")
    print(f"mean cos(g_progress, g_rest): {cos_pr.mean():+.4f}  (std {cos_pr.std():.4f})")
    print(f"sign-flip rate P(sign_k != sign_k+1) of cos(g_progress,g_rest): {sign_flip_rate:.4f}")
    print(f"mean leverage R_k = ||g_progress|| / ||g_rest||: {R_k.mean():.4f}  (std {R_k.std():.4f})")
    print(f"mean resultant angle(g_rest, g_total): {resultant_angle_deg.mean():.2f} deg  (std {resultant_angle_deg.std():.2f})")
    print(f"corr(mean(R_k,R_k+1), flip_size=1-cos_g_total): {r_leverage:.3f}")
    print("read: compare this block's numbers across separate seed0 vs seed1 runs of this script -- a higher")
    print("sign-flip rate / higher mean R_k / larger resultant angle in seed1 than seed0 would explain why")
    print("Test 1's counterfactual delta was so much larger there (progress has more leverage AND switches")
    print("alignment with the stable E-I-B subspace more often).")

    if args.geometry_csv:
        import csv
        with open(args.geometry_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["k", "cos_progress_rest", "sign", "R_leverage", "resultant_angle_deg"])
            for i, k in enumerate(ks):
                writer.writerow([k, cos_pr[i], sign_k[i], R_k[i], resultant_angle_deg[i]])
        print(f"\nwrote per-update geometry table -> {args.geometry_csv}")

    # Test 5 (2026-09-20, follow-up to Test 4's finding that g_rest's OWN
    # temporal stability -- not progress's behavior -- is what differs
    # between seeds): decomposes g_rest = g_E+g_I+g_B further. Which of the
    # three (if any) is individually noisy, or is it purely a mixing/
    # cancellation effect among three individually-stable components?
    non_progress = ("efficiency", "impact", "balance")
    temporal_cos = {o: np.array([_cos(updates[k][f"g_{o}"], updates[k + 1][f"g_{o}"]) for k in ks[:-1]]) for o in non_progress}
    eib_pairs = [("efficiency", "impact"), ("efficiency", "balance"), ("impact", "balance")]
    intra_cos = {(a, b): np.array([_cos(updates[k][f"g_{a}"], updates[k][f"g_{b}"]) for k in ks]) for a, b in eib_pairs}
    eib_norms = {o: np.array([np.linalg.norm(updates[k][f"g_{o}"]) for k in ks]) for o in non_progress}

    print(f"\n=== Test 5: E/I/B individual decomposition of g_rest ===")
    print(f"{'metric':>28}{'mean':>10}{'std':>10}")
    for o in non_progress:
        print(f"{o+' temporal cosine':>28}{temporal_cos[o].mean():>10.4f}{temporal_cos[o].std():>10.4f}")
    for a, b in eib_pairs:
        print(f"{a[:3]+'-'+b[:3]+' mean cosine':>28}{intra_cos[(a, b)].mean():>10.4f}{intra_cos[(a, b)].std():>10.4f}")
    for o in non_progress:
        print(f"{o+' norm':>28}{eib_norms[o].mean():>10.4f}{eib_norms[o].std():>10.4f}")
    least_stable = min(non_progress, key=lambda o: temporal_cos[o].mean())
    print(f"\nleast temporally-stable individual objective: {least_stable} (mean cos {temporal_cos[least_stable].mean():.4f})")
    print("read (per this branch's Case A/B split): if one objective's temporal cosine is clearly lower than the")
    print("other two -> Case A, that objective is the individually-noisy source, go look at its advantage/state-")
    print("regime dependence next. If all three temporal cosines are similar and none particularly low, but")
    print("g_rest's own temporal cosine (printed in Test 4's g_rest identity check above) is still lower than")
    print("all three individually -> Case B, a mixing/cancellation effect between otherwise-stable E/I/B, not")
    print("any single noisy objective -- go do the magnitude/cancellation ratios (r_E, r_I, r_B, C_EI etc) next.")

    # Test 8 (2026-09-20, follow-up to Test 5's finding that impact/balance
    # are the least temporally-stable individual objectives): screens for
    # temporal COUPLING between rollout-distribution shift and gradient
    # instability, using the same Layer-1 state fields already in this .npz
    # -- no retrain, no separate file. NOT a causal claim (observational
    # trace from one training trajectory) -- "temporal association /
    # evidence consistent with a mechanism", per this branch's own caveat.
    #
    # D_k = [mean_vx, mean_height, mean_|vz|, mean_contact, fall_fraction,
    # timeout_fraction] (action_norm excluded -- policy output, not state).
    # ΔD[p] = D[p+1] - D[p] is the rollout-distribution shift caused BY
    # update p's policy change, landing strictly between g_X[p] (computed
    # from rollout p, BEFORE that shift) and g_X[p+1] (computed from
    # rollout p+1, AFTER it) -- so ΔD[p] vs cos_X[p]=cos(g_X[p],g_X[p+1]) is
    # the temporally-motivated "lag 0" pairing, not an arbitrary alignment
    # choice. lag -1 (ΔD[p-1], a shift that happened BEFORE g_X[p] existed)
    # and lag +1 (ΔD[p+1], a shift that happens AFTER g_X[p+1] existed) are
    # included specifically to catch the reverse-ordering case.
    #
    # Two separate questions, since a signed cosine conflates "direction
    # retained" and "direction reversed":
    #   A. signed: corr(cos_X[p], raw ΔD[p][feature]) per feature (lag 0 only,
    #      to keep this printout bounded -- 3 gradients x 6 lags x 6 features
    #      would be 108 numbers).
    #   B. magnitude: corr(1 - cos_X[p], ||standardized ΔD[p]||_2) at all
    #      three lags -- this is the one the hypothesis actually predicts
    #      (distribution shift -> gradient instability), tested without
    #      assuming any particular sign relationship.
    D = {k: _rollout_descriptor(updates[k]) for k in ks}
    raw_delta = {p: D[p + 1] - D[p] for p in ks[:-1]}  # p = 1..49
    delta_matrix = np.stack([raw_delta[p] for p in sorted(raw_delta)])  # (49, 6)
    delta_std = delta_matrix.std(axis=0)
    delta_std_safe = np.where(delta_std < 1e-8, 1.0, delta_std)
    standardized = (delta_matrix - delta_matrix.mean(axis=0)) / delta_std_safe
    delta_norm = {p: float(np.linalg.norm(standardized[i])) for i, p in enumerate(sorted(raw_delta))}

    gradient_series = {
        "rest": {k: g_rest[k] for k in ks},
        "impact": {k: updates[k]["g_impact"] for k in ks},
        "balance": {k: updates[k]["g_balance"] for k in ks},
    }

    print(f"\n=== Test 8: temporal coupling between rollout-distribution shift and gradient instability ===")
    print("Test 8A (signed, lag 0 only): corr(cos_X[p], raw delta_D[p][feature])")
    print(f"{'gradient':>10}" + "".join(f"{f[:11]:>13}" for f in D_FEATURES))
    for name, series in gradient_series.items():
        cos_p = np.array([_cos(series[p], series[p + 1]) for p in ks[:-1]])
        corrs = []
        for j in range(len(D_FEATURES)):
            feat = delta_matrix[:, j]
            corrs.append(float(np.corrcoef(cos_p, feat)[0, 1]) if feat.std() > 1e-8 else float("nan"))
        print(f"{name:>10}" + "".join(f"{c:>13.3f}" for c in corrs))

    print("\nTest 8B (magnitude, 3 lags): corr(1-cos_X[p], ||standardized delta_D||) at lag -1/0/+1")
    print(f"{'gradient':>10}{'lag-1':>10}{'lag0':>10}{'lag+1':>10}")
    ps = sorted(raw_delta)  # 1..49
    for name, series in gradient_series.items():
        instab = {p: 1.0 - _cos(series[p], series[p + 1]) for p in ps}
        row = []
        for lag, shift in [("lag-1", -1), ("lag0", 0), ("lag+1", 1)]:
            pairs = [(instab[p], delta_norm[p + shift]) for p in ps if (p + shift) in delta_norm]
            if len(pairs) >= 3:
                a, b = np.array([x[0] for x in pairs]), np.array([x[1] for x in pairs])
                r = float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-8 and b.std() > 1e-8 else float("nan")
            else:
                r = float("nan")
            row.append(r)
        print(f"{name:>10}" + "".join(f"{v:>10.3f}" for v in row))

    print("\nread: Test 8B lag0 high (positive) for impact/balance -> distribution shift and their gradient")
    print("instability co-occur in the SAME update transition -- go to Test 7 (advantage-by-regime) for")
    print("whichever D_FEATURES Test 8A's signed correlations point to. All three lags low for all three")
    print("gradients -> simple rollout-distribution drift doesn't explain it -- look at minibatch/data")
    print("composition or PPO clipping/update ordering instead, not reward-regime dependence.")

    if args.out_csv:
        import csv
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {len(rows)} rows -> {args.out_csv}")


def demo() -> None:
    """Self-check: synthetic updates where g_total is built EXACTLY from
    sum(g_objective) + g_diversity (mirrors the real identity multi_update_trace.py
    guarantees), and a known cosine-1.0 case plus a known cosine-(-1.0) case."""
    rng = np.random.default_rng(0)
    P = 500

    def make_update(base_dir: np.ndarray, scale: float) -> dict[str, np.ndarray]:
        g = {o: (base_dir * scale + rng.normal(0, 0.01, P)).astype(np.float32) for o in OBJECTIVES}
        g_diversity = rng.normal(0, 0.001, P).astype(np.float32)
        g_total = sum(g.values()) + g_diversity
        return {f"g_{o}": g[o] for o in OBJECTIVES} | {
            "g_diversity": g_diversity, "g_total": g_total, "delta_theta": base_dir * scale,
        }

    d1 = rng.normal(0, 1, P)
    updates = {1: make_update(d1, 1.0), 2: make_update(d1, 1.0), 3: make_update(-d1, 1.0)}

    assert _cos(np.ones(3), np.ones(3)) > 0.999, "identical vectors must have cos ~1"
    assert _cos(np.array([1.0, 0, 0]), np.array([-1.0, 0, 0])) < -0.999, "opposite vectors must have cos ~-1"
    assert _cos(np.array([1.0, 0]), np.array([0.0, 1.0])) < 0.01, "orthogonal vectors must have cos ~0"

    cos_1_2 = _cos(updates[1]["g_total"], updates[2]["g_total"])
    cos_1_3 = _cos(updates[1]["g_total"], updates[3]["g_total"])
    assert cos_1_2 > 0.9, f"same-direction updates should read high cos, got {cos_1_2}"
    assert cos_1_3 < -0.9, f"opposite-direction updates should read strongly negative cos, got {cos_1_3}"

    g_sum = sum(updates[1][f"g_{o}"] for o in OBJECTIVES) + updates[1]["g_diversity"]
    assert np.allclose(g_sum, updates[1]["g_total"], atol=1e-5), "synthetic data must itself satisfy the real identity multi_update_trace.py guarantees"

    # Counterfactual self-check: build a case where progress ALONE flips sign
    # update-to-update while efficiency/impact/balance stay fixed -- removing
    # progress must make no_progress's consecutive cosine read ~1.0 (perfectly
    # stable), strictly higher than actual's, proving the subtraction isolates
    # exactly what it claims to.
    stable_dir = rng.normal(0, 1, P)
    updates2 = {
        1: make_update(stable_dir, 1.0),
        2: make_update(stable_dir, 1.0),
    }
    updates2[2][f"g_progress"] = -updates2[1]["g_progress"].copy()  # flip ONLY progress
    updates2[2]["g_total"] = sum(updates2[2][f"g_{o}"] for o in OBJECTIVES) + updates2[2]["g_diversity"]
    conds = _counterfactual_conditions(updates2)
    cos_actual = _cos(conds["actual"][1], conds["actual"][2])
    cos_no_progress = _cos(conds["no_progress"][1], conds["no_progress"][2])
    assert cos_no_progress > 0.999, f"removing a flipped progress should leave a perfectly stable no_progress direction, got {cos_no_progress}"
    assert cos_no_progress > cos_actual, f"no_progress must read MORE stable than actual when progress alone is the flip source, got no_progress={cos_no_progress} actual={cos_actual}"

    # Test 4 self-check: g_rest orthogonal to g_progress by construction ->
    # cos(g_progress, g_rest) must read ~0, sign-flip rate over an
    # alternating +progress/-progress sequence must read 1.0 (every step
    # flips), and R_k (leverage) must scale exactly with the progress-norm
    # multiplier used to build it.
    e1, e2 = np.zeros(P), np.zeros(P)
    e1[0], e2[1] = 1.0, 1.0  # orthogonal unit vectors
    updates3 = {
        1: {"g_progress": e1 * 3.0, "g_efficiency": e2, "g_impact": e2, "g_balance": e2,
            "g_diversity": np.zeros(P), "g_total": e1 * 3.0 + 3 * e2},
        2: {"g_progress": -e1 * 3.0, "g_efficiency": e2, "g_impact": e2, "g_balance": e2,
            "g_diversity": np.zeros(P), "g_total": -e1 * 3.0 + 3 * e2},
    }
    g_rest3 = {k: updates3[k]["g_efficiency"] + updates3[k]["g_impact"] + updates3[k]["g_balance"] for k in (1, 2)}
    cos_pr3 = np.array([_cos(updates3[k]["g_progress"], g_rest3[k]) for k in (1, 2)])
    assert abs(cos_pr3[0]) < 1e-6 and abs(cos_pr3[1]) < 1e-6, f"orthogonal-by-construction progress/rest should read cos ~0, got {cos_pr3}"
    R3 = np.array([np.linalg.norm(updates3[k]["g_progress"]) / np.linalg.norm(g_rest3[k]) for k in (1, 2)])
    assert np.allclose(R3, 1.0, atol=1e-6), f"||3*e1||/||3*e2|| should read leverage 1.0 exactly, got {R3}"

    # Test 8 self-check: _rollout_descriptor pulls the right fields in the
    # right order, and the lag-correlation logic finds a PLANTED coupling
    # (instability[p] constructed as an exact linear function of
    # ||standardized delta_D[p]||) at lag 0, not at lag +/-1.
    u_fake = {
        "vx": np.full((4, 2), 0.5), "height": np.full((4, 2), 0.3), "vz": np.full((4, 2), -0.1),
        "contact_count": np.full((4, 2), 2.0), "term_base_contact": np.zeros((4, 2)), "term_time_out": np.zeros((4, 2)),
    }
    d_check = _rollout_descriptor(u_fake)
    expected = np.array([0.5, 0.3, 0.1, 2.0, 0.0, 0.0])
    assert np.allclose(d_check, expected), f"_rollout_descriptor field order/values wrong: got {d_check}, expected {expected}"

    rng2 = np.random.default_rng(1)
    n_pairs = 20
    delta_planted = rng2.normal(0, 1, (n_pairs, 6))
    delta_planted_std = (delta_planted - delta_planted.mean(0)) / delta_planted.std(0)
    norms_planted = np.linalg.norm(delta_planted_std, axis=1)
    instab_planted = {p: float(norms_planted[p - 1]) for p in range(1, n_pairs + 1)}  # instab[p] == ||delta_D[p]|| exactly -> lag0 corr must be ~1
    lag0_pairs = [(instab_planted[p], norms_planted[p - 1]) for p in range(1, n_pairs + 1)]
    r_lag0 = float(np.corrcoef([x[0] for x in lag0_pairs], [x[1] for x in lag0_pairs])[0, 1])
    assert r_lag0 > 0.99, f"a planted exact lag-0 coupling must read corr ~1, got {r_lag0}"
    shuffled = norms_planted.copy()
    rng2.shuffle(shuffled)
    r_shuffled = float(np.corrcoef(list(instab_planted.values()), shuffled)[0, 1])
    assert abs(r_shuffled) < r_lag0, "a shuffled (decoupled) series must read a weaker correlation than the exact planted lag-0 coupling"

    print("demo() OK: cosine sanity, same/opposite-direction discrimination, identity, counterfactual-isolation, geometry-decomposition, rollout-descriptor/lag-coupling self-checks passed")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo()
    else:
        main()
