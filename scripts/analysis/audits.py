"""Post-hoc readers for closed experiments' report files.

These were fifteen throwaway `tmp_*.py` scripts at the repo root, each
hardcoding an absolute path to one `runs/<name>-<date>/` report. They are the
only record of how the numbers in the matching `docs/*-verdict.md` were
derived, so they are kept — but as one registry instead of loose files, and
resolving `runs/` relative to this file so they survive a move to another
machine.

Every audit here belongs to an experiment line that is already CLOSED
(`authority_isolated_*` = Experiment 3, `v2b_*`). Nothing in this module is
part of the training path; it only reads reports that were already written.

Usage:
    python scripts/analysis/audits.py <name>
    python scripts/analysis/audits.py --list
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

RUNS = Path(__file__).resolve().parents[2] / "runs"


# --- helpers shared by more than one audit ------------------------------


def col(trace, key):
    """Stack one per-step field of a trace into an array."""
    return np.asarray([x[key] for x in trace])


def mean_abs_diff(a, b, lo, hi):
    """Per-coordinate mean |a - b| over the step window [lo, hi)."""
    return np.abs(np.asarray(a)[lo:hi] - np.asarray(b)[lo:hi]).mean(0)


def top_k(d, names, k):
    """The k largest entries of a per-coordinate vector, biggest first."""
    return [(int(j), names[j], float(d[j])) for j in np.argsort(-d)[:k]]


def _r(x, n=3):
    return round(float(x), n)


# --- base ---------------------------------------------------------------


class Audit:
    """One closed experiment's report reader.

    Subclass, set `name` and `report`, implement `run`. Registration is
    automatic — `AUDITS` is built from `__subclasses__()`, so adding an
    audit means adding a class and nothing else.
    """

    name: str = ""
    report: str = ""  # path relative to runs/

    @property
    def path(self) -> Path:
        return RUNS / self.report

    def load(self):
        p = self.path
        if not p.exists():
            raise SystemExit(f"missing report: {p}")
        if p.suffix == ".npz":
            return np.load(p, allow_pickle=True)
        return json.loads(p.read_text())

    def run(self, data):
        raise NotImplementedError


# --- audits -------------------------------------------------------------


class RepeatedUpdate(Audit):
    """v2b: does composing repeated updates hold the endpoint semantics?"""

    name = "repeated-update"
    report = (
        "v2b_repeated_update_composition_audit-2026-09-24/"
        "repeated_update_composition_report.json"
    )

    def run(self, d):
        print("BASELINE")
        print({k: (_r(v["semantic_score"]), v["pass"]) for k, v in d["baseline_endpoint"].items()})
        for arm, rows in d["arms"].items():
            print("\nARM", arm)
            for r in rows:
                print(
                    "u", r["update"], "label", r["update_label"],
                    "scores", {k: _r(v) for k, v in r["semantic_scores"].items()},
                    "passes", {k: r["endpoint"][k]["pass"] for k in ["T", "A", "O", "S"]},
                    "forget", {k: _r(v) for k, v in r["forgetting_from_previous"].items()},
                    "cosprev", None if r["update_cos_prev"] is None else _r(r["update_cos_prev"]),
                    "disp", _r(r["parameter_displacement_from_theta0"], 4),
                    "sep", _r(r["preference_separation"]["mean_pairwise"], 4),
                    "gnorm", _r(r["raw_grad_norm"]),
                )
            print(
                "forget summary",
                {c: {k: _r(v) for k, v in vals.items()} for c, vals in d["forgetting_summary"][arm].items()},
            )
            for lab in ["T", "A", "O", "S"]:
                sc = [r["semantic_scores"][lab] for r in rows]
                ps = [r["endpoint"][lab]["pass"] for r in rows]
                print(lab, "mean", _r(np.mean(sc)), "passfreq", sum(ps), "/", len(ps),
                      "minmax", _r(min(sc)), _r(max(sc)))


class Retention(Audit):
    """v2b: reference-gradient retention gate — scores, then the projection
    constraints on the retention arm that explain them."""

    name = "retention"
    report = (
        "v2b_reference_gradient_retention_gate-2026-09-24/"
        "reference_gradient_retention_report.json"
    )

    def run(self, d):
        print("ACTIVATION", d["activation"])
        for arm, rows in d["arms"].items():
            print("\nARM", arm)
            for r in rows:
                print(
                    "u", r["update"],
                    "scores", {k: _r(v) for k, v in r["semantic_scores"].items()},
                    "passes", {k: r["endpoint"][k]["pass"] for k in ["T", "A", "O", "S"]},
                    "refs", r["active_references_before_update"],
                    "proj", r["projection"]["projected"],
                    "cos", _r(r["raw_to_applied_cosine"], 4),
                    "disp", _r(r["parameter_displacement_from_theta0"], 4),
                    "sep", _r(r["preference_separation"]["mean_pairwise"], 4),
                )
            print("RETENTION")
            for lab, st in d["retention_stats"][arm].items():
                print(lab, st)

        print("\n=== PROJECTION CONSTRAINTS (retention arm) ===")
        for r in d["arms"]["retention"]:
            if r["update"] not in (1, 2, 3, 4, 6, 7, 8):
                continue
            print("\nu", r["update"], "refs", r["active_references_before_update"],
                  "proj", r["projection"]["projected"], "cos", r["raw_to_applied_cosine"])
            print("pre", {k: _r(v, 6) for k, v in r["projection"]["pre_dot"].items()})
            print("post", {k: _r(v, 6) for k, v in r["projection"]["post_dot"].items()})
            print("passes", {k: r["endpoint"][k]["pass"] for k in ["T", "A", "O", "S"]})
            print("scores", {k: r["semantic_scores"][k] for k in ["T", "A", "O", "S"]})


class ClassBInvariant(Audit):
    """authority-isolated: per-step feature traces for the Class-B failures,
    then their means over the transition window."""

    name = "classb"
    report = "authority_isolated_classB_invariant_audit-2026-09-25/classB_invariant_audit.json"

    TOPS = ["hip_diag", "action_vel_cos", "hip_lr", "hip_frontrear",
            "support_velocity_coupling", "leg_vel_diag", "gravity_hip_coupling", "hip_std"]
    TRANSITION = ["hip_diag", "hip_lr", "hip_frontrear", "action_vel_cos",
                  "gravity_hip_coupling", "support_velocity_coupling"]

    def run(self, r):
        for f in self.TOPS:
            print("\nFEATURE", f)
            rec = r["features"][f]["records"]
            for pref in ["O", "S", "C"]:
                print(" ", pref)
                for x in rec:
                    if x["pref"] == pref and 4 <= x["t"] <= 13:
                        print(x["t"], "fail", _r(x["fail"]), "res", _r(x["rescue"]),
                              "u50", _r(x["u50"]), "ctrl", _r(x["ctrl_mean"]),
                              "zf", _r(x["zf"], 2), "zr", _r(x["zr"], 2))

        print("\n=== TRANSITION WINDOW t=8..10 ===")
        for f in self.TRANSITION:
            print("\n", f)
            for pref in ["O", "S", "C"]:
                rec = [x for x in r["features"][f]["records"] if x["pref"] == pref and 8 <= x["t"] <= 10]
                m = lambda k: float(np.mean([x[k] for x in rec]))  # noqa: E731
                print(pref, "fail", _r(m("fail")), "res", _r(m("rescue")),
                      "u50", _r(m("u50")), "ctrl", _r(m("ctrl_mean")),
                      "fail_absz", _r(np.mean([abs(x["zf"]) for x in rec]), 2),
                      "res_absz", _r(np.mean([abs(x["zr"]) for x in rec]), 2))

        print("\nANGULAR timing")
        for pref in ["O", "S", "C"]:
            rec = [x for x in r["features"]["ang_xy"]["records"] if x["pref"] == pref]
            for t in range(6, 14):
                x = [q for q in rec if q["t"] == t][0]
                print(pref, t, "fail", _r(x["fail"]), "res", _r(x["rescue"]),
                      "ctrl", _r(x["ctrl_mean"]), "zf", _r(x["zf"], 2))


class ResidualLane(Audit):
    """authority-isolated: residual-lane arms — state traces, the action
    coordinates that separate them, headroom/responsiveness, and the FL hip
    values that first diverge."""

    name = "residual-lane"
    report = (
        "authority_isolated_residual_lane_audit-2026-09-25/"
        "authority_isolated_residual_lane_report.json"
    )

    def run(self, r):
        names = r["action_coordinate_names"]
        arms = r["arms"]
        for arm in ["control", "2.5", "2.0", "1.75"]:
            tr = arms[arm]["trace"]
            print("\nARM", arm, "FAIL", arms[arm]["first_fail"])
            for t in range(10, min(17, len(tr))):
                q = tr[t]
                print("t", t, "h", _r(q["height"]), "rp", _r(q["roll"]), _r(q["pitch"]),
                      "ang", np.round(q["ang_vel_b"][:2], 3).tolist(),
                      "a", np.round(q["action"], 3).tolist(),
                      "done", q["done_after_step"])

        base = col(arms["2.5"]["trace"], "action")
        for arm in ["control", "2.0", "1.75"]:
            a = col(arms[arm]["trace"], "action")
            d = mean_abs_diff(a, base, 8, 16)
            print("\nDIFF 2.5 vs", arm)
            for j, nm, v in top_k(d, names, 8):
                print(j, nm, "mean|da|", round(v, 4),
                      "2.5", np.round(base[8:16, j], 3).tolist(),
                      arm, np.round(a[8:16, j], 3).tolist())

        for arm in ["control", "2.5", "2.0", "1.75"]:
            tr = arms[arm]["trace"]
            a, h, z = col(tr, "action"), col(tr, "headroom"), col(tr, "z_transformed")
            d = np.abs(np.diff(a[8:16], axis=0)).mean(0)
            hm, zm = h[8:16].mean(0), np.abs(z[8:16]).mean(0)
            print("\nCOORD", arm)
            for j in range(12):
                print(j, names[j], "head", _r(hm[j], 4), "resp", _r(d[j], 4), "|z|", _r(zm[j]))

        print("\n=== FL HIP, t=4..7 ===")
        for arm in ["2.5", "2.0", "1.75"]:
            print("ARM", arm)
            for t in range(4, 8):
                q = arms[arm]["trace"][t]
                print(t, "FLhip", _r(q["action"][0], 6), "z", _r(q["z_transformed"][0], 6),
                      "h", _r(q["height"], 4), "roll", _r(q["roll"], 4), "pitch", _r(q["pitch"], 4),
                      "angxy", [round(x, 4) for x in q["ang_vel_b"][:2]])


class Mechanism(Audit):
    """authority-isolated: which single/grouped/phase/strength interventions
    rescue each case, and when the repaired trace diverges from its refs."""

    name = "mechanism"
    report = (
        "authority_isolated_residual_mechanism_classification-2026-09-25/"
        "mechanism_classification.json"
    )

    def run(self, r):
        for cname, c in r["cases"].items():
            print("\nCASE", cname)
            print("BASE", {k: v["first_fail"] for k, v in c["baseline"].items()})
            for d in ["u50", "u75"]:
                print("SINGLE", d, [n for n, q in c["single"][d].items() if q["survived"]])
                print("GROUP", d, [n for n, q in c["groups"][d].items() if q["survived"]])
            print("PHASE RESCUES")
            for key, qs in c["phase"].items():
                z = [w for w, q in qs.items() if q["survived"]]
                if z:
                    print(key, z)
            print("STRENGTH RESCUES")
            for key, qs in c["strength"].items():
                z = [a for a, q in qs.items() if q["survived"]]
                if z:
                    print(key, z)

        print("\n=== DIVERGENCE OF repair VS REFS ===")
        for cname, c in r["cases"].items():
            rep = c["baseline"]["repair"]["trace"]
            print("\nCASE", cname)
            for ref_name in ("u50", "u75"):
                ref = c["baseline"][ref_name]["trace"]
                n = min(16, len(rep), len(ref))
                print("vs", ref_name)
                for t in range(n):
                    if t not in (0, 2, 4, 6, 8, 10, 12, 14, 15):
                        continue
                    print(t,
                          "da", _r(np.linalg.norm(np.array(rep[t]["action"]) - np.array(ref[t]["action"]))),
                          "dang", _r(np.linalg.norm(np.array(rep[t]["ang_vel"]) - np.array(ref[t]["ang_vel"]))),
                          "dlin", _r(np.linalg.norm(np.array(rep[t]["lin_vel"]) - np.array(ref[t]["lin_vel"]))),
                          "drp", _r(np.linalg.norm([rep[t]["roll"] - ref[t]["roll"],
                                                    rep[t]["pitch"] - ref[t]["pitch"]])))
                for metric, thr in (("action", 0.25), ("ang_vel", 0.5), ("rp", 0.05)):
                    first = None
                    for t in range(n):
                        if metric == "rp":
                            v = np.linalg.norm([rep[t]["roll"] - ref[t]["roll"],
                                                rep[t]["pitch"] - ref[t]["pitch"]])
                        else:
                            v = np.linalg.norm(np.array(rep[t][metric]) - np.array(ref[t][metric]))
                        if v > thr:
                            first = t
                            break
                    print("first", metric, ">", thr, first)


class PostRepair(Audit):
    """authority-isolated: which joints the repair actually moves, per case
    and shared across cases."""

    name = "postrepair"
    report = "authority_isolated_postrepair_residual_audit-2026-09-25/residual_audit.json"

    def run(self, r):
        names = r["joint_names"]
        all_d = []
        for cname, c in r["cases"].items():
            u, rep = c["u75"]["trace"], c["repair"]["trace"]
            a0, a1 = col(u, "action"), col(rep, "action")
            z0, z1 = col(u, "z"), col(rep, "z")
            d = mean_abs_diff(a1, a0, 4, 14)
            dz = mean_abs_diff(z1, z0, 4, 14)
            all_d.append(d)
            print("\nCASE", cname, "fail u75", c["u75"]["first_fail"],
                  "repair", c["repair"]["first_fail"])
            for j, nm, v in top_k(d, names, 6):
                print(j, nm, "|da|", round(v, 4), "|dz|", _r(dz[j]),
                      "u75", np.round(a0[4:9, j], 3).tolist(),
                      "rep", np.round(a1[4:9, j], 3).tolist())
            for t in (4, 8, 12, 14):
                if t < len(u) and t < len(rep):
                    x, q = u[t], rep[t]
                    print("STATE", t,
                          "h", _r(x["height"]), _r(q["height"]),
                          "rp", _r(x["roll"]), _r(q["roll"]), _r(x["pitch"]), _r(q["pitch"]),
                          "angxy", np.round(x["ang_vel"][:2], 3).tolist(),
                          np.round(q["ang_vel"][:2], 3).tolist())
        m = np.mean(np.stack(all_d), 0)
        print("\nSHARED ACTION DIFF")
        for j in np.argsort(-m):
            print(int(j), names[j], _r(m[j], 4))


class Suite3O(Audit):
    """authority-isolated: suite-3 lane-0 dynamics — which joints separate
    u50 from u75/repair, per time window."""

    name = "suite3o"
    report = "authority_isolated_suite3O_lane0_dynamics_audit-2026-09-25/dynamics_audit.json"

    def run(self, r):
        names = r["joint_names"]
        for pair in (("u50", "u75"), ("u50", "repair")):
            a = col(r["tests"][pair[0]]["trace"], "action")
            b = col(r["tests"][pair[1]]["trace"], "action")
            print("\n", pair)
            for lo, hi in ((0, 3), (4, 8), (6, 10), (9, 13), (11, 14)):
                d = mean_abs_diff(a, b, lo, hi + 1)
                print("window", lo, hi, [(nm, round(v, 3)) for _, nm, v in top_k(d, names, 6)])
        for lab in ("u50", "u75", "repair"):
            tr = r["tests"][lab]["trace"]
            print("\nSTATE", lab, "fail", r["tests"][lab]["first_fail"])
            for t in (4, 6, 8, 10, 12, 14):
                x = tr[t]
                print(t, "h", _r(x["height"]), "ang", np.round(x["ang_vel"], 3).tolist(),
                      "hip", np.round(np.array(x["action"])[:4], 3).tolist())


class Survival(Audit):
    """authority-isolated: per-metric contrast between failing and surviving
    lanes, and each metric's correlation with failure."""

    name = "survival"
    report = (
        "authority_isolated_survival_audit-2026-09-25/"
        "authority_isolated_survival_audit_raw.npz"
    )

    def run(self, z):
        rs = [json.loads(x) for x in z["payload"]]
        idx = {(r["checkpoint"], r["suite"], r["preference"]): r for r in rs}
        names = rs[0]["metric_names"]

        rows = []
        for r in rs:
            if r["checkpoint"] != "u75":
                continue
            x = np.asarray(r["metrics"])
            ft = np.asarray(r["fail_t"])
            done = np.asarray(r["done"])
            for lane in np.where(done)[0]:
                e = ft[lane] + 1
                s = max(0, e - 8)
                c = idx[("u50", r["suite"], r["preference"])]
                rows.append((r["suite"], r["preference"], lane, ft[lane],
                             x[s:e, lane].mean(0),
                             np.asarray(c["metrics"])[s:e, lane].mean(0),
                             x[s:e, ~done].mean((0, 1))))
        print("FAILURES", len(rows))
        for q in rows:
            print(q[0], q[1], "lane", q[2], "t", q[3])
            print(" u75", dict(zip(names, np.round(q[4], 4))))
            print(" u50", dict(zip(names, np.round(q[5], 4))))
            print(" surv", dict(zip(names, np.round(q[6], 4))))
        a = np.stack([x[4] for x in rows])
        b = np.stack([x[5] for x in rows])
        s = np.stack([x[6] for x in rows])
        print("MEAN_U75", dict(zip(names, np.round(a.mean(0), 4))))
        print("MEAN_U50", dict(zip(names, np.round(b.mean(0), 4))))
        print("MEAN_SURV", dict(zip(names, np.round(s.mean(0), 4))))
        print("RATIO75_50", dict(zip(names, np.round(a.mean(0) / (b.mean(0) + 1e-12), 3))))
        print("RATIO75_SURV", dict(zip(names, np.round(a.mean(0) / (s.mean(0) + 1e-12), 3))))

        # failure indicator vs early-window metrics, u75 suites 2-3 only
        lane_rows = []
        for r in rs:
            if r["checkpoint"] != "u75" or r["suite"] < 2:
                continue
            x = np.asarray(r["metrics"])
            done = np.asarray(r["done"])
            for lane in range(8):
                lane_rows.append((int(done[lane]), x[:12, lane].mean(0)))
        y = np.array([x[0] for x in lane_rows])
        m = np.stack([x[1] for x in lane_rows])
        print("EARLY_FAIL_MEAN", dict(zip(names, np.round(m[y == 1].mean(0), 4))))
        print("EARLY_SURV_MEAN", dict(zip(names, np.round(m[y == 0].mean(0), 4))))
        for j, n in enumerate(names):
            c = np.corrcoef(y, m[:, j])[0, 1] if np.std(m[:, j]) > 0 else np.nan
            print("CORR", n, _r(c, 4))


class TailScreen(Audit):
    """authority-isolated: coordinate tail repair — sensitivity retention and
    robustness per snapshot, plus the treatment arm's tail-alpha schedule."""

    name = "tail-screen"
    report = (
        "authority_isolated_coordinate_tail_repair-2026-09-25/"
        "authority_isolated_coordinate_tail_repair_report.json"
    )

    def run(self, r):
        for arm in ("control", "treatment"):
            print("\nARM", arm)
            s0 = r[arm]["snapshots"]["0"]["sensitivity"]
            p0 = s0["pairwise_action_distance"]["mean"]
            j0 = s0["tangent_jacobian_fro_mean"]
            for k in ("0", "5", "10", "20", "25"):
                s = r[arm]["snapshots"][k]
                se, rob, tail = s["sensitivity"], s["robustness"], s["tail"]
                c = next(x for x in rob["rows"] if x["suite"] == 3 and x["preference"] == "C")
                print(k,
                      "pair", _r(se["pairwise_action_distance"]["mean"], 4),
                      "ret", _r(se["pairwise_action_distance"]["mean"] / p0),
                      "jac", _r(se["tangent_jacobian_fro_mean"], 4),
                      "ret", _r(se["tangent_jacobian_fro_mean"] / j0),
                      "minsur", rob["min_survival"], "fails", rob["total_failed_lanes"],
                      "C3", c["survival"], "FL", [round(x, 4) for x in c["flhip_t4_7"]],
                      "tail", _r(tail["mean_excess"]), "active", _r(tail["active_fraction"]))
            if arm == "treatment":
                rows = r[arm]["rows"]
                print("alpha first10", [_r(x["tail_alpha"], 6) for x in rows[:10]])
                print("alpha last10", [_r(x["tail_alpha"], 6) for x in rows[-10:]])
                ratios = [x["tail_alpha"] * x["tail_grad_norm"] / (x["ppo_grad_norm"] + 1e-12)
                          for x in rows]
                print("effective gradient ratios min/mean/max",
                      _r(min(ratios)), _r(np.mean(ratios)), _r(max(ratios)))


AUDITS = {c.name: c for c in Audit.__subclasses__()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("name", nargs="?", choices=sorted(AUDITS), help="audit to run")
    ap.add_argument("--list", action="store_true", help="list audits and their reports")
    args = ap.parse_args(argv)

    if args.list or not args.name:
        for n in sorted(AUDITS):
            cls = AUDITS[n]
            mark = "ok " if (RUNS / cls.report).exists() else "MISSING"
            print(f"{mark:8} {n:16} {cls.report}")
        return

    audit = AUDITS[args.name]()
    audit.run(audit.load())


if __name__ == "__main__":
    main()
