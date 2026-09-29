#!/usr/bin/env python3
"""FC-C short counterfactual continuation (docs/contracts/teacher_v4/teacher-v4-fcc-continuation-contract.md).

Offline. Each FB-2a seed is continued for 50 iterations from its
pre-inversion checkpoint u0 (FC-B) in four arms (A mixed + dual, B fixed R+
full loss, C fixed R+ preference loss, D fixed R+ R-only loss), saving every
10 iterations. Every checkpoint (u0 included) is replayed with traces. Per
checkpoint: raw Delta_R (FC-B metric), tracking-matched Delta_R, Delta_tl,
shared and conditional rotation. Per arm: the endpoint state over u0+30..50.
Per seed: the pre-declared tree D -> C -> B -> A.
"""
import json
from pathlib import Path

import numpy as np

from fc0_audit import windows

REPO = Path(__file__).resolve().parents[6]
ROOT = REPO / "runs/teacher_v4_fc-2026-09-29/fcc"
FCB = REPO / "runs/teacher_v4_fc-2026-09-29/fcb/fcb_trajectory.json"
U0 = {79101: 450, 79102: 300, 79103: 350}
ARMS = ("D", "C", "B", "A")  # tree order
SOURCE = {"D": "R-stream accumulated learning itself", "C": "accumulated R/O interaction",
          "B": "task interaction", "A": "cross-preference / shared-parameter interference"}
REL, TL_MIN, NBINS, MIN_BIN, MIN_BINS = 0.05, 0.40, 8, 5, 3  # 8 bins: residual bias <= 2% for F_rate ~ tl^2, tl shift .2
END = (30, 40, 50)


def matched_dR(Wr, Wc):
    """Ratio of median F_rate (R+ / C) within pooled window-tl quantile bins, locomoting windows only,
    weighted by the smaller bin count. NaN if fewer than MIN_BINS bins have MIN_BIN windows per condition."""
    r, c = Wr["tl"] >= TL_MIN, Wc["tl"] >= TL_MIN
    tl = np.concatenate([Wr["tl"][r], Wc["tl"][c]])
    if len(tl) < 2 * MIN_BIN:
        return float("nan")
    edges = np.quantile(tl, np.linspace(0, 1, NBINS + 1)[1:-1])
    br, bc = np.digitize(Wr["tl"][r], edges), np.digitize(Wc["tl"][c], edges)
    num = den = 0.0; used = 0
    for b in range(NBINS):
        fr, fc = Wr["F_rate"][r][br == b], Wc["F_rate"][c][bc == b]
        if len(fr) >= MIN_BIN and len(fc) >= MIN_BIN:
            n = min(len(fr), len(fc)); num += n * np.median(fr) / max(np.median(fc), 1e-12); den += n; used += 1
    return num / den - 1 if used >= MIN_BINS else float("nan")


def checkpoint_row(rep, z, u):
    W = {c: windows(z[f"{u}|{c}|x"], z[f"{u}|{c}|contact"], z[f"{u}|{c}|ok"]) for c in ("C", "A+", "O+")}
    med = {c: {k: float(np.median(W[c][k])) for k in ("F_rate", "F_O")} for c in W}
    tl = {c: rep[str(u)][c]["tl"] for c in W}
    return {"viable": tl["A+"] >= TL_MIN and tl["C"] >= TL_MIN,
            "dR": med["A+"]["F_rate"] / max(med["C"]["F_rate"], 1e-12) - 1, "dR_matched": matched_dR(W["A+"], W["C"]),
            "dtl": tl["A+"] - tl["C"], "dO": med["O+"]["F_O"] / max(med["C"]["F_O"], 1e-12) - 1,
            "S_shared": med["C"]["F_rate"], "S_cond": med["A+"]["F_rate"] - med["C"]["F_rate"], "tl": tl}


def arm_audit(s, arm):
    d = ROOT / f"seed{s}" / arm / "replay"
    rep = json.load(open(d / "f2a_replay.json"))["checkpoints"]; z = np.load(d / "rotation_traces.npz")
    u0 = U0[s]
    rows = {k: checkpoint_row(rep, z, u0 + k) for k in range(0, 51, 10)}
    end = [rows[k] for k in END]
    if sum(r["viable"] for r in end) < 2:
        state = "collapse (fewer than 2 viable endpoint checkpoints)"
    else:
        v = [r for r in end if r["viable"]]
        raw = float(np.mean([r["dR"] for r in v])); mt = [r["dR_matched"] for r in v if np.isfinite(r["dR_matched"])]
        m = float(np.mean(mt)) if mt else float("nan")
        if not mt:
            state = "unresolved (no matched-tracking value)"
        elif raw >= REL and m >= REL:
            state = "inverted"
        elif raw < REL and m < REL:
            state = "not inverted"
        else:
            state = "split (raw and matched disagree: tracking-confounded)"
    ev = [r for r in end if r["viable"]]
    return {"u0": u0, "rows": {str(k): r for k, r in rows.items()}, "state": state,
            "end_dR": float(np.mean([r["dR"] for r in ev])) if ev else None,
            "end_dR_matched": float(np.nanmean([r["dR_matched"] for r in ev])) if ev and any(np.isfinite(r["dR_matched"]) for r in ev) else None,
            "shared_change": (np.mean([r["S_shared"] for r in end]) / rows[0]["S_shared"] - 1),
            "cond_change": float(np.mean([r["S_cond"] for r in end]) - rows[0]["S_cond"])}


def tree(arms):
    for a in ARMS:
        st = arms[a]["state"]
        if st == "inverted":
            src = SOURCE[a]
            if arms["A"]["state"] != "inverted":
                src += " (arm A did not reproduce the historical inversion)"
            return f"{a}: {src}"
        if st != "not inverted":
            return f"unresolved at arm {a} ({st})"
    return "historical drift not reproduced in any arm / unresolved"


def selfcheck():
    """Fails if matching stops removing a pure tracking shift or stops seeing a true rotation difference."""
    tl = np.random.default_rng(0).uniform(0.4, 1.1, 384); Wc = {"tl": tl, "F_rate": tl ** 2}
    assert abs(matched_dR({"tl": tl + 0.2, "F_rate": (tl + 0.2) ** 2}, Wc)) < 0.03
    assert abs(matched_dR({"tl": tl, "F_rate": 1.1 * tl ** 2}, Wc) - 0.10) < 0.01


def main():
    selfcheck()
    fcb = json.load(open(FCB))["seeds"]
    out = {"schema": "teacher_v4_fcc_probe_v1", "seeds": {}}
    for s in U0:
        arms = {a: arm_audit(s, a) for a in ARMS}
        # determinism check: the u0 replay must match FC-B's replay of the same checkpoint
        ref = next(r["dR"] for r in fcb[str(s)]["rows"] if r["it"] == U0[s])
        out["seeds"][str(s)] = {"arms": arms, "u0_dR_vs_fcb": {a: arms[a]["rows"]["0"]["dR"] - ref for a in ARMS}, "reading": tree(arms)}
    json.dump(out, open(ROOT / "fcc_probe.json", "w"), indent=1)
    for s, v in out["seeds"].items():
        print(s, "u0", U0[int(s)], "->", v["reading"])
        for a in ARMS:
            r = v["arms"][a]
            cells = " ".join("{}:{:+.2f}/{:+.2f}/{:+.2f}".format(k, x["dR"], x["dR_matched"], x["dtl"]) + ("" if x["viable"] else "*")
                             for k, x in r["rows"].items())
            print("  ", a, r["state"], "| shared {:+.0%} cond {:+.3f} |".format(r["shared_change"], r["cond_change"]), cells)
        print("   u0 replay minus FC-B:", {a: round(x, 4) for a, x in v["u0_dR_vs_fcb"].items()})


if __name__ == "__main__":
    main()
