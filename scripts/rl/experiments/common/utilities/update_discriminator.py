#!/usr/bin/env python3
"""Shared scoring for the update-effect audits.

Both the functional-effect audit and the visitation-interaction one ask the
same question — does this metric separate robust-collapse updates from retained
ones — and pair each robust update with a comparable retained one the same way.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

SEEDS = (983001, 984001, 985001)
EPS = 1e-12
MATCH_KEYS = ("seed", "from", "to", "axis", "source_G_sem")


def cos_np(a, b):
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > EPS else 1.0


def auc(y, s):
    """Probability a positive outranks a negative, ties counting a half."""
    y, s = np.asarray(y, int), np.asarray(s, float)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    z = sum(1.0 if s[i] > s[j] else .5 if s[i] == s[j] else 0.0 for i in pos for j in neg)
    return float(z / (len(pos) * len(neg)))


def per_seed_direction(rows, value):
    """Whether the robust median exceeds the retained one within every seed."""
    ok, per = True, {}
    for sd in SEEDS:
        pp = [value(r) for r in rows if r["seed"] == sd and r["Y_robust_collapse"]]
        nn = [value(r) for r in rows if r["seed"] == sd and not r["Y_robust_collapse"]]
        direction = None if not pp or not nn else float(np.median(pp)) > float(np.median(nn))
        per[str(sd)] = {"positive_n": len(pp), "negative_n": len(nn),
                        "robust_median": float(np.median(pp)) if pp else None,
                        "retained_median": float(np.median(nn)) if nn else None,
                        "direction_ok": direction}
        if direction is False:
            ok = False
    return ok, per


def matched_controls(rows):
    """One retained update per robust one: same axis where possible, then the
    nearest source gate margin, without reuse until the pool is exhausted."""
    posrows = [r for r in rows if r["Y_robust_collapse"]]
    negrows = [r for r in rows if not r["Y_robust_collapse"]]
    unused = set(range(len(negrows)))
    matches = []
    for r in posrows:
        same = [i for i in unused if negrows[i]["axis"] == r["axis"]]
        cand = same if same else list(unused)
        reused = False
        if not cand:
            same = [i for i, n in enumerate(negrows) if n["axis"] == r["axis"]]
            cand = same if same else list(range(len(negrows)))
            reused = True
        i = min(cand, key=lambda i: (abs(negrows[i]["source_G_sem"] - r["source_G_sem"]),
                                     negrows[i]["seed"], negrows[i]["from"]))
        unused.discard(i)
        matches.append({"robust": {k: r[k] for k in MATCH_KEYS},
                        "retained": {k: negrows[i][k] for k in MATCH_KEYS},
                        "source_G_abs_diff": abs(negrows[i]["source_G_sem"] - r["source_G_sem"]),
                        "reused": reused})
    return matches
