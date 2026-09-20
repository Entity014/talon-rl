#!/usr/bin/env python3
"""Parses many advantage_decomposition.py output logs into one summary
table (mean +/- std across seeds, grouped by checkpoint x w-condition) --
the aggregation step above advantage_decomposition.py's single-run raw
dump. Built alongside it 2026-09-19, same reasoning: kept as permanent
tooling for any future reward-vector/preference-weighting question.

Expects one log file per (checkpoint_label, w_label, seed), each containing
the exact text advantage_decomposition.py prints (redirect its stdout to a
file). File naming convention:
    <checkpoint_label>_<w_label>_seed<N>.log
where <w_label> is one of progress/balance/uniform.

Usage: python aggregate_decomposition.py <log_dir> [<log_dir> ...]
"""

from __future__ import annotations

import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np

TERM_ROW = re.compile(
    r"^(?P<term>\w+)\s+"
    r"(?P<raw_reward>-?\d+\.\d+)\s+"
    r"(?P<raw_adv>-?\d+\.\d+)\s+"
    r"(?P<return>-?\d+\.\d+)\s+"
    r"(?P<norm_adv>-?\d+\.\d+)\s+"
    r"(?P<w_contrib>-?\d+\.\d+)\s*$"
)
SHARE_ROW = re.compile(r"^\s+(?P<term>\w+):\s+(?P<share>\d+\.\d+)%\s*$")
EPLEN_ROW = re.compile(r"mean episode length this window:\s+(?P<eplen>-?\d+\.\d+)")
FNAME_RE = re.compile(r"(?P<ckpt>.+)_(?P<w>progress|balance|uniform)_seed(?P<seed>\d+)\.log$")


def parse_one(path: str) -> dict | None:
    text = open(path).read()
    terms = {}
    for line in text.splitlines():
        m = TERM_ROW.match(line)
        if m:
            terms[m["term"]] = {
                "raw_reward": float(m["raw_reward"]),
                "raw_adv": float(m["raw_adv"]),
                "return": float(m["return"]),
                "norm_adv": float(m["norm_adv"]),
                "w_contrib": float(m["w_contrib"]),
            }
    shares = {m["term"]: float(m["share"]) for line in text.splitlines() if (m := SHARE_ROW.match(line))}
    eplen_m = EPLEN_ROW.search(text)
    if not terms:
        return None
    return {"terms": terms, "shares": shares, "eplen": float(eplen_m["eplen"]) if eplen_m else None}


def main() -> None:
    dirs = sys.argv[1:] or ["."]
    files = []
    for d in dirs:
        files += glob.glob(os.path.join(d, "*.log"))

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for f in files:
        m = FNAME_RE.search(os.path.basename(f))
        if not m:
            continue
        parsed = parse_one(f)
        if parsed is None:
            continue
        groups[(m["ckpt"], m["w"])].append(parsed)

    if not groups:
        print(f"no matching *_progress|balance|uniform_seedN.log files with decomposition tables found under {dirs}")
        return

    all_terms = sorted({t for g in groups.values() for p in g for t in p["terms"]})

    for (ckpt, w), parsed_list in sorted(groups.items()):
        n = len(parsed_list)
        print(f"\n=== {ckpt} | w={w} (n={n} seeds) ===")
        eplens = [p["eplen"] for p in parsed_list if p["eplen"] is not None]
        if eplens:
            print(f"episode length: {np.mean(eplens):.2f} +/- {np.std(eplens):.2f}")
        print(f"{'term':<12}{'raw reward':>16}{'raw |adv|':>16}{'return':>16}{'norm |adv|':>16}{'influence %':>14}")
        for term in all_terms:
            rows = [p["terms"][term] for p in parsed_list if term in p["terms"]]
            if not rows:
                continue
            shares = [p["shares"].get(term) for p in parsed_list if p["shares"].get(term) is not None]

            def fmt(key: str) -> str:
                vals = [row[key] for row in rows]
                return f"{np.mean(vals):>7.3f}+/-{np.std(vals):<6.3f}"

            share_str = f"{np.mean(shares):.1f}+/-{np.std(shares):.1f}" if shares else "n/a"
            print(f"{term:<12}{fmt('raw_reward'):>16}{fmt('raw_adv'):>16}{fmt('return'):>16}{fmt('norm_adv'):>16}{share_str:>14}")


if __name__ == "__main__":
    main()
