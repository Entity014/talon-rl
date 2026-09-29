#!/usr/bin/env python3
"""FC-A aggregate (docs/contracts/teacher_v4/teacher-v4-fca-r-credit-audit-contract.md): per-seed diagnostics at R+
(checkpoint 600, loco slice, mean-policy virtual steps), causal class led by the virtual steps with appended flags, and the
checkpoint-300 R-vs-T geometry (secondary, no vote)."""
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[6]
ROOT = REPO / "runs/teacher_v4_fc-2026-09-29/fca"
SEEDS = (79101, 79102, 79103)
REL, S_R_WEAK, COS_CONFLICT = 0.05, 0.35, -0.3


def seed_class(j):
    c = j["conditions"]["R+"]; L = c["loco"]; st = c.get("virtual_steps", {})
    if L is None or "primary" in st:
        return {"class": "unresolved (insufficient locomoting samples)", "flags": []}
    if not st.get("null_ok", False):
        return {"class": "uninterpretable (null repeat >= 5%)", "flags": []}
    rel = lambda name: [st[f"{name}@{k}"]["rel"]["F_rate"] for k in (0.01, 0.02)]  # noqa: E731
    lowers = lambda name: all(x <= -REL for x in rel(name))  # noqa: E731
    raises = lambda name: all(x >= REL for x in rel(name))  # noqa: E731
    if raises("R+"):
        cls = "R credit direction wrong"
    elif lowers("R+") and not lowers("mixed+"):
        cls = "local R direction correct; mixed cancellation / multi-objective interaction"
    elif lowers("R+") and lowers("mixed+"):
        cls = "local update correct; problem later / across updates"
    else:
        cls = "unresolved"
    sR = L["contribution"]["R"] / max(L["contribution"]["R"] + L["contribution"]["O"], 1e-12)
    flags = (["credit-weak"] if sR < S_R_WEAK else []) + (["R/O interference"] if L["cos_pair"]["RO"] <= COS_CONFLICT else []) \
        + (["R/T interference"] if L["cos_pair"]["TR"] <= COS_CONFLICT else [])
    return {"class": cls, "flags": flags, "s_R": sR, "cos": L["cos_pair"], "rel_F_rate": {n: rel(n) for n in ("R+", "R-", "mixed+")},
            "null_rel_F_rate": st["null_rel_F_rate"], "logstd_share_of_gR": L.get("logstd_share_of_gR")}


def main():
    per = {s: seed_class(json.load(open(ROOT / f"seed{s}" / "fca_credit_audit.json"))) for s in SEEDS}
    top, n = Counter(v["class"] for v in per.values()).most_common(1)[0]
    flags = Counter(f for v in per.values() for f in v["flags"])
    verdict = (top + ("; " + ", ".join(f for f, k in flags.items() if k >= 2) if any(k >= 2 for k in flags.values()) else "")) if n >= 2 else "no majority"
    sec = {}
    for s in SEEDS:
        p = ROOT / f"seed{s}_300" / "fca_credit_audit.json"
        if p.exists():
            j = json.load(open(p))
            sec[s] = {c: {"lambda": r["lambda"], "cos_TR": (r["loco"] or r["all"])["cos_pair"]["TR"], "contribution_share": (r["loco"] or r["all"])["contribution_share"]}
                      for c, r in j["conditions"].items()}
    out = {"schema": "teacher_v4_fca_aggregate_v1", "per_seed": {str(k): v for k, v in per.items()}, "verdict": verdict, "checkpoint_300_R_vs_T": sec}
    json.dump(out, open(ROOT / "fca_aggregate.json", "w"), indent=1)
    print(json.dumps({"verdict": verdict, "per_seed": {s: (v["class"], v["flags"]) for s, v in per.items()}}, indent=1))


if __name__ == "__main__":
    main()
