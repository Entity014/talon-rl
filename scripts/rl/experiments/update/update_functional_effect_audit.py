#!/usr/bin/env python3
"""Does a robust-collapse update look different in function space, or only in
parameter magnitude?

For each recorded update it compares the source and target checkpoints on a
fixed probe set: how far the parameters moved, how far the actions moved per
preference, how the response to each preference rotated, and how the Jacobian
of action with respect to preference changed. A metric counts as a strong
discriminator only if it separates robust from retained updates by every one
of four criteria, consistently across all three seeds.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rl.core.offline_audit import REPO, RUNS, OfflineAudit

RUN = "update_functional_effect_audit-2026-09-24"
V18 = "prospective_slope_validation-2026-09-24/prospective_slope_validation_report.json"
CONTRACT = "docs/update-functional-effect-audit-contract.md"
PROBE = "fixed_probe_states.npz"
PREFS = {"T": np.array([.7, .1, .1, .1], np.float32),
         "A": np.array([.1, .7, .1, .1], np.float32),
         "O": np.array([.1, .1, .7, .1], np.float32),
         "S": np.array([.1, .1, .1, .7], np.float32),
         "C": np.array([.25, .25, .25, .25], np.float32)}
ORDER = ("T", "A", "O", "S")
SEEDS = (983001, 984001, 985001)
EPS = 1e-12
FUNCTIONAL = {"action_disp_mean", "action_disp_max", "action_change_cv",
              "response_change_mean", "response_rotation_mean", "response_rotation_max",
              "J_change_rel", "J_rotation", "specific_fraction", "D_specific"}
PARAMETER = {"param_norm", "param_rel_norm"}
GATE = {"count": 8, "effect": .5, "auc": .75}


def cos_np(a, b):
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > EPS else 1.0


def auc(y, s):
    y, s = np.asarray(y, int), np.asarray(s, float)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    z = sum(1.0 if s[i] > s[j] else .5 if s[i] == s[j] else 0.0 for i in pos for j in neg)
    return float(z / (len(pos) * len(neg)))


def group(n):
    if n.startswith("preference_embedding"):
        return "preference_embedding"
    if n.startswith("preference_film"):
        return "preference_film"
    if n.startswith("actor_mean"):
        return "action_head"
    if n == "log_std":
        return "log_std"
    return "actor_body"


def effective_rank(s):
    s = np.asarray(s, float)
    s = s[s > 1e-12]
    if len(s) == 0:
        return 0.0
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


class UpdateFunctionalEffectAudit(OfflineAudit):
    """Functional versus parameter-magnitude signature of a collapse update."""

    run = RUN
    report = "update_functional_effect_report.json"
    schema = "update_functional_effect_audit_v1"

    def checkpoint_features(self, path, obs):
        from talon_rl.v2b_actor_critic import V2BSingleSiteFiLMActorCritic

        state = torch.load(path, map_location="cpu", weights_only=False)["model"]
        m = V2BSingleSiteFiLMActorCritic(obs.shape[1], 12)
        m.load_state_dict(state)
        m.eval()
        ot = torch.tensor(obs, dtype=torch.float32)
        acts = {}
        with torch.no_grad():
            for lab, w in PREFS.items():
                wt = torch.tensor(w).repeat(len(obs), 1)
                acts[lab] = m.act_inference_with_preference(ot, wt).cpu().numpy()

        # d action / d preference at the centre preference, per probe state
        from torch.func import jacrev, vmap

        wc = torch.tensor(PREFS["C"], dtype=torch.float32)

        def f_single(o, w):
            return m.act_inference_with_preference(o.unsqueeze(0), w.unsqueeze(0)).squeeze(0)

        jac = vmap(jacrev(f_single, argnums=1), in_dims=(0, None))(ot, wc).detach().cpu().numpy()
        sv = np.linalg.svd(jac, compute_uv=False)
        pars = {n: p.detach().cpu().numpy() for n, p in m.named_parameters()
                if n.startswith(("actor_", "preference_embedding", "preference_film")) or n == "log_std"}
        return {"acts": acts, "jac": jac, "jac_sv_mean": sv.mean(0),
                "jac_effrank_mean": float(np.mean([effective_rank(x) for x in sv])),
                "pars": pars}

    def pair_row(self, s, a, b):
        names = sorted(a["pars"])
        flat_a = np.concatenate([a["pars"][n].ravel() for n in names])
        flat_d = np.concatenate([(b["pars"][n] - a["pars"][n]).ravel() for n in names])
        pn = float(np.linalg.norm(flat_d))
        group_sq = {g: 0.0 for g in ("actor_body", "preference_embedding",
                                     "preference_film", "action_head", "log_std")}
        for n in names:
            group_sq[group(n)] += float(np.sum((b["pars"][n] - a["pars"][n]) ** 2))
        tot = sum(group_sq.values()) + EPS

        dact = {lab: float(np.sqrt(np.mean(np.sum((b["acts"][lab] - a["acts"][lab]) ** 2, axis=1))))
                for lab in (*ORDER, "C")}
        av = np.array(list(dact.values()))

        rc, rr, peraxis = [], [], {}
        for lab in ORDER:
            ra = a["acts"][lab] - a["acts"]["C"]
            rb = b["acts"][lab] - b["acts"]["C"]
            chg = float(np.sqrt(np.mean(np.sum((rb - ra) ** 2, axis=1))))
            co = cos_np(ra, rb)
            rc.append(chg)
            rr.append(1 - co)
            peraxis[lab] = {"response_change_rms": chg, "response_cosine": co,
                            "response_rotation": 1 - co}

        ja, jb = a["jac"], b["jac"]
        jrel = float(np.linalg.norm(jb - ja) / (np.linalg.norm(ja) + EPS))
        jcos = cos_np(ja, jb)
        col_src = np.sqrt(np.mean(np.sum(ja ** 2, axis=1), axis=0))
        col_tgt = np.sqrt(np.mean(np.sum(jb ** 2, axis=1), axis=0))

        # split the action change into what every preference shares and what is its own
        stack = np.stack([b["acts"][lab] - a["acts"][lab] for lab in (*ORDER, "C")], axis=0)
        common = stack.mean(0)
        spec = stack - common[None, ...]
        dcommon = float(np.sqrt(np.mean(np.sum(common ** 2, axis=1))))
        dspec = float(np.sqrt(np.mean(np.sum(spec ** 2, axis=2))))

        metrics = {"param_norm": pn,
                   "param_rel_norm": pn / (float(np.linalg.norm(flat_a)) + EPS),
                   "action_disp_mean": float(av.mean()), "action_disp_max": float(av.max()),
                   "action_change_cv": float(av.std() / (av.mean() + EPS)),
                   "response_change_mean": float(np.mean(rc)),
                   "response_rotation_mean": float(np.mean(rr)),
                   "response_rotation_max": float(np.max(rr)),
                   "J_change_rel": jrel, "J_rotation": 1 - jcos,
                   "specific_fraction": dspec / (dcommon + dspec + EPS),
                   "D_common": dcommon, "D_specific": dspec}
        return {"seed": s["seed"], "from": s["from"], "to": s["to"], "axis": s["axis"],
                "Y_robust_collapse": s["Y_robust_collapse"], "source_G_sem": s["source_G_sem"],
                "metrics": metrics, "action_displacement_by_preference": dact,
                "response_by_axis": peraxis,
                "jacobian_column_relative_change": dict(
                    zip(ORDER, ((col_tgt - col_src) / (col_src + EPS)).tolist())),
                "source_jac_effrank": a["jac_effrank_mean"],
                "target_jac_effrank": b["jac_effrank_mean"],
                "group_displacement_share": {g: v / tot for g, v in group_sq.items()}}

    def discriminator(self, rows, m, y):
        pos = np.array([r["metrics"][m] for r in rows if r["Y_robust_collapse"]], float)
        neg = np.array([r["metrics"][m] for r in rows if not r["Y_robust_collapse"]], float)
        medn = float(np.median(neg))
        iqr = float(np.percentile(neg, 75) - np.percentile(neg, 25))
        count = int(np.sum(pos > medn))
        effect = (float(np.median(pos)) - medn) / max(iqr, 1e-12)
        a = auc(y, [r["metrics"][m] for r in rows])

        seedok, seedrows = True, {}
        for sd in SEEDS:
            pp = [r["metrics"][m] for r in rows if r["seed"] == sd and r["Y_robust_collapse"]]
            nn = [r["metrics"][m] for r in rows if r["seed"] == sd and not r["Y_robust_collapse"]]
            ok = None if not pp or not nn else float(np.median(pp)) > float(np.median(nn))
            seedrows[str(sd)] = {"positive_n": len(pp), "negative_n": len(nn),
                                 "robust_median": float(np.median(pp)) if pp else None,
                                 "retained_median": float(np.median(nn)) if nn else None,
                                 "direction_ok": ok}
            if ok is False:
                seedok = False
        passed = (count >= GATE["count"] and effect >= GATE["effect"]
                  and a >= GATE["auc"] and seedok)
        return {"robust_median": float(np.median(pos)), "retained_median": medn,
                "retained_iqr": iqr, "robust_above_retained_median_n": count,
                "effect_iqr_units": effect, "auc": a, "seed_consistent": seedok,
                "per_seed": seedrows, "strong_discriminator": passed}, passed

    def matched_controls(self, rows):
        """One retained update per robust one: same axis where possible, then
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
            keys = ("seed", "from", "to", "axis", "source_G_sem")
            matches.append({"robust": {k: r[k] for k in keys},
                            "retained": {k: negrows[i][k] for k in keys},
                            "source_G_abs_diff": abs(negrows[i]["source_G_sem"] - r["source_G_sem"]),
                            "reused": reused})
        return matches

    def analyze(self):
        rep = json.loads((RUNS / V18).read_text())
        obs = np.load(self.dir / PROBE, allow_pickle=False)["obs"].astype(np.float32)
        samples = list(rep["samples"])

        cp = {}
        for sd, arr in rep["states"].items():
            for st in arr:
                cp[(int(sd), st["label"])] = REPO / st["checkpoint"]
        paths = ({str(cp[(s["seed"], s["from"])]) for s in samples}
                 | {str(cp[(s["seed"], s["to"])]) for s in samples})
        cache = {}
        for i, p in enumerate(sorted(paths), 1):
            print(f"FEATURE {i}/{len(paths)} {Path(p).relative_to(REPO)}", flush=True)
            cache[p] = self.checkpoint_features(Path(p), obs)

        rows = [self.pair_row(s, cache[str(cp[(s["seed"], s["from"])])],
                              cache[str(cp[(s["seed"], s["to"])])]) for s in samples]

        y = [r["Y_robust_collapse"] for r in rows]
        summary, strong = {}, []
        for m in list(rows[0]["metrics"]):
            summary[m], passed = self.discriminator(rows, m, y)
            if passed:
                strong.append(m)
        funcstrong = [m for m in strong if m in FUNCTIONAL]
        paramstrong = [m for m in strong if m in PARAMETER]
        status = ("FUNCTIONAL UPDATE EFFECT SUPPORTED" if funcstrong and not paramstrong
                  else "PARAMETER-MAGNITUDE EFFECT ONLY" if paramstrong and not funcstrong
                  else "NO CLEAN UPDATE DISCRIMINATOR")
        return {"schema": self.schema, "status": status, "primary_n": len(rows),
                "robust_n": sum(y), "retained_n": len(rows) - sum(y),
                "probe_shape": list(obs.shape), "metrics": summary,
                "strong_discriminators": strong,
                "functional_strong_discriminators": funcstrong,
                "parameter_strong_discriminators": paramstrong,
                "matched_controls": self.matched_controls(rows), "rows": rows,
                "decision": {"training_method_authorized": False}}

    def execute(self):
        rep = super().execute()
        self.write({"status": "FROZEN_BY_HASH", "decision": rep["status"],
                    "report_sha256": self.sha(self.out / self.report),
                    "contract_sha256": self.sha(REPO / CONTRACT),
                    "script_sha256": self.sha(Path(__file__).resolve()),
                    "probe_sha256": self.sha(self.dir / PROBE),
                    "v18_report_sha256": self.sha(RUNS / V18)},
                   "PROVENANCE_MANIFEST.json")
        return rep

    def summarize(self, rep):
        print(json.dumps({k: rep[k] for k in
                          ("status", "strong_discriminators",
                           "functional_strong_discriminators",
                           "parameter_strong_discriminators", "metrics")}, indent=2))


if __name__ == "__main__":
    UpdateFunctionalEffectAudit.main()
