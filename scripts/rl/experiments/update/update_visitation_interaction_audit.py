#!/usr/bin/env python3
"""Is the update's effect larger on the states the updated policy itself visits?

A 2x2 per update and per quarter of the episode: source and target policy,
evaluated on source-visited and target-visited states. The interaction term is
what the target policy does on its own states beyond the two main effects. A
clean interaction has to beat the update-only audit's AUC as well as pass on
its own.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rl.core.offline_audit import REPO, RUNS, OfflineAudit
from rl.experiments.shared.update_discriminator import (
    EPS,
    auc,
    cos_np,
    matched_controls,
    per_seed_direction,
)

RUN = "update_visitation_interaction_audit-2026-09-24"
V18 = "prospective_slope_validation-2026-09-24/prospective_slope_validation_report.json"
V19 = "update_functional_effect_audit-2026-09-24/update_functional_effect_report.json"
CONTRACT = "docs/update-visitation-interaction-audit-contract.md"
COLLECTOR = "scripts/rl/experiments/update/update_visitation_collect_states.py"
ORDER = ("T", "A", "O", "S")
PHASES = ((0, 16), (16, 32), (32, 48), (48, 64))
HEAVY = {"T": np.array([.7, .1, .1, .1], np.float32),
         "A": np.array([.1, .7, .1, .1], np.float32),
         "O": np.array([.1, .1, .7, .1], np.float32),
         "S": np.array([.1, .1, .1, .7], np.float32)}
CENTER = np.array([.25, .25, .25, .25], np.float32)
JAC_CHUNK = 128
ACTION_DIM = 12


def model_hash(path):
    """Hash the weights rather than the file, so a re-save still hits the cache."""
    import hashlib

    d = torch.load(path, map_location="cpu", weights_only=False)["model"]
    h = hashlib.sha256()
    for k in sorted(d):
        h.update(k.encode())
        h.update(d[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def rmsvec(x):
    return float(np.sqrt(np.mean(np.sum(np.asarray(x, float) ** 2, axis=-1))))


class UpdateVisitationInteractionAudit(OfflineAudit):
    """Update effect crossed with which policy's states it is measured on."""

    run = RUN
    report = "update_visitation_interaction_report.json"
    schema = "update_visitation_interaction_audit_v1"

    @property
    def cache(self):
        return self.dir / "state_cache"

    def model(self, path, obs_dim):
        from talon_rl.v2b_actor_critic import V2BSingleSiteFiLMActorCritic

        m = V2BSingleSiteFiLMActorCritic(obs_dim, ACTION_DIM).cuda()
        m.load_state_dict(torch.load(path, map_location="cuda", weights_only=False)["model"])
        m.eval()
        return m

    def eval_cell(self, m, obs_np, axis):
        from torch.func import jacrev, vmap

        o = torch.tensor(obs_np, dtype=torch.float32, device="cuda")
        wh = torch.tensor(HEAVY[axis], device="cuda").repeat(len(o), 1)
        wc = torch.tensor(CENTER, device="cuda").repeat(len(o), 1)
        with torch.no_grad():
            resp = (m.act_inference_with_preference(o, wh)
                    - m.act_inference_with_preference(o, wc)).cpu().numpy()
        w0 = torch.tensor(CENTER, dtype=torch.float32, device="cuda")

        def f_single(x, w):
            return m.act_inference_with_preference(x.unsqueeze(0), w.unsqueeze(0)).squeeze(0)

        # chunked, because the per-state Jacobian does not fit in one go
        js = [vmap(jacrev(f_single, argnums=1), in_dims=(0, None))(o[k:k + JAC_CHUNK], w0)
              .detach().cpu().numpy() for k in range(0, len(o), JAC_CHUNK)]
        jac = np.concatenate(js, axis=0)
        return {"resp": resp, "jac": jac, "resp_norm": rmsvec(resp),
                "J_norm": float(np.sqrt(np.mean(np.sum(jac.astype(float) ** 2, axis=(1, 2)))))}

    def quarters(self, ms, mt, zs, zt, axis, obs_dim):
        rows = []
        for qi, (lo, hi) in enumerate(PHASES, 1):
            ss = zs[:, lo:hi].reshape(-1, obs_dim)
            st = zt[:, lo:hi].reshape(-1, obs_dim)
            cells = {"SS": self.eval_cell(ms, ss, axis), "TS": self.eval_cell(mt, ss, axis),
                     "ST": self.eval_cell(ms, st, axis), "TT": self.eval_cell(mt, st, axis)}
            base = cells["SS"]
            d_resp = {k: 0.0 if k == "SS" else 1 - cos_np(v["resp"], base["resp"])
                      for k, v in cells.items()}
            d_j = {k: 0.0 if k == "SS" else 1 - cos_np(v["jac"], base["jac"])
                   for k, v in cells.items()}
            rows.append({
                "quarter": qi,
                "response_rotation_cells": d_resp,
                "J_rotation_cells": d_j,
                "I_resp_rotation": d_resp["TT"] - d_resp["TS"] - d_resp["ST"],
                "I_J_rotation": d_j["TT"] - d_j["TS"] - d_j["ST"],
                "I_resp_norm": (cells["TT"]["resp_norm"] - cells["TS"]["resp_norm"]
                                - cells["ST"]["resp_norm"] + base["resp_norm"]),
                "I_J_norm": (cells["TT"]["J_norm"] - cells["TS"]["J_norm"]
                             - cells["ST"]["J_norm"] + base["J_norm"]),
                "response_norm_cells": {k: v["resp_norm"] for k, v in cells.items()},
                "J_norm_cells": {k: v["J_norm"] for k, v in cells.items()}})
        return rows

    def family(self, rows, y, field, growthfield, v19_auc):
        q4 = np.array([r["quarters"][3][field] for r in rows])
        q1 = np.array([r["quarters"][0][field] for r in rows])
        growth = np.array([r[growthfield] for r in rows])
        pos, neg = q4[np.array(y) == 1], q4[np.array(y) == 0]
        medn = float(np.median(neg))
        iqr = float(np.percentile(neg, 75) - np.percentile(neg, 25))
        count = int(np.sum(pos > medn))
        a, ag = auc(y, q4), auc(y, growth)
        effect = (float(np.median(pos)) - medn) / max(iqr, EPS)
        seedok, perseed = per_seed_direction(rows, lambda r: r["quarters"][3][field])
        criteria = {"q4_auc_ge0p75": a >= .75,
                    "median_effect_ge0p5_iqr": effect >= .5,
                    "robust_above_retained_median_ge8": count >= 8,
                    "seed_consistent": seedok,
                    "growth_auc_ge0p70": ag >= .70,
                    "beats_v19_update_only_auc_by0p05": a >= v19_auc + .05}
        return {"q1_auc": auc(y, q1), "q4_auc": a, "growth_auc": ag,
                "robust_q4_median": float(np.median(pos)), "retained_q4_median": medn,
                "retained_q4_iqr": iqr, "effect_iqr_units": effect,
                "robust_above_retained_median_n": count, "seed_consistent": seedok,
                "per_seed": perseed, "v19_update_only_auc": v19_auc,
                "criteria": criteria, "passes": all(criteria.values())}

    def analyze(self):
        v18 = json.loads((RUNS / V18).read_text())
        v19 = json.loads((RUNS / V19).read_text())
        samples = v18["samples"]
        cp = {}
        for sd, arr in v18["states"].items():
            for st in arr:
                cp[(int(sd), st["label"])] = REPO / st["checkpoint"]

        models, rows = {}, []
        for ri, s in enumerate(samples, 1):
            axis = s["axis"]
            ps, pt = cp[(s["seed"], s["from"])], cp[(s["seed"], s["to"])]
            zs = np.load(self.cache / f"{model_hash(ps)}_{axis}.npz", allow_pickle=False)["obs"]
            zt = np.load(self.cache / f"{model_hash(pt)}_{axis}.npz", allow_pickle=False)["obs"]
            obs_dim = zs.shape[-1]
            for p in (ps, pt):
                if str(p) not in models:
                    models[str(p)] = self.model(p, obs_dim)
            q = self.quarters(models[str(ps)], models[str(pt)], zs, zt, axis, obs_dim)
            row = {"seed": s["seed"], "from": s["from"], "to": s["to"], "axis": axis,
                   "Y_robust_collapse": s["Y_robust_collapse"],
                   "source_G_sem": s["source_G_sem"], "quarters": q,
                   "growth_resp": q[3]["I_resp_rotation"] - q[0]["I_resp_rotation"],
                   "growth_J": q[3]["I_J_rotation"] - q[0]["I_J_rotation"],
                   "monotonic_increase_count_resp": sum(
                       q[i + 1]["I_resp_rotation"] > q[i]["I_resp_rotation"] for i in range(3)),
                   "monotonic_increase_count_J": sum(
                       q[i + 1]["I_J_rotation"] > q[i]["I_J_rotation"] for i in range(3))}
            rows.append(row)
            print(f'ROW {ri}/{len(samples)} {s["seed"]} {s["from"]}->{s["to"]} {axis} '
                  f'Y={s["Y_robust_collapse"]}', flush=True)

        y = [r["Y_robust_collapse"] for r in rows]
        v19aucs = {m: v19["metrics"][m]["auc"] for m in ("response_rotation_mean", "J_rotation")}
        families = {
            "response_rotation": self.family(rows, y, "I_resp_rotation", "growth_resp",
                                             v19aucs["response_rotation_mean"]),
            "J_rotation": self.family(rows, y, "I_J_rotation", "growth_J",
                                      v19aucs["J_rotation"])}
        secondary = {}
        for name, field in (("resp_norm", "I_resp_norm"), ("J_norm", "I_J_norm")):
            secondary[name] = {
                "q_auc": [auc(y, [r["quarters"][q][field] for r in rows]) for q in range(4)],
                "q_robust_median": [float(np.median([r["quarters"][q][field] for r in rows
                                                     if r["Y_robust_collapse"]])) for q in range(4)],
                "q_retained_median": [float(np.median([r["quarters"][q][field] for r in rows
                                                       if not r["Y_robust_collapse"]])) for q in range(4)]}
        if any(v["passes"] for v in families.values()):
            status = "UPDATE × VISITATION INTERACTION SUPPORTED"
        else:
            amp = any(v["q4_auc"] >= .65 and v["growth_auc"] >= .60
                      and v["robust_q4_median"] > v["retained_q4_median"]
                      for v in families.values())
            status = ("CLOSED-LOOP AMPLIFICATION WITHOUT CLEAN INTERACTION" if amp
                      else "NO INTERACTION EVIDENCE")
        return {"schema": self.schema, "status": status, "primary_n": len(rows),
                "robust_n": sum(y), "retained_n": len(rows) - sum(y),
                "state_cache_count": len(list(self.cache.glob("*.npz"))),
                "families": families, "secondary_magnitude": secondary,
                "matched_controls": matched_controls(rows), "rows": rows,
                "decision": {"training_method_authorized": False}}

    def execute(self):
        rep = super().execute()
        self.write({"status": "FROZEN_BY_HASH", "decision": rep["status"],
                    "report_sha256": self.sha(self.out / self.report),
                    "contract_sha256": self.sha(REPO / CONTRACT),
                    "collector_sha256": self.sha(REPO / COLLECTOR),
                    "analysis_sha256": self.sha(Path(__file__).resolve()),
                    "v18_sha256": self.sha(RUNS / V18),
                    "v19_sha256": self.sha(RUNS / V19),
                    "state_cache_count": len(list(self.cache.glob("*.npz")))},
                   "PROVENANCE_MANIFEST.json")
        return rep

    def summarize(self, rep):
        print(json.dumps({k: rep[k] for k in
                          ("status", "families", "secondary_magnitude")}, indent=2))


if __name__ == "__main__":
    UpdateVisitationInteractionAudit.main()
