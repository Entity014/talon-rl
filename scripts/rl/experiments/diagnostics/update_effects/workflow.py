"""Consolidated experiment stages. Use the first CLI argument to select a former script stage."""
from __future__ import annotations
import argparse

def run_update_effect_collect_probe():
    """Run former update_effect_collect_probe.py stage."""
    from pathlib import Path
    import sys,hashlib
    import numpy as np, torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
    OUT=ROOT/'runs/update_functional_effect_audit-2026-09-24';OUT.mkdir(parents=True,exist_ok=True)
    CKPT=ROOT/'runs/v2b_adam_continuous-2026-09-24/model_75.pt'
    SEEDS=(850001,850002,850003,850004);PHASES=(0,8,16,24,32,40,48,56);NENV=8;STEPS=64
    W=np.array([.25,.25,.25,.25],np.float32)
    def obs_tensor(x):
        if isinstance(x,dict):x=x.get('policy',next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def main():
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({'headless':True,'enable_cameras':False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.conditioning.single_site_film import V2BSingleSiteFiLMActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/'talon_rl/assets/data/Robots/unitree_a1/a1.usd')
            env=gym.make('Isaac-Velocity-Flat-Unitree-A1-v0',cfg=cfg);o,_=env.reset(seed=0);o=obs_tensor(o).cuda();ad=env.unwrapped.action_manager.total_action_dim
            m=V2BSingleSiteFiLMActorCritic(o.shape[-1],ad).cuda();m.load_state_dict(torch.load(CKPT,map_location='cuda',weights_only=False)['model']);m.eval()
            obs=[];meta=[]
            for sd in SEEDS:
                cur,_=env.reset(seed=sd);cur=obs_tensor(cur).cuda();w=torch.tensor(W,device='cuda').repeat(NENV,1)
                with torch.no_grad():
                    for t in range(STEPS):
                        if t in PHASES:
                            obs.append(cur.detach().cpu().numpy());meta.extend([(sd,t,e) for e in range(NENV)])
                        a=m.act_inference_with_preference(cur,w);nxt,_,_,_,_=env.step(a);cur=obs_tensor(nxt).cuda()
            obs=np.concatenate(obs,axis=0);meta=np.asarray(meta,np.int64)
            out=OUT/'fixed_probe_states.npz';np.savez_compressed(out,obs=obs,meta=meta,seeds=np.array(SEEDS),phases=np.array(PHASES),center_w=W)
            print('PROBE_WRITTEN',out,'shape',obs.shape)
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_update_functional_effect_audit():
    """Run former update_functional_effect_audit.py stage."""
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
    
    from rl.core.diagnostics.offline_audit import REPO, RUNS, OfflineAudit
    from rl.experiments.common.utilities.update_discriminator import (
        EPS,
        auc,
        cos_np,
        matched_controls,
        per_seed_direction,
    )
    
    RUN = "update_functional_effect_audit-2026-09-24"
    V18 = "prospective_slope_validation-2026-09-24/prospective_slope_validation_report.json"
    CONTRACT = "docs/contracts/diagnostics/update-functional-effect-audit-contract.md"
    PROBE = "fixed_probe_states.npz"
    PREFS = {"T": np.array([.7, .1, .1, .1], np.float32),
             "A": np.array([.1, .7, .1, .1], np.float32),
             "O": np.array([.1, .1, .7, .1], np.float32),
             "S": np.array([.1, .1, .1, .7], np.float32),
             "C": np.array([.25, .25, .25, .25], np.float32)}
    ORDER = ("T", "A", "O", "S")
    FUNCTIONAL = {"action_disp_mean", "action_disp_max", "action_change_cv",
                  "response_change_mean", "response_rotation_mean", "response_rotation_max",
                  "J_change_rel", "J_rotation", "specific_fraction", "D_specific"}
    PARAMETER = {"param_norm", "param_rel_norm"}
    GATE = {"count": 8, "effect": .5, "auc": .75}
    
    
    
    
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
            from talon_rl.models.conditioning.single_site_film import V2BSingleSiteFiLMActorCritic
    
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
    
            seedok, seedrows = per_seed_direction(rows, lambda r: r["metrics"][m])
            passed = (count >= GATE["count"] and effect >= GATE["effect"]
                      and a >= GATE["auc"] and seedok)
            return {"robust_median": float(np.median(pos)), "retained_median": medn,
                    "retained_iqr": iqr, "robust_above_retained_median_n": count,
                    "effect_iqr_units": effect, "auc": a, "seed_consistent": seedok,
                    "per_seed": seedrows, "strong_discriminator": passed}, passed
    
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
                    "matched_controls": matched_controls(rows), "rows": rows,
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
    
    
    if True:
        UpdateFunctionalEffectAudit.main()

def run_update_visitation_collect_states():
    """Run former update_visitation_collect_states.py stage."""
    from pathlib import Path
    import os,sys,hashlib,json
    import numpy as np, torch
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
    RUN=ROOT/'runs/update_visitation_interaction_audit-2026-09-24'; CACHE=RUN/'state_cache'; V18=ROOT/'runs/prospective_slope_validation-2026-09-24/prospective_slope_validation_report.json'
    ORDER=('T','A','O','S'); NENV=8; STEPS=64; SUITES=4
    PREFS={'T':np.array([.7,.1,.1,.1],np.float32),'A':np.array([.1,.7,.1,.1],np.float32),'O':np.array([.1,.1,.7,.1],np.float32),'S':np.array([.1,.1,.1,.7],np.float32)}
    
    def obs_tensor(x):
        if isinstance(x,dict):x=x.get('policy',next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def mh(path):
        d=torch.load(path,map_location='cpu',weights_only=False)['model'];h=hashlib.sha256()
        for k in sorted(d):h.update(k.encode());h.update(d[k].detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    def items():
        d=json.load(open(V18)); st=d['states']; out=[]; seen=set()
        for s in d['samples']:
            sd=str(s['seed']); axis=s['axis']
            for lab in (s['from'],s['to']):
                rec=next(x for x in st[sd] if x['label']==lab); p=ROOT/rec['checkpoint']; key=(str(p),axis)
                if key not in seen: seen.add(key); out.append((int(sd),lab,axis,p))
        out.sort(key=lambda x:(x[0],int(x[1][1:]) if x[1].startswith('u') else 0,x[2]))
        return out
    
    def main():
        idx=int(os.environ['UV_SHARD_ID']); sd,lab,axis,p=items()[idx]; hh=mh(p); CACHE.mkdir(parents=True,exist_ok=True); cp=CACHE/f'{hh}_{axis}.npz'
        if cp.exists(): print('CACHE_EXISTS',cp.name); return
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({'headless':True,'enable_cameras':False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.conditioning.single_site_film import V2BSingleSiteFiLMActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/'talon_rl/assets/data/Robots/unitree_a1/a1.usd')
            env=gym.make('Isaac-Velocity-Flat-Unitree-A1-v0',cfg=cfg);o,_=env.reset(seed=0);o=obs_tensor(o).cuda();ad=env.unwrapped.action_manager.total_action_dim
            m=V2BSingleSiteFiLMActorCritic(o.shape[-1],ad).cuda();m.load_state_dict(torch.load(p,map_location='cuda',weights_only=False)['model']);m.eval();w=torch.tensor(PREFS[axis],device='cuda').repeat(NENV,1)
            obs=np.zeros((SUITES,STEPS,NENV,o.shape[-1]),np.float32)
            for si in range(SUITES):
                cur,_=env.reset(seed=840001+si);cur=obs_tensor(cur).cuda()
                with torch.no_grad():
                    for t in range(STEPS):
                        obs[si,t]=cur.detach().cpu().numpy();a=m.act_inference_with_preference(cur,w);nxt,_,_,_,_=env.step(a);cur=obs_tensor(nxt).cuda()
            tmp=cp.with_suffix('.tmp.npz');np.savez_compressed(tmp,obs=obs,checkpoint_hash=np.array(hh),checkpoint_path=np.array(str(p.relative_to(ROOT))),seed=np.array(sd),label=np.array(lab),axis=np.array(axis),reset_seeds=np.arange(840001,840005));tmp.replace(cp)
            print('STATE_CACHE_WRITTEN',cp.name,'shape',obs.shape)
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_update_visitation_interaction_audit():
    """Run former update_visitation_interaction_audit.py stage."""
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
    
    from rl.core.diagnostics.offline_audit import REPO, RUNS, OfflineAudit
    from rl.experiments.common.utilities.update_discriminator import (
        EPS,
        auc,
        cos_np,
        matched_controls,
        per_seed_direction,
    )
    
    RUN = "update_visitation_interaction_audit-2026-09-24"
    V18 = "prospective_slope_validation-2026-09-24/prospective_slope_validation_report.json"
    V19 = "update_functional_effect_audit-2026-09-24/update_functional_effect_report.json"
    CONTRACT = "docs/contracts/diagnostics/update-visitation-interaction-audit-contract.md"
    COLLECTOR = "scripts/rl/experiments/diagnostics/update_effects/update_visitation_collect_states.py"
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
            from talon_rl.models.conditioning.single_site_film import V2BSingleSiteFiLMActorCritic
    
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
    
    
    if True:
        UpdateVisitationInteractionAudit.main()

STAGES = {
    "update_effect_collect_probe": run_update_effect_collect_probe,
    "update_functional_effect_audit": run_update_functional_effect_audit,
    "update_visitation_collect_states": run_update_visitation_collect_states,
    "update_visitation_interaction_audit": run_update_visitation_interaction_audit,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
