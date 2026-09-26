"""Consolidated thematic experiment stages."""
from __future__ import annotations
import argparse

def run_authority_isolated_ai_h2_early_window_authority_audit():
    """Run former authority_isolated_ai_h2_early_window_authority_audit.py stage."""
    """Early-window H2 authority: what the semantic path keeps at updates 5 and 10."""
    import json
    import sys
    from pathlib import Path
    
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    
    import rl.experiments.common.utilities.authority_isolated_h1_screen as h1
    from rl.experiments.common.utilities.authority_retention import (
        AuthorityRetentionAudit,
        actions,
        edge_retention,
    )
    
    RUN = "authority_isolated_ai_h2_semantic_path-2026-09-25"
    
    
    class EarlyWindowAuthorityAudit(AuthorityRetentionAudit):
        """Early-window H2 authority: what the semantic path keeps at updates 5 and 10."""
    
        run = "authority_isolated_ai_h2_early_window_compatibility-2026-09-25"
        report = "authority.json"
        schema = "ai_h2_early_window_authority_v1"
        support_seed = 2609252901
        snapshots = (5, 10)
    
        def analyze(self):
            x, sample = self.support()
            ref = self.load_reference()
            base = h1.sensitivity(ref, sample)
            ref_actions = actions(ref, x)
            rep = {"schema": self.schema, "snapshots": {}}
            for snap in self.snapshots:
                m = self.load_policy(f"{RUN}/model_{snap}.pt")
                q = h1.sensitivity(m, sample)
                pair = self.ratio(q, base, "pairwise_action_distance", "mean")
                tan = self.ratio(q, base, "tangent_jacobian_fro_mean")
                edges = edge_retention(ref_actions, actions(m, x))
                rep["snapshots"][str(snap)] = {
                    "global_update": 30 + snap,
                    "pairwise_retention": pair,
                    "tangent_retention": tan,
                    "min_edge": min(edges.values()),
                    "min_edge_name": min(edges, key=edges.get),
                    "edges": edges,
                    "authority_gate": bool(pair >= .9 and tan >= .9),
                    "edge_gate": bool(min(edges.values()) >= .9)}
            return rep
    
        def summarize(self, report):
            print(json.dumps(report, indent=2))
    
    
    if True:
        EarlyWindowAuthorityAudit.main()

def run_authority_isolated_ai_h2_early_window_noregression_eval():
    """Run former authority_isolated_ai_h2_early_window_noregression_eval.py stage."""
    from pathlib import Path
    import json,sys,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    from talon_rl.models.authority.isolated import AuthorityIsolatedActorCritic
    from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
    CK={
    "u35":ROOT/"runs/authority_isolated_ai_h2_semantic_path-2026-09-25/model_5.pt",
    "u40":ROOT/"runs/authority_isolated_ai_h2_semantic_path-2026-09-25/model_10.pt"
    }
    OUT=ROOT/"runs/authority_isolated_ai_h2_early_window_compatibility-2026-09-25";OUT.mkdir(parents=True,exist_ok=True)
    ORDER=("T","A","O","S","C");PREFS={"T":[.7,.1,.1,.1],"A":[.1,.7,.1,.1],"O":[.1,.1,.7,.1],"S":[.1,.1,.1,.7],"C":[.25]*4}
    SEM={"suite2":840003,"suite3":840004};HELD={"suite4":850101,"suite5":850202,"suite6":850303}
    NENV=8;H=64;G=.99
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def ev(y,p):
        y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
        return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def ret(R,D,seg=None):
        out=np.zeros_like(R)
        if seg is None:
            run=np.zeros_like(R[0])
            for t in range(len(R)-1,-1,-1):
                run=R[t]+G*run*(~D[t])[:,None];out[t]=run
        else:
            for st in range(0,len(R),seg):
                en=min(st+seg,len(R));run=np.zeros_like(R[0])
                for t in range(en-1,st-1,-1):
                    run=R[t]+G*run*(~D[t])[:,None];out[t]=run
        return out
    def rollout(env,m,wv,seed):
        from talon_rl.rewards.objectives import normalized_objective_vector
        mgr=env.unwrapped.reward_manager;w=torch.tensor(wv,device="cuda").repeat(NENV,1)
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda();R=[];D=[];V=[];done=np.zeros(NENV,bool);reasons=[[] for _ in range(NENV)]
        with torch.no_grad():
            for t in range(H):
                V.append(m.value_with_preference(cur,w).cpu().numpy());a=m.act_inference_with_preference(cur,w)
                nxt,_,te,tr,_=env.step(a);dd=(te|tr).cpu().numpy();done|=dd
                tm=env.unwrapped.termination_manager
                for i in range(NENV):
                    if dd[i]:
                        reasons[i]=[n for n in tm.active_terms if bool(tm.get_term(n)[i].detach().cpu())]
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                R.append(normalized_objective_vector(terms(raw,names),shape=(NENV,))*env.unwrapped.step_dt)
                D.append(dd);cur=ot(nxt).cuda()
        R=np.asarray(R);D=np.asarray(D,bool);V=np.asarray(V);H32=ret(R,D,32);MC=ret(R,D,None)
        return {"survival":float(1-done.mean()),"fail_count":int(done.sum()),"reasons":reasons,
          "h32_ev":[ev(H32[:,:,j],V[:,:,j]) for j in range(4)],
          "mc64_ev":[ev(MC[:,:,j],V[:,:,j]) for j in range(4)],
          "h32_bias":[float(np.mean(V[:,:,j]-H32[:,:,j])) for j in range(4)],
          "mc64_bias":[float(np.mean(V[:,:,j]-MC[:,:,j])) for j in range(4)]}
    def summarize(rows):
        h=np.array([r["h32_ev"] for r in rows]);m=np.array([r["mc64_ev"] for r in rows])
        hb=np.array([r["h32_bias"] for r in rows]);mb=np.array([r["mc64_bias"] for r in rows])
        return {"h32_ev_mean":float(h.mean()),"h32_negative_fraction":float((h<0).mean()),
          "mc64_ev_mean":float(m.mean()),"mc64_negative_fraction":float((m<0).mean()),
          "h32_mean_abs_bias":float(np.abs(hb).mean()),"mc64_mean_abs_bias":float(np.abs(mb).mean()),
          "min_survival":float(min(r["survival"] for r in rows)),"failed_lanes":int(sum(r["fail_count"] for r in rows))}
    def main():
        from isaaclab.app import AppLauncher
        sv=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv;env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim
            rep={"schema":"authority_isolated_ai_c2_endpoint_eval_v1","models":{}}
            for arm,path in CK.items():
                cls=AuthorityIsolatedActorCritic
                m=cls(o.shape[-1],ad).cuda();m.load_state_dict(torch.load(path,map_location="cuda",weights_only=False)["model"]);m.eval()
                sem=[];held=[];fresh=[]
                for suite,seed in SEM.items():
                    for lab in ORDER:
                        q=rollout(env,m,PREFS[lab],seed);q.update({"suite":suite,"preference":lab});sem.append(q)
                for suite,seed in HELD.items():
                    for lab in ORDER:
                        q=rollout(env,m,PREFS[lab],seed);q.update({"suite":suite,"preference":lab});held.append(q)
                for li,lab in enumerate(ORDER):
                    for si in range(4):
                        seed=9700000+li*1000+si*113
                        q=rollout(env,m,PREFS[lab],seed);q.update({"suite":si,"preference":lab,"seed":seed});fresh.append(q)
                fs=summarize(fresh);ss=summarize(sem);hs=summarize(held)
                critic_gate=bool(fs["h32_ev_mean"]>0 and fs["mc64_ev_mean"]>0 and fs["h32_negative_fraction"]<=.25 and fs["mc64_negative_fraction"]<=.25 and fs["min_survival"]>=.95)
                rep["models"][arm]={"semantic":ss,"heldout":hs,"fresh_critic":fs,"critic_gate":critic_gate,
                                    "semantic_rows":sem,"heldout_rows":held,"fresh_rows":fresh}
                print("SUMMARY",arm,json.dumps(rep["models"][arm],default=str)[:2000],flush=True)
            (OUT/"endpoint_eval.json").write_text(json.dumps(rep,indent=2)+"\n")
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_authority_isolated_ai_h2_endpoint_timeline():
    """Run former authority_isolated_ai_h2_endpoint_timeline.py stage."""
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from pathlib import Path
    import json,sys,hashlib,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"scripts/rl")]
    from rl.experiments.common.utilities.foundation_v2_semantic_eval import evaluate,obs_tensor as ot,PREFS,ORDER,IDX,PHYS,SUITES
    from talon_rl.models.authority.isolated import AuthorityIsolatedActorCritic
    RUN=ROOT/"runs/authority_isolated_ai_h2_semantic_path-2026-09-25"
    OUT=RUN
    SNAPS=(0,5,10,15,20,25)
    GLOBAL={0:30,5:35,10:40,15:45,20:50,25:55}
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def main():
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=8;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda()
            ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
            rows=[]
            for snap in SNAPS:
                m=AuthorityIsolatedActorCritic(o.shape[-1],ad).cuda()
                m.load_state_dict(torch.load(RUN/f"model_{snap}.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
                erows=[]
                for suite in range(SUITES):
                    seed=840001+suite
                    for lab in ("T","A","O","S","C"):
                        q=evaluate(env,m,mgr,robot,PREFS[lab],seed);q.update({"suite":suite,"label":lab});erows.append(q)
                endpoint={};passes={}
                for lab in ORDER:
                    j=IDX[lab];pk=PHYS[lab];oo=[];pp=[];sv=[];do=[];dp=[]
                    for suite in range(SUITES):
                        r=next(x for x in erows if x["suite"]==suite and x["label"]==lab)
                        c=next(x for x in erows if x["suite"]==suite and x["label"]=="C")
                        od=r["normalized_objective_mean"][j]-c["normalized_objective_mean"][j]
                        pd=r["physical"][pk]-c["physical"][pk]
                        oo.append(od>0);pp.append(pd<0);sv.append(r["survival"]);do.append(od);dp.append(pd)
                    endpoint[lab]={"objective_correct_fraction":float(np.mean(oo)),
                        "physical_correct_fraction":float(np.mean(pp)),
                        "mean_objective_delta_vs_center":float(np.mean(do)),
                        "mean_physical_delta_vs_center":float(np.mean(dp)),
                        "min_survival":float(np.min(sv))}
                    passes[lab]=bool(endpoint[lab]["objective_correct_fraction"]>=.75 and endpoint[lab]["physical_correct_fraction"]>=.75 and endpoint[lab]["min_survival"]>=.95)
                row={"snapshot":snap,"global_update":GLOBAL[snap],"endpoint":endpoint,"endpoint_pass":passes,
                     "pass_axes":[a for a in ORDER if passes[a]],"pass_count":int(sum(passes.values())),
                     "min_survival":float(min(x["survival"] for x in erows))}
                rows.append(row);print("SNAP",snap,"GLOBAL",GLOBAL[snap],"PASS",row["pass_axes"],flush=True)
    
            # Timeline semantics.
            first={};retained={};pf=[];fp=[];jacc=[];rot=[]
            for a in ORDER:
                idx=[i for i,r in enumerate(rows) if r["endpoint_pass"][a]]
                first[a]=(rows[idx[0]]["global_update"] if idx else None)
                if idx:
                    k=idx[0];retained[a]=float(np.mean([r["endpoint_pass"][a] for r in rows[k:]]))
                else:retained[a]=None
            for k in range(len(rows)-1):
                a=rows[k];b=rows[k+1]
                sa=set(a["pass_axes"]);sb=set(b["pass_axes"])
                for x in ORDER:
                    if a["endpoint_pass"][x] and not b["endpoint_pass"][x]:pf.append({"axis":x,"from":a["global_update"],"to":b["global_update"]})
                    if (not a["endpoint_pass"][x]) and b["endpoint_pass"][x]:fp.append({"axis":x,"from":a["global_update"],"to":b["global_update"]})
                uni=sa|sb;jacc.append(float(len(sa&sb)/len(uni)) if uni else 1.0)
                if a["pass_count"]>=b["pass_count"] and sa!=sb:rot.append({"from":a["global_update"],"to":b["global_update"],"source":sorted(sa),"target":sorted(sb)})
            source_pass=sum(rows[k]["pass_count"] for k in range(len(rows)-1))
            pfr=float(len(pf)/source_pass) if source_pass else 0.0
            rep={"schema":"authority_isolated_ai_h2_endpoint_timeline_v1","measurement_only":True,
                 "rows":rows,"first_pass_global_update":first,"retained_fraction_after_first_pass":retained,
                 "pass_to_fail_events":pf,"fail_to_pass_events":fp,"pass_to_fail_rate":pfr,
                 "consecutive_pass_set_jaccard":jacc,"winner_rotation_transitions":rot,
                 "max_simultaneous_passes":max(r["pass_count"] for r in rows),
                 "final_pass_count":rows[-1]["pass_count"],"final_pass_axes":rows[-1]["pass_axes"]}
            out=OUT/"h2_endpoint_timeline.json";out.write_text(json.dumps(rep,indent=2)+"\n")
            print(json.dumps({k:rep[k] for k in ("first_pass_global_update","retained_fraction_after_first_pass","pass_to_fail_events","fail_to_pass_events","pass_to_fail_rate","winner_rotation_transitions","max_simultaneous_passes","final_pass_count","final_pass_axes")},indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_authority_isolated_ai_h2_final_authority_audit():
    """Run former authority_isolated_ai_h2_final_authority_audit.py stage."""
    """Final H2 authority: what the semantic-path checkpoint keeps at update 25."""
    import json
    import sys
    from pathlib import Path
    
    import numpy as np
    
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    
    import rl.experiments.common.utilities.authority_isolated_h1_screen as h1
    from rl.experiments.common.utilities.authority_retention import (
        AuthorityRetentionAudit,
        actions,
        edge_retention,
    )
    
    
    class FinalAuthorityAudit(AuthorityRetentionAudit):
        """Final H2 authority: what the semantic-path checkpoint keeps at update 25."""
    
        run = "authority_isolated_ai_h2_final_authority_audit-2026-09-25"
        report = "final_authority_audit.json"
        schema = "ai_h2_final_authority_audit_v1"
        support_seed = 2609252801
        checkpoint = "authority_isolated_ai_h2_semantic_path-2026-09-25/model_25.pt"
    
        def analyze(self):
            x, sample = self.support()
            ref = self.load_reference()
            m = self.load_policy(self.checkpoint)
            base = h1.sensitivity(ref, sample)
            q = h1.sensitivity(m, sample)
            pair = self.ratio(q, base, "pairwise_action_distance", "mean")
            tan = self.ratio(q, base, "tangent_jacobian_fro_mean")
            fun = self.ratio(q, base, "centered_functional_geometry", "specific_rms")
            edges = edge_retention(actions(ref, x), actions(m, x))
            heavy_heavy = [v for k, v in edges.items() if "C" not in k.split("-")]
            heavy_center = [v for k, v in edges.items() if "C" in k.split("-")]
            return {"schema": self.schema,
                    "pairwise_retention": pair,
                    "tangent_retention": tan,
                    "functional_rms_retention": fun,
                    "heavy_heavy_mean": float(np.mean(heavy_heavy)),
                    "heavy_center_mean": float(np.mean(heavy_center)),
                    "min_edge": float(min(edges.values())),
                    "min_edge_name": min(edges, key=edges.get),
                    "edges": edges,
                    "parameter_rank": q["centered_parameter_geometry"]["effective_rank_5pct"],
                    "functional_rank": q["centered_functional_geometry"]["effective_rank_5pct"],
                    "authority_gate": bool(pair >= .9 and tan >= .9),
                    "edge_gate": bool(min(edges.values()) >= .9)}
    
        def summarize(self, report):
            print(json.dumps(report, indent=2))
    
    
    if True:
        FinalAuthorityAudit.main()

def run_authority_isolated_ai_h2_final_noregression_eval():
    """Run former authority_isolated_ai_h2_final_noregression_eval.py stage."""
    from pathlib import Path
    import json,sys,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    from talon_rl.models.authority.isolated import AuthorityIsolatedActorCritic
    from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
    CK={"h2":ROOT/"runs/authority_isolated_ai_h2_semantic_path-2026-09-25/model_25.pt"}
    OUT=ROOT/"runs/authority_isolated_ai_h2_final_noregression_eval-2026-09-25";OUT.mkdir(parents=True,exist_ok=True)
    ORDER=("T","A","O","S","C");PREFS={"T":[.7,.1,.1,.1],"A":[.1,.7,.1,.1],"O":[.1,.1,.7,.1],"S":[.1,.1,.1,.7],"C":[.25]*4}
    SEM={"suite2":840003,"suite3":840004};HELD={"suite4":850101,"suite5":850202,"suite6":850303}
    NENV=8;H=64;G=.99
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def ev(y,p):
        y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
        return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def ret(R,D,seg=None):
        out=np.zeros_like(R)
        if seg is None:
            run=np.zeros_like(R[0])
            for t in range(len(R)-1,-1,-1):
                run=R[t]+G*run*(~D[t])[:,None];out[t]=run
        else:
            for st in range(0,len(R),seg):
                en=min(st+seg,len(R));run=np.zeros_like(R[0])
                for t in range(en-1,st-1,-1):
                    run=R[t]+G*run*(~D[t])[:,None];out[t]=run
        return out
    def rollout(env,m,wv,seed):
        from talon_rl.rewards.objectives import normalized_objective_vector
        mgr=env.unwrapped.reward_manager;w=torch.tensor(wv,device="cuda").repeat(NENV,1)
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda();R=[];D=[];V=[];done=np.zeros(NENV,bool);reasons=[[] for _ in range(NENV)]
        with torch.no_grad():
            for t in range(H):
                V.append(m.value_with_preference(cur,w).cpu().numpy());a=m.act_inference_with_preference(cur,w)
                nxt,_,te,tr,_=env.step(a);dd=(te|tr).cpu().numpy();done|=dd
                tm=env.unwrapped.termination_manager
                for i in range(NENV):
                    if dd[i]:
                        reasons[i]=[n for n in tm.active_terms if bool(tm.get_term(n)[i].detach().cpu())]
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                R.append(normalized_objective_vector(terms(raw,names),shape=(NENV,))*env.unwrapped.step_dt)
                D.append(dd);cur=ot(nxt).cuda()
        R=np.asarray(R);D=np.asarray(D,bool);V=np.asarray(V);H32=ret(R,D,32);MC=ret(R,D,None)
        return {"survival":float(1-done.mean()),"fail_count":int(done.sum()),"reasons":reasons,
          "h32_ev":[ev(H32[:,:,j],V[:,:,j]) for j in range(4)],
          "mc64_ev":[ev(MC[:,:,j],V[:,:,j]) for j in range(4)],
          "h32_bias":[float(np.mean(V[:,:,j]-H32[:,:,j])) for j in range(4)],
          "mc64_bias":[float(np.mean(V[:,:,j]-MC[:,:,j])) for j in range(4)]}
    def summarize(rows):
        h=np.array([r["h32_ev"] for r in rows]);m=np.array([r["mc64_ev"] for r in rows])
        hb=np.array([r["h32_bias"] for r in rows]);mb=np.array([r["mc64_bias"] for r in rows])
        return {"h32_ev_mean":float(h.mean()),"h32_negative_fraction":float((h<0).mean()),
          "mc64_ev_mean":float(m.mean()),"mc64_negative_fraction":float((m<0).mean()),
          "h32_mean_abs_bias":float(np.abs(hb).mean()),"mc64_mean_abs_bias":float(np.abs(mb).mean()),
          "min_survival":float(min(r["survival"] for r in rows)),"failed_lanes":int(sum(r["fail_count"] for r in rows))}
    def main():
        from isaaclab.app import AppLauncher
        sv=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv;env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim
            rep={"schema":"authority_isolated_ai_c2_endpoint_eval_v1","models":{}}
            for arm,path in CK.items():
                cls=AuthorityIsolatedActorCritic
                m=cls(o.shape[-1],ad).cuda();m.load_state_dict(torch.load(path,map_location="cuda",weights_only=False)["model"]);m.eval()
                sem=[];held=[];fresh=[]
                for suite,seed in SEM.items():
                    for lab in ORDER:
                        q=rollout(env,m,PREFS[lab],seed);q.update({"suite":suite,"preference":lab});sem.append(q)
                for suite,seed in HELD.items():
                    for lab in ORDER:
                        q=rollout(env,m,PREFS[lab],seed);q.update({"suite":suite,"preference":lab});held.append(q)
                for li,lab in enumerate(ORDER):
                    for si in range(4):
                        seed=9700000+li*1000+si*113
                        q=rollout(env,m,PREFS[lab],seed);q.update({"suite":si,"preference":lab,"seed":seed});fresh.append(q)
                fs=summarize(fresh);ss=summarize(sem);hs=summarize(held)
                critic_gate=bool(fs["h32_ev_mean"]>0 and fs["mc64_ev_mean"]>0 and fs["h32_negative_fraction"]<=.25 and fs["mc64_negative_fraction"]<=.25 and fs["min_survival"]>=.95)
                rep["models"][arm]={"semantic":ss,"heldout":hs,"fresh_critic":fs,"critic_gate":critic_gate,
                                    "semantic_rows":sem,"heldout_rows":held,"fresh_rows":fresh}
                print("SUMMARY",arm,json.dumps(rep["models"][arm],default=str)[:2000],flush=True)
            (OUT/"endpoint_eval.json").write_text(json.dumps(rep,indent=2)+"\n")
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_authority_isolated_ai_h2_final_semantic_eval():
    """Run former authority_isolated_ai_h2_final_semantic_eval.py stage."""
    from pathlib import Path
    import argparse, hashlib, json, sys
    import numpy as np
    import torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    CKPT=ROOT/"runs/authority_isolated_ai_h2_semantic_path-2026-09-25/model_25.pt"
    OUT_DEFAULT=ROOT/"runs/authority_isolated_ai_h2_final_semantic_eval-2026-09-25"
    NENV=8;STEPS=64;SUITES=4;G=.99
    ORDER=("T","A","O","S")
    PREFS={
     "T":np.array([.7,.1,.1,.1],np.float32),
     "A":np.array([.1,.7,.1,.1],np.float32),
     "O":np.array([.1,.1,.7,.1],np.float32),
     "S":np.array([.1,.1,.1,.7],np.float32),
     "C":np.array([.25,.25,.25,.25],np.float32)}
    IDX={"T":0,"A":1,"O":2,"S":3}
    PHYS={"T":"tracking_error","A":"ang_vel_xy","O":"tilt_deg","S":"action_rate"}
    ALPHAS=(0.0,.25,.5,.75,1.0)
    PATHS=(("T","A"),("T","O"),("T","S"))
    
    def obs_tensor(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def tilt_deg(q):
        _,x,y,_=[q[:,i] for i in range(4)]
        return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def ev(y,p):
        y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
        return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def segret(R,D,st,en):
        out=np.zeros_like(R[st:en]);run=np.zeros_like(R[0])
        for t in range(en-1,st-1,-1):
            run=R[t]+G*run*(~D[t])[:,None];out[t-st]=run
        return out
    
    def evaluate(env,m,mgr,robot,w_np,seed):
        from talon_rl.rewards.objectives import normalized_objective_vector
        w=torch.tensor(w_np,device="cuda").repeat(NENV,1)
        cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda()
        R=[];D=[];V=[];phys=[];prev=torch.zeros((NENV,env.unwrapped.action_manager.total_action_dim),device="cuda")
        done_any=np.zeros(NENV,bool)
        with torch.no_grad():
            for _ in range(STEPS):
                V.append(m.value_with_preference(cur,w).cpu().numpy())
                a=m.act_inference_with_preference(cur,w)
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                R.append(normalized_objective_vector(terms(raw,names),shape=(NENV,))*env.unwrapped.step_dt)
                dd=(te|tr).cpu().numpy();D.append(dd);done_any|=dd
                data=robot.data;cmd=env.unwrapped.command_manager.get_command("base_velocity")
                vx=float((data.root_lin_vel_b[:,0]-cmd[:,0]).abs().mean())
                wz=float((data.root_ang_vel_b[:,2]-cmd[:,2]).abs().mean())
                phys.append({"vx_error":vx,"wz_error":wz,"tracking_error":vx+wz,
                             "ang_vel_xy":float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),
                             "tilt_deg":float(tilt_deg(data.root_quat_w).mean()),
                             "action_rate":float(torch.linalg.vector_norm(a-prev,dim=-1).mean()),
                             "abs_lin_vel_z":float(data.root_lin_vel_b[:,2].abs().mean())})
                prev=a;cur=obs_tensor(nxt).cuda()
        R=np.asarray(R);D=np.asarray(D,bool);V=np.asarray(V)
        H0=segret(R,D,0,32);H1=segret(R,D,32,64)
        pmean={k:float(np.mean([x[k] for x in phys])) for k in phys[0]}
        obj=R.mean((0,1))
        return {"normalized_objective_mean":obj.tolist(),"physical":pmean,
                "survival":float(1-done_any.mean()),
                "critic":{"first_h32_ev":[ev(H0[:,:,j],V[:32,:,j]) for j in range(4)],
                          "second_h32_ev":[ev(H1[:,:,j],V[32:,:,j]) for j in range(4)],
                          "first_h32_bias":[float(np.mean(V[:32,:,j]-H0[:,:,j])) for j in range(4)],
                          "second_h32_bias":[float(np.mean(V[32:,:,j]-H1[:,:,j])) for j in range(4)]}}
    
    def interp(a,b,alpha):return (1-alpha)*PREFS[a]+alpha*PREFS[b]
    
    def monotonic_fraction(vals):
        vals=np.asarray(vals,float);d=np.diff(vals);target=np.sign(vals[-1]-vals[0])
        if target==0:return 0.0
        return float(np.mean(np.sign(d)==target))
    
    def between_fraction(vals):
        vals=np.asarray(vals,float);lo=min(vals[0],vals[-1]);hi=max(vals[0],vals[-1])
        return float(np.mean((vals[1:-1]>=lo-1e-9)&(vals[1:-1]<=hi+1e-9)))
    def sha(p):
        h=hashlib.sha256();h.update(Path(p).read_bytes());return h.hexdigest()
    
    def main():
        class Args: pass
        args=Args();args.checkpoint=CKPT;args.output_dir=OUT_DEFAULT
        args.output_dir.mkdir(parents=True,exist_ok=True)
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]]
        app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.authority.isolated import AuthorityIsolatedActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=0);o=obs_tensor(o).cuda();ad=env.unwrapped.action_manager.total_action_dim
            mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
            m=AuthorityIsolatedActorCritic(o.shape[-1],ad).cuda()
            m.load_state_dict(torch.load(args.checkpoint,map_location="cuda",weights_only=False)["model"]);m.eval()
            rows=[]
            # Matched endpoints + center: exact same reset seed across every preference in a suite.
            for suite in range(SUITES):
                seed=840001+suite
                for lab in ("T","A","O","S","C"):
                    q=evaluate(env,m,mgr,robot,PREFS[lab],seed)
                    q.update({"suite":suite,"kind":"endpoint","label":lab,"w":PREFS[lab].tolist()});rows.append(q)
            # Continuum paths, again matched by suite and path.
            continuum=[]
            for pi,(a,b) in enumerate(PATHS):
                for suite in range(SUITES):
                    seed=850001+pi*1000+suite
                    vals=[]
                    for alpha in ALPHAS:
                        w=interp(a,b,alpha)
                        q=evaluate(env,m,mgr,robot,w,seed)
                        q.update({"path":f"{a}-{b}","a":a,"b":b,"suite":suite,"alpha":alpha,"w":w.tolist()})
                        vals.append(q);continuum.append(q)
            # Endpoint directional correctness vs matched center.
            endpoint={}
            for lab in ORDER:
                j=IDX[lab];pk=PHYS[lab];obj_ok=[];phys_ok=[];surv=[]
                del_obj=[];del_phys=[]
                for suite in range(SUITES):
                    r=next(x for x in rows if x["suite"]==suite and x["label"]==lab)
                    c=next(x for x in rows if x["suite"]==suite and x["label"]=="C")
                    do=r["normalized_objective_mean"][j]-c["normalized_objective_mean"][j]
                    dp=r["physical"][pk]-c["physical"][pk]
                    # Objectives are higher-is-better; physical proxies are lower-is-better.
                    obj_ok.append(do>0);phys_ok.append(dp<0);surv.append(r["survival"])
                    del_obj.append(do);del_phys.append(dp)
                endpoint[lab]={"objective_correct_fraction":float(np.mean(obj_ok)),
                               "physical_correct_fraction":float(np.mean(phys_ok)),
                               "mean_objective_delta_vs_center":float(np.mean(del_obj)),
                               "mean_physical_delta_vs_center":float(np.mean(del_phys)),
                               "min_survival":float(np.min(surv))}
            cont_stats=[]
            for a,b in PATHS:
                for suite in range(SUITES):
                    rr=sorted([x for x in continuum if x["path"]==f"{a}-{b}" and x["suite"]==suite],key=lambda x:x["alpha"])
                    for lab in (a,b):
                        j=IDX[lab];pk=PHYS[lab]
                        ov=[x["normalized_objective_mean"][j] for x in rr]
                        pv=[x["physical"][pk] for x in rr]
                        cont_stats.append({"path":f"{a}-{b}","suite":suite,"axis":lab,
                            "objective_monotonic":monotonic_fraction(ov),
                            "objective_between":between_fraction(ov),
                            # lower physical is better, but monotonic_fraction only tests coherence between endpoints.
                            "physical_monotonic":monotonic_fraction(pv),
                            "physical_between":between_fraction(pv)})
            mono=float(np.mean([(x["objective_monotonic"]+x["physical_monotonic"])/2 for x in cont_stats]))
            between=float(np.mean([(x["objective_between"]+x["physical_between"])/2 for x in cont_stats]))
            # Cross-axis trade-off/collateral summary.
            centers=[x for x in rows if x["label"]=="C"]
            center_track=float(np.mean([x["physical"]["tracking_error"] for x in centers]))
            max_track_ratio=0.0
            for lab in ORDER:
                rr=[x for x in rows if x["label"]==lab]
                tr=float(np.mean([x["physical"]["tracking_error"] for x in rr]))
                max_track_ratio=max(max_track_ratio,tr/(center_track+1e-12))
            all_surv=[x["survival"] for x in rows]+[x["survival"] for x in continuum]
            # Fresh critic aggregate over all endpoint/center matched evaluations.
            cev=[];cbias=[]
            for x in rows:
                cev+=x["critic"]["first_h32_ev"]+x["critic"]["second_h32_ev"]
                cbias+=x["critic"]["first_h32_bias"]+x["critic"]["second_h32_bias"]
            critic={"h32_ev_mean":float(np.mean(cev)),"negative_fraction":float(np.mean(np.asarray(cev)<0)),
                    "mean_abs_bias":float(np.mean(np.abs(cbias)))}
            endpoint_pass={lab:(endpoint[lab]["objective_correct_fraction"]>=.75 and
                                endpoint[lab]["physical_correct_fraction"]>=.75 and
                                endpoint[lab]["min_survival"]>=.95) for lab in ORDER}
            criteria={
                "all_endpoint_axes_correct":bool(all(endpoint_pass.values())),
                "continuum_monotonicity":mono>=.65,
                "continuum_endpoint_between":between>=.65,
                "survival_preserved":float(np.min(all_surv))>=.95,
                "tracking_collateral_bounded":max_track_ratio<=2.0,
                "fresh_critic_not_collapsed":critic["h32_ev_mean"]>0.0 and critic["negative_fraction"]<=.25,
            }
            passed=all(criteria.values())
            report={"schema":"authority_isolated_ai_h2_final_semantic_eval_v1","status":"AI-H2 FINAL SEMANTIC PASS" if passed else "AI-H2 FINAL SEMANTIC FAIL",
                "measurement_only":True,"training_updates":0,"checkpoint":str(args.checkpoint.relative_to(ROOT)),
                "protocol":{"matched_reset":True,"suites":SUITES,"steps":STEPS,"alphas":list(ALPHAS),
                            "paths":[f"{a}-{b}" for a,b in PATHS],"thresholds":{"endpoint_fraction":.75,"continuum_fraction":.65,
                            "min_survival":.95,"max_tracking_ratio_to_center":2.0}},
                "endpoint":endpoint,"endpoint_pass":endpoint_pass,"continuum_summary":{"monotonicity_fraction":mono,
                            "endpoint_between_fraction":between,"details":cont_stats},
                "collateral":{"center_tracking_error":center_track,"max_tracking_ratio_to_center":max_track_ratio},
                "critic":critic,"min_survival_all":float(np.min(all_surv)),"criteria":criteria,
                "decision":{"semantic_pass":bool(passed),"architecture":"authority_isolated_narrow"},
                "endpoint_rows":rows,"continuum_rows":continuum}
            out=args.output_dir/"semantic_report.json";out.write_text(json.dumps(report,indent=2)+"\n")
            print(json.dumps({k:report[k] for k in ("status","endpoint","endpoint_pass","continuum_summary","collateral","critic","min_survival_all","criteria","decision") if k in report and k!="continuum_summary"} | {"continuum_summary":{"monotonicity_fraction":mono,"endpoint_between_fraction":between}},indent=2))
            prov={"status":"FROZEN_BY_HASH","gate":"FOUNDATION-V2-SEMANTIC","verdict":report["status"],"measurement_only":True,
                  "architecture":"authority_isolated_narrow",
                  "artifacts":{str(out.relative_to(ROOT)):{"sha256":sha(out)},
                               str(args.checkpoint.relative_to(ROOT)):{"sha256":sha(args.checkpoint)},
                               str(Path(__file__).resolve().relative_to(ROOT)):{"sha256":sha(Path(__file__).resolve())}}}
            (args.output_dir/"PROVENANCE_MANIFEST.json").write_text(json.dumps(prov,indent=2)+"\n")
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_authority_isolated_ai_h2_semantic_train():
    """Run former authority_isolated_ai_h2_semantic_train.py stage."""
    
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from pathlib import Path
    import json,sys,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"scripts/rl")]
    import rl.experiments.common.utilities.authority_isolated_h1_screen as h1
    from talon_rl.models.authority.isolated import AuthorityIsolatedActorCritic
    
    SOURCE_STATE=ROOT/"runs/authority_isolated_ai_c2_edgec2_narrow-2026-09-25/resume_state.pt"
    SOURCE_MODEL=ROOT/"runs/authority_isolated_ai_c2_edgec2_narrow-2026-09-25/model_10.pt"
    REF_U20=ROOT/"runs/authority_isolated_coverage2_control-2026-09-25/model_20.pt"
    AUTH_SUPPORT=ROOT/"runs/authority_isolated_u20_authority_support-2026-09-25/u20_authority_support.npz"
    TAU_PATH=ROOT/"runs/authority_isolated_tail_budget_calibration-2026-09-25/tail_budget_probe.json"
    OUT=ROOT/"runs/authority_isolated_ai_h2_semantic_path-2026-09-25";OUT.mkdir(parents=True,exist_ok=True)
    
    NENV=h1.NENV;H=h1.H
    GLOBAL_START=30;TARGET=25;SNAPS={0,5,10,15,20,25}
    KAPPA=.05;RHO=.25;BETA0=2.497041993384243;GAMMA=.90
    ALL_PREFS=("T","A","O","S","C")
    EDGE_PAIRS=tuple((ALL_PREFS[i],ALL_PREFS[j]) for i in range(len(ALL_PREFS)) for j in range(i+1,len(ALL_PREFS)))
    
    def tail_metrics(m,obs,w,tau):
        z=m._actor_mean_with_preference(obs,w);ex=torch.relu(z.abs()-tau)
        return ((ex/(tau+1e-6))**2).mean(),(z.abs()>tau).float().mean()
    
    def balanced_indices(origin,phase,gidx):
        rng=np.random.default_rng(2609252401+gidx);ids=[]
        for oi in range(5):
            for pi in range(3):
                pool=np.flatnonzero((origin==oi)&(phase==pi))
                ids.extend(rng.choice(pool,size=8,replace=False).tolist())
        return np.asarray(ids,np.int64)
    
    def edge_floor_loss(m,ref,obs):
        acts={};refs={};n=len(obs)
        for lab in ALL_PREFS:
            w=torch.tensor(h1.PREFS[lab],device="cuda").repeat(n,1)
            acts[lab]=m.act_inference_with_preference(obs,w)
            with torch.no_grad():refs[lab]=ref.act_inference_with_preference(obs,w)
        vals=[]
        for i,j in EDGE_PAIRS:
            d=torch.linalg.vector_norm(acts[i]-acts[j],dim=1)
            dr=torch.linalg.vector_norm(refs[i]-refs[j],dim=1)
            vals.append(torch.relu(GAMMA*dr-d).pow(2))
        return torch.stack(vals,dim=1).mean()
    
    def grad_norm(gs):
        return torch.sqrt(sum((g.detach()**2).sum() for g in gs if g is not None)+1e-12)
    
    def save(path,m,opt,adaptive_pools,anchor_specs,rows,update):
        torch.save({"model":m.state_dict(),"optimizer":opt.state_dict(),"adaptive_pools":adaptive_pools,
          "anchor_specs":anchor_specs,"rows":rows,"update":update,
          "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all(),
          "numpy_rng":np.random.get_state()},path)
    
    def main():
        tau=torch.tensor(json.load(open(TAU_PATH))["tau"],device="cuda",dtype=torch.float32)
        sup=np.load(AUTH_SUPPORT);auth_obs=torch.tensor(sup["obs"],device="cuda");origin=sup["origin"];phase=sup["phase"]
        torch.manual_seed(73001);np.random.seed(73001)
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.foundations.four_objective import vector_gae,scalarized_late_weighted_ppo
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=73001
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=73001);o=h1.ot(o).cuda()
            ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
            m=AuthorityIsolatedActorCritic(o.shape[-1],ad).cuda()
            ref=AuthorityIsolatedActorCritic(o.shape[-1],ad).cuda()
            # u20 source checkpoint was wide architecture; copy actor/family only into narrow reference.
            u20=torch.load(REF_U20,map_location="cuda",weights_only=False)["model"]
            rs=ref.state_dict()
            for k in rs:
                if k.startswith("actor_") or k.startswith("family_") or k=="log_std":rs[k].copy_(u20[k])
            ref.load_state_dict(rs);ref.eval()
            for p in ref.parameters():p.requires_grad_(False)
            for n,p in m.named_parameters():
                if n.startswith("critic_body") or n.startswith("critic_head"):p.requires_grad_(False)
            actor_params=[p for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std" or n.startswith("family_")]
            opt=torch.optim.Adam(actor_params,lr=1e-3)
            state_path=OUT/"resume_state.pt"
            if state_path.exists():
                st=torch.load(state_path,map_location="cpu",weights_only=False);m.load_state_dict(st["model"]);opt.load_state_dict(st["optimizer"])
                adaptive_pools=st["adaptive_pools"];anchor_specs=st["anchor_specs"];rows=st["rows"];cur=int(st["update"])
                torch.set_rng_state(st["torch_rng"]);torch.cuda.set_rng_state_all(st["cuda_rng"]);np.random.set_state(st["numpy_rng"])
                print("RESUME",cur,flush=True)
            else:
                st=torch.load(SOURCE_STATE,map_location="cpu",weights_only=False)
                m.load_state_dict(st["model"]);opt.load_state_dict(st["optimizer"])
                adaptive_pools=st["adaptive_pools"];anchor_specs=st["anchor_specs"];rows=[];cur=0
                torch.save({"model":m.state_dict(),"h2_update":0,"global_update":GLOBAL_START},OUT/"model_0.pt")
                save(state_path,m,opt,adaptive_pools,anchor_specs,rows,cur)
    
            def current_anchor_units():
                q={"early":[],"late":[]}
                for k,seed in anchor_specs:
                    _,w=h1.pref_batch(k,torch.device("cuda"))
                    for u in h1.collect_support64(env,m,w,mgr,seed):q[u["phase"]].append(u)
                return q
    
            for hidx in range(cur+1,TARGET+1):
                gidx=GLOBAL_START+hidx
                _,w=h1.pref_batch(gidx,torch.device("cuda"))
                main=h1.collect_actor(env,m,w,mgr,73001+gidx*211,True)
                _,ws=h1.pref_batch(gidx+17,torch.device("cuda"))
                units=h1.collect_support64(env,m,ws,mgr,73001+200000+gidx*223)
                for u in units:
                    ph=u["phase"];adaptive_pools[ph].append(u)
                    if len(adaptive_pools[ph])>h1.POOL_MAX:adaptive_pools[ph].pop(0)
                h1.fit_expanded_current_policy(m,current_anchor_units(),adaptive_pools)
                with torch.no_grad():
                    vt=m.value_with_preference(main["obs"],main["w"]).reshape(H,NENV,4)
                    nv=m.value_with_preference(main["next_obs"],w)
                    adv,_=vector_gae(main["rt"],vt,nv,main["dt"],lam=.95)
                lp=[]
                for stx in range(0,len(main["obs"]),NENV):
                    lp.append(m.logp_from_pre_tanh_with_preference(main["obs"][stx:stx+NENV],main["w"][stx:stx+NENV],main["u"][stx:stx+NENV]))
                ratio=torch.exp(torch.cat(lp)-main["old"].detach());ratio_err=float((ratio-1).abs().max().cpu())
                if ratio_err>1e-4:raise RuntimeError(f"ratio invariant {ratio_err}")
                ppo=scalarized_late_weighted_ppo(ratio,adv.reshape(-1,4).detach(),main["w"])
                tail,tailfrac=tail_metrics(m,main["obs"],main["w"],tau)
                gp=torch.autograd.grad(ppo,actor_params,retain_graph=True,allow_unused=True)
                gt=torch.autograd.grad(tail,actor_params,retain_graph=True,allow_unused=True)
                gp2=sum((g.detach()**2).sum() for g in gp if g is not None);gt2=sum((g.detach()**2).sum() for g in gt if g is not None)
                gpn=torch.sqrt(gp2+1e-12);gtn=torch.sqrt(gt2+1e-12)
                dot=sum((a.detach()*b.detach()).sum() for a,b in zip(gp,gt) if a is not None and b is not None)
                coeff=torch.minimum(dot/(gt2+1e-12),torch.zeros_like(dot))
                base=[];proj2=torch.zeros((),device="cuda")
                for p,a,b in zip(actor_params,gp,gt):
                    aa=torch.zeros_like(p) if a is None else a;bb=torch.zeros_like(p) if b is None else b
                    pp=aa-coeff*bb;base.append((pp,bb));proj2+=(pp.detach()**2).sum()
                projn=torch.sqrt(proj2+1e-12);tail_scale=KAPPA*projn/(gtn+1e-12)
                gbase=[pp+tail_scale*bb for pp,bb in base];gbn=grad_norm(gbase)
                ids=balanced_indices(origin,phase,gidx);edge=edge_floor_loss(m,ref,auth_obs[ids])
                ge=torch.autograd.grad(edge,actor_params,allow_unused=True);gen=grad_norm(ge)
                alpha=min(BETA0,RHO*float(gbn.cpu())/(float(gen.cpu())+1e-12))
                opt.zero_grad(set_to_none=True)
                for i,p in enumerate(actor_params):
                    gg=gbase[i]
                    if ge[i] is not None:gg=gg+alpha*ge[i]
                    p.grad=gg
                preclip=float(torch.nn.utils.clip_grad_norm_(actor_params,1.0).cpu());opt.step()
                with torch.no_grad():m.log_std.clamp_(m.LOG_STD_MIN,m.LOG_STD_MAX)
                row={"h2_update":hidx,"global_update":gidx,"ppo_loss":float(ppo.detach().cpu()),
                     "tail_loss":float(tail.detach().cpu()),"tail_fraction":float(tailfrac.detach().cpu()),
                     "edge_loss":float(edge.detach().cpu()),"edge_alpha":float(alpha),
                     "weighted_edge_over_base":float(alpha*float(gen.cpu())/(float(gbn.cpu())+1e-12)),
                     "ratio_maxerr":ratio_err,"termination_fraction":main["termination_fraction"],"grad_norm_preclip":preclip}
                rows.append(row);print("UPDATE",hidx,"GLOBAL",gidx,json.dumps(row),flush=True)
                if hidx in SNAPS:
                    torch.save({"model":m.state_dict(),"h2_update":hidx,"global_update":gidx},OUT/f"model_{hidx}.pt")
                save(state_path,m,opt,adaptive_pools,anchor_specs,rows,hidx)
    
            report={"schema":"authority_isolated_ai_h2_semantic_train_v1","start_global_update":GLOBAL_START,
                    "end_global_update":GLOBAL_START+TARGET,"snapshots":sorted(SNAPS),
                    "rho":RHO,"beta0":BETA0,"edge_gamma":GAMMA,
                    "max_ratio_error":max(r["ratio_maxerr"] for r in rows),
                    "max_training_termination_fraction":max(r["termination_fraction"] for r in rows),
                    "mean_weighted_edge_over_base":float(np.mean([r["weighted_edge_over_base"] for r in rows])),
                    "rows":rows}
            (OUT/"training_report.json").write_text(json.dumps(report,indent=2)+"\n")
            print("FINAL",json.dumps({k:v for k,v in report.items() if k!="rows"},indent=2),flush=True)
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

STAGES = {
    "authority_isolated_ai_h2_early_window_authority_audit": run_authority_isolated_ai_h2_early_window_authority_audit,
    "authority_isolated_ai_h2_early_window_noregression_eval": run_authority_isolated_ai_h2_early_window_noregression_eval,
    "authority_isolated_ai_h2_endpoint_timeline": run_authority_isolated_ai_h2_endpoint_timeline,
    "authority_isolated_ai_h2_final_authority_audit": run_authority_isolated_ai_h2_final_authority_audit,
    "authority_isolated_ai_h2_final_noregression_eval": run_authority_isolated_ai_h2_final_noregression_eval,
    "authority_isolated_ai_h2_final_semantic_eval": run_authority_isolated_ai_h2_final_semantic_eval,
    "authority_isolated_ai_h2_semantic_train": run_authority_isolated_ai_h2_semantic_train,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
