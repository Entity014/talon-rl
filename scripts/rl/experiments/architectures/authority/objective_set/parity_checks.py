"""Consolidated experiment stage module."""
from __future__ import annotations
import argparse

def run_objective_set_g0_behavioral_parity():
    """Run former objective_set_g0_behavioral_parity.py stage."""
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from pathlib import Path
    import json,sys,hashlib,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"scripts/rl")]
    import rl.experiments.common.utilities.authority_isolated_h2a_u30_semantic_validity as h2
    from talon_rl.models.authority.objective_set import ObjectiveSetAuthorityIsolatedWideCritic
    
    INIT=ROOT/"runs/objective_set_g0_structural_parity-2026-09-25/objective_set_g0_init.pt"
    BASE=ROOT/"runs/authority_isolated_h2a_u30_semantic_validity-2026-09-25/semantic_report.json"
    OUT=ROOT/"runs/objective_set_g0_behavioral_parity-2026-09-25";OUT.mkdir(parents=True,exist_ok=True)
    
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    
    def main():
        from isaaclab.app import AppLauncher
        sv=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv;env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=h2.NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=0);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim
            mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
            m=ObjectiveSetAuthorityIsolatedWideCritic(o.shape[-1],ad).cuda()
            m.load_state_dict(torch.load(INIT,map_location="cuda",weights_only=False)["model"]);m.eval()
            rows=[]
            for suite in range(h2.SUITES):
                seed=840001+suite
                for lab in ("T","A","O","S","C"):
                    q=h2.evaluate(env,m,mgr,robot,h2.PREFS[lab],seed)
                    q.update({"suite":suite,"kind":"endpoint","label":lab,"w":h2.PREFS[lab].tolist()});rows.append(q)
            continuum=[]
            for pi,(a,b) in enumerate(h2.PATHS):
                for suite in range(h2.SUITES):
                    seed=850001+pi*1000+suite
                    for alpha in h2.ALPHAS:
                        w=h2.interp(a,b,alpha);q=h2.evaluate(env,m,mgr,robot,w,seed)
                        q.update({"path":f"{a}-{b}","a":a,"b":b,"suite":suite,"alpha":alpha,"w":w.tolist()});continuum.append(q)
            endpoint={}
            for lab in h2.ORDER:
                j=h2.IDX[lab];pk=h2.PHYS[lab];oo=[];pp=[];surv=[];do=[];dp=[]
                for suite in range(h2.SUITES):
                    r=next(x for x in rows if x["suite"]==suite and x["label"]==lab)
                    c=next(x for x in rows if x["suite"]==suite and x["label"]=="C")
                    x=r["normalized_objective_mean"][j]-c["normalized_objective_mean"][j]
                    y=r["physical"][pk]-c["physical"][pk]
                    oo.append(x>0);pp.append(y<0);surv.append(r["survival"]);do.append(x);dp.append(y)
                endpoint[lab]={"objective_correct_fraction":float(np.mean(oo)),"physical_correct_fraction":float(np.mean(pp)),
                  "mean_objective_delta_vs_center":float(np.mean(do)),"mean_physical_delta_vs_center":float(np.mean(dp)),"min_survival":float(np.min(surv))}
            cont=[]
            for a,b in h2.PATHS:
                for suite in range(h2.SUITES):
                    rr=sorted([x for x in continuum if x["path"]==f"{a}-{b}" and x["suite"]==suite],key=lambda x:x["alpha"])
                    for lab in (a,b):
                        j=h2.IDX[lab];pk=h2.PHYS[lab]
                        ov=[x["normalized_objective_mean"][j] for x in rr];pv=[x["physical"][pk] for x in rr]
                        cont.append({"path":f"{a}-{b}","suite":suite,"axis":lab,
                          "objective_monotonic":h2.monotonic_fraction(ov),"objective_between":h2.between_fraction(ov),
                          "physical_monotonic":h2.monotonic_fraction(pv),"physical_between":h2.between_fraction(pv)})
            mono=float(np.mean([(x["objective_monotonic"]+x["physical_monotonic"])/2 for x in cont]))
            between=float(np.mean([(x["objective_between"]+x["physical_between"])/2 for x in cont]))
            center_between=[]
            for suite in range(h2.SUITES):
                hs=[next(x for x in rows if x["suite"]==suite and x["label"]==lab) for lab in h2.ORDER]
                cc=next(x for x in rows if x["suite"]==suite and x["label"]=="C")
                for j in range(4):
                    vals=[x["normalized_objective_mean"][j] for x in hs];cv=cc["normalized_objective_mean"][j]
                    center_between.append(min(vals)-1e-9<=cv<=max(vals)+1e-9)
            center=float(np.mean(center_between))
            all_surv=[x["survival"] for x in rows]+[x["survival"] for x in continuum]
            cev=[];cb=[]
            for x in rows:
                cev+=x["critic"]["first_h32_ev"]+x["critic"]["second_h32_ev"]
                cb+=x["critic"]["first_h32_bias"]+x["critic"]["second_h32_bias"]
            critic={"h32_ev_mean":float(np.mean(cev)),"negative_fraction":float(np.mean(np.asarray(cev)<0)),
                    "mean_abs_bias":float(np.mean(np.abs(cb)))}
            epass={lab:(endpoint[lab]["objective_correct_fraction"]>=.75 and endpoint[lab]["physical_correct_fraction"]>=.75 and endpoint[lab]["min_survival"]>=.95) for lab in h2.ORDER}
            criteria={"T_preserved":epass["T"],"A_preserved":epass["A"],"O_preserved":epass["O"],
              "endpoint_survival":min(x["survival"] for x in rows)>=.95,
              "critic_valid":critic["h32_ev_mean"]>0 and critic["negative_fraction"]<=.25,
              "continuum_monotonicity":mono>=.65,"continuum_between":between>=.65,"center_compromise":center>=.75}
            base=json.load(open(BASE))
            drift={"endpoint":{}}
            for lab in h2.ORDER:
                drift["endpoint"][lab]={k:float(endpoint[lab][k]-base["endpoint"][lab][k]) for k in
                  ("objective_correct_fraction","physical_correct_fraction","mean_objective_delta_vs_center","mean_physical_delta_vs_center","min_survival")}
            drift["continuum_monotonicity"]=mono-base["continuum_summary"]["monotonicity_fraction"]
            drift["continuum_between"]=between-base["continuum_summary"]["endpoint_between_fraction"]
            drift["critic_ev"]=critic["h32_ev_mean"]-base["critic"]["h32_ev_mean"]
            rep={"schema":"objective_set_g0_behavioral_parity_v1","status":"G0-B PASS" if all(criteria.values()) else "G0-B FAIL",
              "checkpoint":str(INIT.relative_to(ROOT)),"endpoint":endpoint,"endpoint_pass":epass,
              "continuum":{"monotonicity_fraction":mono,"endpoint_between_fraction":between},"center_compromise_fraction":center,
              "critic":critic,"min_survival_all":float(np.min(all_surv)),"criteria":criteria,"drift_vs_phase1_h2a":drift,
              "decision":{"G1_authorized":bool(all(criteria.values()))}}
            rp=OUT/"g0_behavioral_parity.json";rp.write_text(json.dumps(rep,indent=2)+"\n")
            (OUT/"PROVENANCE_MANIFEST.json").write_text(json.dumps({"status":"FROZEN_BY_HASH","report_sha256":sha(rp),
              "script_sha256":sha(Path(__file__).resolve()),"init_sha256":sha(INIT),"base_report_sha256":sha(BASE)},indent=2)+"\n")
            print(json.dumps(rep,indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_objective_set_g0_structural_parity():
    """Run former objective_set_g0_structural_parity.py stage."""
    from pathlib import Path
    import itertools,json,hashlib,sys
    import numpy as np, torch
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    CK=ROOT/"runs/authority_isolated_simplex_edge_retain-2026-09-25/model_30.pt"
    PROBE=ROOT/"runs/update_functional_effect_audit-2026-09-24/fixed_probe_states.npz"
    OUT=ROOT/"runs/objective_set_g0_structural_parity-2026-09-25";OUT.mkdir(parents=True,exist_ok=True)
    TOL=1e-6
    PREFS={"T":[.7,.1,.1,.1],"A":[.1,.7,.1,.1],"O":[.1,.1,.7,.1],"S":[.1,.1,.1,.7],"C":[.25]*4}
    
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def maxerr(a,b):return float((a-b).abs().max().detach().cpu())
    
    def main():
        from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
        from talon_rl.models.authority.objective_set import ObjectiveSetAuthorityIsolatedWideCritic,initialize_exact_from_phase1,set_from_dense_preference
        probe=torch.tensor(np.load(PROBE)["obs"],dtype=torch.float32)
        old=AuthorityIsolatedWideCritic(probe.shape[-1],12)
        state=torch.load(CK,map_location="cpu",weights_only=False)["model"]
        old.load_state_dict(state);old.eval()
        new=ObjectiveSetAuthorityIsolatedWideCritic(probe.shape[-1],12)
        initialize_exact_from_phase1(new,state);new.eval()
    
        errs={"pre_tanh":0.0,"action":0.0,"critic":0.0,"logp":0.0,
              "perm_pre_tanh":0.0,"perm_action":0.0,"perm_value":0.0}
        finite=True
        with torch.no_grad():
            for lab,wv in PREFS.items():
                w=torch.tensor(wv,dtype=torch.float32).repeat(len(probe),1)
                tok,ww=set_from_dense_preference(w)
                om=old._actor_mean_with_preference(probe,w)
                nm=new.pre_tanh_mean_from_set(probe,tok,ww)
                oa=old.act_inference_with_preference(probe,w)
                na=new.act_inference_from_set(probe,tok,ww)
                ov=old.value_with_preference(probe,w)
                nv=new.query_values_from_set(probe,tok,ww,tok)
                errs["pre_tanh"]=max(errs["pre_tanh"],maxerr(om,nm))
                errs["action"]=max(errs["action"],maxerr(oa,na))
                errs["critic"]=max(errs["critic"],maxerr(ov,nv))
                u=om+0.123*torch.ones_like(om)
                olp=old.logp_from_pre_tanh_with_preference(probe,w,u)
                nlp=new.logp_from_pre_tanh_from_set(probe,tok,ww,u)
                errs["logp"]=max(errs["logp"],maxerr(olp,nlp))
                finite &= bool(torch.isfinite(nm).all() and torch.isfinite(nv).all() and torch.isfinite(nlp).all())
    
                refm=nm;refa=na;refv=nv
                for p in itertools.permutations(range(4)):
                    q=torch.tensor(p)
                    pm=new.pre_tanh_mean_from_set(probe,tok[:,q],ww[:,q])
                    pa=new.act_inference_from_set(probe,tok[:,q],ww[:,q])
                    pv=new.query_values_from_set(probe,tok[:,q],ww[:,q],tok)
                    errs["perm_pre_tanh"]=max(errs["perm_pre_tanh"],maxerr(refm,pm))
                    errs["perm_action"]=max(errs["perm_action"],maxerr(refa,pa))
                    errs["perm_value"]=max(errs["perm_value"],maxerr(refv,pv))
        criteria={
          "pre_tanh_parity":errs["pre_tanh"]<=TOL,
          "action_parity":errs["action"]<=TOL,
          "critic_parity":errs["critic"]<=TOL,
          "logp_parity":errs["logp"]<=TOL,
          "permutation_pre_tanh":errs["perm_pre_tanh"]<=TOL,
          "permutation_action":errs["perm_action"]<=TOL,
          "permutation_value":errs["perm_value"]<=TOL,
          "finite":finite}
        passed=all(criteria.values())
        rep={"schema":"objective_set_g0_structural_parity_v1","status":"G0-A PASS" if passed else "G0-A FAIL",
             "checkpoint":str(CK.relative_to(ROOT)),"probe":str(PROBE.relative_to(ROOT)),
             "tolerance":TOL,"metrics":errs,"criteria":criteria,
             "decision":{"G0_B_authorized":passed}}
        rp=OUT/"g0_structural_parity.json";rp.write_text(json.dumps(rep,indent=2)+"\n")
        torch.save({"model":new.state_dict(),"source_checkpoint":str(CK.relative_to(ROOT))},OUT/"objective_set_g0_init.pt")
        (OUT/"PROVENANCE_MANIFEST.json").write_text(json.dumps({
          "status":"FROZEN_BY_HASH","report_sha256":sha(rp),
          "script_sha256":sha(Path(__file__).resolve()),
          "model_sha256":sha(ROOT/"talon_rl/objective_set_actor_critic.py"),
          "checkpoint_sha256":sha(CK)},indent=2)+"\n")
        print(json.dumps(rep,indent=2))
    if True:main()

STAGES = {
    "objective_set_g0_behavioral_parity": run_objective_set_g0_behavioral_parity,
    "objective_set_g0_structural_parity": run_objective_set_g0_structural_parity,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
