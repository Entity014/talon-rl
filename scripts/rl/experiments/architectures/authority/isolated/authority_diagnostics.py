"""Consolidated thematic experiment stages."""
from __future__ import annotations
import argparse

def run_authority_isolated_Oheavy_support_matrix():
    """Run former authority_isolated_Oheavy_support_matrix.py stage."""
    from pathlib import Path
    import json,sys,torch,numpy as np
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    from talon_rl.models.authority.isolated import AuthorityIsolatedActorCritic
    from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
    U75=ROOT/"runs/authority_isolated_h1-2026-09-25/model_75.pt"
    REP=ROOT/"runs/authority_isolated_projected_tail_descent-2026-09-25/model_10.pt"
    OUT=ROOT/"runs/authority_isolated_Oheavy_support_matrix-2026-09-25";OUT.mkdir(parents=True,exist_ok=True)
    SEEDS=(840003,840004);W=[.1,.1,.7,.1];NENV=8;H=24
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def rollout(env,base,donor,idx=None,window=None):
        w=torch.tensor(W,device="cuda").repeat(NENV,1);obs,_=env.reset(seed=env._audit_seed);obs=ot(obs).cuda()
        first=np.full(NENV,-1,int)
        for t in range(H):
            with torch.no_grad():
                ab=base.act_inference_with_preference(obs,w)
                if donor is None:a=ab
                else:
                    ad=donor.act_inference_with_preference(obs,w);a=ab.clone()
                    active=(window is None or (window[0]<=t<=window[1]))
                    if active:a[:,idx]=ad[:,idx]
            nxt,_,te,tr,_=env.step(a);dd=(te|tr).cpu().numpy().astype(bool)
            hit=(first<0)&dd;first[hit]=t;obs=ot(nxt).cuda()
        return first
    def main():
        from isaaclab.app import AppLauncher
        sv=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv;env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda()
            u=AuthorityIsolatedActorCritic(o.shape[-1],12).cuda();u.load_state_dict(torch.load(U75,map_location="cuda",weights_only=False)["model"]);u.eval()
            r=AuthorityIsolatedWideCritic(o.shape[-1],12).cuda();r.load_state_dict(torch.load(REP,map_location="cuda",weights_only=False)["model"]);r.eval()
            names=list(env.unwrapped.scene["robot"].data.joint_names);rep={"joint_names":names,"suites":{}}
            tests=[("base",None,None),("all",list(range(12)),None)]+[(names[j],[j],None) for j in range(12)]
            tests += [("FL_hip_t0_5",[0],(0,5)),("FL_hip_t6_10",[0],(6,10)),("FL_hip_t11_15",[0],(11,15)),("FL_hip_t4_8",[0],(4,8))]
            for si,seed in enumerate(SEEDS,2):
                env._audit_seed=seed;rows={}
                for lab,idx,win in tests:
                    if lab=="base":ff=rollout(env,r,None)
                    else:ff=rollout(env,r,u,idx,win)
                    rows[lab]={"first_fail":ff.tolist(),"survival":float(np.mean(ff<0))}
                    print("suite",si,lab,rows[lab],flush=True)
                rep["suites"][str(si)]=rows
            (OUT/"support_matrix.json").write_text(json.dumps(rep,indent=2)+"\n")
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_authority_isolated_adversarial_cem_confirm():
    """Run former authority_isolated_adversarial_cem_confirm.py stage."""
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from pathlib import Path
    import json,sys,torch,numpy as np
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"scripts/rl")]
    from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
    from rl.experiments.common.utilities.authority_isolated_adversarial_vulnerability_audit import ot,norm6,set_wrench,clear_wrench,mode_vec
    CK=ROOT/"runs/authority_isolated_projected_tail_descent-2026-09-25/model_10.pt"
    OUT=ROOT/"runs/authority_isolated_adversarial_vulnerability_audit-2026-09-25"
    SEED=840004;NENV=8;H=28;LANES=list(range(1,8));PULSE=(6,10);F=30.;T=3.
    PREFS={"O":[.1,.1,.7,.1],"S":[.1,.1,.1,.7],"C":[.25]*4}
    BASES=["ang_xy_pos","lin_xy_pos","tilt_torque_pos","tilt_force_pos","cross_ang_force_pos"]
    POP=18;ELITE=5;ITERS=4
    def rollout(env,m,wv,coef):
     robot=env.unwrapped.scene["robot"];obs,_=env.reset(seed=SEED);obs=ot(obs).cuda();clear_wrench(robot)
     w=torch.tensor(wv,device="cuda").repeat(NENV,1);fail=np.full(NENV,-1,int);reasons=[[] for _ in range(NENV)]
     peak=np.zeros(NENV);minh=np.full(NENV,1e9)
     for step in range(H):
      d=robot.data;peak=np.maximum(peak,torch.linalg.vector_norm(d.root_ang_vel_b[:,:2],dim=1).detach().cpu().numpy());minh=np.minimum(minh,d.root_pos_w[:,2].detach().cpu().numpy())
      if PULSE[0]<=step<=PULSE[1]:
       ids=torch.tensor(LANES,device="cuda",dtype=torch.long);u=torch.zeros((len(LANES),6),device="cuda")
       for bi,b in enumerate(BASES):u += float(coef[step-PULSE[0],bi])*mode_vec(b,obs,d)[ids]
       set_wrench(robot,u,F,T,ids)
      else:clear_wrench(robot)
      with torch.no_grad():a=m.act_inference_with_preference(obs,w)
      nxt,_,te,tr,_=env.step(a);dd=(te|tr).detach().cpu().numpy().astype(bool);tm=env.unwrapped.termination_manager
      terms={}
      for n in tm.active_terms:
       try:terms[n]=tm.get_term(n).detach().cpu().numpy().astype(bool)
       except:pass
      for i in range(NENV):
       if dd[i] and fail[i]<0:fail[i]=step;reasons[i]=[n for n,v in terms.items() if v[i]]
      obs=ot(nxt).cuda()
     clear_wrench(robot)
     delayed=[i for i in LANES if fail[i]>PULSE[1] and "base_contact" in reasons[i]]
     failed=[i for i in LANES if fail[i]>=0]
     # smooth search objective: failures dominate; otherwise angular excitation and height erosion guide CEM
     score=100.*len(delayed)+30.*(len(failed)-len(delayed))+float(np.mean(peak[LANES]))+20.*max(0.,0.17-float(np.mean(minh[LANES])))
     return score,{"failed":failed,"delayed":delayed,"fail_t":fail.tolist(),"peak_ang":float(np.mean(peak[LANES])),"min_height":float(np.mean(minh[LANES]))}
    def eval_coef(env,m,coef):
     qs={};score=0.
     for p,w in PREFS.items():
      s,q=rollout(env,m,w,coef);score+=s;qs[p]=q
     return score,qs
    def main():
     from isaaclab.app import AppLauncher
     sv=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda()
      m=AuthorityIsolatedWideCritic(o.shape[-1],12).cuda();m.load_state_dict(torch.load(CK,map_location="cuda",weights_only=False)["model"]);m.eval()
      rng=np.random.default_rng(92526);mu=np.zeros((5,len(BASES)));sd=np.ones_like(mu);history=[];best=None
      for it in range(ITERS):
       cand=[]
       for k in range(POP):
        x=rng.normal(mu,sd);sc,qs=eval_coef(env,m,x);cand.append((sc,x,qs))
       cand.sort(key=lambda z:z[0],reverse=True);elite=cand[:ELITE]
       mu=np.mean([x[1] for x in elite],axis=0);sd=np.std([x[1] for x in elite],axis=0)+0.15
       top=elite[0]
       if best is None or top[0]>best[0]:best=top
       rec={"iter":it,"best_score":top[0],"best_outcomes":top[2],"elite_mean":float(np.mean([x[0] for x in elite]))}
       history.append(rec);print("ITER",json.dumps(rec),flush=True)
      rep={"schema":"adversarial_cem_confirm_v1","budget":{"force_N":F,"torque_Nm":T},"bases":BASES,
           "population":POP,"elite":ELITE,"iterations":ITERS,"history":history,
           "best_score":best[0],"best_coefficients":best[1].tolist(),"best_outcomes":best[2],
           "common_delayed_failure":all(len(best[2][p]["delayed"])>0 for p in PREFS)}
      (OUT/"adversarial_cem_confirm.json").write_text(json.dumps(rep,indent=2)+"\n");print("FINAL",json.dumps(rep,indent=2),flush=True)
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_authority_isolated_authority_durability_diagnostic():
    """Run former authority_isolated_authority_durability_diagnostic.py stage."""
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
    from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
    
    CK20=ROOT/"runs/authority_isolated_coverage2_control-2026-09-25/model_20.pt"
    CK30=ROOT/"runs/authority_isolated_ai_c2_continuity-2026-09-25/model_30.pt"
    PROBE=ROOT/"runs/update_functional_effect_audit-2026-09-24/fixed_probe_states.npz"
    OUT=ROOT/"runs/authority_isolated_authority_durability_diagnostic-2026-09-25";OUT.mkdir(parents=True,exist_ok=True)
    
    COMMON_PREFIX=("actor_body","actor_mean")
    FAMILY_PREFIX=("family_",)
    
    def is_common(k):
        return k.startswith(COMMON_PREFIX)
    def is_family(k):
        return k.startswith(FAMILY_PREFIX)
    
    def build(state):
        m=AuthorityIsolatedWideCritic(48,12).cuda();m.load_state_dict(state);m.eval();return m
    
    def hybrid(s20,s30,common_from,family_from):
        out={k:v.clone() for k,v in s20.items()}
        for k in out:
            if is_common(k): out[k]=(s20 if common_from==20 else s30)[k].clone()
            elif is_family(k): out[k]=(s20 if family_from==20 else s30)[k].clone()
            elif k=="log_std": out[k]=s20[k].clone()
            elif k.startswith("critic_"): out[k]=s20[k].clone()
        return out
    
    def rel_delta(a,b,keys):
        n=0.;d=0.
        for k in keys:
            dv=(b[k]-a[k]).float();n+=float((dv*dv).sum())
            av=a[k].float();d+=float((av*av).sum())
        return float(np.sqrt(n)/(np.sqrt(d)+1e-12))
    
    def group_keys(s):
        return {
          "shared_common":[k for k in s if is_common(k)],
          "family_all":[k for k in s if is_family(k)],
          "family_hyper":[k for k in s if k.startswith("family_hyper")],
          "family_generated_bases":[k for k in s if k.startswith("family_B_")],
          "family_base_network":[k for k in s if k.startswith("family_w") and not k.startswith("family_B_") or k.startswith("family_b") and not k.startswith("family_B_")],
          "log_std":["log_std"],
        }
    
    def coeff_stats(m):
        rows={}
        with torch.no_grad():
            for lab in h1.ORDER:
                w=torch.tensor(h1.PREFS[lab],device="cuda")[None,:]
                c=m.family_coefficients(w).squeeze(0)
                p=m.generated_parameter_vector(w).squeeze(0)
                rows[lab]={"coeff_norm":float(c.norm().cpu()),"param_norm":float(p.norm().cpu()),
                           "coeff":c.cpu().numpy().tolist()}
        C=np.array([rows[x]["coeff"] for x in h1.ORDER])
        Cc=C-C.mean(0,keepdims=True);s=np.linalg.svd(Cc,compute_uv=False)
        return {"by_pref":rows,"centered_coeff_singular_values":s.tolist(),
                "centered_coeff_rank5":int(np.sum(s >= (s[0]*.05 if s[0]>0 else np.inf))),
                "centered_coeff_rms":float(np.sqrt(np.mean(np.sum(Cc*Cc,axis=1))))}
    
    def action_ref_error(m,ref,probe):
        errs=[];rels=[]
        with torch.no_grad():
            for lab in h1.ORDER:
                w=torch.tensor(h1.PREFS[lab],device="cuda").repeat(len(probe),1)
                a=m.act_inference_with_preference(probe,w);r=ref.act_inference_with_preference(probe,w)
                e=torch.linalg.vector_norm(a-r,dim=1);rn=torch.linalg.vector_norm(r,dim=1)
                errs.append(float(e.mean().cpu()));rels.append(float((e/(rn+1e-8)).mean().cpu()))
        return {"mean_action_error":float(np.mean(errs)),"mean_relative_action_error":float(np.mean(rels))}
    
    def main():
        s20=torch.load(CK20,map_location="cuda",weights_only=False)["model"]
        s30=torch.load(CK30,map_location="cuda",weights_only=False)["model"]
        probe=torch.tensor(np.load(PROBE)["obs"],device="cuda",dtype=torch.float32)
        states={
          "u20":s20,
          "u30":s30,
          "common30_family20":hybrid(s20,s30,30,20),
          "common20_family30":hybrid(s20,s30,20,30),
        }
        models={k:build(v) for k,v in states.items()}
        rep={"schema":"authority_durability_diagnostic_v1","models":{},"parameter_drift":{}}
        for k,m in models.items():
            rep["models"][k]={"sensitivity":h1.sensitivity(m,probe),
                              "coefficients":coeff_stats(m),
                              "deviation_from_u20":action_ref_error(m,models["u20"],probe)}
        keys=group_keys(s20)
        for g,ks in keys.items():rep["parameter_drift"][g]={"relative_l2_u20_to_u30":rel_delta(s20,s30,ks),"num_tensors":len(ks)}
        # causal retention ratios
        base=rep["models"]["u20"]["sensitivity"];r={}
        for k in ("u30","common30_family20","common20_family30"):
            q=rep["models"][k]["sensitivity"]
            r[k]={
              "pairwise_retention":q["pairwise_action_distance"]["mean"]/(base["pairwise_action_distance"]["mean"]+1e-12),
              "action_tangent_retention":q["tangent_jacobian_fro_mean"]/(base["tangent_jacobian_fro_mean"]+1e-12),
              "coeff_pairwise_retention":q["coefficient_pairwise_distance_heavy"]["mean"]/(base["coefficient_pairwise_distance_heavy"]["mean"]+1e-12),
              "generated_param_pairwise_retention":q["generated_parameter_pairwise_distance_heavy"]["mean"]/(base["generated_parameter_pairwise_distance_heavy"]["mean"]+1e-12),
              "param_tangent_retention":q["parameter_tangent_jacobian_fro"]/(base["parameter_tangent_jacobian_fro"]+1e-12),
              "functional_specific_rms_retention":q["centered_functional_geometry"]["specific_rms"]/(base["centered_functional_geometry"]["specific_rms"]+1e-12),
              "parameter_specific_energy_retention":q["centered_parameter_geometry"]["specific_energy"]/(base["centered_parameter_geometry"]["specific_energy"]+1e-12),
            }
        rep["retention"]=r
        (OUT/"authority_durability_diagnostic.json").write_text(json.dumps(rep,indent=2)+"\n")
        for k,v in r.items():print(k,json.dumps(v),flush=True)
        print("DRIFT",json.dumps(rep["parameter_drift"]),flush=True)
        for k in states:
            s=rep["models"][k]["sensitivity"]
            print("GEOM",k,"func_rank",s["centered_functional_geometry"]["effective_rank_5pct"],
                  "param_rank",s["centered_parameter_geometry"]["effective_rank_5pct"],
                  "func_s2/s1",s["centered_functional_geometry"]["s2_over_s1"],
                  "param_s2/s1",s["centered_parameter_geometry"]["s2_over_s1"],flush=True)
    if True:main()

def run_authority_isolated_authority_matched_state_audit():
    """Run former authority_isolated_authority_matched_state_audit.py stage."""
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
    from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
    
    CK20=ROOT/"runs/authority_isolated_coverage2_control-2026-09-25/model_20.pt"
    CK30=ROOT/"runs/authority_isolated_ai_c2_continuity-2026-09-25/model_30.pt"
    OUT=ROOT/"runs/authority_isolated_authority_matched_state_audit-2026-09-25";OUT.mkdir(parents=True,exist_ok=True)
    SEEDS={"suite2":840003,"suite3":840004,"suite4":850101,"suite5":850202,"suite6":850303}
    NENV=8;H=64
    
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    
    def build(state):
        m=AuthorityIsolatedWideCritic(48,12).cuda();m.load_state_dict(state);m.eval();return m
    
    def hybrid(s20,s30,common_from,family_from):
        out={k:v.clone() for k,v in s20.items()}
        for k in out:
            if k.startswith("actor_body") or k.startswith("actor_mean"):
                out[k]=(s20 if common_from==20 else s30)[k].clone()
            elif k.startswith("family_"):
                out[k]=(s20 if family_from==20 else s30)[k].clone()
            elif k=="log_std":out[k]=s20[k].clone()
        return out
    
    def collect_states(env,m,seed,wv):
        w=torch.tensor(wv,device="cuda").repeat(NENV,1)
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda();arr=[]
        with torch.no_grad():
            for t in range(H):
                arr.append(cur.detach().cpu())
                a=m.act_inference_with_preference(cur,w)
                nxt,_,_,_,_=env.step(a);cur=ot(nxt).cuda()
        return torch.stack(arr) # H,E,D
    
    def metrics(m,states,maxn=384):
        x=states.reshape(-1,states.shape[-1])
        if len(x)>maxn:
            idx=torch.linspace(0,len(x)-1,maxn).long();x=x[idx]
        return h1.sensitivity(m,x.cuda())
    
    def ret(q,b):
        return {
          "pairwise":q["pairwise_action_distance"]["mean"]/(b["pairwise_action_distance"]["mean"]+1e-12),
          "tangent":q["tangent_jacobian_fro_mean"]/(b["tangent_jacobian_fro_mean"]+1e-12),
          "functional_specific_rms":q["centered_functional_geometry"]["specific_rms"]/(b["centered_functional_geometry"]["specific_rms"]+1e-12),
          "functional_rank":q["centered_functional_geometry"]["effective_rank_5pct"],
          "functional_s2_s1":q["centered_functional_geometry"]["s2_over_s1"],
        }
    
    def main():
        from isaaclab.app import AppLauncher
        sv=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);env.reset(seed=0)
            s20=torch.load(CK20,map_location="cuda",weights_only=False)["model"];s30=torch.load(CK30,map_location="cuda",weights_only=False)["model"]
            models={"u20":build(s20),"u30":build(s30),
                    "common30_family20":build(hybrid(s20,s30,30,20)),
                    "common20_family30":build(hybrid(s20,s30,20,30))}
            buckets={"initial":[],"early":[],"late":[]}
            by_origin={lab:{"early":[],"late":[]} for lab in h1.ORDER}
            by_suite={s:{"early":[],"late":[]} for s in SEEDS}
            for suite,seed in SEEDS.items():
                for lab in h1.ORDER:
                    st=collect_states(env,models["u20"],seed,h1.PREFS[lab])
                    buckets["initial"].append(st[0]);buckets["early"].append(st[:32]);buckets["late"].append(st[32:])
                    by_origin[lab]["early"].append(st[:32]);by_origin[lab]["late"].append(st[32:])
                    by_suite[suite]["early"].append(st[:32]);by_suite[suite]["late"].append(st[32:])
            def cat(xs):return torch.cat([x.reshape(-1,48) for x in xs],0)
            sets={k:cat(v) for k,v in buckets.items()}
            for lab in h1.ORDER:
                for ph in ("early","late"):sets[f"origin_{lab}_{ph}"]=cat(by_origin[lab][ph])
            for suite in SEEDS:
                for ph in ("early","late"):sets[f"{suite}_{ph}"]=cat(by_suite[suite][ph])
            rep={"schema":"authority_matched_state_audit_v1","sets":{}}
            for name,states in sets.items():
                base=metrics(models["u20"],states);entry={"n_states":int(len(states)),"u20":base,"models":{}}
                for mn in ("u30","common30_family20","common20_family30"):
                    q=metrics(models[mn],states);entry["models"][mn]={"sensitivity":q,"retention":ret(q,base)}
                rep["sets"][name]=entry
                if name in ("initial","early","late") or name.startswith("origin_"):
                    print(name,{mn:entry["models"][mn]["retention"] for mn in entry["models"]},flush=True)
            (OUT/"matched_state_audit.json").write_text(json.dumps(rep,indent=2)+"\n")
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_authority_isolated_build_u20_authority_support():
    """Run former authority_isolated_build_u20_authority_support.py stage."""
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from pathlib import Path
    import json,sys,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"scripts/rl")]
    import rl.experiments.common.utilities.authority_isolated_h1_screen as h1
    from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
    CK=ROOT/"runs/authority_isolated_coverage2_control-2026-09-25/model_20.pt"
    OUT=ROOT/"runs/authority_isolated_u20_authority_support-2026-09-25";OUT.mkdir(parents=True,exist_ok=True)
    SEEDS=(840003,840004,850101,850202,850303);NENV=8;H=64;PER_CELL=128
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def main():
        from isaaclab.app import AppLauncher
        sv=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv;env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda()
            m=AuthorityIsolatedWideCritic(o.shape[-1],12).cuda();m.load_state_dict(torch.load(CK,map_location="cuda",weights_only=False)["model"]);m.eval()
            rng=np.random.default_rng(2609252301);obs_out=[];origin=[];phase=[];seed_out=[]
            for li,lab in enumerate(h1.ORDER):
                pool={"initial":[],"early":[],"late":[]}
                for seed in SEEDS:
                    w=torch.tensor(h1.PREFS[lab],device="cuda").repeat(NENV,1);cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
                    with torch.no_grad():
                        for t in range(H):
                            key="initial" if t<4 else ("early" if t<32 else "late")
                            pool[key].append(cur.detach().cpu())
                            a=m.act_inference_with_preference(cur,w);nxt,_,_,_,_=env.step(a);cur=ot(nxt).cuda()
                for pi,key in enumerate(("initial","early","late")):
                    X=torch.cat(pool[key],0).numpy()
                    idx=rng.choice(len(X),size=min(PER_CELL,len(X)),replace=False)
                    obs_out.append(X[idx]);origin.extend([li]*len(idx));phase.extend([pi]*len(idx));seed_out.extend([-1]*len(idx))
            obs=np.concatenate(obs_out).astype(np.float32);origin=np.asarray(origin,np.int64);phase=np.asarray(phase,np.int64)
            np.savez_compressed(OUT/"u20_authority_support.npz",obs=obs,origin=origin,phase=phase)
            counts={}
            for li,lab in enumerate(h1.ORDER):
                counts[lab]={}
                for pi,key in enumerate(("initial","early","late")):counts[lab][key]=int(((origin==li)&(phase==pi)).sum())
            rep={"schema":"u20_authority_support_v1","checkpoint":str(CK.relative_to(ROOT)),"seeds":list(SEEDS),
                 "per_cell_target":PER_CELL,"n_states":int(len(obs)),"counts":counts,
                 "phases":{"initial":"t0-3","early":"t4-31","late":"t32-63"}}
            (OUT/"support_manifest.json").write_text(json.dumps(rep,indent=2)+"\n");print(json.dumps(rep,indent=2),flush=True)
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_authority_isolated_direction_amplitude_audit():
    """Run former authority_isolated_direction_amplitude_audit.py stage."""
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
    from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
    
    CK={
    "u20":ROOT/"runs/authority_isolated_coverage2_control-2026-09-25/model_20.pt",
    "control":ROOT/"runs/authority_isolated_functional_deltaa_control-2026-09-25/model_30.pt",
    "retain":ROOT/"runs/authority_isolated_functional_deltaa_retain-2026-09-25/model_30.pt",
    }
    SUP=ROOT/"runs/authority_isolated_u20_authority_support-2026-09-25/u20_authority_support.npz"
    OUT=ROOT/"runs/authority_isolated_direction_amplitude_audit-2026-09-25"
    OUT.mkdir(parents=True,exist_ok=True)
    HEAVY=("T","A","O","S")
    EPS=1e-8
    
    def load_model(path):
        s=torch.load(path,map_location="cuda",weights_only=False)["model"]
        m=AuthorityIsolatedWideCritic(48,12).cuda();m.load_state_dict(s);m.eval()
        return m
    
    def delta(m,x,lab):
        n=len(x)
        w=torch.tensor(h1.PREFS[lab],device="cuda").repeat(n,1)
        c=torch.tensor(h1.PREFS["C"],device="cuda").repeat(n,1)
        with torch.no_grad():
            return m.act_inference_with_preference(x,w)-m.act_inference_with_preference(x,c)
    
    def qstats(x):
        a=np.asarray(x,float)
        return {
          "mean":float(np.mean(a)),
          "median":float(np.median(a)),
          "p10":float(np.quantile(a,.10)),
          "p25":float(np.quantile(a,.25)),
          "p75":float(np.quantile(a,.75)),
          "p90":float(np.quantile(a,.90)),
          "below_0.9":float(np.mean(a<.9)),
          "below_0.8":float(np.mean(a<.8)),
          "below_0.7":float(np.mean(a<.7)),
        }
    
    def pearson(a,b):
        a=np.asarray(a,float);b=np.asarray(b,float)
        if np.std(a)<1e-12 or np.std(b)<1e-12:return 0.0
        return float(np.corrcoef(a,b)[0,1])
    
    def spear(a,b):
        a=np.asarray(a,float);b=np.asarray(b,float)
        ra=np.argsort(np.argsort(a));rb=np.argsort(np.argsort(b))
        return pearson(ra,rb)
    
    def main():
        d=np.load(SUP);X=torch.tensor(d["obs"],device="cuda")
        origin=d["origin"];phase=d["phase"]
        ms={k:load_model(v) for k,v in CK.items()}
        rows=[]
        phase_names=("initial","early","late")
        origin_names=h1.ORDER
    
        for lab in HEAVY:
            r=delta(ms["u20"],X,lab)
            rn=torch.linalg.vector_norm(r,dim=1)
            for name in ("control","retain"):
                q=delta(ms[name],X,lab)
                qn=torch.linalg.vector_norm(q,dim=1)
                cos=((q*r).sum(1)/(qn*rn+EPS)).clamp(-1,1)
                mag=qn/(rn+EPS)
                mpar=(q*r).sum(1)/(rn.pow(2)+EPS)
                orth=torch.sqrt(torch.clamp(qn.pow(2)-(mpar*rn).pow(2),min=0.0))/(rn+EPS)
                err=torch.linalg.vector_norm(q-r,dim=1)/(rn+EPS)
                for i in range(len(X)):
                    rows.append({
                      "model":name,"axis":lab,
                      "origin":origin_names[int(origin[i])],
                      "phase":phase_names[int(phase[i])],
                      "mag_ratio":float(mag[i].cpu()),
                      "cos_dir":float(cos[i].cpu()),
                      "m_parallel":float(mpar[i].cpu()),
                      "orth_ratio":float(orth[i].cpu()),
                      "relative_error":float(err[i].cpu()),
                      "ref_norm":float(rn[i].cpu()),
                    })
    
        # cell-level authority metrics: each state subset gets pairwise/tangent sensitivity.
        cell=[]
        rng=np.random.default_rng(2609252601)
        for oi,on in enumerate(origin_names):
            for pi,pn in enumerate(phase_names):
                idx=np.flatnonzero((origin==oi)&(phase==pi))
                # deterministic cap for sensitivity cost
                use=rng.choice(idx,size=min(96,len(idx)),replace=False)
                xx=X[use]
                b=h1.sensitivity(ms["u20"],xx)
                for name in ("control","retain"):
                    q=h1.sensitivity(ms[name],xx)
                    cell.append({
                      "model":name,"origin":on,"phase":pn,
                      "pairwise_retention":q["pairwise_action_distance"]["mean"]/(b["pairwise_action_distance"]["mean"]+1e-12),
                      "tangent_retention":q["tangent_jacobian_fro_mean"]/(b["tangent_jacobian_fro_mean"]+1e-12),
                      "functional_rms_retention":q["centered_functional_geometry"]["specific_rms"]/(b["centered_functional_geometry"]["specific_rms"]+1e-12),
                    })
    
        # Aggregate state metrics by matching model/origin/phase and join with cell authority.
        summary={}
        for name in ("control","retain"):
            summary[name]={"overall":{},"by_axis":{},"by_phase":{},"by_origin":{}}
            rr=[r for r in rows if r["model"]==name]
            for metric in ("mag_ratio","cos_dir","m_parallel","orth_ratio","relative_error"):
                summary[name]["overall"][metric]=qstats([r[metric] for r in rr])
            for lab in HEAVY:
                zz=[r for r in rr if r["axis"]==lab]
                summary[name]["by_axis"][lab]={m:qstats([r[m] for r in zz]) for m in ("mag_ratio","cos_dir","m_parallel","orth_ratio","relative_error")}
            for pn in phase_names:
                zz=[r for r in rr if r["phase"]==pn]
                summary[name]["by_phase"][pn]={m:qstats([r[m] for r in zz]) for m in ("mag_ratio","cos_dir","m_parallel","orth_ratio","relative_error")}
            for on in origin_names:
                zz=[r for r in rr if r["origin"]==on]
                summary[name]["by_origin"][on]={m:qstats([r[m] for r in zz]) for m in ("mag_ratio","cos_dir","m_parallel","orth_ratio","relative_error")}
    
        # correlation across the 15 matched origin×phase cells
        corr={}
        for name in ("control","retain"):
            crows=[c for c in cell if c["model"]==name]
            metrics={m:[] for m in ("mag_ratio","cos_dir","m_parallel","orth_ratio","relative_error")}
            pair=[];tan=[]
            for c in crows:
                zz=[r for r in rows if r["model"]==name and r["origin"]==c["origin"] and r["phase"]==c["phase"]]
                for m in metrics:metrics[m].append(float(np.median([r[m] for r in zz])))
                pair.append(c["pairwise_retention"]);tan.append(c["tangent_retention"])
            corr[name]={}
            for m,v in metrics.items():
                corr[name][m]={
                  "pearson_pairwise":pearson(v,pair),
                  "spearman_pairwise":spear(v,pair),
                  "pearson_tangent":pearson(v,tan),
                  "spearman_tangent":spear(v,tan),
                }
    
        rep={"schema":"direction_amplitude_authority_audit_v1","summary":summary,"cell_authority":cell,"correlations":corr}
        (OUT/"direction_amplitude_audit.json").write_text(json.dumps(rep,indent=2)+"\n")
        for name in ("control","retain"):
            print("\nMODEL",name)
            print("OVERALL",json.dumps(summary[name]["overall"],indent=2))
            print("BY_AXIS")
            for lab in HEAVY:
                s=summary[name]["by_axis"][lab]
                print(lab,"mag_med",s["mag_ratio"]["median"],"cos_med",s["cos_dir"]["median"],"mpar_med",s["m_parallel"]["median"],
                      "mpar<.9",s["m_parallel"]["below_0.9"])
            print("CORR",json.dumps(corr[name],indent=2))
    if True:main()

def run_authority_isolated_flhip_phase_rescue():
    """Run former authority_isolated_flhip_phase_rescue.py stage."""
    from pathlib import Path
    import json,sys,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    OUT=ROOT/"runs/authority_isolated_flhip_phase_rescue-2026-09-25";OUT.mkdir(parents=True,exist_ok=True)
    CKPT=ROOT/"runs/authority_isolated_h1-2026-09-25/model_75.pt";SEED=840004;NENV=8;LANE=0;H=24;J=0
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def tf(z,c):return c*torch.tanh(z/c)
    def run(env,m,steps,label):
        w=torch.full((NENV,4),.25,device="cuda");obs,_=env.reset(seed=SEED);obs=ot(obs).cuda();ff=None;vals=[]
        for t in range(H):
            with torch.no_grad():
                z=m._actor_mean_with_preference(obs,w);a20=torch.tanh(tf(z,2.0));a25=torch.tanh(tf(z,2.5));a=a20.clone()
                if t in steps:a[:,J]=a25[:,J]
            vals.append({"t":t,"a20":float(a20[LANE,J]),"a25":float(a25[LANE,J]),"used":float(a[LANE,J])})
            nxt,_,te,tr,_=env.step(a);dd=(te|tr).cpu().numpy()
            if dd[LANE] and ff is None:ff=t
            obs=ot(nxt).cuda()
        return {"label":label,"steps":sorted(steps),"first_fail":ff,"survived":ff is None,"vals":vals}
    def main():
        from isaaclab.app import AppLauncher
        sv=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv;env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.authority.isolated import AuthorityIsolatedActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda()
            m=AuthorityIsolatedActorCritic(o.shape[-1],12).cuda();m.load_state_dict(torch.load(CKPT,map_location="cuda",weights_only=False)["model"]);m.eval()
            tests=[("none",set()),("all0_15",set(range(16))),("early0_7",set(range(8))),("mid8_10",set(range(8,11))),
              ("late11_13",set(range(11,14))),("critical14_15",{14,15}),("pre0_10",set(range(11))),("pre11_15",set(range(11,16)))]
            tests += [(f"single_t{t}",{t}) for t in range(16)]
            rows=[]
            for label,steps in tests:
                r=run(env,m,steps,label);rows.append(r);print(label,r["survived"],r["first_fail"],flush=True)
            (OUT/"authority_isolated_flhip_phase_rescue_report.json").write_text(json.dumps({"rows":rows},indent=2)+"\n")
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_authority_isolated_flhip_reverse_phase():
    """Run former authority_isolated_flhip_reverse_phase.py stage."""
    from pathlib import Path
    import json,sys,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    OUT=ROOT/"runs/authority_isolated_flhip_reverse_phase-2026-09-25";OUT.mkdir(parents=True,exist_ok=True)
    CKPT=ROOT/"runs/authority_isolated_h1-2026-09-25/model_75.pt";SEED=840004;NENV=8;LANE=0;H=24;J=0
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def tf(z,c):return c*torch.tanh(z/c)
    def run(env,m,steps,label):
        w=torch.full((NENV,4),.25,device="cuda");obs,_=env.reset(seed=SEED);obs=ot(obs).cuda();ff=None
        for t in range(H):
            with torch.no_grad():
                z=m._actor_mean_with_preference(obs,w);a25=torch.tanh(tf(z,2.5));a20=torch.tanh(tf(z,2.0));a=a25.clone()
                if t in steps:a[:,J]=a20[:,J]
            nxt,_,te,tr,_=env.step(a);dd=(te|tr).cpu().numpy()
            if dd[LANE] and ff is None:ff=t
            obs=ot(nxt).cuda()
        return {"label":label,"steps":sorted(steps),"first_fail":ff,"survived":ff is None}
    def main():
        from isaaclab.app import AppLauncher
        sv=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv;env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.authority.isolated import AuthorityIsolatedActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda()
            m=AuthorityIsolatedActorCritic(o.shape[-1],12).cuda();m.load_state_dict(torch.load(CKPT,map_location="cuda",weights_only=False)["model"]);m.eval()
            tests=[("none",set()),("all0_15",set(range(16)))] + [(f"single_t{t}",{t}) for t in range(16)]
            rows=[]
            for label,steps in tests:
                r=run(env,m,steps,label);rows.append(r);print(label,r["survived"],r["first_fail"],flush=True)
            (OUT/"authority_isolated_flhip_reverse_phase_report.json").write_text(json.dumps({"rows":rows},indent=2)+"\n")
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_authority_isolated_flhip_window_reverse():
    """Run former authority_isolated_flhip_window_reverse.py stage."""
    from pathlib import Path
    import json,sys,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    OUT=ROOT/"runs/authority_isolated_flhip_window_reverse-2026-09-25";OUT.mkdir(parents=True,exist_ok=True)
    CKPT=ROOT/"runs/authority_isolated_h1-2026-09-25/model_75.pt";SEED=840004;NENV=8;LANE=0;H=24;J=0
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def tf(z,c):return c*torch.tanh(z/c)
    def run(env,m,steps,label):
        w=torch.full((NENV,4),.25,device="cuda");obs,_=env.reset(seed=SEED);obs=ot(obs).cuda();ff=None
        for t in range(H):
            with torch.no_grad():
                z=m._actor_mean_with_preference(obs,w);a25=torch.tanh(tf(z,2.5));a20=torch.tanh(tf(z,2.0));a=a25.clone()
                if t in steps:a[:,J]=a20[:,J]
            nxt,_,te,tr,_=env.step(a);dd=(te|tr).cpu().numpy()
            if dd[LANE] and ff is None:ff=t
            obs=ot(nxt).cuda()
        return {"label":label,"steps":sorted(steps),"first_fail":ff,"survived":ff is None}
    def main():
        from isaaclab.app import AppLauncher
        sv=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv;env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.authority.isolated import AuthorityIsolatedActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda()
            m=AuthorityIsolatedActorCritic(o.shape[-1],12).cuda();m.load_state_dict(torch.load(CKPT,map_location="cuda",weights_only=False)["model"]);m.eval()
            tests=[]
            for s,e in [(0,3),(4,7),(8,11),(12,15),(0,7),(4,11),(8,15),(0,11),(4,15),(0,15)]:
                tests.append((f"{s}_{e}",set(range(s,e+1))))
            rows=[]
            for label,steps in tests:
                r=run(env,m,steps,label);rows.append(r);print(label,r["survived"],r["first_fail"],flush=True)
            (OUT/"authority_isolated_flhip_window_reverse_report.json").write_text(json.dumps({"rows":rows},indent=2)+"\n")
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_authority_isolated_functional_deltaa_endpoint_audit():
    """Run former authority_isolated_functional_deltaa_endpoint_audit.py stage."""
    """Does the functional-delta-a retain arm keep u20's preference geometry?
    
    Measures the per-preference action offset from centre against the u20
    reference, then sensitivity retention overall and per support phase.
    """
    import json
    import sys
    from pathlib import Path
    
    import numpy as np
    import torch
    
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    
    import rl.experiments.common.utilities.authority_isolated_h1_screen as h1
    from rl.core.diagnostics.offline_audit import RUNS
    from rl.experiments.common.utilities.authority_retention import SUPPORT, AuthorityRetentionAudit
    
    HEAVY = ("T", "A", "O", "S")
    PHASES = ("initial", "early", "late")
    ARMS = ("control", "retain")
    
    
    class FunctionalDeltaAEndpointAudit(AuthorityRetentionAudit):
        """Preference geometry kept by the functional-delta-a arms at update 30."""
    
        run = "authority_isolated_functional_deltaa_endpoint_audit-2026-09-25"
        report = "endpoint_audit.json"
        schema = "functional_deltaa_endpoint_audit_v1"
        support_seed = 2609252501
        checkpoints = {
            "u20": "authority_isolated_coverage2_control-2026-09-25/model_20.pt",
            "control": "authority_isolated_functional_deltaa_control-2026-09-25/model_30.pt",
            "retain": "authority_isolated_functional_deltaa_retain-2026-09-25/model_30.pt"}
    
        def delta(self, m, x, lab):
            """Action offset of a preference from the centre preference."""
            n = len(x)
            w = torch.tensor(h1.PREFS[lab], device="cuda").repeat(n, 1)
            c = torch.tensor(h1.PREFS["C"], device="cuda").repeat(n, 1)
            with torch.no_grad():
                return m.act_inference_with_preference(x, w) - m.act_inference_with_preference(x, c)
    
        def analyze(self):
            d = np.load(RUNS / SUPPORT)
            x_all = torch.tensor(d["obs"], device="cuda")
            origin, phase = d["origin"], d["phase"]
            ms = {k: self.load_wide(v) for k, v in self.checkpoints.items()}
    
            rep = {"schema": self.schema, "models": {}}
            for name in ARMS:
                mse, rel = [], []
                for lab in HEAVY:
                    r = self.delta(ms["u20"], x_all, lab)
                    q = self.delta(ms[name], x_all, lab)
                    mse.append(float(((q - r) ** 2).mean(1).mean().cpu()))
                    rel.append(float((torch.linalg.vector_norm(q - r, dim=1)
                                      / (torch.linalg.vector_norm(r, dim=1) + 1e-8)).mean().cpu()))
                rep["models"][name] = {"deltaa_mse": float(np.mean(mse)),
                                       "deltaa_relative_error": float(np.mean(rel)),
                                       "by_axis_mse": dict(zip(HEAVY, mse))}
    
            # 384 states: up to 128 per phase, balanced over origins. The generator
            # is reused by the per-phase draw below, so the order of the two must
            # not change.
            rng = np.random.default_rng(self.support_seed)
            ids = []
            for pi in range(3):
                pools = []
                for oi in range(5):
                    z = np.flatnonzero((origin == oi) & (phase == pi))
                    pools.extend(rng.choice(z, size=25, replace=False).tolist())
                ids.extend(pools[:128] if len(pools) >= 128 else pools)
            x = x_all[np.asarray(ids[:384])]
    
            base = h1.sensitivity(ms["u20"], x)
            rep["u20_sensitivity"] = base
            for name in ARMS:
                q = h1.sensitivity(ms[name], x)
                m = rep["models"][name]
                m["sensitivity"] = q
                m["retention"] = {
                    "pairwise": self.ratio(q, base, "pairwise_action_distance", "mean"),
                    "tangent": self.ratio(q, base, "tangent_jacobian_fro_mean"),
                    "functional_specific_rms": self.ratio(q, base, "centered_functional_geometry", "specific_rms"),
                    "parameter_specific_energy": self.ratio(q, base, "centered_parameter_geometry", "specific_energy"),
                    "parameter_rank": q["centered_parameter_geometry"]["effective_rank_5pct"],
                    "functional_rank": q["centered_functional_geometry"]["effective_rank_5pct"]}
                m["authority_gate"] = bool(m["retention"]["pairwise"] >= .9
                                           and m["retention"]["tangent"] >= .9)
    
            rep["phase"] = {}
            for pi, pn in enumerate(PHASES):
                z = np.flatnonzero(phase == pi)
                xx = x_all[rng.choice(z, size=min(256, len(z)), replace=False)]
                b = h1.sensitivity(ms["u20"], xx)
                rep["phase"][pn] = {}
                for name in ARMS:
                    q = h1.sensitivity(ms[name], xx)
                    rep["phase"][pn][name] = {
                        "pairwise": self.ratio(q, b, "pairwise_action_distance", "mean"),
                        "tangent": self.ratio(q, b, "tangent_jacobian_fro_mean")}
            return rep
    
        def summarize(self, report):
            print(json.dumps({k: {"mse": v["deltaa_mse"], "relerr": v["deltaa_relative_error"],
                                  "retention": v["retention"], "gate": v["authority_gate"]}
                              for k, v in report["models"].items()}, indent=2), flush=True)
            print("PHASE", json.dumps(report["phase"], indent=2), flush=True)
    
    
    if True:
        FunctionalDeltaAEndpointAudit.main()

def run_authority_isolated_functional_deltaa_retention_train():
    """Run former authority_isolated_functional_deltaa_retention_train.py stage."""
    
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from pathlib import Path
    import argparse,json,sys
    import numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"scripts/rl")]
    import rl.experiments.common.utilities.authority_isolated_h1_screen as h1
    from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
    
    SOURCE_STATE=ROOT/"runs/authority_isolated_coverage2_control-2026-09-25/resume_state.pt"
    SOURCE_MODEL=ROOT/"runs/authority_isolated_coverage2_control-2026-09-25/model_20.pt"
    SUPPORT=ROOT/"runs/authority_isolated_u20_authority_support-2026-09-25/u20_authority_support.npz"
    TAU_PATH=ROOT/"runs/authority_isolated_tail_budget_calibration-2026-09-25/tail_budget_probe.json"
    PROBE=ROOT/"runs/update_functional_effect_audit-2026-09-24/fixed_probe_states.npz"
    NENV=h1.NENV;H=h1.H;KAPPA=.05;RHO=.25;BETA0=2.497041993384243;EPS=1e-12
    HEAVY=("T","A","O","S")
    
    def tail_metrics(m,obs,w,tau):
        z=m._actor_mean_with_preference(obs,w);ex=torch.relu(z.abs()-tau)
        return ((ex/(tau+1e-6))**2).mean(),(z.abs()>tau).float().mean()
    
    def audit(m,probe,tau,out,tag,snaps,arm):
        m.eval();sens=h1.sensitivity(m,probe)
        P=torch.tensor(np.load(PROBE)["obs"],device="cuda")
        wm=torch.tensor(h1.PREFS["C"],device="cuda").repeat(len(P),1)
        with torch.no_grad():tl,tf=tail_metrics(m,P,wm,tau)
        snaps[str(tag)]={"sensitivity":sens,"probe_tail_loss":float(tl.cpu()),"probe_tail_fraction":float(tf.cpu())}
        torch.save({"model":m.state_dict(),"update":int(tag),"arm":arm},out/f"model_{tag}.pt");m.train()
    
    def save_state(path,m,opt,adaptive_pools,anchor_specs,rows,snaps,update):
        torch.save({"model":m.state_dict(),"optimizer":opt.state_dict(),"adaptive_pools":adaptive_pools,
          "anchor_specs":anchor_specs,"rows":rows,"snaps":snaps,"update":update,
          "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all(),"numpy_rng":np.random.get_state()},path)
    
    def balanced_indices(origin,phase,update):
        rng=np.random.default_rng(2609252401+update);ids=[]
        for oi in range(5):
            for pi in range(3):
                pool=np.flatnonzero((origin==oi)&(phase==pi))
                ids.extend(rng.choice(pool,size=8,replace=False).tolist())
        return np.asarray(ids,np.int64)
    
    def deltaa_loss(m,ref,obs):
        n=len(obs);c=torch.tensor(h1.PREFS["C"],device="cuda").repeat(n,1)
        x=obs.repeat_interleave(4,0)
        wh=torch.tensor(np.stack([h1.PREFS[k] for k in HEAVY]),device="cuda").repeat(n,1)
        cc=c.repeat_interleave(4,0)
        a=m.act_inference_with_preference(x,wh);ac=m.act_inference_with_preference(x,cc)
        with torch.no_grad():
            ar=ref.act_inference_with_preference(x,wh);acr=ref.act_inference_with_preference(x,cc)
        return ((a-ac)-(ar-acr)).pow(2).mean()
    
    def grad_norm(gs):
        return torch.sqrt(sum((g.detach()**2).sum() for g in gs if g is not None)+EPS)
    
    def main():
        ap=argparse.ArgumentParser();ap.add_argument("--arm",choices=["control","retain"],required=True)
        ap.add_argument("--seed",type=int,default=73001);ap.add_argument("--target-updates",type=int,default=30)
        args=ap.parse_args()
        out=ROOT/f"runs/authority_isolated_functional_deltaa_{args.arm}-2026-09-25";out.mkdir(parents=True,exist_ok=True)
        state_path=out/"resume_state.pt";tau=torch.tensor(json.load(open(TAU_PATH))["tau"],device="cuda",dtype=torch.float32)
        sd=np.load(SUPPORT);ref_obs=torch.tensor(sd["obs"],device="cuda");origin=sd["origin"];phase=sd["phase"]
        torch.manual_seed(args.seed);np.random.seed(args.seed)
        from isaaclab.app import AppLauncher
        sv=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv;env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.foundations.four_objective import vector_gae,scalarized_late_weighted_ppo
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=args.seed
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=args.seed);o=h1.ot(o).cuda();probe=o.detach().clone()
            ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
            m=AuthorityIsolatedWideCritic(o.shape[-1],ad).cuda()
            ref=AuthorityIsolatedWideCritic(o.shape[-1],ad).cuda();ref.load_state_dict(torch.load(SOURCE_MODEL,map_location="cuda",weights_only=False)["model"]);ref.eval()
            for p in ref.parameters():p.requires_grad_(False)
            for n,p in m.named_parameters():
                if n.startswith("critic_body") or n.startswith("critic_head"):p.requires_grad_(False)
            actor_params=[p for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std" or n.startswith("family_")]
            opt=torch.optim.Adam(actor_params,lr=1e-3)
    
            if state_path.exists():
                st=torch.load(state_path,map_location="cpu",weights_only=False);m.load_state_dict(st["model"]);opt.load_state_dict(st["optimizer"])
                adaptive_pools=st["adaptive_pools"];anchor_specs=st["anchor_specs"];rows=st["rows"];snaps=st["snaps"];cur=int(st["update"])
                torch.set_rng_state(st["torch_rng"]);torch.cuda.set_rng_state_all(st["cuda_rng"]);np.random.set_state(st["numpy_rng"])
            else:
                st=torch.load(SOURCE_STATE,map_location="cpu",weights_only=False);m.load_state_dict(st["model"]);opt.load_state_dict(st["optimizer"])
                adaptive_pools=st["adaptive_pools"];anchor_specs=st["anchor_specs"];rows=[];snaps={"20":st["snaps"]["20"]};cur=20
                audit(m,probe,tau,out,20,snaps,args.arm);save_state(state_path,m,opt,adaptive_pools,anchor_specs,rows,snaps,cur)
    
            def current_anchor_units():
                q={"early":[],"late":[]}
                for k,seed in anchor_specs:
                    _,w=h1.pref_batch(k,torch.device("cuda"))
                    for u in h1.collect_support64(env,m,w,mgr,seed):q[u["phase"]].append(u)
                return q
    
            for uidx in range(cur+1,args.target_updates+1):
                _,w=h1.pref_batch(uidx,torch.device("cuda"))
                main=h1.collect_actor(env,m,w,mgr,args.seed+uidx*211,True)
                _,ws=h1.pref_batch(uidx+17,torch.device("cuda"));units=h1.collect_support64(env,m,ws,mgr,args.seed+200000+uidx*223)
                for u in units:
                    ph=u["phase"];adaptive_pools[ph].append(u)
                    if len(adaptive_pools[ph])>h1.POOL_MAX:adaptive_pools[ph].pop(0)
                h1.fit_expanded_current_policy(m,current_anchor_units(),adaptive_pools)
                with torch.no_grad():
                    vt=m.value_with_preference(main["obs"],main["w"]).reshape(H,NENV,4);nv=m.value_with_preference(main["next_obs"],w)
                    adv,_=vector_gae(main["rt"],vt,nv,main["dt"],lam=.95)
                lp=[]
                for stx in range(0,len(main["obs"]),NENV):
                    lp.append(m.logp_from_pre_tanh_with_preference(main["obs"][stx:stx+NENV],main["w"][stx:stx+NENV],main["u"][stx:stx+NENV]))
                ratio=torch.exp(torch.cat(lp)-main["old"].detach());ratio_err=float((ratio-1).abs().max().detach().cpu())
                if ratio_err>1e-4:raise RuntimeError(f"ratio invariant {ratio_err}")
                ppo=scalarized_late_weighted_ppo(ratio,adv.reshape(-1,4).detach(),main["w"])
                tail,tailfrac=tail_metrics(m,main["obs"],main["w"],tau)
                gp=torch.autograd.grad(ppo,actor_params,retain_graph=True,allow_unused=True)
                gt=torch.autograd.grad(tail,actor_params,retain_graph=(args.arm=="retain"),allow_unused=True)
                gp2=sum((g.detach()**2).sum() for g in gp if g is not None);gt2=sum((g.detach()**2).sum() for g in gt if g is not None)
                gpn=torch.sqrt(gp2+EPS);gtn=torch.sqrt(gt2+EPS)
                dot=sum((a.detach()*b.detach()).sum() for a,b in zip(gp,gt) if a is not None and b is not None)
                coeff=torch.minimum(dot/(gt2+EPS),torch.zeros_like(dot));base=[];proj2=torch.zeros((),device="cuda")
                for p,a,b in zip(actor_params,gp,gt):
                    aa=torch.zeros_like(p) if a is None else a;bb=torch.zeros_like(p) if b is None else b
                    pp=aa-coeff*bb;base.append([pp,bb]);proj2+=(pp.detach()**2).sum()
                projn=torch.sqrt(proj2+EPS);tail_scale=KAPPA*projn/(gtn+EPS)
                gbase=[pp+tail_scale*bb for pp,bb in base];gbn=grad_norm(gbase)
    
                retain=torch.zeros((),device="cuda");gr=[None]*len(actor_params);alpha=0.;grn=torch.zeros((),device="cuda")
                if args.arm=="retain":
                    ids=balanced_indices(origin,phase,uidx);retain=deltaa_loss(m,ref,ref_obs[ids])
                    gr=torch.autograd.grad(retain,actor_params,allow_unused=True);grn=grad_norm(gr)
                    alpha=min(BETA0,RHO*float(gbn.detach().cpu())/(float(grn.detach().cpu())+EPS))
                opt.zero_grad(set_to_none=True)
                for i,p in enumerate(actor_params):
                    gg=gbase[i]
                    if args.arm=="retain" and gr[i] is not None:gg=gg+alpha*gr[i]
                    p.grad=gg
                preclip=float(torch.nn.utils.clip_grad_norm_(actor_params,1.0).detach().cpu());opt.step()
                with torch.no_grad():m.log_std.clamp_(m.LOG_STD_MIN,m.LOG_STD_MAX)
                row={"update":uidx,"ppo_loss":float(ppo.detach().cpu()),"tail_loss":float(tail.detach().cpu()),"tail_fraction":float(tailfrac.detach().cpu()),
                     "retain_loss":float(retain.detach().cpu()),"retain_alpha":float(alpha),"base_grad_norm":float(gbn.cpu()),"retain_grad_norm":float(grn.cpu()),
                     "weighted_retain_over_base":float(alpha*float(grn.cpu())/(float(gbn.cpu())+EPS)) if args.arm=="retain" else 0.,
                     "ratio_maxerr":ratio_err,"termination_fraction":main["termination_fraction"],"grad_norm_preclip":preclip}
                rows.append(row);print("UPDATE",args.arm,uidx,json.dumps(row),flush=True)
                if uidx==args.target_updates:audit(m,probe,tau,out,uidx,snaps,args.arm)
                save_state(state_path,m,opt,adaptive_pools,anchor_specs,rows,snaps,uidx)
    
            s0=snaps["20"]["sensitivity"];sf=snaps[str(args.target_updates)]["sensitivity"]
            rep={"schema":"functional_deltaa_durability_v1","arm":args.arm,"rho":RHO,"beta0":BETA0,"updates":args.target_updates,
                 "summary":{"fixed_probe_pairwise_retention":sf["pairwise_action_distance"]["mean"]/(s0["pairwise_action_distance"]["mean"]+EPS),
                            "fixed_probe_tangent_retention":sf["tangent_jacobian_fro_mean"]/(s0["tangent_jacobian_fro_mean"]+EPS),
                            "max_ratio_error":max(r["ratio_maxerr"] for r in rows),"max_termination_fraction":max(r["termination_fraction"] for r in rows),
                            "mean_weighted_retain_over_base":float(np.mean([r["weighted_retain_over_base"] for r in rows]))},
                 "rows":rows,"snapshots":snaps}
            (out/"training_report.json").write_text(json.dumps(rep,indent=2)+"\n");print("FINAL",args.arm,json.dumps(rep["summary"]),flush=True)
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_authority_isolated_h0_gate():
    """Run former authority_isolated_h0_gate.py stage."""
    from pathlib import Path
    import json,sys,hashlib,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
    BASE=ROOT/'runs/v2b_adam_continuous-2026-09-24/model_75.pt'
    STUD=ROOT/'runs/authority_isolation_feasibility-2026-09-25/student_step_5000.pt'
    PROBE=ROOT/'runs/update_functional_effect_audit-2026-09-24/fixed_probe_states.npz'
    OUT=ROOT/'runs/authority_isolated_h0-2026-09-25'
    OUT.mkdir(parents=True,exist_ok=True)
    P={'T':[.7,.1,.1,.1],'A':[.1,.7,.1,.1],'O':[.1,.1,.7,.1],
       'S':[.1,.1,.1,.7],'C':[.25]*4}
    TOL=1e-6
    
    def ot(x):
        if isinstance(x,dict):
            x=x.get('policy',next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    
    def sha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    
    def main():
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]]
        app=AppLauncher({'headless':True,'enable_cameras':False}).app
        sys.argv=saved;env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.conditioning.single_site_film import V2BSingleSiteFiLMActorCritic
            from talon_rl.models.authority.isolated import AuthorityIsolatedActorCritic,initialize_from_transfer
            cfg=UnitreeA1FlatEnvCfg()
            cfg.scene.num_envs=8;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/'talon_rl/assets/data/Robots/unitree_a1/a1.usd')
            env=gym.make('Isaac-Velocity-Flat-Unitree-A1-v0',cfg=cfg)
            o,_=env.reset(seed=0);o=ot(o).cuda()
            ad=env.unwrapped.action_manager.total_action_dim
            vb=torch.load(BASE,map_location='cuda',weights_only=False)['model']
            ss=torch.load(STUD,map_location='cuda',weights_only=False)['model']
            t=V2BSingleSiteFiLMActorCritic(o.shape[-1],ad).cuda()
            t.load_state_dict(vb);t.eval()
            m=AuthorityIsolatedActorCritic(o.shape[-1],ad).cuda()
            initialize_from_transfer(m,ss,vb);m.eval()
    
            init=OUT/'authority_isolated_h0_init.pt'
            torch.save({'model':m.state_dict()},init)
            mr=AuthorityIsolatedActorCritic(o.shape[-1],ad).cuda()
            mr.load_state_dict(torch.load(init,map_location='cuda',weights_only=False)['model'])
            mr.eval()
            probe=torch.tensor(np.load(PROBE)['obs'],device='cuda')
            mx={'critic':0.,'ratio':0.,'roundtrip_action':0.,'roundtrip_value':0.}
            finite=True
            for lab,wv in P.items():
                w=torch.tensor(wv,device='cuda').repeat(len(probe),1)
                with torch.no_grad():
                    vt=t.value_with_preference(probe,w)
                    vm=m.value_with_preference(probe,w)
                    am=m.act_inference_with_preference(probe,w)
                    ar=mr.act_inference_with_preference(probe,w)
                    vr=mr.value_with_preference(probe,w)
                mx['critic']=max(mx['critic'],float((vt-vm).abs().max().cpu()))
                mx['roundtrip_action']=max(mx['roundtrip_action'],float((am-ar).abs().max().cpu()))
                mx['roundtrip_value']=max(mx['roundtrip_value'],float((vm-vr).abs().max().cpu()))
                torch.manual_seed(1234)
                a,lp,u=m.act_with_preference_latent(probe,w)
                lp2=m.logp_from_pre_tanh_with_preference(probe,w,u)
                ratio=torch.exp(lp2-lp.detach())
                mx['ratio']=max(mx['ratio'],float((ratio-1).abs().max().cpu()))
                finite &= bool(torch.isfinite(a).all() and torch.isfinite(lp).all())
    
            from torch.func import jacrev,vmap
            w0=torch.tensor([.25]*4,device='cuda')
            D=torch.tensor([[1.,-1,0,0],[1.,0,-1,0],[1.,0,0,-1]],device='cuda').T
            Q,_=torch.linalg.qr(D,mode='reduced')
            def tf(x,w):
                return t._pre_tanh_dist_with_preference(x.unsqueeze(0),w.unsqueeze(0)).mean.squeeze(0)
            def mf(x,w):
                return m._pre_tanh_dist_with_preference(x.unsqueeze(0),w.unsqueeze(0)).mean.squeeze(0)
            jt=vmap(jacrev(tf,argnums=1),in_dims=(0,None))(probe,w0)@Q
            jm=vmap(jacrev(mf,argnums=1),in_dims=(0,None))(probe,w0)@Q
            jrel=float((torch.linalg.vector_norm(jm-jt)/(torch.linalg.vector_norm(jt)+1e-12)).cpu())
            jcos=float(((jt.flatten()*jm.flatten()).sum()/
                        (torch.linalg.vector_norm(jt)*torch.linalg.vector_norm(jm)+1e-12)).cpu())
            names=[n for n,_ in m.named_modules()]
            isolated=(m.actor_body[0].in_features==48 and
                      not any('preference_embedding' in n or 'preference_film' in n for n in names))
    
            surv=[]
            for _,wv in P.items():
                for seed in (840001,840002,840003,840004):
                    cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
                    w=torch.tensor(wv,device='cuda').repeat(len(cur),1)
                    alive=torch.ones(len(cur),dtype=torch.bool,device='cuda')
                    with torch.no_grad():
                        for _ in range(64):
                            a=m.act_inference_with_preference(cur,w)
                            nxt,_,term,trunc,_=env.step(a)
                            alive &= ~(term|trunc)
                            cur=ot(nxt).cuda()
                    surv.append(float(alive.float().mean().cpu()))
            survival=float(np.mean(surv))
            feas=json.load(open(ROOT/'runs/authority_isolation_feasibility-2026-09-25/authority_isolation_feasibility_report_repaired.json'))
            held=next(x for x in feas['splits'] if x['name']=='heldout_states_x_heldout_preferences')
            transfer_ok=(held['pre_tanh_rmse']<=.02 and held['mean_action_l2']<=.03 and
                         held['p95_action_l2']<=.06 and held['max_abs_action_coordinate_error']<=.10 and
                         held['preference_separation_relative_error']<=.10)
            crit={
              'transfer_bounds':transfer_ok,
              'tangent_jacobian':jrel<=.15 and jcos>=.95,
              'actor_graph_isolated':isolated,
              'critic_unchanged':mx['critic']<=TOL,
              'ppo_ratio':mx['ratio']<=TOL,
              'roundtrip':mx['roundtrip_action']<=TOL and mx['roundtrip_value']<=TOL,
              'finite_semantics':finite,
              'survival':survival>=.95,
            }
            passed=all(crit.values())
            rep={'schema':'authority_isolated_h0_v1','status':'PASS' if passed else 'FAIL',
                 'criteria':crit,
                 'metrics':{'tangent_jacobian_relative_error':jrel,
                            'tangent_jacobian_cosine':jcos,
                            'survival_mean':survival,**mx},
                 'decision':{'AI_H1_authorized':passed}}
            rp=OUT/'authority_isolated_h0_report.json'
            rp.write_text(json.dumps(rep,indent=2)+'\n')
            (OUT/'PROVENANCE_MANIFEST.json').write_text(json.dumps({
              'status':'FROZEN_BY_HASH',
              'report_sha256':sha(rp),
              'script_sha256':sha(Path(__file__).resolve()),
              'model_sha256':sha(ROOT/'talon_rl/authority_isolated_actor_critic.py'),
              'init_sha256':sha(init)},indent=2)+'\n')
            print(json.dumps(rep,indent=2))
        finally:
            if env is not None:
                env.close()
            app.close()
    
    if True:
        main()

def run_authority_isolated_overexpansion_causal_audit():
    """Run former authority_isolated_overexpansion_causal_audit.py stage."""
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from pathlib import Path
    import json,sys,itertools
    import numpy as np, torch
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"scripts/rl")]
    import rl.experiments.common.utilities.authority_isolated_h1_screen as h1
    from talon_rl.models.authority.isolated import AuthorityIsolatedActorCritic
    from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
    
    CK20=ROOT/"runs/authority_isolated_coverage2_control-2026-09-25/model_20.pt"
    CK30=ROOT/"runs/authority_isolated_simplex_edge_retain-2026-09-25/model_30.pt"
    CK40={
     "narrow":ROOT/"runs/authority_isolated_ai_c2_formal_edge_narrow-2026-09-25/model_10.pt",
     "wide":ROOT/"runs/authority_isolated_ai_c2_formal_edge_wide-2026-09-25/model_10.pt",
    }
    SUP=ROOT/"runs/authority_isolated_u20_authority_support-2026-09-25/u20_authority_support.npz"
    OUT=ROOT/"runs/authority_isolated_overexpansion_causal_audit-2026-09-25";OUT.mkdir(parents=True,exist_ok=True)
    ORDER=h1.ORDER;PREFS=h1.PREFS;LAMBDAS=(1.0,.75,.5,.25,0.0)
    SEEDS=[9700000,9700113,9700226,9700339,9701000,9701113,9701226,9701339,9702000,9702113,9702226,9702339,9703000,9703113,9703226,9703339,9704000,9704113,9704226,9704339]
    NENV=8;H=64;EPS=1e-8;PAIRS=list(itertools.combinations(ORDER,2))
    
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    
    def load_wide(path):
        s=torch.load(path,map_location="cuda",weights_only=False)["model"];m=AuthorityIsolatedWideCritic(48,12).cuda();m.load_state_dict(s);m.eval();return m
    def load40(arm,path):
        s=torch.load(path,map_location="cuda",weights_only=False)["model"]
        cls=AuthorityIsolatedActorCritic if arm=="narrow" else AuthorityIsolatedWideCritic
        m=cls(48,12).cuda();m.load_state_dict(s);m.eval();return m
    
    class FunctionalInterp:
        def __init__(self,m30,m40,lam):
            self.m30=m30;self.m40=m40;self.lam=float(lam);self.ACTION_CLIP=m40.ACTION_CLIP
        def pre_tanh_with_preference(self,obs,w):
            c=torch.tensor(PREFS["C"],device=obs.device,dtype=w.dtype).repeat(len(obs),1)
            z40c=self.m40._actor_mean_with_preference(obs,c)
            z30c=self.m30._actor_mean_with_preference(obs,c)
            z40w=self.m40._actor_mean_with_preference(obs,w)
            z30w=self.m30._actor_mean_with_preference(obs,w)
            dz30=z30w-z30c;dz40=z40w-z40c
            return z40c+dz30+self.lam*(dz40-dz30)
        def act_inference_with_preference(self,obs,w):
            return torch.tanh(self.pre_tanh_with_preference(obs,w))*self.ACTION_CLIP
    
    def authority_metrics(pol,m20,X):
        acts={};ref={}
        with torch.no_grad():
            for lab in ORDER:
                w=torch.tensor(PREFS[lab],device="cuda").repeat(len(X),1)
                acts[lab]=pol.act_inference_with_preference(X,w)
                ref[lab]=m20.act_inference_with_preference(X,w)
        ds=[];rs=[];edges={}
        for i,j in PAIRS:
            d=torch.linalg.vector_norm(acts[i]-acts[j],dim=1)
            r=torch.linalg.vector_norm(ref[i]-ref[j],dim=1)
            ds.append(float(d.mean().cpu()));rs.append(float(r.mean().cpu()))
            edges[f"{i}-{j}"]=float(torch.sqrt((d.pow(2).sum()+1e-12)/(r.pow(2).sum()+1e-12)).cpu())
        pair=float(np.mean(ds)/(np.mean(rs)+1e-12))
        # tangent at center
        from torch.func import jacrev,vmap
        wc=torch.tensor(PREFS["C"],device="cuda")
        D=torch.tensor([[1.,-1,0,0],[1.,0,-1,0],[1.,0,0,-1]],device="cuda").T;Q,_=torch.linalg.qr(D,mode="reduced")
        def afun(o,w):return pol.act_inference_with_preference(o.unsqueeze(0),w.unsqueeze(0)).squeeze(0)
        def rfun(o,w):return m20.act_inference_with_preference(o.unsqueeze(0),w.unsqueeze(0)).squeeze(0)
        J=vmap(jacrev(afun,argnums=1),in_dims=(0,None))(X,wc)@Q
        R=vmap(jacrev(rfun,argnums=1),in_dims=(0,None))(X,wc)@Q
        tan=float((torch.linalg.matrix_norm(J,ord="fro",dim=(1,2)).mean()/(torch.linalg.matrix_norm(R,ord="fro",dim=(1,2)).mean()+1e-12)).detach().cpu())
        vals=list(edges.values())
        return {"pairwise_retention":pair,"tangent_retention":tan,"min_edge":min(vals),"min_edge_name":min(edges,key=edges.get),
                "mean_edge":float(np.mean(vals)),"edges":edges}
    
    def rollout(env,pol,m40,lab,seed):
        w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1)
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        done=np.zeros(NENV,bool);ttf=np.full(NENV,H,np.int32);reason=[[] for _ in range(NENV)]
        prev_a=None
        max_ang=np.zeros(NENV);max_tilt=np.zeros(NENV);max_jv=np.zeros(NENV)
        max_anorm=np.zeros(NENV);max_arate=np.zeros(NENV);min_head=np.ones(NENV)*9.;sat_steps=np.zeros(NENV)
        raw40_sat=np.zeros(NENV);raw40_absmax=np.zeros(NENV)
        traces=[[] for _ in range(NENV)]
        with torch.no_grad():
            for t in range(H):
                a=pol.act_inference_with_preference(cur,w)
                zeff=pol.pre_tanh_with_preference(cur,w)
                aa=a.detach().cpu().numpy();oo=cur.detach().cpu().numpy()
                ang=np.linalg.norm(oo[:,3:6],axis=1);tilt=np.linalg.norm(oo[:,6:8],axis=1);jv=np.linalg.norm(oo[:,24:36],axis=1)
                an=np.linalg.norm(aa,axis=1);head=np.min(1-np.abs(aa),axis=1)
                ar=np.zeros(NENV) if prev_a is None else np.linalg.norm(aa-prev_a,axis=1)
                z=zeff.detach().cpu().numpy()
                max_ang=np.maximum(max_ang,ang);max_tilt=np.maximum(max_tilt,tilt);max_jv=np.maximum(max_jv,jv)
                max_anorm=np.maximum(max_anorm,an);max_arate=np.maximum(max_arate,ar);min_head=np.minimum(min_head,head)
                sat_steps+=(np.abs(aa)>=.95).mean(axis=1);raw40_sat+=(np.abs(np.tanh(z))>=.95).mean(axis=1);raw40_absmax=np.maximum(raw40_absmax,np.max(np.abs(z),axis=1))
                for i in range(NENV):
                    if not done[i]:
                        traces[i].append([t,float(ang[i]),float(tilt[i]),float(jv[i]),float(an[i]),float(ar[i]),float(head[i])])
                nxt,_,te,tr,_=env.step(a);dd=(te|tr).cpu().numpy();tm=env.unwrapped.termination_manager
                new=dd & (~done)
                for i in np.flatnonzero(new):
                    ttf[i]=t+1;reason[i]=[n for n in tm.active_terms if bool(tm.get_term(n)[i].detach().cpu())]
                done|=dd;prev_a=aa;cur=ot(nxt).cuda()
        lanes=[]
        for i in range(NENV):
            lanes.append({"lane":i,"survived":bool(not done[i]),"ttf":int(ttf[i]),"reasons":reason[i],
              "max_ang_vel":float(max_ang[i]),"max_tilt_xy":float(max_tilt[i]),"max_joint_vel_norm":float(max_jv[i]),
              "max_action_norm":float(max_anorm[i]),"max_action_rate":float(max_arate[i]),"min_action_headroom":float(min_head[i]),
              "action_sat_fraction":float(sat_steps[i]/H),"u40_raw_sat_fraction":float(raw40_sat[i]/H),"u40_raw_absmax":float(raw40_absmax[i]),
              "trace":traces[i]})
        return {"survival":float(np.mean([x["survived"] for x in lanes])),
                "fail_count":int(sum(not x["survived"] for x in lanes)),
                "mean_ttf":float(np.mean([x["ttf"] for x in lanes if not x["survived"]])) if any(not x["survived"] for x in lanes) else H,
                "lanes":lanes}
    
    def main():
        from isaaclab.app import AppLauncher
        sv=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv;env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);env.reset(seed=0)
            m20=load_wide(CK20);m30=load_wide(CK30)
            sd=np.load(SUP);Xall=torch.tensor(sd["obs"],device="cuda");origin=sd["origin"];phase=sd["phase"]
            rng=np.random.default_rng(2609252801);ids=[]
            for oi in range(5):
                for pi in range(3):
                    z=np.flatnonzero((origin==oi)&(phase==pi));ids.extend(rng.choice(z,size=20,replace=False).tolist())
            X=Xall[np.asarray(ids)]
            rep={"schema":"authority_overexpansion_causal_audit_v1","lambdas":list(LAMBDAS),"arms":{}}
            for arm,path in CK40.items():
                m40=load40(arm,path);arep={"authority":{},"fresh":{}}
                for lam in LAMBDAS:
                    pol=FunctionalInterp(m30,m40,lam)
                    arep["authority"][str(lam)]=authority_metrics(pol,m20,X)
                    rows=[]
                    for li,lab in enumerate(ORDER):
                        for si in range(4):
                            seed=9700000+li*1000+si*113
                            q=rollout(env,pol,m40,lab,seed);q.update({"preference":lab,"suite":si,"seed":seed});rows.append(q)
                    arep["fresh"][str(lam)]=rows
                    summ={"min_survival":min(r["survival"] for r in rows),"failed_lanes":sum(r["fail_count"] for r in rows),
                          "failed_cases":sum(r["fail_count"]>0 for r in rows),
                          "mean_failed_ttf":float(np.mean([r["mean_ttf"] for r in rows if r["fail_count"]])) if any(r["fail_count"] for r in rows) else H}
                    arep["fresh"][str(lam)+"_summary"]=summ
                    print("RESULT",arm,lam,json.dumps({"authority":arep["authority"][str(lam)],"fresh":summ}),flush=True)
                rep["arms"][arm]=arep
            rep["interpolation_space"]="pre_tanh_preference_response"
            (OUT/"overexpansion_causal_audit.json").write_text(json.dumps(rep,indent=2)+"\n")
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_authority_isolated_policy_output_neighborhood_audit():
    """Run former authority_isolated_policy_output_neighborhood_audit.py stage."""
    from pathlib import Path
    import json,sys,math,torch,numpy as np
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"scripts/rl")]
    from talon_rl.models.authority.isolated import AuthorityIsolatedActorCritic
    from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
    CK={
    "u50":ROOT/"runs/authority_isolated_h1-2026-09-25/model_50.pt",
    "u75":ROOT/"runs/authority_isolated_h1-2026-09-25/model_75.pt",
    "repair":ROOT/"runs/authority_isolated_projected_tail_descent-2026-09-25/model_10.pt"}
    OUT=ROOT/"runs/authority_isolated_policy_output_neighborhood_audit-2026-09-25";OUT.mkdir(parents=True,exist_ok=True)
    SEEDS={"suite2":840003,"suite3":840004};PREFS={"O":[.1,.1,.7,.1],"S":[.1,.1,.1,.7],"C":[.25]*4}
    NENV=8;H=32;PERT_END=15;EPS=[.01,.02,.04,.08]
    SPATIAL={
    "global":[1]*12,
    "left_right":[1,-1,1,-1,1,-1,1,-1,1,-1,1,-1],
    "front_rear":[1,1,-1,-1,1,1,-1,-1,1,1,-1,-1],
    "diagonal":[1,-1,-1,1,1,-1,-1,1,1,-1,-1,1],
    "hip_vs_distal":[1,1,1,1,-.5,-.5,-.5,-.5,-.5,-.5,-.5,-.5],
    "hips":[1,1,1,1,0,0,0,0,0,0,0,0],
    "thighs":[0,0,0,0,1,1,1,1,0,0,0,0],
    "calves":[0,0,0,0,0,0,0,0,1,1,1,1]}
    TEMP={
    "const":lambda t:1.0,
    "ramp":lambda t:(t/PERT_END)*2-1,
    "half_sine":lambda t:math.sin(math.pi*t/PERT_END),
    "full_sine":lambda t:math.sin(2*math.pi*t/PERT_END),
    "two_lobe":lambda t:math.sin(2*math.pi*t/PERT_END)*math.sin(math.pi*t/PERT_END)}
    RANDOM_N=8
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def get_z(m,obs,w):return m._actor_mean_with_preference(obs,w)
    def scaled_delta(z,d,eps):
        # per-env binary search scalar so realized RMS action deviation approximately eps
        d=d/z.new_tensor(torch.linalg.vector_norm(d,dim=1,keepdim=True)).clamp_min(1e-8)
        lo=torch.zeros((len(z),1),device=z.device);hi=torch.full_like(lo,8.0)
        a0=torch.tanh(z)
        for _ in range(12):
            mid=(lo+hi)/2
            da=torch.tanh(z+mid*d)-a0
            rms=torch.sqrt((da*da).mean(1,keepdim=True))
            lo=torch.where(rms<eps,mid,lo);hi=torch.where(rms>=eps,mid,hi)
        delta=lo*d;ap=torch.tanh(z+delta)
        realized=torch.sqrt(((ap-a0)**2).mean(1))
        return ap,realized
    def terminate_terms(env):
        tm=env.unwrapped.termination_manager;out={}
        for n in tm.active_terms:
            try:out[n]=tm.get_term(n).detach().cpu().numpy().astype(bool)
            except:pass
        return out
    def rollout(env,m,wv,seed,spatial=None,temp=None,eps=0.,random_seq=None):
        obs,_=env.reset(seed=seed);obs=ot(obs).cuda();w=torch.tensor(wv,device="cuda").repeat(NENV,1)
        fail=np.full(NENV,-1,int);reasons=[[] for _ in range(NENV)];realized=[]
        sb=None
        if spatial is not None:
            sb=torch.tensor(SPATIAL[spatial],device="cuda",dtype=torch.float32)[None,:].repeat(NENV,1)
        for t in range(H):
            with torch.no_grad():
                z=get_z(m,obs,w)
                if t<=PERT_END and (spatial is not None or random_seq is not None):
                    if random_seq is not None:
                        d=torch.tensor(random_seq[t],device="cuda",dtype=torch.float32)[None,:].repeat(NENV,1)
                    else:
                        d=sb*float(TEMP[temp](t))
                        # avoid exact-zero direction at temporal nodes
                        if torch.linalg.vector_norm(d[0])<1e-8:d=sb*1e-3
                    a,rr=scaled_delta(z,d,eps);realized.append(float(rr.mean().cpu()))
                else:a=torch.tanh(z)
            nxt,_,te,tr,_=env.step(a);dd=(te|tr).detach().cpu().numpy().astype(bool);terms=terminate_terms(env)
            for i in range(NENV):
                if dd[i] and fail[i]<0:
                    fail[i]=t;reasons[i]=[n for n,v in terms.items() if v[i]]
            obs=ot(nxt).cuda()
        delayed=[i for i in range(NENV) if fail[i]>=4 and "base_contact" in reasons[i]]
        post=[i for i in range(NENV) if fail[i]>PERT_END and "base_contact" in reasons[i]]
        return {"fail_t":fail.tolist(),"reasons":reasons,"delayed_lanes":delayed,"post_lanes":post,
                "realized_eps_mean":float(np.mean(realized)) if realized else 0.}
    def main():
        from isaaclab.app import AppLauncher
        sv=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv;env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda();ad=12
            models={}
            for lab,path in CK.items():
                cls=AuthorityIsolatedWideCritic if lab=="repair" else AuthorityIsolatedActorCritic
                m=cls(o.shape[-1],ad).cuda();m.load_state_dict(torch.load(path,map_location="cuda",weights_only=False)["model"]);m.eval();models[lab]=m
            rng=np.random.default_rng(260926)
            random_bank=[]
            for k in range(RANDOM_N):
                knots=rng.normal(size=(5,12))
                seq=[]
                for t in range(PERT_END+1):
                    u=t/PERT_END*4;j=min(3,int(math.floor(u)));a=u-j
                    x=(1-a)*knots[j]+a*knots[j+1];seq.append(x/np.maximum(np.linalg.norm(x),1e-12))
                random_bank.append(np.array(seq,dtype=np.float32))
            rep={"schema":"policy_output_neighborhood_audit_v1","eps":EPS,"baseline":{},"results":{}}
            # baselines
            for mlab,m in models.items():
                rep["baseline"][mlab]={}
                for suite,seed in SEEDS.items():
                    rep["baseline"][mlab][suite]={}
                    for pref,wv in PREFS.items():
                        q=rollout(env,m,wv,seed);rep["baseline"][mlab][suite][pref]=q
                        print("BASE",mlab,suite,pref,q["fail_t"],flush=True)
            # common survivor support
            common={}
            for suite in SEEDS:
                common[suite]={}
                for pref in PREFS:
                    lanes=[]
                    for i in range(NENV):
                        if all(rep["baseline"][ml][suite][pref]["fail_t"][i]<0 for ml in models):lanes.append(i)
                    common[suite][pref]=lanes
            rep["common_survivors"]=common
            for eps in EPS:
                ek=f"{eps:.2f}";rep["results"][ek]={}
                for mlab,m in models.items():
                    rep["results"][ek][mlab]={}
                    for suite,seed in SEEDS.items():
                        rep["results"][ek][mlab][suite]={}
                        for pref,wv in PREFS.items():
                            struct={}
                            for sp in SPATIAL:
                                for tp in TEMP:
                                    key=sp+"__"+tp;struct[key]=rollout(env,m,wv,seed,sp,tp,eps)
                            rnd=[rollout(env,m,wv,seed,eps=eps,random_seq=s) for s in random_bank]
                            lanes=common[suite][pref]
                            def rate(q,field):
                                return sum(i in q[field] for i in lanes)/len(lanes) if lanes else 0.
                            best=max(struct,key=lambda k:rate(struct[k],"delayed_lanes"))
                            rec={"best_mode":best,
                                 "best_delayed_rate":rate(struct[best],"delayed_lanes"),
                                 "best_post_rate":rate(struct[best],"post_lanes"),
                                 "best_fail_t":struct[best]["fail_t"],
                                 "best_realized_eps":struct[best]["realized_eps_mean"],
                                 "random_delayed_mean":float(np.mean([rate(x,"delayed_lanes") for x in rnd])),
                                 "random_delayed_max":float(np.max([rate(x,"delayed_lanes") for x in rnd])),
                                 "random_post_mean":float(np.mean([rate(x,"post_lanes") for x in rnd]))}
                            rep["results"][ek][mlab][suite][pref]=rec
                            print("RES",ek,mlab,suite,pref,json.dumps(rec),flush=True)
            # aggregate ordering per eps
            summary={}
            for ek in rep["results"]:
                s={}
                for mlab in models:
                    vals=[];rnd=[]
                    for suite in SEEDS:
                        for pref in PREFS:
                            q=rep["results"][ek][mlab][suite][pref];vals.append(q["best_delayed_rate"]);rnd.append(q["random_delayed_mean"])
                    s[mlab]={"adv_mean":float(np.mean(vals)),"random_mean":float(np.mean(rnd))}
                ordering=s["u50"]["adv_mean"] < s["repair"]["adv_mean"] < s["u75"]["adv_mean"]
                recurrence=sum(rep["results"][ek]["u75"][suite][pref]["best_delayed_rate"]>0 for suite in SEEDS for pref in PREFS)
                stronger=s["u75"]["adv_mean"]>s["u75"]["random_mean"]+1e-9
                summary[ek]={"actors":s,"desired_ordering":ordering,"u75_recurrent_cells":recurrence,"u75_adv_gt_random":stronger,
                             "pass":bool(ordering and recurrence>=3 and stronger)}
            rep["summary"]=summary;rep["gate_pass"]=any(x["pass"] for x in summary.values())
            (OUT/"policy_output_neighborhood_audit.json").write_text(json.dumps(rep,indent=2)+"\n")
            print("SUMMARY",json.dumps(summary,indent=2),flush=True);print("GATE_PASS",rep["gate_pass"],flush=True)
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_authority_isolated_policy_output_neighborhood_staged():
    """Run former authority_isolated_policy_output_neighborhood_staged.py stage."""
    from pathlib import Path
    import json,sys,math,torch,numpy as np
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"scripts/rl")]
    from talon_rl.models.authority.isolated import AuthorityIsolatedActorCritic
    from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
    CK={"u50":ROOT/"runs/authority_isolated_h1-2026-09-25/model_50.pt","u75":ROOT/"runs/authority_isolated_h1-2026-09-25/model_75.pt","repair":ROOT/"runs/authority_isolated_projected_tail_descent-2026-09-25/model_10.pt"}
    OUT=ROOT/"runs/authority_isolated_policy_output_neighborhood_audit-2026-09-25";OUT.mkdir(parents=True,exist_ok=True)
    SEEDS={"suite2":840003,"suite3":840004};PREFS={"O":[.1,.1,.7,.1],"S":[.1,.1,.1,.7],"C":[.25]*4}
    NENV=8;H=24;PERT_END=15;EPS=[.01,.02,.04,.08];RANDOM_N=4
    SPATIAL={"global":[1]*12,"left_right":[1,-1,1,-1,1,-1,1,-1,1,-1,1,-1],"front_rear":[1,1,-1,-1,1,1,-1,-1,1,1,-1,-1],"diagonal":[1,-1,-1,1,1,-1,-1,1,1,-1,-1,1],"hip_vs_distal":[1,1,1,1,-.5,-.5,-.5,-.5,-.5,-.5,-.5,-.5],"hips":[1,1,1,1,0,0,0,0,0,0,0,0],"thighs":[0,0,0,0,1,1,1,1,0,0,0,0],"calves":[0,0,0,0,0,0,0,0,1,1,1,1]}
    TEMP={"const":lambda t:1.0,"ramp":lambda t:(t/PERT_END)*2-1,"half_sine":lambda t:math.sin(math.pi*t/PERT_END),"full_sine":lambda t:math.sin(2*math.pi*t/PERT_END),"two_lobe":lambda t:math.sin(2*math.pi*t/PERT_END)*math.sin(math.pi*t/PERT_END)}
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def scaled_delta(z,d,eps):
        n=torch.linalg.vector_norm(d,dim=1,keepdim=True).clamp_min(1e-8);d=d/n;a0=torch.tanh(z)
        lo=torch.zeros((len(z),1),device=z.device);hi=torch.full_like(lo,8.0)
        for _ in range(10):
            mid=(lo+hi)/2;da=torch.tanh(z+mid*d)-a0;rms=torch.sqrt((da*da).mean(1,keepdim=True))
            lo=torch.where(rms<eps,mid,lo);hi=torch.where(rms>=eps,mid,hi)
        a=torch.tanh(z+lo*d);rr=torch.sqrt(((a-a0)**2).mean(1));return a,rr
    def terms(env):
        tm=env.unwrapped.termination_manager;out={}
        for n in tm.active_terms:
            try:out[n]=tm.get_term(n).detach().cpu().numpy().astype(bool)
            except:pass
        return out
    def rollout(env,m,wv,seed,eps=0.,mode=None,random_seq=None):
        obs,_=env.reset(seed=seed);obs=ot(obs).cuda();w=torch.tensor(wv,device="cuda").repeat(NENV,1)
        fail=np.full(NENV,-1,int);why=[[] for _ in range(NENV)];rr=[]
        if mode:
            sp,tp=mode.split("__");sb=torch.tensor(SPATIAL[sp],device="cuda",dtype=torch.float32)[None,:].repeat(NENV,1)
        for t in range(H):
            with torch.no_grad():
                z=m._actor_mean_with_preference(obs,w)
                if t<=PERT_END and (mode or random_seq is not None):
                    if random_seq is not None:d=torch.tensor(random_seq[t],device="cuda",dtype=torch.float32)[None,:].repeat(NENV,1)
                    else:
                        amp=float(TEMP[tp](t));d=sb*(amp if abs(amp)>1e-4 else 1e-3)
                    a,r=scaled_delta(z,d,eps);rr.append(float(r.mean().cpu()))
                else:a=torch.tanh(z)
            nxt,_,te,tr,_=env.step(a);dd=(te|tr).detach().cpu().numpy().astype(bool);tm=terms(env)
            for i in range(NENV):
                if dd[i] and fail[i]<0:fail[i]=t;why[i]=[n for n,v in tm.items() if v[i]]
            obs=ot(nxt).cuda()
        delayed=[i for i in range(NENV) if fail[i]>=4 and "base_contact" in why[i]]
        return {"fail_t":fail.tolist(),"delayed":delayed,"realized":float(np.mean(rr)) if rr else 0.}
    def main():
        from isaaclab.app import AppLauncher
        sv=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv;env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda()
            models={}
            for lab,path in CK.items():
                cls=AuthorityIsolatedWideCritic if lab=="repair" else AuthorityIsolatedActorCritic;m=cls(o.shape[-1],12).cuda();m.load_state_dict(torch.load(path,map_location="cuda",weights_only=False)["model"]);m.eval();models[lab]=m
            baseline={}
            for ml,m in models.items():
                baseline[ml]={}
                for suite,seed in SEEDS.items():
                    baseline[ml][suite]={}
                    for pref,wv in PREFS.items():baseline[ml][suite][pref]=rollout(env,m,wv,seed);print("BASE",ml,suite,pref,baseline[ml][suite][pref]["fail_t"],flush=True)
            common={suite:{pref:[i for i in range(NENV) if all(baseline[ml][suite][pref]["fail_t"][i]<0 for ml in models)] for pref in PREFS} for suite in SEEDS}
            modes=[s+"__"+t for s in SPATIAL for t in TEMP]
            rng=np.random.default_rng(260926);random_bank=[]
            for k in range(RANDOM_N):
                knots=rng.normal(size=(5,12));seq=[]
                for t in range(PERT_END+1):
                    u=t/PERT_END*4;j=min(3,int(math.floor(u)));a=u-j;x=(1-a)*knots[j]+a*knots[j+1];seq.append(x/np.maximum(np.linalg.norm(x),1e-12))
                random_bank.append(np.array(seq,dtype=np.float32))
            rep={"schema":"policy_output_neighborhood_staged_v1","baseline":baseline,"common_survivors":common,"eps":{}}
            for eps in EPS:
                ek=f"{eps:.2f}"
                # development search: u75 / suite3 / aggregate O,S,C only
                mode_scores={}
                for mode in modes:
                    vals=[]
                    for pref,wv in PREFS.items():
                        q=rollout(env,models["u75"],wv,SEEDS["suite3"],eps,mode=mode);lanes=common["suite3"][pref]
                        vals.append(sum(i in q["delayed"] for i in lanes)/len(lanes) if lanes else 0.)
                    mode_scores[mode]=float(np.mean(vals))
                best=max(modes,key=lambda m:mode_scores[m]);print("DEV",ek,best,mode_scores[best],flush=True)
                val={}
                for ml,m in models.items():
                    val[ml]={}
                    for suite,seed in SEEDS.items():
                        val[ml][suite]={}
                        for pref,wv in PREFS.items():
                            q=rollout(env,m,wv,seed,eps,mode=best);lanes=common[suite][pref];rate=sum(i in q["delayed"] for i in lanes)/len(lanes) if lanes else 0.
                            val[ml][suite][pref]={"rate":rate,"fail_t":q["fail_t"],"realized":q["realized"]}
                            print("VAL",ek,ml,suite,pref,rate,q["fail_t"],flush=True)
                rnd={}
                # random control only on u75, same common support
                for suite,seed in SEEDS.items():
                    rnd[suite]={}
                    for pref,wv in PREFS.items():
                        lanes=common[suite][pref];rates=[]
                        for seq in random_bank:
                            q=rollout(env,models["u75"],wv,seed,eps,random_seq=seq);rates.append(sum(i in q["delayed"] for i in lanes)/len(lanes) if lanes else 0.)
                        rnd[suite][pref]={"mean":float(np.mean(rates)),"max":float(np.max(rates))}
                agg={}
                for ml in models:agg[ml]=float(np.mean([val[ml][s][p]["rate"] for s in SEEDS for p in PREFS]))
                rndmean=float(np.mean([rnd[s][p]["mean"] for s in SEEDS for p in PREFS]))
                ordering=agg["u50"]<agg["repair"]<agg["u75"];recur=sum(val["u75"][s][p]["rate"]>0 for s in SEEDS for p in PREFS)
                passed=bool(ordering and recur>=3 and agg["u75"]>rndmean+1e-9)
                rep["eps"][ek]={"best_mode":best,"development_score":mode_scores[best],"validation":val,"u75_random":rnd,"aggregate":agg,"random_mean":rndmean,"desired_ordering":ordering,"u75_recurrent_cells":recur,"pass":passed}
                print("EPS_SUM",ek,json.dumps(rep["eps"][ek]["aggregate"]),rndmean,ordering,recur,passed,flush=True)
            rep["gate_pass"]=any(v["pass"] for v in rep["eps"].values())
            (OUT/"policy_output_neighborhood_staged.json").write_text(json.dumps(rep,indent=2)+"\n");print("GATE",rep["gate_pass"],flush=True)
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_authority_isolated_precontact_compatibility_audit():
    """Run former authority_isolated_precontact_compatibility_audit.py stage."""
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
    from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
    
    CK30=ROOT/"runs/authority_isolated_simplex_edge_retain-2026-09-25/model_30.pt"
    CK40={
     "narrow":ROOT/"runs/authority_isolated_ai_c2_formal_edge_narrow-2026-09-25/model_10.pt",
     "wide":ROOT/"runs/authority_isolated_ai_c2_formal_edge_wide-2026-09-25/model_10.pt",
    }
    OUT=ROOT/"runs/authority_isolated_precontact_compatibility_audit-2026-09-25";OUT.mkdir(parents=True,exist_ok=True)
    PREFS=h1.PREFS;NENV=8;H=64;WINDOW=10
    
    CASES={
     "narrow":[("T",9700226,6)],
     "wide":[("T",9700226,6),("A",9701226,5),("S",9703226,3),("C",9704339,0)],
    }
    
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    
    def load_state(path):
        return torch.load(path,map_location="cuda",weights_only=False)["model"]
    
    def actor_state_for_cls(src,template):
        out={k:v.clone() for k,v in template.items()}
        for k in out:
            if (k.startswith("actor_") or k.startswith("family_") or k=="log_std") and k in src and out[k].shape==src[k].shape:
                out[k]=src[k].clone()
        return out
    
    def hybrid(s30,s40,common_from,family_from):
        out={k:v.clone() for k,v in s30.items()}
        cs=s30 if common_from==30 else s40
        fs=s30 if family_from==30 else s40
        for k in out:
            if k.startswith("actor_body") or k.startswith("actor_mean"):
                out[k]=cs[k].clone()
            elif k.startswith("family_"):
                out[k]=fs[k].clone()
            elif k=="log_std":
                out[k]=s40[k].clone()
        return out
    
    def build(cls,state):
        m=cls(48,12).cuda();m.load_state_dict(state);m.eval();return m
    
    def rollout_case(env,m,lab,seed,lane):
        w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1)
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        hist=[];ttf=H;reason=[]
        with torch.no_grad():
            for t in range(H):
                z=m._actor_mean_with_preference(cur,w)
                a=torch.tanh(z)*m.ACTION_CLIP
                hist.append({"t":t,"obs":cur[lane].detach().cpu(),"z":z[lane].detach().cpu(),"a":a[lane].detach().cpu()})
                nxt,_,te,tr,_=env.step(a);dd=(te|tr).cpu().numpy();tm=env.unwrapped.termination_manager
                if dd[lane] and ttf==H:
                    ttf=t+1;reason=[n for n in tm.active_terms if bool(tm.get_term(n)[lane].detach().cpu())]
                cur=ot(nxt).cuda()
                if ttf<H and t+1>=ttf: break
        return hist,ttf,reason
    
    def eval_matrix(models,states,lab):
        x=torch.stack([q["obs"] for q in states]).cuda()
        w=torch.tensor(PREFS[lab],device="cuda").repeat(len(x),1)
        out={}
        with torch.no_grad():
            for name,m in models.items():
                z=m._actor_mean_with_preference(x,w)
                a=torch.tanh(z)*m.ACTION_CLIP
                out[name]={"z":z.cpu().numpy(),"a":a.cpu().numpy()}
        z00=out["C30F30"]["z"];z10=out["C40F30"]["z"];z01=out["C30F40"]["z"];z11=out["C40F40"]["z"]
        a00=out["C30F30"]["a"];a10=out["C40F30"]["a"];a01=out["C30F40"]["a"];a11=out["C40F40"]["a"]
        iz=z11-z10-z01+z00;ia=a11-a10-a01+a00
        rows=[]
        for k,q in enumerate(states):
            rows.append({
              "t":int(q["t"]),
              "z_norms":{n:float(np.linalg.norm(out[n]["z"][k])) for n in out},
              "a_norms":{n:float(np.linalg.norm(out[n]["a"][k])) for n in out},
              "interaction_z_norm":float(np.linalg.norm(iz[k])),
              "interaction_a_norm":float(np.linalg.norm(ia[k])),
              "interaction_z":iz[k].tolist(),
              "interaction_a":ia[k].tolist(),
              "z_synergy_sign":np.sign(iz[k]).astype(int).tolist(),
              "a_synergy_sign":np.sign(ia[k]).astype(int).tolist(),
              "single_common_z_delta":(z10[k]-z00[k]).tolist(),
              "single_family_z_delta":(z01[k]-z00[k]).tolist(),
              "full_z_delta":(z11[k]-z00[k]).tolist(),
              "single_common_a_delta":(a10[k]-a00[k]).tolist(),
              "single_family_a_delta":(a01[k]-a00[k]).tolist(),
              "full_a_delta":(a11[k]-a00[k]).tolist(),
            })
        return rows
    
    def main():
        from isaaclab.app import AppLauncher
        sv=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv;env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);env.reset(seed=0)
            s30raw=load_state(CK30)
            rep={"schema":"precontact_common_family_compatibility_v1","cases":{}}
            aggregate={"z_abs_sum":np.zeros(12),"a_abs_sum":np.zeros(12),"z_sign_sum":np.zeros(12),"a_sign_sum":np.zeros(12),"count":0}
            for arm,path in CK40.items():
                cls=AuthorityIsolatedActorCritic if arm=="narrow" else AuthorityIsolatedWideCritic
                template=cls(48,12).cuda().state_dict()
                s30=actor_state_for_cls(s30raw,template);s40=actor_state_for_cls(load_state(path),template)
                states={
                  "C30F30":s30,
                  "C40F30":hybrid(s30,s40,40,30),
                  "C30F40":hybrid(s30,s40,30,40),
                  "C40F40":s40,
                }
                models={k:build(cls,v) for k,v in states.items()}
                for lab,seed,lane in CASES[arm]:
                    hist,ttf,reason=rollout_case(env,models["C40F40"],lab,seed,lane)
                    start=max(0,len(hist)-WINDOW);sel=hist[start:]
                    rows=eval_matrix(models,sel,lab)
                    key=f"{arm}:{lab}:{seed}:lane{lane}"
                    rep["cases"][key]={"arm":arm,"preference":lab,"seed":seed,"lane":lane,"ttf":ttf,"reason":reason,"rows":rows}
                    for rr in rows:
                        z=np.array(rr["interaction_z"]);a=np.array(rr["interaction_a"])
                        aggregate["z_abs_sum"]+=np.abs(z);aggregate["a_abs_sum"]+=np.abs(a)
                        aggregate["z_sign_sum"]+=np.sign(z);aggregate["a_sign_sum"]+=np.sign(a);aggregate["count"]+=1
                    print("CASE",key,"ttf",ttf,"rows",len(rows),
                          "mean_Iz",np.mean([r["interaction_z_norm"] for r in rows]),
                          "mean_Ia",np.mean([r["interaction_a_norm"] for r in rows]),flush=True)
            c=max(aggregate["count"],1)
            rep["aggregate"]={
              "mean_abs_interaction_z_per_coord":(aggregate["z_abs_sum"]/c).tolist(),
              "mean_abs_interaction_a_per_coord":(aggregate["a_abs_sum"]/c).tolist(),
              "mean_sign_interaction_z_per_coord":(aggregate["z_sign_sum"]/c).tolist(),
              "mean_sign_interaction_a_per_coord":(aggregate["a_sign_sum"]/c).tolist(),
              "samples":aggregate["count"],
            }
            (OUT/"precontact_compatibility_audit.json").write_text(json.dumps(rep,indent=2)+"\n")
            print("AGG",json.dumps(rep["aggregate"]),flush=True)
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_authority_isolated_s2O_l7_rescue():
    """Run former authority_isolated_s2O_l7_rescue.py stage."""
    from pathlib import Path
    import json,sys,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    from talon_rl.models.authority.isolated import AuthorityIsolatedActorCritic
    from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
    U75=ROOT/"runs/authority_isolated_h1-2026-09-25/model_75.pt"
    REP=ROOT/"runs/authority_isolated_projected_tail_descent-2026-09-25/model_10.pt"
    OUT=ROOT/"runs/authority_isolated_s2O_l7_rescue-2026-09-25";OUT.mkdir(parents=True,exist_ok=True)
    SEED=840003;LAB=[.1,.1,.7,.1];LANE=7;NENV=8;H=24
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def run(env,base,donor,idx,label):
        w=torch.tensor(LAB,device="cuda").repeat(NENV,1);obs,_=env.reset(seed=SEED);obs=ot(obs).cuda();ff=None
        for t in range(H):
            with torch.no_grad():
                ab=base.act_inference_with_preference(obs,w);ad=donor.act_inference_with_preference(obs,w);a=ab.clone()
                if idx:a[:,idx]=ad[:,idx]
            nxt,_,te,tr,_=env.step(a);dd=(te|tr).cpu().numpy().astype(bool)
            if dd[LANE] and ff is None:ff=t
            obs=ot(nxt).cuda()
        return {"label":label,"indices":idx,"survived":ff is None,"first_fail":ff}
    def main():
        from isaaclab.app import AppLauncher
        sv=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv;env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda()
            u=AuthorityIsolatedActorCritic(o.shape[-1],12).cuda();u.load_state_dict(torch.load(U75,map_location="cuda",weights_only=False)["model"]);u.eval()
            r=AuthorityIsolatedWideCritic(o.shape[-1],12).cuda();r.load_state_dict(torch.load(REP,map_location="cuda",weights_only=False)["model"]);r.eval()
            names=list(env.unwrapped.scene["robot"].data.joint_names)
            tests=[("none",[]),("all",list(range(12))),("FL_hip",[0]),("FL_calf",[8]),("RR_hip",[3]),("FLhip+FLcalf",[0,8]),("front_left",[0,4,8]),("front",[0,1,4,5,8,9])]
            rows=[]
            for lab,idx in tests:
                q=run(env,r,u,idx,lab);rows.append(q);print(q,flush=True)
            (OUT/"rescue.json").write_text(json.dumps({"joint_names":names,"rows":rows},indent=2)+"\n")
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_authority_isolated_simplex_edge_endpoint_audit():
    """Run former authority_isolated_simplex_edge_endpoint_audit.py stage."""
    """Simplex-edge endpoint authority for the functional-deltaa control versus the
    simplex-edge retain arm, both wide-critic, at update 30."""
    import sys
    from pathlib import Path
    
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    
    from rl.experiments.common.utilities.simplex_edge_endpoint import SimplexEdgeEndpointAudit
    
    
    class SimplexEdgeRetainAudit(SimplexEdgeEndpointAudit):
        """Functional-deltaa control versus the simplex-edge retain arm at update 30."""
    
        run = "authority_isolated_simplex_edge_endpoint_audit-2026-09-25"
        narrow = ("control", "authority_isolated_functional_deltaa_control-2026-09-25/model_30.pt")
        narrow_kind = "wide"
        wide = ("edge", "authority_isolated_simplex_edge_retain-2026-09-25/model_30.pt")
    
    
    if True:
        SimplexEdgeRetainAudit.main()

def run_authority_isolated_simplex_edge_retention_train():
    """Run former authority_isolated_simplex_edge_retention_train.py stage."""
    
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from pathlib import Path
    import argparse,json,sys
    import numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"scripts/rl")]
    import rl.experiments.common.utilities.authority_isolated_h1_screen as h1
    from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
    
    SOURCE_STATE=ROOT/"runs/authority_isolated_coverage2_control-2026-09-25/resume_state.pt"
    SOURCE_MODEL=ROOT/"runs/authority_isolated_coverage2_control-2026-09-25/model_20.pt"
    SUPPORT=ROOT/"runs/authority_isolated_u20_authority_support-2026-09-25/u20_authority_support.npz"
    TAU_PATH=ROOT/"runs/authority_isolated_tail_budget_calibration-2026-09-25/tail_budget_probe.json"
    PROBE=ROOT/"runs/update_functional_effect_audit-2026-09-24/fixed_probe_states.npz"
    NENV=h1.NENV;H=h1.H;KAPPA=.05;RHO=.25;BETA0=2.497041993384243;EPS=1e-12
    HEAVY=("T","A","O","S")
    
    def tail_metrics(m,obs,w,tau):
        z=m._actor_mean_with_preference(obs,w);ex=torch.relu(z.abs()-tau)
        return ((ex/(tau+1e-6))**2).mean(),(z.abs()>tau).float().mean()
    
    def audit(m,probe,tau,out,tag,snaps,arm):
        m.eval();sens=h1.sensitivity(m,probe)
        P=torch.tensor(np.load(PROBE)["obs"],device="cuda")
        wm=torch.tensor(h1.PREFS["C"],device="cuda").repeat(len(P),1)
        with torch.no_grad():tl,tf=tail_metrics(m,P,wm,tau)
        snaps[str(tag)]={"sensitivity":sens,"probe_tail_loss":float(tl.cpu()),"probe_tail_fraction":float(tf.cpu())}
        torch.save({"model":m.state_dict(),"update":int(tag),"arm":arm},out/f"model_{tag}.pt");m.train()
    
    def save_state(path,m,opt,adaptive_pools,anchor_specs,rows,snaps,update):
        torch.save({"model":m.state_dict(),"optimizer":opt.state_dict(),"adaptive_pools":adaptive_pools,
          "anchor_specs":anchor_specs,"rows":rows,"snaps":snaps,"update":update,
          "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all(),"numpy_rng":np.random.get_state()},path)
    
    def balanced_indices(origin,phase,update):
        rng=np.random.default_rng(2609252401+update);ids=[]
        for oi in range(5):
            for pi in range(3):
                pool=np.flatnonzero((origin==oi)&(phase==pi))
                ids.extend(rng.choice(pool,size=8,replace=False).tolist())
        return np.asarray(ids,np.int64)
    
    ALL_PREFS=("T","A","O","S","C")
    EDGE_PAIRS=tuple((ALL_PREFS[i],ALL_PREFS[j]) for i in range(len(ALL_PREFS)) for j in range(i+1,len(ALL_PREFS)))
    GAMMA=.90
    
    def edge_floor_loss(m,ref,obs):
        n=len(obs)
        acts={};refs={}
        for lab in ALL_PREFS:
            w=torch.tensor(h1.PREFS[lab],device="cuda").repeat(n,1)
            acts[lab]=m.act_inference_with_preference(obs,w)
            with torch.no_grad():
                refs[lab]=ref.act_inference_with_preference(obs,w)
        terms=[]
        for i,j in EDGE_PAIRS:
            d=torch.linalg.vector_norm(acts[i]-acts[j],dim=1)
            dr=torch.linalg.vector_norm(refs[i]-refs[j],dim=1)
            terms.append(torch.relu(GAMMA*dr-d).pow(2))
        return torch.stack(terms,dim=1).mean()
    
    def grad_norm(gs):
        return torch.sqrt(sum((g.detach()**2).sum() for g in gs if g is not None)+EPS)
    
    def main():
        ap=argparse.ArgumentParser();ap.add_argument("--arm",choices=["control","retain"],required=True)
        ap.add_argument("--seed",type=int,default=73001);ap.add_argument("--target-updates",type=int,default=30)
        args=ap.parse_args()
        out=ROOT/f"runs/authority_isolated_simplex_edge_{args.arm}-2026-09-25";out.mkdir(parents=True,exist_ok=True)
        state_path=out/"resume_state.pt";tau=torch.tensor(json.load(open(TAU_PATH))["tau"],device="cuda",dtype=torch.float32)
        sd=np.load(SUPPORT);ref_obs=torch.tensor(sd["obs"],device="cuda");origin=sd["origin"];phase=sd["phase"]
        torch.manual_seed(args.seed);np.random.seed(args.seed)
        from isaaclab.app import AppLauncher
        sv=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv;env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.foundations.four_objective import vector_gae,scalarized_late_weighted_ppo
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=args.seed
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=args.seed);o=h1.ot(o).cuda();probe=o.detach().clone()
            ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
            m=AuthorityIsolatedWideCritic(o.shape[-1],ad).cuda()
            ref=AuthorityIsolatedWideCritic(o.shape[-1],ad).cuda();ref.load_state_dict(torch.load(SOURCE_MODEL,map_location="cuda",weights_only=False)["model"]);ref.eval()
            for p in ref.parameters():p.requires_grad_(False)
            for n,p in m.named_parameters():
                if n.startswith("critic_body") or n.startswith("critic_head"):p.requires_grad_(False)
            actor_params=[p for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std" or n.startswith("family_")]
            opt=torch.optim.Adam(actor_params,lr=1e-3)
    
            if state_path.exists():
                st=torch.load(state_path,map_location="cpu",weights_only=False);m.load_state_dict(st["model"]);opt.load_state_dict(st["optimizer"])
                adaptive_pools=st["adaptive_pools"];anchor_specs=st["anchor_specs"];rows=st["rows"];snaps=st["snaps"];cur=int(st["update"])
                torch.set_rng_state(st["torch_rng"]);torch.cuda.set_rng_state_all(st["cuda_rng"]);np.random.set_state(st["numpy_rng"])
            else:
                st=torch.load(SOURCE_STATE,map_location="cpu",weights_only=False);m.load_state_dict(st["model"]);opt.load_state_dict(st["optimizer"])
                adaptive_pools=st["adaptive_pools"];anchor_specs=st["anchor_specs"];rows=[];snaps={"20":st["snaps"]["20"]};cur=20
                audit(m,probe,tau,out,20,snaps,args.arm);save_state(state_path,m,opt,adaptive_pools,anchor_specs,rows,snaps,cur)
    
            def current_anchor_units():
                q={"early":[],"late":[]}
                for k,seed in anchor_specs:
                    _,w=h1.pref_batch(k,torch.device("cuda"))
                    for u in h1.collect_support64(env,m,w,mgr,seed):q[u["phase"]].append(u)
                return q
    
            for uidx in range(cur+1,args.target_updates+1):
                _,w=h1.pref_batch(uidx,torch.device("cuda"))
                main=h1.collect_actor(env,m,w,mgr,args.seed+uidx*211,True)
                _,ws=h1.pref_batch(uidx+17,torch.device("cuda"));units=h1.collect_support64(env,m,ws,mgr,args.seed+200000+uidx*223)
                for u in units:
                    ph=u["phase"];adaptive_pools[ph].append(u)
                    if len(adaptive_pools[ph])>h1.POOL_MAX:adaptive_pools[ph].pop(0)
                h1.fit_expanded_current_policy(m,current_anchor_units(),adaptive_pools)
                with torch.no_grad():
                    vt=m.value_with_preference(main["obs"],main["w"]).reshape(H,NENV,4);nv=m.value_with_preference(main["next_obs"],w)
                    adv,_=vector_gae(main["rt"],vt,nv,main["dt"],lam=.95)
                lp=[]
                for stx in range(0,len(main["obs"]),NENV):
                    lp.append(m.logp_from_pre_tanh_with_preference(main["obs"][stx:stx+NENV],main["w"][stx:stx+NENV],main["u"][stx:stx+NENV]))
                ratio=torch.exp(torch.cat(lp)-main["old"].detach());ratio_err=float((ratio-1).abs().max().detach().cpu())
                if ratio_err>1e-4:raise RuntimeError(f"ratio invariant {ratio_err}")
                ppo=scalarized_late_weighted_ppo(ratio,adv.reshape(-1,4).detach(),main["w"])
                tail,tailfrac=tail_metrics(m,main["obs"],main["w"],tau)
                gp=torch.autograd.grad(ppo,actor_params,retain_graph=True,allow_unused=True)
                gt=torch.autograd.grad(tail,actor_params,retain_graph=(args.arm=="retain"),allow_unused=True)
                gp2=sum((g.detach()**2).sum() for g in gp if g is not None);gt2=sum((g.detach()**2).sum() for g in gt if g is not None)
                gpn=torch.sqrt(gp2+EPS);gtn=torch.sqrt(gt2+EPS)
                dot=sum((a.detach()*b.detach()).sum() for a,b in zip(gp,gt) if a is not None and b is not None)
                coeff=torch.minimum(dot/(gt2+EPS),torch.zeros_like(dot));base=[];proj2=torch.zeros((),device="cuda")
                for p,a,b in zip(actor_params,gp,gt):
                    aa=torch.zeros_like(p) if a is None else a;bb=torch.zeros_like(p) if b is None else b
                    pp=aa-coeff*bb;base.append([pp,bb]);proj2+=(pp.detach()**2).sum()
                projn=torch.sqrt(proj2+EPS);tail_scale=KAPPA*projn/(gtn+EPS)
                gbase=[pp+tail_scale*bb for pp,bb in base];gbn=grad_norm(gbase)
    
                retain=torch.zeros((),device="cuda");gr=[None]*len(actor_params);alpha=0.;grn=torch.zeros((),device="cuda")
                if args.arm=="retain":
                    ids=balanced_indices(origin,phase,uidx);retain=edge_floor_loss(m,ref,ref_obs[ids])
                    gr=torch.autograd.grad(retain,actor_params,allow_unused=True);grn=grad_norm(gr)
                    alpha=min(BETA0,RHO*float(gbn.detach().cpu())/(float(grn.detach().cpu())+EPS))
                opt.zero_grad(set_to_none=True)
                for i,p in enumerate(actor_params):
                    gg=gbase[i]
                    if args.arm=="retain" and gr[i] is not None:gg=gg+alpha*gr[i]
                    p.grad=gg
                preclip=float(torch.nn.utils.clip_grad_norm_(actor_params,1.0).detach().cpu());opt.step()
                with torch.no_grad():m.log_std.clamp_(m.LOG_STD_MIN,m.LOG_STD_MAX)
                row={"update":uidx,"ppo_loss":float(ppo.detach().cpu()),"tail_loss":float(tail.detach().cpu()),"tail_fraction":float(tailfrac.detach().cpu()),
                     "retain_loss":float(retain.detach().cpu()),"retain_alpha":float(alpha),"base_grad_norm":float(gbn.cpu()),"retain_grad_norm":float(grn.cpu()),
                     "weighted_retain_over_base":float(alpha*float(grn.cpu())/(float(gbn.cpu())+EPS)) if args.arm=="retain" else 0.,
                     "ratio_maxerr":ratio_err,"termination_fraction":main["termination_fraction"],"grad_norm_preclip":preclip}
                rows.append(row);print("UPDATE",args.arm,uidx,json.dumps(row),flush=True)
                if uidx==args.target_updates:audit(m,probe,tau,out,uidx,snaps,args.arm)
                save_state(state_path,m,opt,adaptive_pools,anchor_specs,rows,snaps,uidx)
    
            s0=snaps["20"]["sensitivity"];sf=snaps[str(args.target_updates)]["sensitivity"]
            rep={"schema":"simplex_edge_durability_v1","arm":args.arm,"rho":RHO,"beta0":BETA0,"gamma":GAMMA,"updates":args.target_updates,
                 "summary":{"fixed_probe_pairwise_retention":sf["pairwise_action_distance"]["mean"]/(s0["pairwise_action_distance"]["mean"]+EPS),
                            "fixed_probe_tangent_retention":sf["tangent_jacobian_fro_mean"]/(s0["tangent_jacobian_fro_mean"]+EPS),
                            "max_ratio_error":max(r["ratio_maxerr"] for r in rows),"max_termination_fraction":max(r["termination_fraction"] for r in rows),
                            "mean_weighted_retain_over_base":float(np.mean([r["weighted_retain_over_base"] for r in rows]))},
                 "rows":rows,"snapshots":snaps}
            (out/"training_report.json").write_text(json.dumps(rep,indent=2)+"\n");print("FINAL",args.arm,json.dumps(rep["summary"]),flush=True)
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_authority_isolated_suite3O_hip_margin_audit():
    """Run former authority_isolated_suite3O_hip_margin_audit.py stage."""
    """How much of the u50 hip command does u75 need borrowed back to survive?
    
    On the one failing suite-3 orientation lane, replaces the u75 policy's hip
    coordinate with a blend toward u50's, by blend weight and by time window. The
    smallest blend and the narrowest window that still survives bound where the
    failure lives.
    """
    import json
    import sys
    from pathlib import Path
    
    import torch
    
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    
    from rl.core.diagnostics.isaac_audit import RUNS, IsaacAudit, obs_tensor
    
    RUN = "authority_isolated_h1-2026-09-25"
    SEED = 840004
    PREFERENCE = [.1, .1, .7, .1]
    LANE = 0
    HORIZON = 24
    JOINTS = [(0, "FL_hip"), (3, "RR_hip")]
    ALPHAS = [.1, .25, .5, .75, 1.0]
    WINDOWS = [(0, 5), (4, 8), (6, 10), (9, 13), (11, 15)]
    
    
    class Suite3OHipMarginAudit(IsaacAudit):
        """Hip-coordinate margin between u50 and u75 on the failing suite-3 lane."""
    
        run = "authority_isolated_suite3O_hip_margin_audit-2026-09-25"
        report = "hip_margin.json"
    
        def policy(self, name, obs_dim):
            from talon_rl.models.authority.isolated import AuthorityIsolatedActorCritic
    
            m = AuthorityIsolatedActorCritic(obs_dim, 12).cuda()
            m.load_state_dict(torch.load(RUNS / RUN / name, map_location="cuda",
                                         weights_only=False)["model"])
            m.eval()
            return m
    
        def trial(self, env, base, donor, j, alpha=1.0, window=None):
            """Roll `base`, replacing coordinate j with a blend toward `donor`."""
            w = torch.tensor(PREFERENCE, device="cuda").repeat(self.num_envs, 1)
            obs, _ = env.reset(seed=SEED)
            obs = obs_tensor(obs).cuda()
            first_fail = None
            for t in range(HORIZON):
                with torch.no_grad():
                    a = base.act_inference_with_preference(obs, w)
                    q = donor.act_inference_with_preference(obs, w)
                    if window is None or window[0] <= t <= window[1]:
                        a[:, j] = (1 - alpha) * a[:, j] + alpha * q[:, j]
                nxt, _, te, tr, _ = env.step(a)
                done = (te | tr).cpu().numpy().astype(bool)
                if done[LANE] and first_fail is None:
                    first_fail = t
                obs = obs_tensor(nxt).cuda()
            return {"survived": first_fail is None, "first_fail": first_fail}
    
        def rollout(self, env, obs):
            base = self.policy("model_75.pt", obs.shape[-1])
            donor = self.policy("model_50.pt", obs.shape[-1])
            rows = []
            for j, name in JOINTS:
                for alpha in ALPHAS:
                    q = self.trial(env, base, donor, j, alpha, None)
                    q.update(joint=name, alpha=alpha, window="all")
                    rows.append(q)
                    print(q, flush=True)
                for win in WINDOWS:
                    q = self.trial(env, base, donor, j, 1.0, win)
                    q.update(joint=name, alpha=1.0, window=f"{win[0]}-{win[1]}")
                    rows.append(q)
                    print(q, flush=True)
            rep = {"rows": rows}
            self.write(rep)
            return rep
    
    
    if True:
        Suite3OHipMarginAudit.main()

def run_authority_isolated_suite3O_lane0_dynamics_audit():
    """Run former authority_isolated_suite3O_lane0_dynamics_audit.py stage."""
    """Which part of u50's command keeps the failing suite-3O lane upright?
    
    Rolls u50, u75 and the repaired policy on the one lane that fails, then rolls
    u75 again with u50's command substituted one joint at a time, blended by
    weight, and restricted to time windows. Each trial keeps a per-step trace of
    height, angular velocity and action, so a rescue can be read as a trajectory
    rather than only as a survival bit.
    """
    import json
    import sys
    from pathlib import Path
    
    import torch
    
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    
    from rl.core.diagnostics.isaac_audit import RUNS, IsaacAudit, obs_tensor
    
    H1 = "authority_isolated_h1-2026-09-25"
    REPAIR = "authority_isolated_projected_tail_descent-2026-09-25/model_10.pt"
    SEED = 840004
    PREFERENCE = [.1, .1, .7, .1]
    LANE = 0
    HORIZON = 24
    ALPHAS = [.1, .25, .5, .75, 1.0]
    WINDOWS = [(0, 5), (4, 8), (6, 10), (9, 13), (11, 15)]
    
    
    class Suite3OLane0DynamicsAudit(IsaacAudit):
        """Per-joint, per-weight and per-window rescue of the failing lane."""
    
        run = "authority_isolated_suite3O_lane0_dynamics_audit-2026-09-25"
        report = "dynamics_audit.json"
    
        def actor(self, path, obs_dim):
            from talon_rl.models.authority.isolated import AuthorityIsolatedActorCritic
    
            m = AuthorityIsolatedActorCritic(obs_dim, 12).cuda()
            m.load_state_dict(torch.load(RUNS / path, map_location="cuda",
                                         weights_only=False)["model"])
            m.eval()
            return m
    
        def wide(self, path, obs_dim):
            from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
    
            m = AuthorityIsolatedWideCritic(obs_dim, 12).cuda()
            m.load_state_dict(torch.load(RUNS / path, map_location="cuda",
                                         weights_only=False)["model"])
            m.eval()
            return m
    
        def trial(self, env, base, donor=None, idx=None, alpha=None, window=None):
            """Roll `base`, optionally taking part of `donor`'s command instead.
    
            `idx` substitutes those coordinates outright, `alpha` blends the whole
            command, and neither means take the donor's command entirely.
            """
            w = torch.tensor(PREFERENCE, device="cuda").repeat(self.num_envs, 1)
            obs, _ = env.reset(seed=SEED)
            obs = obs_tensor(obs).cuda()
            first_fail, trace = None, []
            for t in range(HORIZON):
                d = env.unwrapped.scene["robot"].data
                with torch.no_grad():
                    ab = base.act_inference_with_preference(obs, w)
                    a = ab.clone()
                    if donor is not None and (window is None or window[0] <= t <= window[1]):
                        ad = donor.act_inference_with_preference(obs, w)
                        if idx is None and alpha is None:
                            a = ad
                        elif idx is not None:
                            a[:, idx] = ad[:, idx]
                        else:
                            a = (1 - alpha) * ab + alpha * ad
                trace.append({"t": t, "height": float(d.root_pos_w[LANE, 2]),
                              "ang_vel": d.root_ang_vel_b[LANE].cpu().tolist(),
                              "action": a[LANE].cpu().tolist()})
                nxt, _, te, tr, _ = env.step(a)
                done = (te | tr).cpu().numpy().astype(bool)
                if done[LANE] and first_fail is None:
                    first_fail = t
                obs = obs_tensor(nxt).cuda()
            return {"survived": first_fail is None, "first_fail": first_fail, "trace": trace}
    
        def rollout(self, env, obs):
            od = obs.shape[-1]
            u50 = self.actor(f"{H1}/model_50.pt", od)
            u75 = self.actor(f"{H1}/model_75.pt", od)
            repaired = self.wide(REPAIR, od)
    
            names = list(env.unwrapped.scene["robot"].data.joint_names)
            res = {"joint_names": names, "tests": {}}
    
            def record(lab, result):
                res["tests"][lab] = result
                print(lab, result["first_fail"], flush=True)
    
            for lab, m in (("u50", u50), ("u75", u75), ("repair", repaired)):
                record(lab, self.trial(env, m))
            for j, n in enumerate(names):
                record("u75_plus_u50_" + n, self.trial(env, u75, u50, [j]))
            for a in ALPHAS:
                record(f"u75_u50_interp_{a}", self.trial(env, u75, u50, alpha=a))
            for win in WINDOWS:
                record(f"u50_all_t{win[0]}_{win[1]}",
                       self.trial(env, u75, u50, idx=list(range(12)), window=win))
            self.write(res)
            return res
    
    
    if True:
        Suite3OLane0DynamicsAudit.main()

def run_authority_isolated_survival_audit():
    """Run former authority_isolated_survival_audit.py stage."""
    from pathlib import Path
    import json,sys,hashlib
    import numpy as np, torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    OUT=ROOT/"runs/authority_isolated_survival_audit-2026-09-25"
    OUT.mkdir(parents=True,exist_ok=True)
    CKPTS={"u50":ROOT/"runs/authority_isolated_h1-2026-09-25/model_50.pt",
           "u75":ROOT/"runs/authority_isolated_h1-2026-09-25/model_75.pt"}
    ORDER=("T","A","O","S","C")
    PREFS={"T":[.7,.1,.1,.1],"A":[.1,.7,.1,.1],"O":[.1,.1,.7,.1],
           "S":[.1,.1,.1,.7],"C":[.25,.25,.25,.25]}
    SEEDS=(840001,840002,840003,840004);NENV=8;H=64
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def qtilt(q):
        r=torch.atan2(2*(q[:,0]*q[:,1]+q[:,2]*q[:,3]),1-2*(q[:,1]**2+q[:,2]**2))
        p=torch.asin(torch.clamp(2*(q[:,0]*q[:,2]-q[:,3]*q[:,1]),-1,1))
        return torch.rad2deg(torch.maximum(r.abs(),p.abs()))
    def local_authority(m,obs,w):
        wc=torch.full_like(w,.25)
        with torch.no_grad():
            aw=m.act_inference_with_preference(obs,w);ac=m.act_inference_with_preference(obs,wc)
            eff=torch.linalg.vector_norm(aw-ac,dim=1)
        eps=1e-3
        D=torch.tensor([[1.,-1,0,0],[1.,0,-1,0],[1.,0,0,-1]],device=obs.device).T
        Q,_=torch.linalg.qr(D,mode="reduced")
        vals=[]
        with torch.no_grad():
            for j in range(3):
                wp=w+eps*Q[:,j];wm=w-eps*Q[:,j]
                ap=m.act_inference_with_preference(obs,wp)
                am=m.act_inference_with_preference(obs,wm)
                vals.append((ap-am)/(2*eps))
        J=torch.stack(vals,dim=2)
        return eff,torch.linalg.matrix_norm(J,ord="fro",dim=(1,2))
    def reason_terms(env):
        tm=env.unwrapped.termination_manager
        out={}
        for n in tm.active_terms:
            try: out[n]=tm.get_term(n).detach().cpu().numpy().astype(bool)
            except Exception: pass
        return out
    def rollout(env,m,lab,seed):
        w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1)
        obs,_=env.reset(seed=seed);obs=ot(obs).cuda()
        prev=torch.zeros((NENV,env.unwrapped.action_manager.total_action_dim),device="cuda")
        done=np.zeros(NENV,bool);fail_t=np.full(NENV,-1,int);why=[[] for _ in range(NENV)]
        rows=[]
        with torch.no_grad():
            gp=torch.linalg.vector_norm(m.generated_parameter_vector(w),dim=1).cpu().numpy()
        for t in range(H):
            data=env.unwrapped.scene["robot"].data
            tilt=qtilt(data.root_quat_w)
            with torch.no_grad():
                pre=m._actor_mean_with_preference(obs,w)
                a=torch.tanh(pre)*m.ACTION_CLIP
            eff,jac=local_authority(m,obs,w)
            sat=(a.abs()>=.95).float().mean(dim=1)
            margin=1-a.abs().amax(dim=1)
            met=np.stack([data.root_pos_w[:,2].detach().cpu().numpy(),
              tilt.detach().cpu().numpy(),
              torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=1).detach().cpu().numpy(),
              torch.linalg.vector_norm(data.joint_vel,dim=1).detach().cpu().numpy(),
              torch.linalg.vector_norm(a,dim=1).detach().cpu().numpy(),
              torch.linalg.vector_norm(a-prev,dim=1).detach().cpu().numpy(),
              sat.detach().cpu().numpy(),margin.detach().cpu().numpy(),
              torch.linalg.vector_norm(pre,dim=1).detach().cpu().numpy(),
              eff.detach().cpu().numpy(),jac.detach().cpu().numpy(),gp],axis=1)
            rows.append(met)
            nxt,_,te,tr,_=env.step(a);dd=(te|tr).detach().cpu().numpy().astype(bool)
            terms=reason_terms(env)
            for i in range(NENV):
                if dd[i] and not done[i]:
                    fail_t[i]=t;why[i]=[n for n,v in terms.items() if i<len(v) and v[i]]
            done|=dd;prev=a;obs=ot(nxt).cuda()
        X=np.asarray(rows)
        return {"checkpoint":None,"preference":lab,"seed":seed,"suite":SEEDS.index(seed),
          "survival":float(1-done.mean()),"done":done.tolist(),"fail_t":fail_t.tolist(),
          "reason":why,"metric_names":["height","tilt_deg","ang_xy","joint_vel","action",
          "action_rate","sat_frac","action_margin","pre_tanh","family_effect","local_jac",
          "generated_param"],"metrics":X.tolist()}
    def summarize_rollouts(rs):
        out=[]
        for r in rs:
            X=np.asarray(r["metrics"]);ft=np.asarray(r["fail_t"]);done=np.asarray(r["done"])
            pf=[];sv=[]
            for i in range(NENV):
                if done[i]:
                    e=ft[i]+1;s=max(0,e-8);pf.append(X[s:e,i])
                else: sv.append(X[-8:,i])
            pf=np.concatenate(pf) if pf else np.empty((0,X.shape[2]))
            sv=np.concatenate(sv) if sv else np.empty((0,X.shape[2]))
            def stats(A):
                if not len(A):return None
                return {"mean":A.mean(0).tolist(),"p95":np.percentile(A,95,axis=0).tolist()}
            out.append({"checkpoint":r["checkpoint"],"suite":r["suite"],"preference":r["preference"],
              "survival":r["survival"],"fail_count":int(done.sum()),
              "failure_reasons":r["reason"],"prefail8":stats(pf),"survived_tail8":stats(sv)})
        return out
    def main():
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
            o,_=env.reset(seed=0);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim
            allr=[]
            for ck,path in CKPTS.items():
                m=AuthorityIsolatedActorCritic(o.shape[-1],ad).cuda()
                m.load_state_dict(torch.load(path,map_location="cuda",weights_only=False)["model"]);m.eval()
                for seed in SEEDS:
                    for lab in ORDER:
                        r=rollout(env,m,lab,seed);r["checkpoint"]=ck;allr.append(r)
                        print(ck,r["suite"],lab,r["survival"],r["fail_t"],r["reason"],flush=True)
            summ=summarize_rollouts(allr)
            report={"schema":"authority_isolated_survival_audit_v1","read_only":True,
              "checkpoints":{k:str(v.relative_to(ROOT)) for k,v in CKPTS.items()},
              "semantic_seeds":list(SEEDS),"metric_names":allr[0]["metric_names"],
              "summary":summ}
            (OUT/"authority_isolated_survival_audit_report.json").write_text(json.dumps(report,indent=2)+"\n")
            np.savez_compressed(OUT/"authority_isolated_survival_audit_raw.npz",
              payload=np.asarray([json.dumps(x) for x in allr],dtype=object))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_authority_isolated_u30_u40_path_swap_audit():
    """Run former authority_isolated_u30_u40_path_swap_audit.py stage."""
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from pathlib import Path
    import json,sys,itertools,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"scripts/rl")]
    import rl.experiments.common.utilities.authority_isolated_h1_screen as h1
    from talon_rl.models.authority.isolated import AuthorityIsolatedActorCritic
    from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
    
    CK20=ROOT/"runs/authority_isolated_coverage2_control-2026-09-25/model_20.pt"
    CK30=ROOT/"runs/authority_isolated_simplex_edge_retain-2026-09-25/model_30.pt"
    CK40={
     "narrow":ROOT/"runs/authority_isolated_ai_c2_formal_edge_narrow-2026-09-25/model_10.pt",
     "wide":ROOT/"runs/authority_isolated_ai_c2_formal_edge_wide-2026-09-25/model_10.pt",
    }
    SUP=ROOT/"runs/authority_isolated_u20_authority_support-2026-09-25/u20_authority_support.npz"
    OUT=ROOT/"runs/authority_isolated_u30_u40_path_swap_audit-2026-09-25";OUT.mkdir(parents=True,exist_ok=True)
    ORDER=h1.ORDER;PREFS=h1.PREFS;PAIRS=list(itertools.combinations(ORDER,2))
    NENV=8;H=64
    
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    
    def load_state(path):
        return torch.load(path,map_location="cuda",weights_only=False)["model"]
    
    def build(cls,state):
        m=cls(48,12).cuda();m.load_state_dict(state);m.eval();return m
    
    def hybrid_state(s30,s40,common_from,family_from):
        # Start from u30 full state. Only actor function matters; critic remains u30-compatible where possible.
        out={k:v.clone() for k,v in s30.items()}
        cs=s30 if common_from==30 else s40
        fs=s30 if family_from==30 else s40
        for k in out:
            if k.startswith("actor_body") or k.startswith("actor_mean"):
                out[k]=cs[k].clone()
            elif k.startswith("family_"):
                out[k]=fs[k].clone()
            elif k=="log_std":
                out[k]=s40[k].clone()  # deterministic rollout unaffected; keep endpoint stochastic scale
        return out
    
    def actor_only_state_for_cls(src,target_template):
        out={k:v.clone() for k,v in target_template.items()}
        for k in out:
            if (k.startswith("actor_") or k.startswith("family_") or k=="log_std") and k in src and out[k].shape==src[k].shape:
                out[k]=src[k].clone()
        return out
    
    def authority_metrics(m,m20,X):
        A={};R={}
        with torch.no_grad():
            for lab in ORDER:
                w=torch.tensor(PREFS[lab],device="cuda").repeat(len(X),1)
                A[lab]=m.act_inference_with_preference(X,w)
                R[lab]=m20.act_inference_with_preference(X,w)
        ds=[];rs=[];edges={}
        for i,j in PAIRS:
            d=torch.linalg.vector_norm(A[i]-A[j],dim=1)
            r=torch.linalg.vector_norm(R[i]-R[j],dim=1)
            ds.append(float(d.mean().cpu()));rs.append(float(r.mean().cpu()))
            edges[f"{i}-{j}"]=float(torch.sqrt((d.pow(2).sum()+1e-12)/(r.pow(2).sum()+1e-12)).cpu())
        pair=float(np.mean(ds)/(np.mean(rs)+1e-12))
        from torch.func import jacrev,vmap
        wc=torch.tensor(PREFS["C"],device="cuda")
        D=torch.tensor([[1.,-1,0,0],[1.,0,-1,0],[1.,0,0,-1]],device="cuda").T;Q,_=torch.linalg.qr(D,mode="reduced")
        def f(o,w):return m.act_inference_with_preference(o.unsqueeze(0),w.unsqueeze(0)).squeeze(0)
        def rf(o,w):return m20.act_inference_with_preference(o.unsqueeze(0),w.unsqueeze(0)).squeeze(0)
        J=vmap(jacrev(f,argnums=1),in_dims=(0,None))(X,wc)@Q
        RJ=vmap(jacrev(rf,argnums=1),in_dims=(0,None))(X,wc)@Q
        tan=float((torch.linalg.matrix_norm(J,ord="fro",dim=(1,2)).mean()/(torch.linalg.matrix_norm(RJ,ord="fro",dim=(1,2)).mean()+1e-12)).cpu())
        vals=list(edges.values())
        return {"pairwise_retention":pair,"tangent_retention":tan,"mean_edge":float(np.mean(vals)),
                "min_edge":float(min(vals)),"min_edge_name":min(edges,key=edges.get),"edges":edges}
    
    def rollout(env,m,lab,seed):
        w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1)
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        done=np.zeros(NENV,bool);ttf=np.full(NENV,H,np.int32);reasons=[[] for _ in range(NENV)]
        prev=None
        traces=[[] for _ in range(NENV)]
        max_z=np.zeros(NENV);sat=np.zeros(NENV);max_an=np.zeros(NENV);max_ar=np.zeros(NENV)
        with torch.no_grad():
            for t in range(H):
                z=m._actor_mean_with_preference(cur,w);a=torch.tanh(z)*m.ACTION_CLIP
                oo=cur.detach().cpu().numpy();aa=a.detach().cpu().numpy();zz=z.detach().cpu().numpy()
                ang=np.linalg.norm(oo[:,3:6],axis=1);tilt=np.linalg.norm(oo[:,6:9],axis=1)
                lv=np.linalg.norm(oo[:,0:3],axis=1);jv=np.linalg.norm(oo[:,24:36],axis=1)
                ar=np.zeros(NENV) if prev is None else np.linalg.norm(aa-prev,axis=1)
                max_z=np.maximum(max_z,np.max(np.abs(zz),axis=1));sat+=(np.abs(np.tanh(zz))>=.95).mean(axis=1)
                max_an=np.maximum(max_an,np.linalg.norm(aa,axis=1));max_ar=np.maximum(max_ar,ar)
                for i in range(NENV):
                    if not done[i]:
                        traces[i].append([t,float(lv[i]),float(ang[i]),float(tilt[i]),float(jv[i]),float(np.linalg.norm(aa[i])),float(ar[i])])
                nxt,_,te,tr,_=env.step(a);dd=(te|tr).cpu().numpy();tm=env.unwrapped.termination_manager
                new=dd&(~done)
                for i in np.flatnonzero(new):
                    ttf[i]=t+1;reasons[i]=[n for n in tm.active_terms if bool(tm.get_term(n)[i].detach().cpu())]
                done|=dd;prev=aa;cur=ot(nxt).cuda()
        lanes=[]
        for i in range(NENV):
            lanes.append({"lane":i,"survived":bool(not done[i]),"ttf":int(ttf[i]),"reasons":reasons[i],
              "max_pre_tanh_abs":float(max_z[i]),"sat_fraction":float(sat[i]/H),
              "max_action_norm":float(max_an[i]),"max_action_rate":float(max_ar[i]),"trace":traces[i]})
        return {"survival":float(np.mean([q["survived"] for q in lanes])),"fail_count":int(sum(not q["survived"] for q in lanes)),
                "lanes":lanes}
    
    def main():
        from isaaclab.app import AppLauncher
        sv=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv;env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);env.reset(seed=0)
            s20=load_state(CK20);s30=load_state(CK30);m20=build(AuthorityIsolatedWideCritic,s20)
            sd=np.load(SUP);Xall=torch.tensor(sd["obs"],device="cuda");origin=sd["origin"];phase=sd["phase"]
            rng=np.random.default_rng(2609252901);ids=[]
            for oi in range(5):
                for pi in range(3):
                    z=np.flatnonzero((origin==oi)&(phase==pi));ids.extend(rng.choice(z,size=20,replace=False).tolist())
            X=Xall[np.asarray(ids)]
            rep={"schema":"u30_u40_path_swap_audit_v1","arms":{}}
            for arm,path in CK40.items():
                cls=AuthorityIsolatedActorCritic if arm=="narrow" else AuthorityIsolatedWideCritic
                s40=load_state(path)
                # Build all actors in arm's class to avoid critic-shape issues.
                template=cls(48,12).cuda().state_dict()
                s30a=actor_only_state_for_cls(s30,template)
                s40a=actor_only_state_for_cls(s40,template)
                hc=hybrid_state(s30a,s40a,40,30)  # common u40 + family u30
                hf=hybrid_state(s30a,s40a,30,40)  # common u30 + family u40
                models={"u30":build(cls,s30a),"u40":build(cls,s40a),"H_common":build(cls,hc),"H_family":build(cls,hf)}
                arep={"authority":{},"fresh":{}}
                for name,m in models.items():
                    arep["authority"][name]=authority_metrics(m,m20,X)
                    rows=[]
                    for li,lab in enumerate(ORDER):
                        for si in range(4):
                            seed=9700000+li*1000+si*113
                            q=rollout(env,m,lab,seed);q.update({"preference":lab,"suite":si,"seed":seed});rows.append(q)
                    arep["fresh"][name]=rows
                    summ={"min_survival":min(x["survival"] for x in rows),
                          "failed_lanes":sum(x["fail_count"] for x in rows),
                          "failed_cases":sum(x["fail_count"]>0 for x in rows)}
                    arep["fresh"][name+"_summary"]=summ
                    print("RESULT",arm,name,json.dumps({"authority":arep["authority"][name],"fresh":summ}),flush=True)
                rep["arms"][arm]=arep
            (OUT/"path_swap_audit.json").write_text(json.dumps(rep,indent=2)+"\n")
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_authority_isolation_feasibility_audit():
    """Run former authority_isolation_feasibility_audit.py stage."""
    from pathlib import Path
    import json,hashlib,sys
    import numpy as np, torch
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
    BASE=ROOT/'runs/v2b_adam_continuous-2026-09-24/model_75.pt'
    STATE_CACHE=ROOT/'runs/update_visitation_interaction_audit-2026-09-24/state_cache'
    PROBE=ROOT/'runs/update_functional_effect_audit-2026-09-24/fixed_probe_states.npz'
    OUT=ROOT/'runs/authority_isolation_feasibility-2026-09-25';OUT.mkdir(parents=True,exist_ok=True)
    FIT_N=8192;HELD_N=4096;BATCH=2048;STEPS=5000;LR=3e-4
    ANCHORS=np.array([[.7,.1,.1,.1],[.1,.7,.1,.1],[.1,.1,.7,.1],[.1,.1,.1,.7],[.25,.25,.25,.25]],np.float32)
    
    def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def evenly(x,n):
        if len(x)<=n:return x.copy()
        idx=np.linspace(0,len(x)-1,n,dtype=np.int64)
        return x[idx]
    def load_states():
        train=[];held=[]
        for p in sorted(STATE_CACHE.glob('*.npz')):
            z=np.load(p,allow_pickle=False);sd=int(z['seed']);x=z['obs'].reshape(-1,z['obs'].shape[-1]).astype(np.float32)
            if sd in (983001,984001): train.append(x)
            elif sd==985001: held.append(x)
        tr=np.concatenate(train,0);ho=np.concatenate(held,0)
        return evenly(tr,FIT_N),evenly(ho,HELD_N)
    def prefs():
        r1=np.random.default_rng(26092501);r2=np.random.default_rng(26092502)
        fit=np.concatenate([ANCHORS,r1.dirichlet(np.ones(4),59).astype(np.float32)],0)
        held=r2.dirichlet(np.ones(4),64).astype(np.float32)
        return fit,held
    def teacher_mean(m,o,w): return m._pre_tanh_dist_with_preference(o,w).mean
    def outputs_teacher(m,states,prefs,batch=4096):
        out=[]
        with torch.no_grad():
            for w in prefs:
                ww=torch.tensor(w,device='cuda').repeat(len(states),1)
                vals=[]
                for i in range(0,len(states),batch):
                    o=torch.tensor(states[i:i+batch],device='cuda')
                    vals.append(teacher_mean(m,o,ww[i:i+batch]).cpu())
                out.append(torch.cat(vals,0).numpy())
        return np.stack(out,1)
    def outputs_student(m,states,prefs,batch=4096):
        out=[]
        with torch.no_grad():
            for w in prefs:
                ww=torch.tensor(w,device='cuda').repeat(len(states),1)
                vals=[]
                for i in range(0,len(states),batch):
                    o=torch.tensor(states[i:i+batch],device='cuda')
                    vals.append(m.pre_tanh_mean(o,ww[i:i+batch]).cpu())
                out.append(torch.cat(vals,0).numpy())
        return np.stack(out,1)
    def sep_score(actions):
        # exact mean pairwise Euclidean action separation, averaged over states.
        t=torch.tensor(actions)
        # [N,P,A] -> pairwise distances [N,P,P]
        d=torch.cdist(t,t)
        p=d.shape[1]
        mask=torch.triu(torch.ones(p,p,dtype=torch.bool),diagonal=1)
        return float(d[:,mask].mean())
    def jacobians_teacher(m,states,batch=128):
        from torch.func import jacrev,vmap
        w0=torch.tensor([.25,.25,.25,.25],device='cuda')
        def f(o,w): return teacher_mean(m,o.unsqueeze(0),w.unsqueeze(0)).squeeze(0)
        js=[]
        for i in range(0,len(states),batch):
            o=torch.tensor(states[i:i+batch],device='cuda')
            js.append(vmap(jacrev(f,argnums=1),in_dims=(0,None))(o,w0).detach().cpu())
        return torch.cat(js).numpy()
    def jacobians_student(m,states,batch=128):
        from torch.func import jacrev,vmap
        w0=torch.tensor([.25,.25,.25,.25],device='cuda')
        def f(o,w): return m.pre_tanh_mean(o.unsqueeze(0),w.unsqueeze(0)).squeeze(0)
        js=[]
        for i in range(0,len(states),batch):
            o=torch.tensor(states[i:i+batch],device='cuda')
            js.append(vmap(jacrev(f,argnums=1),in_dims=(0,None))(o,w0).detach().cpu())
        return torch.cat(js).numpy()
    def cosine(a,b):
        a=a.ravel().astype(np.float64);b=b.ravel().astype(np.float64)
        return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
    def eval_split(tm,sm,states,prefs,name,with_jac=False):
        yt=outputs_teacher(tm,states,prefs);ys=outputs_student(sm,states,prefs)
        diff=ys-yt
        at=np.tanh(yt);ass=np.tanh(ys);ad=ass-at
        l2=np.linalg.norm(ad,axis=-1)
        res={
          'name':name,
          'n_states':len(states),'n_preferences':len(prefs),
          'pre_tanh_rmse':float(np.sqrt(np.mean(diff**2))),
          'mean_action_l2':float(np.mean(l2)),
          'p95_action_l2':float(np.percentile(l2,95)),
          'max_abs_action_coordinate_error':float(np.max(np.abs(ad))),
        }
        st=sep_score(at);ss=sep_score(ass)
        res['teacher_pairwise_preference_separation']=st
        res['student_pairwise_preference_separation']=ss
        res['preference_separation_relative_error']=float(abs(ss-st)/(abs(st)+1e-12))
        if with_jac:
            jt=jacobians_teacher(tm,states);js=jacobians_student(sm,states)
            res['jacobian_relative_fro_error']=float(np.linalg.norm(js-jt)/(np.linalg.norm(jt)+1e-12))
            res['jacobian_cosine']=cosine(js,jt)
        return res
    def main():
        from talon_rl.models.conditioning.single_site_film import V2BSingleSiteFiLMActorCritic
        from talon_rl.models.authority.isolated import AuthorityIsolatedActor
        fit_states,held_states=load_states();fit_prefs,held_prefs=prefs()
        probe=np.load(PROBE,allow_pickle=False)['obs'].astype(np.float32)
        state=torch.load(BASE,map_location='cuda',weights_only=False)['model']
        tm=V2BSingleSiteFiLMActorCritic(48,12).cuda();tm.load_state_dict(state);tm.eval()
        sm=AuthorityIsolatedActor(48,12).cuda();sm.initialize_state_path_from_v2b(state);sm.train()
        opt=torch.optim.Adam(sm.parameters(),lr=LR,weight_decay=0)
        rng=np.random.default_rng(26092503)
        checkpoints={0:OUT/'student_step_0.pt',500:OUT/'student_step_500.pt',1000:OUT/'student_step_1000.pt',2500:OUT/'student_step_2500.pt',5000:OUT/'student_step_5000.pt'}
        torch.save({'model':sm.state_dict(),'step':0},checkpoints[0])
        losses=[]
        for step in range(1,STEPS+1):
            si=rng.integers(0,len(fit_states),size=BATCH)
            pi=rng.integers(0,len(fit_prefs),size=BATCH)
            o=torch.tensor(fit_states[si],device='cuda')
            w=torch.tensor(fit_prefs[pi],device='cuda')
            with torch.no_grad(): y=teacher_mean(tm,o,w)
            pred=sm.pre_tanh_mean(o,w)
            loss=(pred-y).pow(2).mean()
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            if step%100==0: losses.append({'step':step,'loss':float(loss.detach().cpu())})
            if step in checkpoints: torch.save({'model':sm.state_dict(),'step':step},checkpoints[step])
        sm.eval()
        fit_fit=eval_split(tm,sm,fit_states,fit_prefs,'fit_states_x_fit_preferences',False)
        held_fit=eval_split(tm,sm,held_states,fit_prefs,'heldout_states_x_fit_preferences',False)
        fit_held=eval_split(tm,sm,fit_states,held_prefs,'fit_states_x_heldout_preferences',False)
        held_held=eval_split(tm,sm,held_states,held_prefs,'heldout_states_x_heldout_preferences',True)
        probe_held=eval_split(tm,sm,probe,held_prefs,'fixed_probe_x_heldout_preferences',True)
        ratio=held_held['pre_tanh_rmse']/(fit_fit['pre_tanh_rmse']+1e-12)
        c={
          'held_rmse':held_held['pre_tanh_rmse']<=.02,
          'held_mean_action':held_held['mean_action_l2']<=.03,
          'held_p95_action':held_held['p95_action_l2']<=.06,
          'held_max_coord':held_held['max_abs_action_coordinate_error']<=.10,
          'held_sep':held_held['preference_separation_relative_error']<=.10,
          'held_jac_rel':held_held['jacobian_relative_fro_error']<=.15,
          'held_jac_cos':held_held['jacobian_cosine']>=.95,
          'probe_mean_action':probe_held['mean_action_l2']<=.04,
          'probe_sep':probe_held['preference_separation_relative_error']<=.15,
          'generalization_ratio':ratio<=1.5,
        }
        if all(c.values()): status='FUNCTION-PRESERVING-ISH TRANSFER FEASIBLE'
        elif held_held['mean_action_l2']<=.08 and held_held['jacobian_cosine']>=.85: status='TRANSFER APPROXIMATE BUT NOT FUNCTION-PRESERVING'
        else: status='AUTHORITY ISOLATION TRANSFER NOT FEASIBLE'
        rep={
          'schema':'authority_isolation_feasibility_v1','status':status,
          'teacher_checkpoint':str(BASE.relative_to(ROOT)),
          'state_counts':{'fit':len(fit_states),'heldout':len(held_states),'probe':len(probe)},
          'preference_counts':{'fit':len(fit_prefs),'heldout':len(held_prefs)},
          'optimization':{'optimizer':'Adam','lr':LR,'batch_size':BATCH,'steps':STEPS,'weight_decay':0},
          'splits':[fit_fit,held_fit,fit_held,held_held,probe_held],
          'generalization_rmse_ratio':ratio,'criteria':c,'loss_trace':losses,
          'decision':{'rl_training_authorized':False}
        }
        rp=OUT/'authority_isolation_feasibility_report.json';rp.write_text(json.dumps(rep,indent=2)+'\n')
        (OUT/'PROVENANCE_MANIFEST.json').write_text(json.dumps({
          'status':'FROZEN_BY_HASH','decision':status,
          'report_sha256':sha(rp),'contract_sha256':sha(ROOT/'docs/contracts/authority/authority-isolation-feasibility-contract.md'),
          'script_sha256':sha(Path(__file__).resolve()),'architecture_sha256':sha(ROOT/'talon_rl/authority_isolated_actor.py'),
          'teacher_sha256':sha(BASE),'final_student_sha256':sha(checkpoints[5000])
        },indent=2)+'\n')
        print(json.dumps(rep,indent=2))
    if True: main()

STAGES = {
    "authority_isolated_Oheavy_support_matrix": run_authority_isolated_Oheavy_support_matrix,
    "authority_isolated_adversarial_cem_confirm": run_authority_isolated_adversarial_cem_confirm,
    "authority_isolated_authority_durability_diagnostic": run_authority_isolated_authority_durability_diagnostic,
    "authority_isolated_authority_matched_state_audit": run_authority_isolated_authority_matched_state_audit,
    "authority_isolated_build_u20_authority_support": run_authority_isolated_build_u20_authority_support,
    "authority_isolated_direction_amplitude_audit": run_authority_isolated_direction_amplitude_audit,
    "authority_isolated_flhip_phase_rescue": run_authority_isolated_flhip_phase_rescue,
    "authority_isolated_flhip_reverse_phase": run_authority_isolated_flhip_reverse_phase,
    "authority_isolated_flhip_window_reverse": run_authority_isolated_flhip_window_reverse,
    "authority_isolated_functional_deltaa_endpoint_audit": run_authority_isolated_functional_deltaa_endpoint_audit,
    "authority_isolated_functional_deltaa_retention_train": run_authority_isolated_functional_deltaa_retention_train,
    "authority_isolated_h0_gate": run_authority_isolated_h0_gate,
    "authority_isolated_overexpansion_causal_audit": run_authority_isolated_overexpansion_causal_audit,
    "authority_isolated_policy_output_neighborhood_audit": run_authority_isolated_policy_output_neighborhood_audit,
    "authority_isolated_policy_output_neighborhood_staged": run_authority_isolated_policy_output_neighborhood_staged,
    "authority_isolated_precontact_compatibility_audit": run_authority_isolated_precontact_compatibility_audit,
    "authority_isolated_s2O_l7_rescue": run_authority_isolated_s2O_l7_rescue,
    "authority_isolated_simplex_edge_endpoint_audit": run_authority_isolated_simplex_edge_endpoint_audit,
    "authority_isolated_simplex_edge_retention_train": run_authority_isolated_simplex_edge_retention_train,
    "authority_isolated_suite3O_hip_margin_audit": run_authority_isolated_suite3O_hip_margin_audit,
    "authority_isolated_suite3O_lane0_dynamics_audit": run_authority_isolated_suite3O_lane0_dynamics_audit,
    "authority_isolated_survival_audit": run_authority_isolated_survival_audit,
    "authority_isolated_u30_u40_path_swap_audit": run_authority_isolated_u30_u40_path_swap_audit,
    "authority_isolation_feasibility_audit": run_authority_isolation_feasibility_audit,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
