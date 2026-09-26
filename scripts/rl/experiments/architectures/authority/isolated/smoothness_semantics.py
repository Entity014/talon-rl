"""Consolidated thematic experiment stages."""
from __future__ import annotations
import argparse

def run_authority_isolated_s_gradient_alignment_fd_check():
    """Run former authority_isolated_s_gradient_alignment_fd_check.py stage."""
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from pathlib import Path
    import sys,json,torch,numpy as np
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"scripts/rl")]
    from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
    import rl.experiments.common.utilities.authority_isolated_h1_screen as h1
    CK=ROOT/"runs/authority_isolated_simplex_edge_retain-2026-09-25/model_30.pt"
    OUT=ROOT/"runs/authority_isolated_s_local_gradient_alignment_audit-2026-09-25"
    NENV=8
    WC=torch.tensor(h1.PREFS["C"],dtype=torch.float32);DS=torch.tensor(h1.PREFS["S"]-h1.PREFS["C"],dtype=torch.float32)
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
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);obs,_=env.reset(seed=840001);obs=ot(obs).cuda()
      m=AuthorityIsolatedWideCritic(obs.shape[-1],12).cuda();m.load_state_dict(torch.load(CK,map_location="cuda",weights_only=False)["model"]);m.eval()
      prev=torch.zeros((NENV,12),device="cuda");w=WC.cuda().repeat(NENV,1);d=DS.cuda().repeat(NENV,1)
      def f(ww): return m.act_inference_with_preference(obs,ww)
      a,jd=torch.func.jvp(f,(w,),(d,))
      G=(-.02*(a-prev)*jd).sum(-1)
      epsvals=[1e-3,1e-2,5e-2]
      out={"G":G.detach().cpu().tolist(),"eps":{}}
      with torch.no_grad():
       r0=-.01*((a-prev)**2).sum(-1)
       for e in epsvals:
        ap=f(w+e*d);am=f(w-e*d)
        rp=-.01*((ap-prev)**2).sum(-1);rm=-.01*((am-prev)**2).sum(-1)
        fd=(rp-rm)/(2*e)
        out["eps"][str(e)]={"fd":fd.cpu().tolist(),"max_abs_err":float((fd-G).abs().max().cpu()),"mean_abs_err":float((fd-G).abs().mean().cpu())}
      (OUT/"finite_difference_check.json").write_text(json.dumps(out,indent=2)+"\n")
      print(json.dumps({e:q["max_abs_err"] for e,q in out["eps"].items()}),flush=True)
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_authority_isolated_s_local_gradient_alignment_audit():
    """Run former authority_isolated_s_local_gradient_alignment_audit.py stage."""
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from pathlib import Path
    import json,sys,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"scripts/rl")]
    from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
    import rl.experiments.common.utilities.authority_isolated_h1_screen as h1
    
    CK=ROOT/"runs/authority_isolated_simplex_edge_retain-2026-09-25/model_30.pt"
    LABELS=ROOT/"runs/authority_isolated_s_sign_context_audit-2026-09-25/s_sign_context_audit.json"
    OUT=ROOT/"runs/authority_isolated_s_local_gradient_alignment_audit-2026-09-25";OUT.mkdir(parents=True,exist_ok=True)
    NENV=8;H=64
    WC=torch.tensor(h1.PREFS["C"],dtype=torch.float32)
    WS=torch.tensor(h1.PREFS["S"],dtype=torch.float32)
    DS=WS-WC
    
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    
    def local_alignment(m,obs,prev):
        w=WC.to(obs.device).repeat(len(obs),1)
        d=DS.to(obs.device).repeat(len(obs),1)
        def f(ww):
            return m.act_inference_with_preference(obs,ww)
        a,jd=torch.func.jvp(f,(w,),(d,))
        ga=-0.02*(a-prev)
        contrib=ga*jd
        G=contrib.sum(-1)
        return a.detach(),jd.detach(),ga.detach(),contrib.detach(),G.detach()
    
    def run_center(env,m,seed):
        obs,_=env.reset(seed=seed);obs=ot(obs).cuda()
        prev=torch.zeros((NENV,12),device="cuda")
        rows=[]
        with torch.no_grad():
            pass
        for t in range(H):
            a,jd,ga,contrib,G=local_alignment(m,obs,prev)
            rows.append({
              "t":t,
              "G":G.cpu().numpy().tolist(),
              "a":a.cpu().numpy().tolist(),
              "prev":prev.cpu().numpy().tolist(),
              "Jd":jd.cpu().numpy().tolist(),
              "grad_a":ga.cpu().numpy().tolist(),
              "contrib":contrib.cpu().numpy().tolist(),
            })
            with torch.no_grad():
                nxt,_,_,_,_=env.step(a)
            prev=a.detach();obs=ot(nxt).cuda()
        return rows
    
    def stats(vals):
        a=np.asarray(vals,float)
        return {"mean":float(a.mean()),"median":float(np.median(a)),
                "negative_fraction":float(np.mean(a<0)),"positive_fraction":float(np.mean(a>0)),
                "p10":float(np.quantile(a,.1)),"p90":float(np.quantile(a,.9))}
    
    def auc(y,s):
        y=np.asarray(y);s=np.asarray(s,float);pos=s[y==1];neg=s[y==0]
        if len(pos)==0 or len(neg)==0:return None
        return float(np.mean([(a>b)+.5*(a==b) for a in pos for b in neg]))
    
    def main():
        labels=json.load(open(LABELS))["samples"]
        lab={(x["suite"],x["lane"]):x for x in labels}
        from isaaclab.app import AppLauncher
        sv=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv;env=None
        try:
          import gymnasium as gym,isaaclab_tasks
          from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
          cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
          env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda()
          m=AuthorityIsolatedWideCritic(o.shape[-1],12).cuda();m.load_state_dict(torch.load(CK,map_location="cuda",weights_only=False)["model"]);m.eval()
          samples=[];raw={}
          for suite in range(4):
            seed=840001+suite
            rows=run_center(env,m,seed);raw[str(suite)]=rows
            for lane in range(NENV):
              G=np.array([r["G"][lane] for r in rows])
              sample={"suite":suite,"lane":lane,"label_correct":lab[(suite,lane)]["label_correct"],
                      "delta_S_reward":lab[(suite,lane)]["delta_S_reward"],
                      "G_t0":float(G[0]),"G_t0_3_mean":float(G[:4].mean()),"G_t0_7_mean":float(G[:8].mean()),
                      "G_t0_11_mean":float(G[:12].mean()),"G_t0_15_mean":float(G[:16].mean()),
                      "G_t0_3_negfrac":float(np.mean(G[:4]<0)),"G_t0_7_negfrac":float(np.mean(G[:8]<0)),
                      "G_t0_11_negfrac":float(np.mean(G[:12]<0)),"G_t0_15_negfrac":float(np.mean(G[:16]<0)),
                      "G_series":G.tolist()}
              # coordinate contribution summaries over source/early windows
              C=np.array([[r["contrib"][lane][j] for j in range(12)] for r in rows])
              sample["coord_G_t0_3"]=C[:4].mean(0).tolist()
              sample["coord_G_t0_7"]=C[:8].mean(0).tolist()
              samples.append(sample)
            print("SUITE",suite,"mean G0-7",np.mean([x["G_t0_7_mean"] for x in samples if x["suite"]==suite]),flush=True)
    
          y=np.array([x["label_correct"] for x in samples])
          summaries={}
          for key in ("G_t0","G_t0_3_mean","G_t0_7_mean","G_t0_11_mean","G_t0_15_mean",
                      "G_t0_3_negfrac","G_t0_7_negfrac","G_t0_11_negfrac","G_t0_15_negfrac"):
            v=np.array([x[key] for x in samples],float)
            summaries[key]={"correct":stats(v[y==1]),"wrong":stats(v[y==0]),"auc_correct":auc(y,v),
                            "mean_difference_correct_minus_wrong":float(v[y==1].mean()-v[y==0].mean())}
          # leave-one-suite-out sign threshold G>0, no learned parameters
          loso={}
          for key in ("G_t0","G_t0_3_mean","G_t0_7_mean","G_t0_11_mean","G_t0_15_mean"):
            folds=[]
            for suite in range(4):
              idx=np.array([x["suite"]==suite for x in samples])
              pred=np.array([x[key]>0 for x in samples])[idx].astype(int);yy=y[idx]
              rec=[]
              for c in (0,1):
                mm=yy==c
                if mm.any():rec.append(float(np.mean(pred[mm]==c)))
              folds.append({"suite":suite,"balanced_accuracy":float(np.mean(rec)),"accuracy":float(np.mean(pred==yy))})
            loso[key]={"folds":folds,"mean_balanced_accuracy":float(np.mean([f["balanced_accuracy"] for f in folds]))}
          # continuous trajectory-level relation
          try:
            from scipy.stats import spearmanr
            cont={}
            target=np.array([x["delta_S_reward"] for x in samples],float)
            for key in ("G_t0","G_t0_3_mean","G_t0_7_mean","G_t0_11_mean","G_t0_15_mean"):
              rho,p=spearmanr(np.array([x[key] for x in samples]),target)
              cont[key]={"spearman":float(rho),"p":float(p)}
          except Exception as e:cont={"error":str(e)}
    
          # coordinate recurrence: correct-wrong delta in early G contribution
          coord={}
          names=['FL_hip','FR_hip','RL_hip','RR_hip','FL_thigh','FR_thigh','RL_thigh','RR_thigh','FL_calf','FR_calf','RL_calf','RR_calf']
          for win in ("coord_G_t0_3","coord_G_t0_7"):
            A=np.array([x[win] for x in samples],float)
            d=A[y==1].mean(0)-A[y==0].mean(0)
            coord[win]={"correct_minus_wrong":{names[i]:float(d[i]) for i in range(12)},
                        "top_abs":[names[i] for i in np.argsort(-np.abs(d))[:6]]}
    
          rep={"schema":"s_local_simplex_gradient_alignment_v1",
               "direction":{"w_center":WC.tolist(),"w_S":WS.tolist(),"d_center_to_S":DS.tolist()},
               "reward_gradient":"-0.02*(a_t-a_{t-1}) for weighted action_rate_l2",
               "state_source":"center-policy trajectory only; same previous center action",
               "n_samples":len(samples),"samples":samples,"summaries":summaries,"sign_rule_loso":loso,
               "continuous_relation":cont,"coordinate_summary":coord}
          (OUT/"s_local_gradient_alignment.json").write_text(json.dumps(rep,indent=2)+"\n")
          for k,q in summaries.items():
            print(k,"corr_mean",q["correct"]["mean"],"wrong_mean",q["wrong"]["mean"],"AUC",q["auc_correct"],flush=True)
          print("LOSO",json.dumps({k:v["mean_balanced_accuracy"] for k,v in loso.items()}),flush=True)
          print("CONT",json.dumps(cont),flush=True)
          print("COORD",json.dumps(coord),flush=True)
        finally:
          if env is not None:env.close()
          app.close()
    if True:main()

def run_authority_isolated_s_phase_local_h16_sensitivity():
    """Run former authority_isolated_s_phase_local_h16_sensitivity.py stage."""
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from pathlib import Path
    import json,sys,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"scripts/rl")]
    from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
    import rl.experiments.common.utilities.authority_isolated_h1_screen as h1
    
    CK=ROOT/"runs/authority_isolated_simplex_edge_retain-2026-09-25/model_30.pt"
    LABELS=ROOT/"runs/authority_isolated_s_sign_context_audit-2026-09-25/s_sign_context_audit.json"
    OUT=ROOT/"runs/authority_isolated_s_phase_local_h16_sensitivity-2026-09-25";OUT.mkdir(parents=True,exist_ok=True)
    NENV=8;H=16
    PHASES=(0,4,8,12,16)
    EPS=(.005,.01,.02)
    WC=np.array(h1.PREFS["C"],np.float32)
    DS=np.array(h1.PREFS["S"]-h1.PREFS["C"],np.float32)
    
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    
    def branch(env,m,seed,phase,w_np):
        wc=torch.tensor(WC,device="cuda").repeat(NENV,1)
        wb=torch.tensor(w_np,device="cuda").repeat(NENV,1)
        obs,_=env.reset(seed=seed);obs=ot(obs).cuda()
        prev=torch.zeros((NENV,12),device="cuda")
        with torch.no_grad():
            for _ in range(phase):
                a=m.act_inference_with_preference(obs,wc)
                nxt,_,_,_,_=env.step(a);prev=a;obs=ot(nxt).cuda()
            source_obs=obs.detach().cpu().numpy().copy()
            source_prev=prev.detach().cpu().numpy().copy()
            rr=[]
            for _ in range(H):
                a=m.act_inference_with_preference(obs,wb)
                da=a-prev
                rr.append((-.01*(da*da).sum(-1)).cpu().numpy())
                nxt,_,_,_,_=env.step(a);prev=a;obs=ot(nxt).cuda()
        return np.stack(rr),source_obs,source_prev
    
    def auc(y,s):
        y=np.asarray(y);s=np.asarray(s,float);p=s[y==1];n=s[y==0]
        if len(p)==0 or len(n)==0:return None
        return float(np.mean([(a>b)+.5*(a==b) for a in p for b in n]))
    
    def balacc(y,p):
        vals=[]
        for c in (0,1):
            m=y==c
            if m.any():vals.append(float(np.mean(p[m]==c)))
        return float(np.mean(vals))
    
    def spear(a,b):
        from scipy.stats import spearmanr
        r,p=spearmanr(a,b)
        return {"rho":float(r) if np.isfinite(r) else 0.0,"p":float(p) if np.isfinite(p) else 1.0}
    
    def summarize(meta,d):
        y=np.array([x["label_correct"] for x in meta],int)
        target=np.array([x["delta_S_reward"] for x in meta],float)
        pred=(d>0).astype(int)
        folds=[]
        for suite in range(4):
            idx=np.array([x["suite"]==suite for x in meta])
            folds.append({"heldout_suite":suite,
              "balanced_accuracy":balacc(y[idx],pred[idx]),
              "accuracy":float(np.mean(y[idx]==pred[idx])),
              "sign_agreement":float(np.mean(np.sign(d[idx])==np.sign(target[idx]))),
              "positive_fraction_pred":float(np.mean(pred[idx])),
              "positive_fraction_true":float(np.mean(y[idx]))})
        return {"auc_correct":auc(y,d),"spearman_full_delta":spear(d,target),
          "global_sign_agreement":float(np.mean(np.sign(d)==np.sign(target))),
          "global_balanced_accuracy":balacc(y,pred),
          "mean_loso_balanced_accuracy":float(np.mean([f["balanced_accuracy"] for f in folds])),
          "folds":folds,
          "correct_mean":float(d[y==1].mean()),"wrong_mean":float(d[y==0].mean()),
          "correct_median":float(np.median(d[y==1])),"wrong_median":float(np.median(d[y==0]))}
    
    def main():
        labels=json.load(open(LABELS))["samples"];lab={(x["suite"],x["lane"]):x for x in labels}
        from isaaclab.app import AppLauncher
        sv=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv;env=None
        try:
          import gymnasium as gym,isaaclab_tasks
          from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
          cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
          env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda()
          m=AuthorityIsolatedWideCritic(o.shape[-1],12).cuda();m.load_state_dict(torch.load(CK,map_location="cuda",weights_only=False)["model"]);m.eval()
          meta=[];vals={str(p):{str(e):[] for e in EPS} for p in PHASES}
          source_checks={str(p):[] for p in PHASES}
          for suite in range(4):
            seed=840001+suite
            for lane in range(NENV):
                meta.append({"suite":suite,"lane":lane,"label_correct":lab[(suite,lane)]["label_correct"],
                             "delta_S_reward":lab[(suite,lane)]["delta_S_reward"]})
            for phase in PHASES:
              ref_obs=None;ref_prev=None
              for e in EPS:
                rp,op,pp=branch(env,m,seed,phase,WC+e*DS)
                rm,om,pm=branch(env,m,seed,phase,WC-e*DS)
                # exact source-state/history equality across +/- branch
                mm=max(float(np.max(np.abs(op-om))),float(np.max(np.abs(pp-pm))))
                if ref_obs is None:ref_obs=op;ref_prev=pp
                else:mm=max(mm,float(np.max(np.abs(ref_obs-op))),float(np.max(np.abs(ref_prev-pp))))
                source_checks[str(phase)].append({"suite":suite,"epsilon":e,"max_source_mismatch":mm})
                d=(rp.sum(0)-rm.sum(0))/(2*e)
                vals[str(phase)][str(e)].extend(d.tolist())
              print("SUITE",suite,"PHASE",phase,"D005_mean",float(np.mean(vals[str(phase)]["0.005"][-NENV:])),flush=True)
          summaries={};convergence={}
          base_eps=str(EPS[0])
          for phase in PHASES:
            ps=str(phase);d0=np.array(vals[ps][base_eps],float)
            summaries[ps]=summarize(meta,d0)
            convergence[ps]={}
            for e in EPS[1:]:
              de=np.array(vals[ps][str(e)],float)
              convergence[ps][str(e)]={"pearson_vs_eps005":float(np.corrcoef(d0,de)[0,1]) if np.std(d0)>1e-12 and np.std(de)>1e-12 else 0.0,
                "sign_agreement_vs_eps005":float(np.mean(np.sign(d0)==np.sign(de))),
                "median_relative_abs_diff":float(np.median(np.abs(d0-de)/(np.abs(d0)+1e-8)))}
          max_mismatch=max(q["max_source_mismatch"] for v in source_checks.values() for q in v)
          rep={"schema":"s_phase_local_h16_sensitivity_v1","checkpoint":str(CK.relative_to(ROOT)),
            "source_phases":list(PHASES),"horizon":H,"epsilons":list(EPS),"primary_epsilon":EPS[0],
            "source_contract":"center replay to phase; identical reset/command/previous-action history; branch only after source",
            "max_source_state_history_mismatch":max_mismatch,"source_checks":source_checks,
            "meta":meta,"values":vals,"summaries":summaries,"epsilon_convergence":convergence}
          (OUT/"phase_local_h16_sensitivity.json").write_text(json.dumps(rep,indent=2)+"\n")
          for p in PHASES:
            q=summaries[str(p)]
            print("PHASE_RESULT",p,"AUC",q["auc_correct"],"rho",q["spearman_full_delta"],
              "sign",q["global_sign_agreement"],"LOSO_BA",q["mean_loso_balanced_accuracy"],
              "folds",[round(x["balanced_accuracy"],3) for x in q["folds"]],
              "eps",convergence[str(p)],flush=True)
          print("MAX_SOURCE_MISMATCH",max_mismatch,flush=True)
        finally:
          if env is not None:env.close()
          app.close()
    if True:main()

def run_authority_isolated_s_semantic_alignment_audit():
    """Run former authority_isolated_s_semantic_alignment_audit.py stage."""
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from pathlib import Path
    import json,sys,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"scripts/rl")]
    from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
    from talon_rl.rewards.objectives import normalized_objective_vector
    import rl.experiments.common.utilities.authority_isolated_h1_screen as h1
    
    CK=ROOT/"runs/authority_isolated_simplex_edge_retain-2026-09-25/model_30.pt"
    OUT=ROOT/"runs/authority_isolated_s_semantic_alignment_audit-2026-09-25";OUT.mkdir(parents=True,exist_ok=True)
    NENV=8;H=64
    PREFS=h1.PREFS
    POINTS={
     "C":PREFS["C"],
     "S":PREFS["S"],
     "TS50":.5*PREFS["T"]+.5*PREFS["S"],
     "TS75":.25*PREFS["T"]+.75*PREFS["S"],
     "AS50":.5*PREFS["A"]+.5*PREFS["S"],
     "AS75":.25*PREFS["A"]+.75*PREFS["S"],
     "OS50":.5*PREFS["O"]+.5*PREFS["S"],
     "OS75":.25*PREFS["O"]+.75*PREFS["S"],
    }
    JOINTS=['FL_hip','FR_hip','RL_hip','RR_hip','FL_thigh','FR_thigh','RL_thigh','RR_thigh','FL_calf','FR_calf','RL_calf','RR_calf']
    
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    
    def run(env,m,mgr,w_np,seed):
        w=torch.tensor(w_np,device="cuda").repeat(NENV,1)
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        prev=torch.zeros((NENV,12),device="cuda");prev2=prev.clone()
        done_any=np.zeros(NENV,bool)
        agg={k:[] for k in ["reward_s_weighted","reward_s_raw_proxy","l2","l2sq","l1","maxcoord","jerk_l2","dof_vel","dof_acc","torque"]}
        coord_sq=[];coord_abs=[]
        reward_terms=[]
        with torch.no_grad():
          for t in range(H):
            a=m.act_inference_with_preference(cur,w)
            da=a-prev
            dda=(a-prev)-(prev-prev2)
            nxt,_,te,tr,_=env.step(a)
            raw=mgr._step_reward.detach().clone();names=list(mgr.active_terms);td=terms(raw,names)
            # weighted reward term from manager is exactly what objective code consumes
            sweighted=td["action_rate_l2"].cpu().numpy()
            agg["reward_s_weighted"].append(sweighted)
            # exact unweighted action_rate_l2 kernel reconstructed from applied actions
            sq=(da*da).sum(-1)
            agg["reward_s_raw_proxy"].append(sq.cpu().numpy())
            agg["l2"].append(torch.linalg.vector_norm(da,dim=-1).cpu().numpy())
            agg["l2sq"].append(sq.cpu().numpy())
            agg["l1"].append(da.abs().sum(-1).cpu().numpy())
            agg["maxcoord"].append(da.abs().max(-1).values.cpu().numpy())
            agg["jerk_l2"].append(torch.linalg.vector_norm(dda,dim=-1).cpu().numpy())
            robot=env.unwrapped.scene["robot"].data
            agg["dof_vel"].append(torch.linalg.vector_norm(robot.joint_vel,dim=-1).cpu().numpy())
            agg["dof_acc"].append(torch.linalg.vector_norm(robot.joint_acc,dim=-1).cpu().numpy())
            agg["torque"].append(torch.linalg.vector_norm(robot.applied_torque,dim=-1).cpu().numpy())
            coord_sq.append((da*da).mean(0).cpu().numpy());coord_abs.append(da.abs().mean(0).cpu().numpy())
            reward_terms.append({k:float(v.mean().cpu()) for k,v in td.items() if k in ("action_rate_l2","dof_acc_l2","dof_torques_l2","lin_vel_z_l2","feet_air_time")})
            dd=(te|tr).cpu().numpy();done_any|=dd
            prev2=prev;prev=a;cur=ot(nxt).cuda()
        out={k:float(np.mean(v)) for k,v in agg.items()}
        out["survival"]=float(1-done_any.mean())
        out["coord_sq_mean"]=np.mean(coord_sq,0).tolist()
        out["coord_abs_mean"]=np.mean(coord_abs,0).tolist()
        # exact consistency: weighted term / (-.01) vs reconstructed squared action rate
        out["reward_kernel_consistency_ratio"]=float((-out["reward_s_weighted"]/0.01)/(out["l2sq"]+1e-12))
        out["reward_term_means"]={k:float(np.mean([x[k] for x in reward_terms])) for k in reward_terms[0]}
        return out
    
    def main():
        from isaaclab.app import AppLauncher
        sv=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv;env=None
        try:
          import gymnasium as gym,isaaclab_tasks
          from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
          cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
          env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda()
          m=AuthorityIsolatedWideCritic(o.shape[-1],12).cuda();m.load_state_dict(torch.load(CK,map_location="cuda",weights_only=False)["model"]);m.eval()
          mgr=env.unwrapped.reward_manager
          rep={"schema":"s_semantic_alignment_v1","definition":{
            "S_objective_terms":["action_rate_l2"],"action_rate_weight":-0.01,
            "evaluator_proxy":"mean ||a_t-a_{t-1}||_2",
            "training_kernel":"sum_j (a_tj-a_{t-1,j})^2"},"suites":[]}
          for suite in range(4):
            seed=840001+suite
            rows={}
            for name,w in POINTS.items(): rows[name]=run(env,m,mgr,w,seed)
            base=rows["C"]
            delta={}
            for name,q in rows.items():
              if name=="C":continue
              delta[name]={k:q[k]-base[k] for k in ("reward_s_weighted","l2","l2sq","l1","maxcoord","jerk_l2","dof_vel","dof_acc","torque")}
              delta[name]["coord_sq_delta"]=(np.array(q["coord_sq_mean"])-np.array(base["coord_sq_mean"])).tolist()
            rep["suites"].append({"suite":suite,"seed":seed,"rows":rows,"delta_vs_C":delta})
            print("SUITE",suite,"S-C",json.dumps(delta["S"]),flush=True)
          # aggregate S vs C and joint contributions
          keys=("reward_s_weighted","l2","l2sq","l1","maxcoord","jerk_l2","dof_vel","dof_acc","torque")
          agg={}
          for k in keys:
            ds=[x["delta_vs_C"]["S"][k] for x in rep["suites"]]
            agg[k]={"mean_delta":float(np.mean(ds)),"correct_fraction":float(np.mean(np.array(ds)>0)) if k=="reward_s_weighted" else float(np.mean(np.array(ds)<0))}
          jd=np.mean([np.array(x["delta_vs_C"]["S"]["coord_sq_delta"]) for x in rep["suites"]],0)
          agg["joint_sq_delta_mean"]={JOINTS[i]:float(jd[i]) for i in range(12)}
          agg["joint_sq_top_abs"]=[JOINTS[i] for i in np.argsort(-np.abs(jd))[:6]]
          # neighbor monotonic tendency toward S: compare distance from C as S weight increases .5 -> .75 -> .7 heavy endpoint not linear ordering globally,
          # report raw rows rather than force a monotonic claim.
          rep["aggregate_S_vs_C"]=agg
          (OUT/"s_semantic_alignment.json").write_text(json.dumps(rep,indent=2)+"\n")
          print("AGG",json.dumps(agg,indent=2),flush=True)
        finally:
          if env is not None:env.close()
          app.close()
    if True:main()

def run_authority_isolated_s_short_horizon_epsilon_convergence():
    """Run former authority_isolated_s_short_horizon_epsilon_convergence.py stage."""
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from pathlib import Path
    import json,sys,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"scripts/rl")]
    from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
    import rl.experiments.common.utilities.authority_isolated_h1_screen as h1
    CK=ROOT/"runs/authority_isolated_simplex_edge_retain-2026-09-25/model_30.pt"
    LABELS=ROOT/"runs/authority_isolated_s_sign_context_audit-2026-09-25/s_sign_context_audit.json"
    OUT=ROOT/"runs/authority_isolated_s_short_horizon_epsilon_convergence-2026-09-25";OUT.mkdir(parents=True,exist_ok=True)
    NENV=8;HMAX=32;HORIZONS=(8,16,32);EPS=(.005,.01,.02,.05)
    WC=np.array(h1.PREFS["C"],np.float32);DS=np.array(h1.PREFS["S"]-h1.PREFS["C"],np.float32)
    def ot(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def rollout(env,m,seed,w_np):
     w=torch.tensor(w_np,device="cuda",dtype=torch.float32).repeat(NENV,1);obs,_=env.reset(seed=seed);obs=ot(obs).cuda()
     prev=torch.zeros((NENV,12),device="cuda");rr=[]
     with torch.no_grad():
      for _ in range(HMAX):
       a=m.act_inference_with_preference(obs,w);da=a-prev;rr.append((-.01*(da*da).sum(-1)).cpu().numpy())
       nxt,_,_,_,_=env.step(a);prev=a;obs=ot(nxt).cuda()
     return np.stack(rr)
    def auc(y,s):
     y=np.asarray(y);s=np.asarray(s);p=s[y==1];n=s[y==0]
     return float(np.mean([(a>b)+.5*(a==b) for a in p for b in n]))
    def spear(a,b):
     from scipy.stats import spearmanr
     r,p=spearmanr(a,b);return float(r),float(p)
    def main():
     labels=json.load(open(LABELS))["samples"];lab={(x["suite"],x["lane"]):x for x in labels}
     from isaaclab.app import AppLauncher
     sv=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda()
      m=AuthorityIsolatedWideCritic(o.shape[-1],12).cuda();m.load_state_dict(torch.load(CK,map_location="cuda",weights_only=False)["model"]);m.eval()
      vals={str(e):{str(H):[] for H in HORIZONS} for e in EPS};meta=[]
      for suite in range(4):
       seed=840001+suite
       for e in EPS:
        rp=rollout(env,m,seed,WC+e*DS);rm=rollout(env,m,seed,WC-e*DS)
        for H in HORIZONS:
         d=(rp[:H].sum(0)-rm[:H].sum(0))/(2*e);vals[str(e)][str(H)].extend(d.tolist())
       for lane in range(NENV):meta.append({"suite":suite,"lane":lane,"label_correct":lab[(suite,lane)]["label_correct"],"delta_S_reward":lab[(suite,lane)]["delta_S_reward"]})
       print("SUITE",suite,"done",flush=True)
      y=np.array([x["label_correct"] for x in meta]);target=np.array([x["delta_S_reward"] for x in meta])
      summary={}
      for H in HORIZONS:
       summary[str(H)]={}
       for e in EPS:
        d=np.array(vals[str(e)][str(H)])
        r,p=spear(d,target)
        summary[str(H)][str(e)]={"auc":auc(y,d),"spearman":r,"p":p,"sign_agreement":float(np.mean(np.sign(d)==np.sign(target))),
          "mean_correct":float(d[y==1].mean()),"mean_wrong":float(d[y==0].mean())}
       # convergence against smallest epsilon
       base=np.array(vals[str(EPS[0])][str(H)])
       conv={}
       for e in EPS[1:]:
        q=np.array(vals[str(e)][str(H)])
        conv[str(e)]={"pearson_vs_eps005":float(np.corrcoef(base,q)[0,1]),"sign_agreement_vs_eps005":float(np.mean(np.sign(base)==np.sign(q))),
          "median_relative_abs_diff":float(np.median(np.abs(base-q)/(np.abs(base)+1e-8)))}
       summary[str(H)]["convergence"]=conv
      rep={"schema":"s_short_horizon_epsilon_convergence_v1","epsilons":list(EPS),"horizons":list(HORIZONS),"meta":meta,"values":vals,"summary":summary}
      (OUT/"epsilon_convergence.json").write_text(json.dumps(rep,indent=2)+"\n")
      print(json.dumps(summary,indent=2),flush=True)
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_authority_isolated_s_short_horizon_sensitivity_audit():
    """Run former authority_isolated_s_short_horizon_sensitivity_audit.py stage."""
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from pathlib import Path
    import json,sys,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"scripts/rl")]
    from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
    import rl.experiments.common.utilities.authority_isolated_h1_screen as h1
    
    CK=ROOT/"runs/authority_isolated_simplex_edge_retain-2026-09-25/model_30.pt"
    LABELS=ROOT/"runs/authority_isolated_s_sign_context_audit-2026-09-25/s_sign_context_audit.json"
    OUT=ROOT/"runs/authority_isolated_s_short_horizon_sensitivity_audit-2026-09-25";OUT.mkdir(parents=True,exist_ok=True)
    
    NENV=8;HMAX=32
    HORIZONS=(1,4,8,16,32)
    EPS_PRIMARY=.05
    EPS_CONTROL=.02
    WC=np.array(h1.PREFS["C"],np.float32)
    DS=np.array(h1.PREFS["S"]-h1.PREFS["C"],np.float32)
    
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    
    def rollout(env,m,seed,w_np):
        w=torch.tensor(w_np,device="cuda",dtype=torch.float32).repeat(NENV,1)
        obs,_=env.reset(seed=seed);obs=ot(obs).cuda()
        prev=torch.zeros((NENV,12),device="cuda")
        step_reward=[]
        with torch.no_grad():
            for t in range(HMAX):
                a=m.act_inference_with_preference(obs,w)
                da=a-prev
                # exact weighted S kernel: -0.01 sum_j (a_t-a_{t-1})^2
                rs=-.01*(da*da).sum(-1)
                step_reward.append(rs.cpu().numpy())
                nxt,_,_,_,_=env.step(a)
                prev=a;obs=ot(nxt).cuda()
        return np.stack(step_reward,0) # H,E
    
    def auc(y,s):
        y=np.asarray(y);s=np.asarray(s,float);pos=s[y==1];neg=s[y==0]
        if len(pos)==0 or len(neg)==0:return None
        return float(np.mean([(a>b)+.5*(a==b) for a in pos for b in neg]))
    
    def spearman(x,y):
        try:
            from scipy.stats import spearmanr
            r,p=spearmanr(x,y)
            return {"rho":float(r) if np.isfinite(r) else 0.0,"p":float(p) if np.isfinite(p) else 1.0}
        except Exception as e:return {"error":str(e)}
    
    def balacc(y,p):
        vals=[]
        for c in (0,1):
            m=y==c
            if m.any():vals.append(float(np.mean(p[m]==c)))
        return float(np.mean(vals))
    
    def summarize(samples,key):
        y=np.array([x["label_correct"] for x in samples],int)
        full=np.array([x["delta_S_reward"] for x in samples],float)
        d=np.array([x[key] for x in samples],float)
        pred=(d>0).astype(int)
        correct=d[y==1];wrong=d[y==0]
        folds=[]
        for suite in range(4):
            idx=np.array([x["suite"]==suite for x in samples])
            folds.append({"suite":suite,"balanced_accuracy":balacc(y[idx],pred[idx]),
                          "accuracy":float(np.mean(y[idx]==pred[idx])),
                          "sign_agreement":float(np.mean(np.sign(d[idx])==np.sign(full[idx])))})
        return {
          "correct":{"mean":float(correct.mean()),"median":float(np.median(correct)),"positive_fraction":float(np.mean(correct>0))},
          "wrong":{"mean":float(wrong.mean()),"median":float(np.median(wrong)),"negative_fraction":float(np.mean(wrong<0))},
          "auc_correct":auc(y,d),
          "spearman_full_delta":spearman(d,full),
          "global_sign_agreement":float(np.mean(np.sign(d)==np.sign(full))),
          "mean_fold_balanced_accuracy":float(np.mean([f["balanced_accuracy"] for f in folds])),
          "folds":folds,
        }
    
    def main():
        label_rows=json.load(open(LABELS))["samples"]
        lab={(x["suite"],x["lane"]):x for x in label_rows}
        from isaaclab.app import AppLauncher
        sv=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv;env=None
        try:
          import gymnasium as gym,isaaclab_tasks
          from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
          cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
          env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda()
          m=AuthorityIsolatedWideCritic(o.shape[-1],12).cuda();m.load_state_dict(torch.load(CK,map_location="cuda",weights_only=False)["model"]);m.eval()
    
          samples=[]
          raw={}
          for suite in range(4):
            seed=840001+suite
            raw[str(suite)]={}
            sensitivities={}
            for eps in (EPS_PRIMARY,EPS_CONTROL):
              wp=WC+eps*DS;wm=WC-eps*DS
              rp=rollout(env,m,seed,wp);rm=rollout(env,m,seed,wm)
              sens={}
              for H in HORIZONS:
                # cumulative exact weighted S return over first H steps
                jp=rp[:H].sum(0);jm=rm[:H].sum(0)
                sens[H]=(jp-jm)/(2*eps)
              sensitivities[eps]=sens
              raw[str(suite)][str(eps)]={"plus_step_reward":rp.tolist(),"minus_step_reward":rm.tolist()}
            for lane in range(NENV):
              rec={"suite":suite,"seed":seed,"lane":lane,
                   "label_correct":lab[(suite,lane)]["label_correct"],
                   "delta_S_reward":lab[(suite,lane)]["delta_S_reward"]}
              for H in HORIZONS:
                rec[f"D_H{H}_eps{EPS_PRIMARY}"]=float(sensitivities[EPS_PRIMARY][H][lane])
                rec[f"D_H{H}_eps{EPS_CONTROL}"]=float(sensitivities[EPS_CONTROL][H][lane])
              samples.append(rec)
            print("SUITE",suite,"primary", {H:float(np.mean(sensitivities[EPS_PRIMARY][H])) for H in HORIZONS},flush=True)
    
          summaries={}
          consistency={}
          for H in HORIZONS:
            kp=f"D_H{H}_eps{EPS_PRIMARY}";kc=f"D_H{H}_eps{EPS_CONTROL}"
            summaries[str(H)]=summarize(samples,kp)
            a=np.array([x[kp] for x in samples]);b=np.array([x[kc] for x in samples])
            consistency[str(H)]={"pearson":float(np.corrcoef(a,b)[0,1]) if np.std(a)>1e-12 and np.std(b)>1e-12 else 0.0,
                                 "median_relative_abs_diff":float(np.median(np.abs(a-b)/(np.abs(a)+1e-8))),
                                 "sign_agreement":float(np.mean(np.sign(a)==np.sign(b)))}
          rep={"schema":"s_short_horizon_closed_loop_sensitivity_v1",
               "checkpoint":str(CK.relative_to(ROOT)),
               "source_state":"reset t0; matched simulator reset/command/previous-action history",
               "direction":{"w_center":WC.tolist(),"d_center_to_S":DS.tolist()},
               "finite_difference":{"primary_epsilon":EPS_PRIMARY,"control_epsilon":EPS_CONTROL,"central":True},
               "horizons":list(HORIZONS),"n_samples":len(samples),"samples":samples,
               "summaries":summaries,"epsilon_consistency":consistency}
          (OUT/"s_short_horizon_sensitivity.json").write_text(json.dumps(rep,indent=2)+"\n")
          for H in HORIZONS:
            q=summaries[str(H)]
            print("H",H,"AUC",q["auc_correct"],"rho",q["spearman_full_delta"],"sign",q["global_sign_agreement"],
                  "LOSO_BA",q["mean_fold_balanced_accuracy"],
                  "correct",q["correct"],"wrong",q["wrong"],"eps",consistency[str(H)],flush=True)
        finally:
          if env is not None:env.close()
          app.close()
    if True:main()

def run_authority_isolated_s_sign_context_audit():
    """Run former authority_isolated_s_sign_context_audit.py stage."""
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from pathlib import Path
    import json,sys,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"scripts/rl")]
    from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
    import rl.experiments.common.utilities.authority_isolated_h1_screen as h1
    
    CK=ROOT/"runs/authority_isolated_simplex_edge_retain-2026-09-25/model_30.pt"
    OUT=ROOT/"runs/authority_isolated_s_sign_context_audit-2026-09-25";OUT.mkdir(parents=True,exist_ok=True)
    NENV=8;H=64;PREFS=h1.PREFS
    JOINTS=['FL_hip','FR_hip','RL_hip','RR_hip','FL_thigh','FR_thigh','RL_thigh','RR_thigh','FL_calf','FR_calf','RL_calf','RR_calf']
    LEG_IDX=[[0,4,8],[1,5,9],[2,6,10],[3,7,11]]
    
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    
    def contact(env):
        try:
            f=env.unwrapped.scene.sensors["contact_forces"].data.net_forces_w
            if f.ndim==4:f=f[:,-1]
            c=(torch.linalg.vector_norm(f,dim=-1)>1.0).float()
            return c.mean(-1)
        except:
            return torch.zeros(NENV,device="cuda")
    
    def feat(env,obs,prev_action=None):
        d=env.unwrapped.scene["robot"].data
        pg=obs[:,6:9];cmd=obs[:,9:12];jv=d.joint_vel
        pa=obs[:,36:48] if prev_action is None else prev_action
        legv=torch.stack([torch.linalg.vector_norm(jv[:,ix],dim=1) for ix in LEG_IDX],1)
        lega=torch.stack([torch.linalg.vector_norm(pa[:,ix],dim=1) for ix in LEG_IDX],1)
        F={
          "cmd_x":cmd[:,0],"cmd_y":cmd[:,1],"cmd_yaw":cmd[:,2],"cmd_speed":torch.linalg.vector_norm(cmd[:,:2],dim=1),
          "pg_x":pg[:,0],"pg_y":pg[:,1],"pg_xy":torch.linalg.vector_norm(pg[:,:2],dim=1),
          "base_lin_x":d.root_lin_vel_b[:,0],"base_lin_y":d.root_lin_vel_b[:,1],"base_lin_z":d.root_lin_vel_b[:,2],
          "base_lin_xy":torch.linalg.vector_norm(d.root_lin_vel_b[:,:2],dim=1),
          "base_ang_x":d.root_ang_vel_b[:,0],"base_ang_y":d.root_ang_vel_b[:,1],"base_ang_z":d.root_ang_vel_b[:,2],
          "base_ang_xy":torch.linalg.vector_norm(d.root_ang_vel_b[:,:2],dim=1),
          "joint_vel_norm":torch.linalg.vector_norm(jv,dim=1),"joint_vel_std":jv.std(dim=1),
          "leg_vel_std":legv.std(dim=1),"leg_vel_lr":(legv[:,0]+legv[:,2])-(legv[:,1]+legv[:,3]),
          "leg_vel_diag":(legv[:,0]+legv[:,3])-(legv[:,1]+legv[:,2]),
          "prev_action_norm":torch.linalg.vector_norm(pa,dim=1),"prev_action_std":pa.std(dim=1),
          "leg_action_std":lega.std(dim=1),"leg_action_lr":(lega[:,0]+lega[:,2])-(lega[:,1]+lega[:,3]),
          "leg_action_diag":(lega[:,0]+lega[:,3])-(lega[:,1]+lega[:,2]),
          "contact_frac":contact(env),
        }
        return {k:v.detach().cpu().numpy() for k,v in F.items()}
    
    def run(env,m,w_np,seed,collect_features=False):
        w=torch.tensor(w_np,device="cuda").repeat(NENV,1)
        obs,_=env.reset(seed=seed);obs=ot(obs).cuda()
        source=feat(env,obs)
        early=[];reward=np.zeros(NENV,np.float64);l2sq=np.zeros(NENV,np.float64)
        prev=torch.zeros((NENV,12),device="cuda");states=[];acts=[]
        with torch.no_grad():
          for t in range(H):
            if collect_features and t<4: early.append(feat(env,obs,prev))
            a=m.act_inference_with_preference(obs,w);da=a-prev
            if collect_features and t<12:
                states.append(obs.detach().cpu().numpy());acts.append(a.detach().cpu().numpy())
            nxt,_,_,_,_=env.step(a)
            names=list(env.unwrapped.reward_manager.active_terms);raw=env.unwrapped.reward_manager._step_reward.detach()
            idx=names.index("action_rate_l2")
            reward+=raw[:,idx].cpu().numpy()
            l2sq+=(da*da).sum(-1).cpu().numpy()
            prev=a;obs=ot(nxt).cuda()
        early_mean={}
        if collect_features:
          for k in source:
            early_mean[k]=np.mean([q[k] for q in early],axis=0)
        return {"source":source,"early":early_mean,"reward":reward/H,"l2sq":l2sq/H,
                "states":states,"actions":acts}
    
    def logistic_loso(X,y,groups):
        try:
          from sklearn.linear_model import LogisticRegression
          from sklearn.preprocessing import StandardScaler
          from sklearn.metrics import balanced_accuracy_score,roc_auc_score
        except Exception as e:
          return {"available":False,"error":str(e)}
        preds=[];probs=[];ys=[];folds=[]
        for g in sorted(set(groups)):
          tr=groups!=g;te=groups==g
          if len(np.unique(y[tr]))<2:
            folds.append({"suite":int(g),"status":"train_single_class"});continue
          sc=StandardScaler().fit(X[tr]);clf=LogisticRegression(C=1.0,max_iter=2000,class_weight="balanced").fit(sc.transform(X[tr]),y[tr])
          p=clf.predict_proba(sc.transform(X[te]))[:,1];pr=(p>=.5).astype(int)
          ba=float(balanced_accuracy_score(y[te],pr)) if len(np.unique(y[te]))>1 else float(np.mean(pr==y[te]))
          folds.append({"suite":int(g),"balanced_accuracy":ba,"n":int(te.sum()),"positive_fraction":float(y[te].mean())})
          preds.extend(pr.tolist());probs.extend(p.tolist());ys.extend(y[te].tolist())
        out={"available":True,"folds":folds,"overall_accuracy":float(np.mean(np.array(preds)==np.array(ys))) if ys else None,
             "mean_fold_balanced_accuracy":float(np.mean([f["balanced_accuracy"] for f in folds if "balanced_accuracy" in f]))}
        if ys and len(set(ys))>1:
          out["pooled_auc"]=float(roc_auc_score(ys,probs))
        return out
    
    def main():
        from isaaclab.app import AppLauncher
        sv=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv;env=None
        try:
          import gymnasium as gym,isaaclab_tasks
          from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
          cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
          env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda()
          m=AuthorityIsolatedWideCritic(o.shape[-1],12).cuda();m.load_state_dict(torch.load(CK,map_location="cuda",weights_only=False)["model"]);m.eval()
          samples=[];traj={}
          for suite in range(4):
            seed=840001+suite
            C=run(env,m,PREFS["C"],seed,True);S=run(env,m,PREFS["S"],seed,True)
            for lane in range(NENV):
              dr=float(S["reward"][lane]-C["reward"][lane])
              dl=float(S["l2sq"][lane]-C["l2sq"][lane])
              rec={"suite":suite,"seed":seed,"lane":lane,"delta_S_reward":dr,"delta_l2sq":dl,"label_correct":int(dr>0),
                   "source":{k:float(v[lane]) for k,v in C["source"].items()},
                   "control_early":{k:float(v[lane]) for k,v in C["early"].items()}}
              samples.append(rec)
            traj[str(suite)]={"C_states":[x.tolist() for x in C["states"]],"S_states":[x.tolist() for x in S["states"]],
                              "C_actions":[x.tolist() for x in C["actions"]],"S_actions":[x.tolist() for x in S["actions"]]}
            print("SUITE",suite,"labels",[x["label_correct"] for x in samples if x["suite"]==suite],
                  "mean_dS",np.mean([x["delta_S_reward"] for x in samples if x["suite"]==suite]),flush=True)
          names=list(samples[0]["source"])
          families={
            "command":["cmd_x","cmd_y","cmd_yaw","cmd_speed"],
            "orientation":["pg_x","pg_y","pg_xy","base_ang_x","base_ang_y","base_ang_z","base_ang_xy"],
            "linear_velocity":["base_lin_x","base_lin_y","base_lin_z","base_lin_xy"],
            "joint_velocity":["joint_vel_norm","joint_vel_std","leg_vel_std","leg_vel_lr","leg_vel_diag"],
            "action_context":["prev_action_norm","prev_action_std","leg_action_std","leg_action_lr","leg_action_diag"],
            "contact":["contact_frac"],
            "all":names,
          }
          y=np.array([x["label_correct"] for x in samples]);groups=np.array([x["suite"] for x in samples])
          results={}
          for window in ("source","control_early"):
            results[window]={}
            for fam,fs in families.items():
              X=np.array([[x[window][k] for k in fs] for x in samples],float)
              results[window][fam]={"features":fs,"loso":logistic_loso(X,y,groups)}
          # univariate rank correlations with continuous delta reward
          try:
            from scipy.stats import spearmanr
            univ={}
            target=np.array([x["delta_S_reward"] for x in samples])
            for window in ("source","control_early"):
              univ[window]={}
              for k in names:
                v=np.array([x[window][k] for x in samples]);rho,p=spearmanr(v,target)
                univ[window][k]={"spearman":float(rho) if np.isfinite(rho) else 0.0,"p":float(p) if np.isfinite(p) else 1.0}
          except Exception as e: univ={"error":str(e)}
          rep={"schema":"s_sign_context_audit_v1","n_samples":len(samples),"positive_fraction":float(y.mean()),
               "samples":samples,"feature_families":results,"univariate":univ}
          (OUT/"s_sign_context_audit.json").write_text(json.dumps(rep,indent=2)+"\n")
          # compact print
          for window in results:
            print("\nWINDOW",window)
            for fam,q in results[window].items():
              z=q["loso"];print(fam,"BA",z.get("mean_fold_balanced_accuracy"),"AUC",z.get("pooled_auc"),"ACC",z.get("overall_accuracy"))
          if "error" not in univ:
            for window in univ:
              top=sorted(univ[window].items(),key=lambda kv:-abs(kv[1]["spearman"]))[:8]
              print("TOP",window,[(k,round(v["spearman"],3),round(v["p"],3)) for k,v in top],flush=True)
        finally:
          if env is not None:env.close()
          app.close()
    if True:main()

def run_authority_isolated_s_trajectory_temporal_decomposition():
    """Run former authority_isolated_s_trajectory_temporal_decomposition.py stage."""
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from pathlib import Path
    import json,sys,hashlib,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"scripts/rl")]
    from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
    import rl.experiments.common.utilities.authority_isolated_h1_screen as h1
    CK=ROOT/"runs/authority_isolated_simplex_edge_retain-2026-09-25/model_30.pt"
    OUT=ROOT/"runs/authority_isolated_s_trajectory_temporal_decomposition-2026-09-25";OUT.mkdir(parents=True,exist_ok=True)
    CONTRACT=ROOT/"docs/contracts/authority/authority-isolated-s-trajectory-temporal-decomposition-contract.md"
    NENV,H=8,64; CPS=(4,8,16,32,64); PREFS=h1.PREFS
    
    def ot(x):
        if isinstance(x,dict): x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def contact(env):
        try:
            f=env.unwrapped.scene.sensors["contact_forces"].data.net_forces_w
            if f.ndim==4:f=f[:,-1]
            return (torch.linalg.vector_norm(f,dim=-1)>1.0).float().mean(-1)
        except Exception:return torch.zeros(NENV,device="cuda")
    def rollout(env,m,w_np,seed):
        w=torch.tensor(w_np,device="cuda").repeat(NENV,1);obs,_=env.reset(seed=seed);obs=ot(obs).cuda()
        prev=torch.zeros((NENV,env.unwrapped.action_manager.total_action_dim),device="cuda")
        R=[];A=[];AR=[];S=[];C=[];D=[];first=np.full(NENV,-1,int)
        with torch.no_grad():
            for t in range(H):
                s=obs.clone();a=m.act_inference_with_preference(obs,w);da=a-prev
                R.append((-.01*(da*da).sum(-1)).cpu().numpy());A.append(a.cpu().numpy())
                AR.append(torch.linalg.vector_norm(da,dim=-1).cpu().numpy());S.append(s.cpu().numpy());C.append(contact(env).cpu().numpy())
                nxt,_,te,tr,_=env.step(a);d=(te|tr).cpu().numpy();D.append(d)
                for i in np.where((first<0)&d)[0]:first[i]=t
                prev=a;obs=ot(nxt).cuda()
        return {k:np.stack(v) for k,v in {"r":R,"a":A,"ar":AR,"s":S,"c":C,"d":D}.items()}|{"first":first}
    def sgn(x,t):return 1 if x>t else (-1 if x<-t else 0)
    def classify(cum):
        tau=max(1e-6,.10*float(np.max(np.abs(cum))));v={h:float(cum[h-1]) for h in CPS};q={h:sgn(v[h],tau) for h in CPS}
        if q[64]<0 and q[8]<0 and all(q[h]<=0 for h in (16,32,64)):cl="A"
        elif q[64]<0 and any(q[h]>0 for h in (4,8,16)):cl="B"
        elif q[64]>0 and q[4]<=0 and q[8]<=0 and any(q[h]>0 for h in (16,32,64)):cl="C"
        else:cl="D"
        nz=[q[h] for h in CPS if q[h]!=0];trans=sum(nz[i]!=nz[i-1] for i in range(1,len(nz)))
        first=next((i+1 for i,x in enumerate(cum) if abs(x)>tau),None)
        return cl,tau,v,q,trans,first
    def main():
        from isaaclab.app import AppLauncher
        sv=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv;env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim
            m=AuthorityIsolatedWideCritic(o.shape[-1],ad).cuda();m.load_state_dict(torch.load(CK,map_location="cuda",weights_only=False)["model"]);m.eval()
            samples=[];traces={}
            for suite in range(4):
                seed=840001+suite;C=rollout(env,m,PREFS["C"],seed);S=rollout(env,m,PREFS["S"],seed);traces[str(suite)]={"lanes":{}}
                for lane in range(NENV):
                    dr=S["r"][:,lane]-C["r"][:,lane];cum=np.cumsum(dr);dar=S["ar"][:,lane]-C["ar"][:,lane]
                    adiv=np.linalg.norm(S["a"][:,lane]-C["a"][:,lane],axis=-1);sdiv=np.linalg.norm(S["s"][:,lane]-C["s"][:,lane],axis=-1)
                    cl,tau,v,q,ntr,first=classify(cum);final=float(cum[-1])
                    samples.append({"suite":suite,"seed":seed,"lane":lane,"final_delta_J_S":final,"final_correct":bool(final>0),"class":cl,"deadband":tau,
                      "checkpoint_cumulative_delta":{str(k):x for k,x in v.items()},"checkpoint_sign":{str(k):int(x) for k,x in q.items()},
                      "meaningful_checkpoint_sign_transitions":ntr,"first_meaningful_cumulative_step":first,
                      "C_first_termination_step":int(C["first"][lane]),"S_first_termination_step":int(S["first"][lane])})
                    traces[str(suite)]["lanes"][str(lane)]={"delta_r_S":dr.tolist(),"cumulative_delta_J_S":cum.tolist(),"delta_action_rate":dar.tolist(),
                      "action_separation_l2":adiv.tolist(),"state_separation_l2":sdiv.tolist(),"C_contact":C["c"][:,lane].tolist(),"S_contact":S["c"][:,lane].tolist(),
                      "C_terminated":C["d"][:,lane].astype(int).tolist(),"S_terminated":S["d"][:,lane].astype(int).tolist()}
                print("SUITE",suite,"classes",[x["class"] for x in samples if x["suite"]==suite],flush=True)
            classes=("A","B","C","D");wrong=[x for x in samples if not x["final_correct"]];correct=[x for x in samples if x["final_correct"]]
            cnt=lambda g:{c:sum(x["class"]==c for x in g) for c in classes}
            per={}
            for s in range(4):
                ss=[x for x in samples if x["suite"]==s];ww=[x for x in ss if not x["final_correct"]]
                per[str(s)]={"n":len(ss),"wrong_n":len(ww),"class_counts":cnt(ss),"wrong_class_counts":cnt(ww)}
            frac=lambda c:(sum(x["class"]==c for x in wrong)/len(wrong)) if wrong else 0.0
            support={}
            for c in ("A","B"):
                sw=[s for s in range(4) if per[str(s)]["wrong_n"]>0];sp=sum(per[str(s)]["wrong_class_counts"][c]>0 for s in sw)
                support[c]={"wrong_fraction":frac(c),"suites_with_wrong":len(sw),"suites_present":sp,"majority_rule":bool(frac(c)>=.60 and sp>=3)}
            auth=[c for c in ("A","B") if support[c]["majority_rule"]]
            med=lambda g,k:np.median(np.stack([np.asarray(traces[str(x["suite"])]["lanes"][str(x["lane"])][k]) for x in g]),0).tolist() if g else []
            desc={n:{k:med(g,k) for k in ("cumulative_delta_J_S","delta_action_rate","action_separation_l2","state_separation_l2")} for n,g in (("final_wrong",wrong),("final_correct",correct))}
            decision={"recurring_temporal_mechanism_supported":bool(auth),"supported_class":auth[0] if len(auth)==1 else (auth or None),
              "authorize_one_S_repair_candidate":bool(auth),"stop_S_mechanism_mining":not bool(auth),"H2a_remains_incomplete":True,"H2b_authorized":False}
            rep={"schema":"authority_isolated_s_trajectory_temporal_decomposition_v1","measurement_only":True,"training_updates":0,"checkpoint":str(CK.relative_to(ROOT)),
              "protocol":{"suites":4,"lanes_per_suite":NENV,"steps":H,"matched_reset":True,"checkpoints":list(CPS),
              "classification_deadband":"max(1e-6,0.10*max_abs_cumulative_delta_J_S)"},"samples":samples,
              "summary":{"n":len(samples),"wrong_n":len(wrong),"correct_n":len(correct),"overall_class_counts":cnt(samples),"wrong_class_counts":cnt(wrong),
              "correct_class_counts":cnt(correct),"per_suite":per,"wrong_class_fractions":{c:frac(c) for c in classes},"mechanism_support":support},
              "descriptive_medians":desc,"traces":traces,"decision":decision}
            out=OUT/"trajectory_temporal_decomposition.json";out.write_text(json.dumps(rep,indent=2)+"\n")
            (OUT/"PROVENANCE_MANIFEST.json").write_text(json.dumps({"status":"FROZEN_BY_HASH","gate":"S-TRAJECTORY-TEMPORAL-DECOMPOSITION","measurement_only":True,
              "decision":decision,"artifacts":{str(out.relative_to(ROOT)):{"sha256":sha(out)},str(CK.relative_to(ROOT)):{"sha256":sha(CK)},
              str(Path(__file__).resolve().relative_to(ROOT)):{"sha256":sha(Path(__file__).resolve())},str(CONTRACT.relative_to(ROOT)):{"sha256":sha(CONTRACT)}}},indent=2)+"\n")
            print(json.dumps({"summary":rep["summary"],"decision":decision},indent=2),flush=True)
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_authority_isolated_saturation_causal_audit():
    """Run former authority_isolated_saturation_causal_audit.py stage."""
    from pathlib import Path
    import json,sys,hashlib
    import numpy as np, torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    OUT=ROOT/"runs/authority_isolated_saturation_causal_audit-2026-09-25";OUT.mkdir(parents=True,exist_ok=True)
    CKPT=ROOT/"runs/authority_isolated_h1-2026-09-25/model_75.pt"
    PROBE=ROOT/"runs/update_functional_effect_audit-2026-09-24/fixed_probe_states.npz"
    ORDER=("T","A","O","S","C")
    PREFS={"T":[.7,.1,.1,.1],"A":[.1,.7,.1,.1],"O":[.1,.1,.7,.1],"S":[.1,.1,.1,.7],"C":[.25]*4}
    SEEDS=(840001,840002,840003,840004);ALPHAS=(1.0,.9,.8,.7);NENV=8;H=64
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def qtilt(q):
        r=torch.atan2(2*(q[:,0]*q[:,1]+q[:,2]*q[:,3]),1-2*(q[:,1]**2+q[:,2]**2))
        p=torch.asin(torch.clamp(2*(q[:,0]*q[:,2]-q[:,3]*q[:,1]),-1,1))
        return torch.rad2deg(torch.maximum(r.abs(),p.abs()))
    def act(m,obs,w,alpha):
        z=m._actor_mean_with_preference(obs,w)
        return torch.tanh(alpha*z)*m.ACTION_CLIP,z
    def reason_terms(env):
        tm=env.unwrapped.termination_manager;out={}
        for n in tm.active_terms:
            try:out[n]=tm.get_term(n).detach().cpu().numpy().astype(bool)
            except Exception:pass
        return out
    def fixed_probe_metrics(m,probe,alpha):
        acts={}
        with torch.no_grad():
            for lab in ORDER:
                w=torch.tensor(PREFS[lab],device="cuda").repeat(len(probe),1)
                acts[lab]=act(m,probe,w,alpha)[0]
        vals=[]
        for i,a in enumerate(ORDER):
            for b in ORDER[i+1:]:
                vals.append(torch.linalg.vector_norm(acts[a]-acts[b],dim=1).mean().item())
        pair=float(np.mean(vals))
        wc=torch.tensor(PREFS["C"],device="cuda")
        D=torch.tensor([[1.,-1,0,0],[1.,0,-1,0],[1.,0,0,-1]],device="cuda").T
        Q,_=torch.linalg.qr(D,mode="reduced");eps=1e-3;js=[]
        with torch.no_grad():
            for o in probe:
                cols=[]
                for j in range(3):
                    wp=(wc+eps*Q[:,j]).unsqueeze(0);wm=(wc-eps*Q[:,j]).unsqueeze(0)
                    ap=act(m,o.unsqueeze(0),wp,alpha)[0];am=act(m,o.unsqueeze(0),wm,alpha)[0]
                    cols.append(((ap-am)/(2*eps)).squeeze(0))
                J=torch.stack(cols,dim=1);js.append(torch.linalg.matrix_norm(J,ord="fro").item())
        dev=[]
        if alpha!=1.0:
            with torch.no_grad():
                for lab in ORDER:
                    w=torch.tensor(PREFS[lab],device="cuda").repeat(len(probe),1)
                    a0=act(m,probe,w,1.0)[0];aa=acts[lab]
                    dev.extend(torch.linalg.vector_norm(aa-a0,dim=1).cpu().numpy().tolist())
        return {"pairwise":pair,"jacobian":float(np.mean(js)),
          "action_dev_mean":float(np.mean(dev)) if dev else 0.0,
          "action_dev_p95":float(np.percentile(dev,95)) if dev else 0.0}
    def rollout(env,m,lab,seed,alpha):
        w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1)
        obs,_=env.reset(seed=seed);obs=ot(obs).cuda();prev=torch.zeros((NENV,12),device="cuda")
        done=np.zeros(NENV,bool);ft=np.full(NENV,-1,int);why=[[] for _ in range(NENV)]
        acc=[]
        for t in range(H):
            data=env.unwrapped.scene["robot"].data
            with torch.no_grad():a,z=act(m,obs,w,alpha)
            met=np.stack([
              data.root_pos_w[:,2].detach().cpu().numpy(),
              qtilt(data.root_quat_w).detach().cpu().numpy(),
              torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=1).detach().cpu().numpy(),
              torch.linalg.vector_norm(a,dim=1).detach().cpu().numpy(),
              torch.linalg.vector_norm(a-prev,dim=1).detach().cpu().numpy(),
              (a.abs()>=.95).float().mean(dim=1).detach().cpu().numpy(),
              torch.linalg.vector_norm(z,dim=1).detach().cpu().numpy(),
              torch.linalg.vector_norm(alpha*z,dim=1).detach().cpu().numpy()],axis=1)
            acc.append(met)
            nxt,_,te,tr,_=env.step(a);dd=(te|tr).detach().cpu().numpy().astype(bool);terms=reason_terms(env)
            for i in range(NENV):
                if dd[i] and not done[i]:
                    ft[i]=t;why[i]=[n for n,v in terms.items() if v[i]]
            done|=dd;prev=a;obs=ot(nxt).cuda()
        X=np.asarray(acc)
        return {"alpha":alpha,"suite":SEEDS.index(seed),"seed":seed,"preference":lab,
          "survival":float(1-done.mean()),"done":done.tolist(),"fail_t":ft.tolist(),"reason":why,
          "metric_names":["height","tilt_deg","ang_xy","action","action_rate","sat_frac","pre_tanh_raw","pre_tanh_scaled"],
          "metric_mean":X.mean((0,1)).tolist()}
    def main():
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.authority.isolated import AuthorityIsolatedActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=0);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim
            m=AuthorityIsolatedActorCritic(o.shape[-1],ad).cuda();m.load_state_dict(torch.load(CKPT,map_location="cuda",weights_only=False)["model"]);m.eval()
            probe=torch.tensor(np.load(PROBE)["obs"],device="cuda")
            fixed={str(a):fixed_probe_metrics(m,probe,a) for a in ALPHAS}
            ctrl=fixed["1.0"]
            for k,v in fixed.items():
                v["pairwise_retention"]=v["pairwise"]/(ctrl["pairwise"]+1e-12)
                v["jacobian_retention"]=v["jacobian"]/(ctrl["jacobian"]+1e-12)
            rows=[]
            for a in ALPHAS:
                for seed in SEEDS:
                    for lab in ORDER:
                        r=rollout(env,m,lab,seed,a);rows.append(r)
                        print(a,r["suite"],lab,r["survival"],r["fail_t"],r["reason"],flush=True)
            by={}
            for a in ALPHAS:
                rr=[r for r in rows if r["alpha"]==a]
                minsur=min(r["survival"] for r in rr);means=np.mean([r["metric_mean"] for r in rr],axis=0)
                by[str(a)]={"min_survival":float(minsur),"mean_metrics":dict(zip(rr[0]["metric_names"],means.tolist())),
                  "fixed_probe":fixed[str(a)]}
            sat0=by["1.0"]["mean_metrics"]["sat_frac"]
            candidates=[]
            for a in ALPHAS[1:]:
                q=by[str(a)]
                ok=(q["min_survival"]>=.95 and q["mean_metrics"]["sat_frac"]<sat0 and
                    q["fixed_probe"]["pairwise_retention"]>=.90 and q["fixed_probe"]["jacobian_retention"]>=.90)
                q["causal_support_gate"]=bool(ok)
                if ok:candidates.append(a)
            primary=max(candidates) if candidates else None
            report={"schema":"authority_isolated_saturation_causal_audit_v1","read_only":True,
              "checkpoint":str(CKPT.relative_to(ROOT)),"alphas":list(ALPHAS),"by_alpha":by,
              "primary_causal_candidate":primary,"rows":rows}
            rp=OUT/"authority_isolated_saturation_causal_audit_report.json";rp.write_text(json.dumps(report,indent=2)+"\n")
            print("DECISION",json.dumps({"primary":primary,"by_alpha":{k:{"survival":v["min_survival"],
              "sat":v["mean_metrics"]["sat_frac"],"pair_ret":v["fixed_probe"]["pairwise_retention"],
              "jac_ret":v["fixed_probe"]["jacobian_retention"],"gate":v.get("causal_support_gate")} for k,v in by.items()}},indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_authority_isolated_semantic_robustness_temporal_audit():
    """Run former authority_isolated_semantic_robustness_temporal_audit.py stage."""
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from pathlib import Path
    import json,sys,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"scripts/rl")]
    from talon_rl.models.authority.isolated import AuthorityIsolatedActorCritic
    from talon_rl.rewards.objectives import normalized_objective_vector
    from rl.experiments.common.utilities.foundation_v2_semantic_eval import obs_tensor,tilt_deg,PREFS,IDX,PHYS
    
    RUN=ROOT/"runs/authority_isolated_ai_h2_semantic_path-2026-09-25"
    OUT=ROOT/"runs/authority_isolated_semantic_robustness_temporal_audit-2026-09-25";OUT.mkdir(parents=True,exist_ok=True)
    CK={"u35":RUN/"model_5.pt","u40":RUN/"model_10.pt"}
    SEM_SEEDS=(840001,840002,840003,840004)
    ORDER=("T","A","O","S")
    FRESH={"T":9700000,"A":9701000,"O":9702000,"S":9703000,"C":9704000}
    H=64;NENV=8
    
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    
    def rollout(env,m,mgr,robot,w_np,seed):
        w=torch.tensor(w_np,device="cuda").repeat(NENV,1)
        cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda()
        ad=env.unwrapped.action_manager.total_action_dim
        prev=torch.zeros((NENV,ad),device="cuda")
        out={k:[] for k in ("obj","tracking_error","ang_vel_xy","tilt_deg","action_rate","abs_lin_vel_z","height","joint_vel_norm","action_norm","done")}
        reasons=[[] for _ in range(NENV)]
        with torch.no_grad():
            for t in range(H):
                a=m.act_inference_with_preference(cur,w)
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                obj=normalized_objective_vector(terms(raw,names),shape=(NENV,))*env.unwrapped.step_dt
                data=robot.data;cmd=env.unwrapped.command_manager.get_command("base_velocity")
                track=(data.root_lin_vel_b[:,0]-cmd[:,0]).abs()+(data.root_ang_vel_b[:,2]-cmd[:,2]).abs()
                ang=torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1)
                tilt=tilt_deg(data.root_quat_w)
                ar=torch.linalg.vector_norm(a-prev,dim=-1)
                vz=data.root_lin_vel_b[:,2].abs()
                height=data.root_pos_w[:,2]
                jv=torch.linalg.vector_norm(data.joint_vel,dim=-1)
                an=torch.linalg.vector_norm(a,dim=-1)
                dd=(te|tr)
                tm=env.unwrapped.termination_manager
                active=list(tm.active_terms)
                termvals={n:tm.get_term(n).detach().cpu().numpy().astype(bool) for n in active}
                for i in range(NENV):
                    if bool(dd[i]) and not reasons[i]:
                        reasons[i]=[n for n,v in termvals.items() if v[i]]
                for k,v in {
                    "obj":obj,"tracking_error":track.cpu().numpy(),"ang_vel_xy":ang.cpu().numpy(),
                    "tilt_deg":tilt.cpu().numpy(),"action_rate":ar.cpu().numpy(),"abs_lin_vel_z":vz.cpu().numpy(),
                    "height":height.cpu().numpy(),"joint_vel_norm":jv.cpu().numpy(),"action_norm":an.cpu().numpy(),
                    "done":dd.cpu().numpy().astype(bool)}.items():
                    out[k].append(np.asarray(v))
                prev=a;cur=obs_tensor(nxt).cuda()
        for k in out:out[k]=np.asarray(out[k])
        out["reasons"]=reasons
        return out
    
    def trailing5(x):
        x=np.asarray(x,float);y=np.full_like(x,np.nan,dtype=float)
        for t in range(4,len(x)):y[t]=np.mean(x[t-4:t+1])
        return y
    
    def first_3(mask):
        m=np.asarray(mask,bool)
        for t in range(2,len(m)):
            if m[t] and m[t-1] and m[t-2]:return int(t-2)
        return None
    
    def semantic_rel(heavy,center,lab):
        j=IDX[lab]
        obj=(heavy["obj"][:,:,j]-center["obj"][:,:,j]).mean(1)
        if lab=="T":phys=(center["tracking_error"]-heavy["tracking_error"]).mean(1)
        elif lab=="A":phys=(center["ang_vel_xy"]-heavy["ang_vel_xy"]).mean(1)
        elif lab=="O":phys=(center["tilt_deg"]-heavy["tilt_deg"]).mean(1)
        else:phys=(center["action_rate"]-heavy["action_rate"]).mean(1)
        o5=trailing5(obj);p5=trailing5(phys)
        joint=(o5<=0)&(p5<=0)
        return {"obj":obj,"phys":phys,"obj5":o5,"phys5":p5,
                "obj_onset":first_3(o5<=0),"phys_onset":first_3(p5<=0),"joint_onset":first_3(joint)}
    
    STAB_KEYS=("tilt_deg","ang_vel_xy","abs_lin_vel_z","joint_vel_norm","action_norm")
    def stability_onset(ref,cur):
        bad=np.zeros((H,NENV,len(STAB_KEYS)),bool);thresholds={}
        for ki,k in enumerate(STAB_KEYS):
            # reference scale from cross-env dispersion around u35 per-step median
            med=np.median(ref[k],axis=1,keepdims=True)
            disp=np.abs(ref[k]-med)
            thr=max(float(np.quantile(disp,.95)),1e-6);thresholds[k]=thr
            bad[:,:,ki]=np.abs(cur[k]-ref[k])>thr
        # height uses deviation from matched u35, threshold from u35 cross-env dispersion
        medh=np.median(ref["height"],axis=1,keepdims=True)
        hthr=max(float(np.quantile(np.abs(ref["height"]-medh),.95)),1e-6);thresholds["height"]=hthr
        hbad=np.abs(cur["height"]-ref["height"])>hthr
        count=bad.sum(2)+hbad.astype(int)
        any2=count>=2
        env_on=[]
        for i in range(NENV):
            on=first_3(any2[:,i])
            contact=np.flatnonzero(cur["done"][:,i])
            ct=int(contact[0]) if len(contact) else None
            vals=[x for x in (on,ct) if x is not None]
            env_on.append(min(vals) if vals else None)
        valid=[x for x in env_on if x is not None]
        return {"env_onsets":env_on,"first_any":min(valid) if valid else None,
                "fraction_with_event":float(len(valid)/NENV),"thresholds":thresholds}
    
    def classify(sem,stab):
        if sem is None and stab is None:return "none"
        if sem is not None and stab is None:return "semantic-without-local-instability"
        if sem is None and stab is not None:return "instability-without-semantic-loss"
        d=sem-stab
        if d<=-3:return "semantic-first"
        if d>=3:return "robustness-first"
        return "co-emergent"
    
    def pack_trace(x):
        return [None if not np.isfinite(v) else float(v) for v in np.asarray(x).reshape(-1)]
    
    def main():
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]]
        app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=0);o=obs_tensor(o).cuda();ad=env.unwrapped.action_manager.total_action_dim
            mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
            models={}
            for name,path in CK.items():
                m=AuthorityIsolatedActorCritic(o.shape[-1],ad).cuda()
                m.load_state_dict(torch.load(path,map_location="cuda",weights_only=False)["model"]);m.eval();models[name]=m
    
            rep={"schema":"semantic_robustness_temporal_ordering_v1","primary":[],"fresh":[]}
            # Primary matched semantic suites.
            for seed in SEM_SEEDS:
                traces={}
                for ck,m in models.items():
                    traces[ck]={}
                    for lab in ("T","A","O","S","C"):
                        traces[ck][lab]=rollout(env,m,mgr,robot,PREFS[lab],seed)
                for lab in ORDER:
                    s35=semantic_rel(traces["u35"][lab],traces["u35"]["C"],lab)
                    s40=semantic_rel(traces["u40"][lab],traces["u40"]["C"],lab)
                    st_h=stability_onset(traces["u35"][lab],traces["u40"][lab])
                    st_c=stability_onset(traces["u35"]["C"],traces["u40"]["C"])
                    st_candidates=[x for x in (st_h["first_any"],st_c["first_any"]) if x is not None]
                    st=min(st_candidates) if st_candidates else None
                    dobj=s40["obj5"]-s35["obj5"];dphys=s40["phys5"]-s35["phys5"]
                    diff_on=first_3((dobj<0)&(dphys<0))
                    row={"seed":seed,"axis":lab,
                         "u35_semantic_onset":s35["joint_onset"],"u40_semantic_onset":s40["joint_onset"],
                         "u40_objective_onset":s40["obj_onset"],"u40_physical_onset":s40["phys_onset"],
                         "differential_semantic_onset":diff_on,
                         "stability_onset":st,"heavy_stability":st_h,"center_stability":st_c,
                         "classification":classify(diff_on,st),
                         "u35_survival":float(1-traces["u35"][lab]["done"].any(0).mean()),
                         "u40_survival":float(1-traces["u40"][lab]["done"].any(0).mean()),
                         "u35_obj5":pack_trace(s35["obj5"]),"u35_phys5":pack_trace(s35["phys5"]),
                         "u40_obj5":pack_trace(s40["obj5"]),"u40_phys5":pack_trace(s40["phys5"]),
                         "delta_obj5":pack_trace(dobj),"delta_phys5":pack_trace(dphys)}
                    rep["primary"].append(row)
                    print("PRIMARY",seed,lab,row["classification"],"sem",s40["joint_onset"],"stab",st,flush=True)
    
            # Fresh transfer: preference-specific fresh seed and matched center on same seed.
            for lab in ORDER:
                seed=FRESH[lab];traces={}
                for ck,m in models.items():
                    traces[ck]={}
                    for qlab in (lab,"C"):
                        traces[ck][qlab]=rollout(env,m,mgr,robot,PREFS[qlab],seed)
                s35=semantic_rel(traces["u35"][lab],traces["u35"]["C"],lab)
                s40=semantic_rel(traces["u40"][lab],traces["u40"]["C"],lab)
                dobj=s40["obj5"]-s35["obj5"];dphys=s40["phys5"]-s35["phys5"];diff_on=first_3((dobj<0)&(dphys<0))
                st_h=stability_onset(traces["u35"][lab],traces["u40"][lab])
                st_c=stability_onset(traces["u35"]["C"],traces["u40"]["C"])
                cand=[x for x in (st_h["first_any"],st_c["first_any"]) if x is not None];st=min(cand) if cand else None
                row={"seed":seed,"axis":lab,"semantic_onset":s40["joint_onset"],"differential_semantic_onset":diff_on,
                     "objective_onset":s40["obj_onset"],"physical_onset":s40["phys_onset"],
                     "stability_onset":st,"classification":classify(diff_on,st),
                     "u35_survival":float(1-traces["u35"][lab]["done"].any(0).mean()),
                     "u40_survival":float(1-traces["u40"][lab]["done"].any(0).mean()),
                     "heavy_first_done":next((int(t) for t in range(H) if traces["u40"][lab]["done"][t].any()),None),
                     "heavy_reasons":traces["u40"][lab]["reasons"],
                     "heavy_stability":st_h,"center_stability":st_c,
                     "u40_obj5":pack_trace(s40["obj5"]),"u40_phys5":pack_trace(s40["phys5"])}
                rep["fresh"].append(row)
                print("FRESH",lab,seed,row["classification"],"sem",row["semantic_onset"],"stab",st,"surv",row["u40_survival"],flush=True)
    
            from collections import Counter
            rep["summary"]={
              "primary_class_counts":dict(Counter(r["classification"] for r in rep["primary"])),
              "fresh_class_counts":dict(Counter(r["classification"] for r in rep["fresh"])),
              "primary_semantic_without_instability":sum(r["classification"]=="semantic-without-local-instability" for r in rep["primary"]),
              "primary_u40_survival_min":min(r["u40_survival"] for r in rep["primary"]),
              "fresh_u40_survival_min":min(r["u40_survival"] for r in rep["fresh"])
            }
            (OUT/"temporal_ordering_report.json").write_text(json.dumps(rep,indent=2)+"\n")
            print("SUMMARY",json.dumps(rep["summary"],indent=2),flush=True)
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_authority_isolated_smooth_compression_causal_audit():
    """Run former authority_isolated_smooth_compression_causal_audit.py stage."""
    from pathlib import Path
    import json,sys,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    OUT=ROOT/"runs/authority_isolated_smooth_compression_causal_audit-2026-09-25";OUT.mkdir(parents=True,exist_ok=True)
    CKPT=ROOT/"runs/authority_isolated_h1-2026-09-25/model_75.pt"
    PROBE=ROOT/"runs/update_functional_effect_audit-2026-09-24/fixed_probe_states.npz"
    ORDER=("T","A","O","S","C")
    PREFS={"T":[.7,.1,.1,.1],"A":[.1,.7,.1,.1],"O":[.1,.1,.7,.1],"S":[.1,.1,.1,.7],"C":[.25]*4}
    SEEDS=(840001,840002,840003,840004);ARMS=("control",3.0,2.5,2.0,1.75);NENV=8;H=64
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def qtilt(q):
        r=torch.atan2(2*(q[:,0]*q[:,1]+q[:,2]*q[:,3]),1-2*(q[:,1]**2+q[:,2]**2))
        p=torch.asin(torch.clamp(2*(q[:,0]*q[:,2]-q[:,3]*q[:,1]),-1,1))
        return torch.rad2deg(torch.maximum(r.abs(),p.abs()))
    def transform(z,arm):
        return z if arm=="control" else float(arm)*torch.tanh(z/float(arm))
    def act(m,obs,w,arm):
        z=m._actor_mean_with_preference(obs,w);zt=transform(z,arm)
        return torch.tanh(zt)*m.ACTION_CLIP,z,zt
    def reasons(env):
        out={};tm=env.unwrapped.termination_manager
        for n in tm.active_terms:
            try:out[n]=tm.get_term(n).detach().cpu().numpy().astype(bool)
            except Exception:pass
        return out
    def probe_metrics(m,probe,arm):
        acts={}
        with torch.no_grad():
            for lab in ORDER:
                w=torch.tensor(PREFS[lab],device="cuda").repeat(len(probe),1);acts[lab]=act(m,probe,w,arm)[0]
        ds=[]
        for i,a in enumerate(ORDER):
            for b in ORDER[i+1:]:ds.append(torch.linalg.vector_norm(acts[a]-acts[b],dim=1).mean().item())
        wc=torch.tensor(PREFS["C"],device="cuda");D=torch.tensor([[1.,-1,0,0],[1.,0,-1,0],[1.,0,0,-1]],device="cuda").T
        Q,_=torch.linalg.qr(D,mode="reduced");eps=1e-3;js=[]
        with torch.no_grad():
            for o in probe:
                cols=[]
                for j in range(3):
                    ap=act(m,o[None],(wc+eps*Q[:,j])[None],arm)[0]
                    am=act(m,o[None],(wc-eps*Q[:,j])[None],arm)[0]
                    cols.append(((ap-am)/(2*eps)).squeeze(0))
                js.append(torch.linalg.matrix_norm(torch.stack(cols,dim=1),ord="fro").item())
        dev=[]
        if arm!="control":
            with torch.no_grad():
                for lab in ORDER:
                    w=torch.tensor(PREFS[lab],device="cuda").repeat(len(probe),1)
                    a0=act(m,probe,w,"control")[0];aa=acts[lab]
                    dev.extend(torch.linalg.vector_norm(aa-a0,dim=1).cpu().tolist())
        return {"pairwise":float(np.mean(ds)),"jacobian":float(np.mean(js)),
          "action_dev_mean":float(np.mean(dev)) if dev else 0.0,
          "action_dev_p95":float(np.percentile(dev,95)) if dev else 0.0}
    def rollout(env,m,lab,seed,arm):
        w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1);obs,_=env.reset(seed=seed);obs=ot(obs).cuda()
        prev=torch.zeros((NENV,12),device="cuda");done=np.zeros(NENV,bool);ft=np.full(NENV,-1,int);why=[[] for _ in range(NENV)];acc=[]
        for t in range(H):
            data=env.unwrapped.scene["robot"].data
            with torch.no_grad():a,z,zt=act(m,obs,w,arm)
            acc.append(np.stack([data.root_pos_w[:,2].cpu().numpy(),qtilt(data.root_quat_w).cpu().numpy(),
              torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=1).cpu().numpy(),
              torch.linalg.vector_norm(a,dim=1).cpu().numpy(),torch.linalg.vector_norm(a-prev,dim=1).cpu().numpy(),
              (a.abs()>=.95).float().mean(dim=1).cpu().numpy(),torch.linalg.vector_norm(z,dim=1).cpu().numpy(),
              torch.linalg.vector_norm(zt,dim=1).cpu().numpy()],axis=1))
            nxt,_,te,tr,_=env.step(a);dd=(te|tr).cpu().numpy().astype(bool);rr=reasons(env)
            for i in range(NENV):
                if dd[i] and not done[i]:ft[i]=t;why[i]=[n for n,v in rr.items() if v[i]]
            done|=dd;prev=a;obs=ot(nxt).cuda()
        X=np.asarray(acc)
        return {"arm":str(arm),"suite":SEEDS.index(seed),"preference":lab,"survival":float(1-done.mean()),
          "done":done.tolist(),"fail_t":ft.tolist(),"reason":why,
          "metric_names":["height","tilt_deg","ang_xy","action","action_rate","sat_frac","pre_raw","pre_transformed"],
          "metric_mean":X.mean((0,1)).tolist()}
    def main():
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved;env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.authority.isolated import AuthorityIsolatedActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda()
            m=AuthorityIsolatedActorCritic(o.shape[-1],env.unwrapped.action_manager.total_action_dim).cuda()
            m.load_state_dict(torch.load(CKPT,map_location="cuda",weights_only=False)["model"]);m.eval()
            probe=torch.tensor(np.load(PROBE)["obs"],device="cuda")
            pm={str(a):probe_metrics(m,probe,a) for a in ARMS};ctrl=pm["control"]
            for v in pm.values():
                v["pairwise_retention"]=v["pairwise"]/(ctrl["pairwise"]+1e-12);v["jacobian_retention"]=v["jacobian"]/(ctrl["jacobian"]+1e-12)
            rows=[]
            for arm in ARMS:
                for seed in SEEDS:
                    for lab in ORDER:
                        r=rollout(env,m,lab,seed,arm);rows.append(r);print(arm,r["suite"],lab,r["survival"],r["fail_t"],flush=True)
            by={}
            for arm in ARMS:
                rr=[r for r in rows if r["arm"]==str(arm)];mm=np.mean([r["metric_mean"] for r in rr],axis=0)
                by[str(arm)]={"min_survival":float(min(r["survival"] for r in rr)),
                  "fail_count":int(sum(sum(r["done"]) for r in rr)),"mean_metrics":dict(zip(rr[0]["metric_names"],mm.tolist())),"fixed_probe":pm[str(arm)]}
            sat0=by["control"]["mean_metrics"]["sat_frac"];cand=[]
            for arm in ARMS[1:]:
                q=by[str(arm)];ok=q["min_survival"]>=.95 and q["mean_metrics"]["sat_frac"]<sat0 and q["fixed_probe"]["pairwise_retention"]>=.9 and q["fixed_probe"]["jacobian_retention"]>=.9
                q["causal_support_gate"]=bool(ok)
                if ok:cand.append(float(arm))
            primary=max(cand) if cand else None
            rep={"schema":"authority_isolated_smooth_compression_causal_v1","read_only":True,"by_arm":by,"primary":primary,"rows":rows}
            (OUT/"authority_isolated_smooth_compression_causal_report.json").write_text(json.dumps(rep,indent=2)+"\n")
            print("DECISION",json.dumps({"primary":primary,"by_arm":by},indent=2),flush=True)
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

STAGES = {
    "authority_isolated_s_gradient_alignment_fd_check": run_authority_isolated_s_gradient_alignment_fd_check,
    "authority_isolated_s_local_gradient_alignment_audit": run_authority_isolated_s_local_gradient_alignment_audit,
    "authority_isolated_s_phase_local_h16_sensitivity": run_authority_isolated_s_phase_local_h16_sensitivity,
    "authority_isolated_s_semantic_alignment_audit": run_authority_isolated_s_semantic_alignment_audit,
    "authority_isolated_s_short_horizon_epsilon_convergence": run_authority_isolated_s_short_horizon_epsilon_convergence,
    "authority_isolated_s_short_horizon_sensitivity_audit": run_authority_isolated_s_short_horizon_sensitivity_audit,
    "authority_isolated_s_sign_context_audit": run_authority_isolated_s_sign_context_audit,
    "authority_isolated_s_trajectory_temporal_decomposition": run_authority_isolated_s_trajectory_temporal_decomposition,
    "authority_isolated_saturation_causal_audit": run_authority_isolated_saturation_causal_audit,
    "authority_isolated_semantic_robustness_temporal_audit": run_authority_isolated_semantic_robustness_temporal_audit,
    "authority_isolated_smooth_compression_causal_audit": run_authority_isolated_smooth_compression_causal_audit,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
