"""Consolidated thematic experiment stages."""
from __future__ import annotations
import argparse

def run_authority_isolated_classB_alt_rescue_invariant():
    """Run former authority_isolated_classB_alt_rescue_invariant.py stage."""
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from pathlib import Path
    import json,sys,torch,numpy as np
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"scripts/rl")]
    from talon_rl.models.authority.isolated import AuthorityIsolatedActorCritic
    from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
    from rl.experiments.common.utilities.authority_isolated_classB_invariant_audit import ot,features
    U50=ROOT/"runs/authority_isolated_h1-2026-09-25/model_50.pt";REP=ROOT/"runs/authority_isolated_projected_tail_descent-2026-09-25/model_10.pt"
    OUT=ROOT/"runs/authority_isolated_classB_invariant_audit-2026-09-25";NENV=8;H=14;SEED=840004
    CASES={"O":{"w":[.1,.1,.7,.1],"idx":[0]},"S":{"w":[.1,.1,.1,.7],"idx":[3]},"C":{"w":[.25]*4,"idx":[3]}}
    TOP=["support_velocity_coupling","hip_diag","hip_lr","hip_frontrear","action_vel_cos"]
    def roll(env,b,d,c,rescue):
     w=torch.tensor(c["w"],device="cuda").repeat(NENV,1);obs,_=env.reset(seed=SEED);obs=ot(obs).cuda();pc=torch.zeros(NENV,device="cuda");rows=[];ff=None
     for t in range(H):
      with torch.no_grad():
       a=b.act_inference_with_preference(obs,w)
       if rescue and 6<=t<=10:
        ad=d.act_inference_with_preference(obs,w);a[:,c["idx"]]=ad[:,c["idx"]]
      F,pc2=features(env,obs,a,pc);rows.append(F);nxt,_,te,tr,_=env.step(a);dd=(te|tr).cpu().numpy().astype(bool)
      if dd[0] and ff is None:ff=t
      obs=ot(nxt).cuda();pc=pc2
     return rows,ff
    def main():
     from isaaclab.app import AppLauncher
     sv=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda()
      u=AuthorityIsolatedActorCritic(o.shape[-1],12).cuda();u.load_state_dict(torch.load(U50,map_location="cuda",weights_only=False)["model"]);u.eval()
      b=AuthorityIsolatedWideCritic(o.shape[-1],12).cuda();b.load_state_dict(torch.load(REP,map_location="cuda",weights_only=False)["model"]);b.eval()
      out={}
      for pref,c in CASES.items():
       F,ff=roll(env,b,u,c,False);R,rf=roll(env,b,u,c,True);q={"base_fail":ff,"rescue_fail":rf}
       for f in TOP:
        vals=[]
        for t in range(8,11):
         v=F[t][f];rv=R[t][f];mu=np.mean(v[1:]);sd=np.std(v[1:])+1e-4
         vals.append((v[0],rv[0],mu,(v[0]-mu)/sd,(rv[0]-mu)/sd))
        q[f]={"fail":float(np.mean([x[0] for x in vals])),"rescue":float(np.mean([x[1] for x in vals])),"ctrl":float(np.mean([x[2] for x in vals])),
              "fail_absz":float(np.mean([abs(x[3]) for x in vals])),"rescue_absz":float(np.mean([abs(x[4]) for x in vals]))}
       out[pref]=q
      (OUT/"classB_alt_rescue_invariant.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_authority_isolated_classB_pca_basin_gate():
    """Run former authority_isolated_classB_pca_basin_gate.py stage."""
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from pathlib import Path
    import json,sys,torch,numpy as np
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"scripts/rl")]
    from talon_rl.models.authority.isolated import AuthorityIsolatedActorCritic
    from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
    from rl.experiments.common.utilities.authority_isolated_classB_invariant_audit import ot,features
    U50=ROOT/"runs/authority_isolated_h1-2026-09-25/model_50.pt"
    U75=ROOT/"runs/authority_isolated_h1-2026-09-25/model_75.pt"
    REP=ROOT/"runs/authority_isolated_projected_tail_descent-2026-09-25/model_10.pt"
    OUT=ROOT/"runs/authority_isolated_classB_trajectory_representation_gate-2026-09-25";OUT.mkdir(parents=True,exist_ok=True)
    NENV=8;H=24
    PREFS={"O":[.1,.1,.7,.1],"S":[.1,.1,.1,.7],"C":[.25]*4}
    BASE_FEATURES=["pg_x","pg_y","ang_x","ang_y","ang_z","lin_x","lin_y","lin_z",
     "hip_lr","hip_frontrear","hip_diag","hip_std","leg_action_lr","leg_action_diag",
     "leg_vel_lr","leg_vel_diag","action_vel_cos","hip_vel_cos","contact_frac","contact_change"]
    WINDOWS={"t6_10":(6,10),"t6_13":(6,13)}
    def collect(env,base,wv,seed,donor=None,mode=None,idx=None,win=(6,10)):
        w=torch.tensor(wv,device="cuda").repeat(NENV,1);obs,_=env.reset(seed=seed);obs=ot(obs).cuda()
        pc=torch.zeros(NENV,device="cuda");rows=[];fail=np.full(NENV,-1,int)
        for t in range(H):
            with torch.no_grad():
                a=base.act_inference_with_preference(obs,w)
                if donor is not None and win[0]<=t<=win[1]:
                    q=donor.act_inference_with_preference(obs,w)
                    if mode=="whole":a=q
                    elif mode=="single":a[:,idx]=q[:,idx]
            F,pc2=features(env,obs,a,pc)
            rows.append({k:F[k] for k in BASE_FEATURES})
            nxt,_,te,tr,_=env.step(a);dd=(te|tr).detach().cpu().numpy().astype(bool)
            hit=(fail<0)&dd;fail[hit]=t;obs=ot(nxt).cuda();pc=pc2
        return rows,fail
    def vec(rows,lane,lo,hi):
        parts=[]
        for f in BASE_FEATURES:
            x=np.array([rows[t][f][lane] for t in range(lo,hi+1)],dtype=np.float64)
            tt=np.arange(len(x),dtype=np.float64)
            slope=float(np.polyfit(tt,x,1)[0]) if len(x)>1 else 0.
            parts += [float(x.mean()),slope,float(x.var()),float(x.max()-x.min()),float(x[-1]-x[0])]
        return np.array(parts,dtype=np.float64)
    def standardize(X):
        mu=X.mean(0);sd=X.std(0);sd[sd<1e-6]=1.;return (X-mu)/sd,mu,sd
    def pca_fit(Z,k=3):
        _,_,vt=np.linalg.svd(Z,full_matrices=False)
        return vt[:min(k,vt.shape[0])]
    def centroid_fit(P,y):
        return P[y==0].mean(0),P[y==1].mean(0)
    def basin_score(P,c0,c1):
        return np.linalg.norm(P-c0,axis=1)-np.linalg.norm(P-c1,axis=1)
    def main():
        from isaaclab.app import AppLauncher
        sv=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv;env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda()
            u50=AuthorityIsolatedActorCritic(o.shape[-1],12).cuda();u50.load_state_dict(torch.load(U50,map_location="cuda",weights_only=False)["model"]);u50.eval()
            u75=AuthorityIsolatedActorCritic(o.shape[-1],12).cuda();u75.load_state_dict(torch.load(U75,map_location="cuda",weights_only=False)["model"]);u75.eval()
            rep=AuthorityIsolatedWideCritic(o.shape[-1],12).cuda();rep.load_state_dict(torch.load(REP,map_location="cuda",weights_only=False)["model"]);rep.eval()
            raw={}
            for pref,wv in PREFS.items():
                raw[pref]={}
                for name,base,donor,mode,idx in [
                    ("fail",rep,None,None,None),("u50",u50,None,None,None),
                    ("whole_rescue",rep,u50,"whole",None),
                    ("single_rescue",rep,u50,"single",0 if pref=="O" else 3)]:
                    rows,ff=collect(env,base,wv,840004,donor,mode,idx)
                    raw[pref][name]={"rows":rows,"fail":ff.tolist()}
                    print("COLLECT",pref,name,ff[0],flush=True)
            # negative controls suite2 O
            raw["NEG"]={}
            for name,lane,donor,idx in [("A_fail",7,None,None),("A_rescue",7,u75,0),("C_fail",2,None,None),("C_rescue",2,u50,0)]:
                rows,ff=collect(env,rep,PREFS["O"],840003,donor,"single" if donor else None,idx)
                raw["NEG"][name]={"rows":rows,"lane":lane,"fail":ff.tolist()}
                print("NEG",name,ff[lane],flush=True)
            report={"schema":"classB_pca_basin_gate_v1","pca_components":3,"windows":{}}
            for wname,(lo,hi) in WINDOWS.items():
                samples=[]
                # matched retained controls = repaired lanes1..7 in suite3, exclude any actually failed (normally only lane0)
                for pref in PREFS:
                    R=raw[pref]["fail"]
                    for lane in range(1,8):
                        if R["fail"][lane]<0:
                            samples.append({"pref":pref,"kind":"matched_survivor","label":0,"vec":vec(R["rows"],lane,lo,hi)})
                    for kind,label in [("fail",1),("u50",0),("whole_rescue",0),("single_rescue",0)]:
                        samples.append({"pref":pref,"kind":kind,"label":label,"vec":vec(raw[pref][kind]["rows"],0,lo,hi)})
                folds={}
                for testpref in PREFS:
                    train=[s for s in samples if s["pref"]!=testpref and s["kind"] in ("matched_survivor","fail","u50")]
                    # final low-dimensional gate: fit PCA only on natural train trajectories, then nearest class centroid in k=3 space
                    X=np.stack([s["vec"] for s in train]);y=np.array([s["label"] for s in train])
                    Z,mu,sd=standardize(X);V=pca_fit(Z,3);P=Z@V.T;c0,c1=centroid_fit(P,y)
                    test=[s for s in samples if s["pref"]==testpref]
                    vals=[]
                    for s in test:
                        pp=((s["vec"]-mu)/sd)[None,:]@V.T
                        sc=float(basin_score(pp,c0,c1)[0])
                        vals.append({"kind":s["kind"],"label":s["label"],"score":sc,"pred_fail":bool(sc>0)})
                    neg=[]
                    for n,q in raw["NEG"].items():
                        z=vec(q["rows"],q["lane"],lo,hi);pp=((z-mu)/sd)[None,:]@V.T
                        sc=float(basin_score(pp,c0,c1)[0])
                        neg.append({"kind":n,"score":sc,"pred_fail":bool(sc>0)})
                    folds[testpref]={"test":vals,"negative_controls":neg,
                        "fail_correct":next(v for v in vals if v["kind"]=="fail")["pred_fail"],
                        "whole_rescue_survivor":not next(v for v in vals if v["kind"]=="whole_rescue")["pred_fail"],
                        "single_rescue_survivor":not next(v for v in vals if v["kind"]=="single_rescue")["pred_fail"]}
                report["windows"][wname]=folds
            # compact gate
            gates={}
            for wname,folds in report["windows"].items():
                cross=all(f["fail_correct"] and f["whole_rescue_survivor"] and f["single_rescue_survivor"] for f in folds.values())
                # negative controls should not all collapse into Class-B fail; require both A/C rescues not predicted fail in >=2/3 folds
                ar=sum(not next(x for x in f["negative_controls"] if x["kind"]=="A_rescue")["pred_fail"] for f in folds.values())
                cr=sum(not next(x for x in f["negative_controls"] if x["kind"]=="C_rescue")["pred_fail"] for f in folds.values())
                gates[wname]={"cross_preference_core":cross,"A_rescue_survivor_folds":ar,"C_rescue_survivor_folds":cr,
                              "gate":bool(cross and ar>=2 and cr>=2)}
            report["gates"]=gates
            (OUT/"pca_basin_gate.json").write_text(json.dumps(report,indent=2)+"\n")
            print("GATES",json.dumps(gates,indent=2),flush=True)
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_authority_isolated_classB_rescue_survival_validation():
    """Run former authority_isolated_classB_rescue_survival_validation.py stage."""
    from pathlib import Path
    import json,sys,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    from talon_rl.models.authority.isolated import AuthorityIsolatedActorCritic
    from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
    U50=ROOT/"runs/authority_isolated_h1-2026-09-25/model_50.pt";REP=ROOT/"runs/authority_isolated_projected_tail_descent-2026-09-25/model_10.pt"
    OUT=ROOT/"runs/authority_isolated_classB_invariant_audit-2026-09-25";SEED=840004;NENV=8;H=24
    CASES={"O":[.1,.1,.7,.1],"S":[.1,.1,.1,.7],"C":[.25]*4}
    ALT={"O":[0],"S":[3],"C":[3]}
    def ot(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def run(env,b,d,wv,kind):
     w=torch.tensor(wv,device="cuda").repeat(NENV,1);obs,_=env.reset(seed=SEED);obs=ot(obs).cuda();ff=None
     for t in range(H):
      with torch.no_grad():
       a=b.act_inference_with_preference(obs,w)
       if 6<=t<=10:
        q=d.act_inference_with_preference(obs,w)
        if kind=="whole":a=q
        elif kind=="single":a[:,ALT[cur]]=q[:,ALT[cur]]
      nxt,_,te,tr,_=env.step(a);dd=(te|tr).cpu().numpy().astype(bool)
      if dd[0] and ff is None:ff=t
      obs=ot(nxt).cuda()
     return ff
    def main():
     from isaaclab.app import AppLauncher
     sv=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda()
      u=AuthorityIsolatedActorCritic(o.shape[-1],12).cuda();u.load_state_dict(torch.load(U50,map_location="cuda",weights_only=False)["model"]);u.eval()
      b=AuthorityIsolatedWideCritic(o.shape[-1],12).cuda();b.load_state_dict(torch.load(REP,map_location="cuda",weights_only=False)["model"]);b.eval()
      global cur;out={}
      for cur,wv in CASES.items():
       out[cur]={}
       for kind in ["base","whole","single"]:
        if kind=="base":
         # donor unused, no replacement because impossible window branch
         w=torch.tensor(wv,device="cuda").repeat(NENV,1);obs,_=env.reset(seed=SEED);obs=ot(obs).cuda();ff=None
         for t in range(H):
          with torch.no_grad():a=b.act_inference_with_preference(obs,w)
          nxt,_,te,tr,_=env.step(a);dd=(te|tr).cpu().numpy().astype(bool)
          if dd[0] and ff is None:ff=t
          obs=ot(nxt).cuda()
        else:ff=run(env,b,u,wv,kind)
        out[cur][kind]=ff;print(cur,kind,ff,flush=True)
      (OUT/"rescue_survival_validation.json").write_text(json.dumps(out,indent=2)+"\n")
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_authority_isolated_classB_specificity_check():
    """Run former authority_isolated_classB_specificity_check.py stage."""
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from pathlib import Path
    import json,sys,torch,numpy as np
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"scripts/rl")]
    from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
    from rl.experiments.common.utilities.authority_isolated_classB_invariant_audit import ot,features
    CK=ROOT/"runs/authority_isolated_projected_tail_descent-2026-09-25/model_10.pt"
    OUT=ROOT/"runs/authority_isolated_classB_invariant_audit-2026-09-25"
    SEED=840003;W=[.1,.1,.7,.1];NENV=8;H=14
    TOP=["hip_diag","action_vel_cos","hip_lr","hip_frontrear","support_velocity_coupling","leg_vel_diag","gravity_hip_coupling","hip_std"]
    def main():
     from isaaclab.app import AppLauncher
     sv=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda()
      m=AuthorityIsolatedWideCritic(o.shape[-1],12).cuda();m.load_state_dict(torch.load(CK,map_location="cuda",weights_only=False)["model"]);m.eval()
      w=torch.tensor(W,device="cuda").repeat(NENV,1);obs,_=env.reset(seed=SEED);obs=ot(obs).cuda();pc=torch.zeros(NENV,device="cuda");rows=[]
      for t in range(H):
       with torch.no_grad():a=m.act_inference_with_preference(obs,w)
       F,pc2=features(env,obs,a,pc);rows.append(F);nxt,_,_,_,_=env.step(a);obs=ot(nxt).cuda();pc=pc2
      out={}
      controls=[0,1,3,4,5,6]
      for f in TOP:
       q={}
       for lane,name in [(7,"classA_lane7"),(2,"classC_lane2")]:
        zs=[]
        for t in range(6,11):
         v=rows[t][f];mu=np.mean(v[controls]);sd=np.std(v[controls])+1e-4;zs.append(float((v[lane]-mu)/sd))
        q[name]={"mean_abs_z":float(np.mean(np.abs(zs))),"zs":zs}
       out[f]=q
      (OUT/"specificity_suite2_O.json").write_text(json.dumps(out,indent=2)+"\n")
      print(json.dumps(out,indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_authority_isolated_classB_trajectory_representation_gate():
    """Run former authority_isolated_classB_trajectory_representation_gate.py stage."""
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from pathlib import Path
    import json,sys,torch,numpy as np
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"scripts/rl")]
    from talon_rl.models.authority.isolated import AuthorityIsolatedActorCritic
    from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
    from rl.experiments.common.utilities.authority_isolated_classB_invariant_audit import ot,features
    U50=ROOT/"runs/authority_isolated_h1-2026-09-25/model_50.pt"
    U75=ROOT/"runs/authority_isolated_h1-2026-09-25/model_75.pt"
    REP=ROOT/"runs/authority_isolated_projected_tail_descent-2026-09-25/model_10.pt"
    OUT=ROOT/"runs/authority_isolated_classB_trajectory_representation_gate-2026-09-25";OUT.mkdir(parents=True,exist_ok=True)
    NENV=8;H=24
    PREFS={"O":[.1,.1,.7,.1],"S":[.1,.1,.1,.7],"C":[.25]*4}
    BASE_FEATURES=["pg_x","pg_y","ang_x","ang_y","ang_z","lin_x","lin_y","lin_z",
     "hip_lr","hip_frontrear","hip_diag","hip_std","leg_action_lr","leg_action_diag",
     "leg_vel_lr","leg_vel_diag","action_vel_cos","hip_vel_cos","contact_frac","contact_change"]
    WINDOWS={"t6_10":(6,10),"t6_13":(6,13)}
    def collect(env,base,wv,seed,donor=None,mode=None,idx=None,win=(6,10)):
        w=torch.tensor(wv,device="cuda").repeat(NENV,1);obs,_=env.reset(seed=seed);obs=ot(obs).cuda()
        pc=torch.zeros(NENV,device="cuda");rows=[];fail=np.full(NENV,-1,int)
        for t in range(H):
            with torch.no_grad():
                a=base.act_inference_with_preference(obs,w)
                if donor is not None and win[0]<=t<=win[1]:
                    q=donor.act_inference_with_preference(obs,w)
                    if mode=="whole":a=q
                    elif mode=="single":a[:,idx]=q[:,idx]
            F,pc2=features(env,obs,a,pc)
            rows.append({k:F[k] for k in BASE_FEATURES})
            nxt,_,te,tr,_=env.step(a);dd=(te|tr).detach().cpu().numpy().astype(bool)
            hit=(fail<0)&dd;fail[hit]=t;obs=ot(nxt).cuda();pc=pc2
        return rows,fail
    def vec(rows,lane,lo,hi):
        parts=[]
        for f in BASE_FEATURES:
            x=np.array([rows[t][f][lane] for t in range(lo,hi+1)],dtype=np.float64)
            tt=np.arange(len(x),dtype=np.float64)
            slope=float(np.polyfit(tt,x,1)[0]) if len(x)>1 else 0.
            parts += [float(x.mean()),slope,float(x.var()),float(x.max()-x.min()),float(x[-1]-x[0])]
        return np.array(parts,dtype=np.float64)
    def standardize(X):
        mu=X.mean(0);sd=X.std(0);sd[sd<1e-6]=1.;return (X-mu)/sd,mu,sd
    def lda_fit(X,y):
        X0=X[y==0];X1=X[y==1];m0=X0.mean(0);m1=X1.mean(0)
        S=np.cov(X0,rowvar=False)+np.cov(X1,rowvar=False)+1e-2*np.eye(X.shape[1])
        w=np.linalg.solve(S,m1-m0);thr=.5*np.dot(w,m0+m1)
        return w,thr
    def score_model(w,thr,X):return X@w-thr
    def main():
        from isaaclab.app import AppLauncher
        sv=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv;env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda()
            u50=AuthorityIsolatedActorCritic(o.shape[-1],12).cuda();u50.load_state_dict(torch.load(U50,map_location="cuda",weights_only=False)["model"]);u50.eval()
            u75=AuthorityIsolatedActorCritic(o.shape[-1],12).cuda();u75.load_state_dict(torch.load(U75,map_location="cuda",weights_only=False)["model"]);u75.eval()
            rep=AuthorityIsolatedWideCritic(o.shape[-1],12).cuda();rep.load_state_dict(torch.load(REP,map_location="cuda",weights_only=False)["model"]);rep.eval()
            raw={}
            for pref,wv in PREFS.items():
                raw[pref]={}
                for name,base,donor,mode,idx in [
                    ("fail",rep,None,None,None),("u50",u50,None,None,None),
                    ("whole_rescue",rep,u50,"whole",None),
                    ("single_rescue",rep,u50,"single",0 if pref=="O" else 3)]:
                    rows,ff=collect(env,base,wv,840004,donor,mode,idx)
                    raw[pref][name]={"rows":rows,"fail":ff.tolist()}
                    print("COLLECT",pref,name,ff[0],flush=True)
            # negative controls suite2 O
            raw["NEG"]={}
            for name,lane,donor,idx in [("A_fail",7,None,None),("A_rescue",7,u75,0),("C_fail",2,None,None),("C_rescue",2,u50,0)]:
                rows,ff=collect(env,rep,PREFS["O"],840003,donor,"single" if donor else None,idx)
                raw["NEG"][name]={"rows":rows,"lane":lane,"fail":ff.tolist()}
                print("NEG",name,ff[lane],flush=True)
            report={"schema":"classB_trajectory_representation_gate_v1","windows":{}}
            for wname,(lo,hi) in WINDOWS.items():
                samples=[]
                # matched retained controls = repaired lanes1..7 in suite3, exclude any actually failed (normally only lane0)
                for pref in PREFS:
                    R=raw[pref]["fail"]
                    for lane in range(1,8):
                        if R["fail"][lane]<0:
                            samples.append({"pref":pref,"kind":"matched_survivor","label":0,"vec":vec(R["rows"],lane,lo,hi)})
                    for kind,label in [("fail",1),("u50",0),("whole_rescue",0),("single_rescue",0)]:
                        samples.append({"pref":pref,"kind":kind,"label":label,"vec":vec(raw[pref][kind]["rows"],0,lo,hi)})
                folds={}
                for testpref in PREFS:
                    train=[s for s in samples if s["pref"]!=testpref and s["kind"] in ("matched_survivor","fail","u50")]
                    # train descriptive basin discriminator only on natural fail/survive, not rescues
                    X=np.stack([s["vec"] for s in train]);y=np.array([s["label"] for s in train])
                    Z,mu,sd=standardize(X);ww,thr=lda_fit(Z,y)
                    test=[s for s in samples if s["pref"]==testpref]
                    vals=[]
                    for s in test:
                        sc=float(score_model(ww,thr,((s["vec"]-mu)/sd)[None,:])[0])
                        vals.append({"kind":s["kind"],"label":s["label"],"score":sc,"pred_fail":bool(sc>0)})
                    neg=[]
                    for n,q in raw["NEG"].items():
                        z=vec(q["rows"],q["lane"],lo,hi);sc=float(score_model(ww,thr,((z-mu)/sd)[None,:])[0])
                        neg.append({"kind":n,"score":sc,"pred_fail":bool(sc>0)})
                    folds[testpref]={"test":vals,"negative_controls":neg,
                        "fail_correct":next(v for v in vals if v["kind"]=="fail")["pred_fail"],
                        "whole_rescue_survivor":not next(v for v in vals if v["kind"]=="whole_rescue")["pred_fail"],
                        "single_rescue_survivor":not next(v for v in vals if v["kind"]=="single_rescue")["pred_fail"]}
                report["windows"][wname]=folds
            # compact gate
            gates={}
            for wname,folds in report["windows"].items():
                cross=all(f["fail_correct"] and f["whole_rescue_survivor"] and f["single_rescue_survivor"] for f in folds.values())
                # negative controls should not all collapse into Class-B fail; require both A/C rescues not predicted fail in >=2/3 folds
                ar=sum(not next(x for x in f["negative_controls"] if x["kind"]=="A_rescue")["pred_fail"] for f in folds.values())
                cr=sum(not next(x for x in f["negative_controls"] if x["kind"]=="C_rescue")["pred_fail"] for f in folds.values())
                gates[wname]={"cross_preference_core":cross,"A_rescue_survivor_folds":ar,"C_rescue_survivor_folds":cr,
                              "gate":bool(cross and ar>=2 and cr>=2)}
            report["gates"]=gates
            (OUT/"trajectory_representation_gate.json").write_text(json.dumps(report,indent=2)+"\n")
            print("GATES",json.dumps(gates,indent=2),flush=True)
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_authority_isolated_crossclass_invariant_rescue():
    """Run former authority_isolated_crossclass_invariant_rescue.py stage."""
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from pathlib import Path
    import json,sys,torch,numpy as np
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"scripts/rl")]
    from talon_rl.models.authority.isolated import AuthorityIsolatedActorCritic
    from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
    from rl.experiments.common.utilities.authority_isolated_classB_invariant_audit import ot,features
    U50=ROOT/"runs/authority_isolated_h1-2026-09-25/model_50.pt";U75=ROOT/"runs/authority_isolated_h1-2026-09-25/model_75.pt"
    REP=ROOT/"runs/authority_isolated_projected_tail_descent-2026-09-25/model_10.pt"
    OUT=ROOT/"runs/authority_isolated_classB_invariant_audit-2026-09-25"
    NENV=8;H=14
    CASES={
     "A_s2O_l7":{"seed":840003,"w":[.1,.1,.7,.1],"lane":7,"donor":"u75","idx":[0],"win":(6,10),"controls":[0,1,3,4,5,6]},
     "C_s2O_l2":{"seed":840003,"w":[.1,.1,.7,.1],"lane":2,"donor":"u50","idx":[0],"win":(6,10),"controls":[0,1,3,4,5,6]}}
    TOP=["support_velocity_coupling","hip_diag","hip_lr","hip_frontrear","action_vel_cos"]
    def rollout(env,base,donor,case,rescue):
     w=torch.tensor(case["w"],device="cuda").repeat(NENV,1);obs,_=env.reset(seed=case["seed"]);obs=ot(obs).cuda();pc=torch.zeros(NENV,device="cuda");rows=[]
     for t in range(H):
      with torch.no_grad():
       a=base.act_inference_with_preference(obs,w)
       if rescue and case["win"][0]<=t<=case["win"][1]:
        ad=donor.act_inference_with_preference(obs,w);a[:,case["idx"]]=ad[:,case["idx"]]
      F,pc2=features(env,obs,a,pc);rows.append(F);nxt,_,_,_,_=env.step(a);obs=ot(nxt).cuda();pc=pc2
     return rows
    def main():
     from isaaclab.app import AppLauncher
     sv=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda()
      u50=AuthorityIsolatedActorCritic(o.shape[-1],12).cuda();u50.load_state_dict(torch.load(U50,map_location="cuda",weights_only=False)["model"]);u50.eval()
      u75=AuthorityIsolatedActorCritic(o.shape[-1],12).cuda();u75.load_state_dict(torch.load(U75,map_location="cuda",weights_only=False)["model"]);u75.eval()
      rep=AuthorityIsolatedWideCritic(o.shape[-1],12).cuda();rep.load_state_dict(torch.load(REP,map_location="cuda",weights_only=False)["model"]);rep.eval()
      dm={"u50":u50,"u75":u75};out={}
      for name,c in CASES.items():
       fail=rollout(env,rep,dm[c["donor"]],c,False);res=rollout(env,rep,dm[c["donor"]],c,True);q={}
       for f in TOP:
        vals=[]
        for t in range(8,11):
         fv=fail[t][f];rv=res[t][f];mu=np.mean(fv[c["controls"]]);sd=np.std(fv[c["controls"]])+1e-4
         vals.append((float(fv[c["lane"]]),float(rv[c["lane"]]),float(mu),float((fv[c["lane"]]-mu)/sd),float((rv[c["lane"]]-mu)/sd)))
        q[f]={"fail":float(np.mean([x[0] for x in vals])),"rescue":float(np.mean([x[1] for x in vals])),"ctrl":float(np.mean([x[2] for x in vals])),
              "fail_absz":float(np.mean([abs(x[3]) for x in vals])),"rescue_absz":float(np.mean([abs(x[4]) for x in vals]))}
       out[name]=q
      (OUT/"crossclass_rescue_invariant.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

STAGES = {
    "authority_isolated_classB_alt_rescue_invariant": run_authority_isolated_classB_alt_rescue_invariant,
    "authority_isolated_classB_pca_basin_gate": run_authority_isolated_classB_pca_basin_gate,
    "authority_isolated_classB_rescue_survival_validation": run_authority_isolated_classB_rescue_survival_validation,
    "authority_isolated_classB_specificity_check": run_authority_isolated_classB_specificity_check,
    "authority_isolated_classB_trajectory_representation_gate": run_authority_isolated_classB_trajectory_representation_gate,
    "authority_isolated_crossclass_invariant_rescue": run_authority_isolated_crossclass_invariant_rescue,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
