"""Consolidated thematic experiment stages."""
from __future__ import annotations
import argparse

def run_authority_isolated_postrepair_residual_audit():
    """Run former authority_isolated_postrepair_residual_audit.py stage."""
    from pathlib import Path
    import json,sys,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    from talon_rl.models.authority.isolated import AuthorityIsolatedActorCritic
    from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
    U75=ROOT/"runs/authority_isolated_h1-2026-09-25/model_75.pt"
    REP=ROOT/"runs/authority_isolated_projected_tail_descent-2026-09-25/model_10.pt"
    OUT=ROOT/"runs/authority_isolated_postrepair_residual_audit-2026-09-25";OUT.mkdir(parents=True,exist_ok=True)
    PREFS={"O":[.1,.1,.7,.1],"S":[.1,.1,.1,.7],"C":[.25]*4}
    CASES=[("s2O_l2",840003,"O",2),("s2O_l7",840003,"O",7),("s3O_l0",840004,"O",0),("s3S_l0",840004,"S",0),("s3C_l0",840004,"C",0)]
    NENV=8;H=20
    TAU=torch.tensor(json.load(open(ROOT/"runs/authority_isolated_tail_budget_calibration-2026-09-25/tail_budget_probe.json"))["tau"],device="cuda")
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def quat_rp(q):
        r=torch.atan2(2*(q[:,0]*q[:,1]+q[:,2]*q[:,3]),1-2*(q[:,1]**2+q[:,2]**2))
        p=torch.asin(torch.clamp(2*(q[:,0]*q[:,2]-q[:,3]*q[:,1]),-1,1))
        return r,p
    def termmap(env):
        out={};tm=env.unwrapped.termination_manager
        for n in tm.active_terms:
            try:out[n]=tm.get_term(n).detach().cpu().numpy().astype(bool)
            except:pass
        return out
    def run(env,m,seed,lab,lane):
        w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1);obs,_=env.reset(seed=seed);obs=ot(obs).cuda()
        tr=[];ff=None
        for t in range(H):
            d=env.unwrapped.scene["robot"].data;r,p=quat_rp(d.root_quat_w)
            with torch.no_grad():z=m._actor_mean_with_preference(obs,w);a=torch.tanh(z)
            row={"t":t,"height":float(d.root_pos_w[lane,2]),"roll":float(r[lane]),"pitch":float(p[lane]),
                 "lin_vel":d.root_lin_vel_b[lane].cpu().tolist(),"ang_vel":d.root_ang_vel_b[lane].cpu().tolist(),
                 "joint_vel":d.joint_vel[lane].cpu().tolist(),"z":z[lane].cpu().tolist(),"action":a[lane].cpu().tolist(),
                 "headroom":(1-a[lane].abs()).cpu().tolist(),"tail_mask":(z[lane].abs()>TAU).cpu().tolist()}
            nxt,_,te,trm,_=env.step(a);dd=(te|trm).cpu().numpy().astype(bool);tm=termmap(env)
            row["done"]=bool(dd[lane]);row["terms"]=[n for n,v in tm.items() if v[lane]]
            if dd[lane] and ff is None:ff=t
            tr.append(row);obs=ot(nxt).cuda()
        return {"first_fail":ff,"trace":tr}
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
            names=list(env.unwrapped.scene["robot"].data.joint_names);rep={"joint_names":names,"cases":{}}
            for name,seed,lab,lane in CASES:
                rep["cases"][name]={"seed":seed,"preference":lab,"lane":lane,
                    "u75":run(env,u,seed,lab,lane),"repair":run(env,r,seed,lab,lane)}
                print(name,"u75",rep["cases"][name]["u75"]["first_fail"],"repair",rep["cases"][name]["repair"]["first_fail"],flush=True)
            (OUT/"residual_audit.json").write_text(json.dumps(rep,indent=2)+"\n")
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_authority_isolated_residual_lane_audit():
    """Run former authority_isolated_residual_lane_audit.py stage."""
    from pathlib import Path
    import json,sys,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    OUT=ROOT/"runs/authority_isolated_residual_lane_audit-2026-09-25";OUT.mkdir(parents=True,exist_ok=True)
    CKPT=ROOT/"runs/authority_isolated_h1-2026-09-25/model_75.pt"
    SEED=840004;LAB="C";W=[.25,.25,.25,.25];ARMS=("control",2.5,2.0,1.75);NENV=8;LANE=0;H=24
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def transform(z,arm):return z if arm=="control" else float(arm)*torch.tanh(z/float(arm))
    def quat_rp(q):
        roll=torch.atan2(2*(q[:,0]*q[:,1]+q[:,2]*q[:,3]),1-2*(q[:,1]**2+q[:,2]**2))
        pitch=torch.asin(torch.clamp(2*(q[:,0]*q[:,2]-q[:,3]*q[:,1]),-1,1))
        return roll,pitch
    def reasons(env):
        out={};tm=env.unwrapped.termination_manager
        for n in tm.active_terms:
            try:out[n]=tm.get_term(n).detach().cpu().numpy().astype(bool)
            except Exception:pass
        return out
    def run_arm(env,m,arm):
        w=torch.tensor(W,device="cuda").repeat(NENV,1)
        obs,_=env.reset(seed=SEED);obs=ot(obs).cuda();prev=torch.zeros((NENV,12),device="cuda")
        trace=[];first_fail=None
        for t in range(H):
            data=env.unwrapped.scene["robot"].data
            q=data.root_quat_w;roll,pitch=quat_rp(q)
            with torch.no_grad():
                z=m._actor_mean_with_preference(obs,w);zt=transform(z,arm);a=torch.tanh(zt)*m.ACTION_CLIP
            i=LANE
            row={"t":t,"post_failure":first_fail is not None,
              "height":float(data.root_pos_w[i,2]),"roll":float(roll[i]),"pitch":float(pitch[i]),
              "ang_vel_b":data.root_ang_vel_b[i].detach().cpu().tolist(),
              "lin_vel_b":data.root_lin_vel_b[i].detach().cpu().tolist(),
              "joint_pos":data.joint_pos[i].detach().cpu().tolist(),
              "joint_vel":data.joint_vel[i].detach().cpu().tolist(),
              "z_raw":z[i].detach().cpu().tolist(),"z_transformed":zt[i].detach().cpu().tolist(),
              "action":a[i].detach().cpu().tolist(),
              "headroom":(1-a[i].abs()).detach().cpu().tolist(),
              "action_delta":(a[i]-prev[i]).detach().cpu().tolist(),
              "sat_mask":(a[i].abs()>=.95).detach().cpu().tolist()}
            nxt,_,te,tr,_=env.step(a);dd=(te|tr).detach().cpu().numpy().astype(bool);rr=reasons(env)
            row["done_after_step"]=bool(dd[i]);row["termination_terms"]=[n for n,v in rr.items() if v[i]]
            if dd[i] and first_fail is None:first_fail=t
            trace.append(row);prev=a;obs=ot(nxt).cuda()
        return {"arm":str(arm),"first_fail":first_fail,"trace":trace}
    def summarize(arms,joint_names):
        ctrl=arms["control"]["trace"];out={}
        for arm,r in arms.items():
            tr=r["trace"];end=r["first_fail"] if r["first_fail"] is not None else 20
            end=min(end,20);start=max(0,end-6)
            Z=np.array([x["z_transformed"] for x in tr[start:end+1]])
            A=np.array([x["action"] for x in tr[start:end+1]])
            Hm=np.array([x["headroom"] for x in tr[start:end+1]])
            D=np.array([x["action_delta"] for x in tr[start:end+1]])
            out[arm]={"first_fail":r["first_fail"],"window":[start,end],
              "mean_abs_action":np.mean(np.abs(A),axis=0).tolist(),
              "mean_headroom":np.mean(Hm,axis=0).tolist(),
              "mean_abs_delta":np.mean(np.abs(D),axis=0).tolist(),
              "mean_abs_z_transformed":np.mean(np.abs(Z),axis=0).tolist()}
        return out
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
            robot=env.unwrapped.scene["robot"]
            joint_names=list(robot.data.joint_names)
            arms={}
            for arm in ARMS:
                arms[str(arm)]=run_arm(env,m,arm)
                print("ARM",arm,"FAIL",arms[str(arm)]["first_fail"],flush=True)
            rep={"schema":"authority_isolated_residual_lane_v1","read_only":True,
              "seed":SEED,"suite":3,"preference":LAB,"lane":LANE,"joint_names":joint_names,
              "action_coordinate_names":joint_names,"arms":arms,"summary":summarize(arms,joint_names)}
            (OUT/"authority_isolated_residual_lane_report.json").write_text(json.dumps(rep,indent=2)+"\n")
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_authority_isolated_residual_mechanism_classification():
    """Run former authority_isolated_residual_mechanism_classification.py stage."""
    from pathlib import Path
    import json,sys,torch,numpy as np
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    from talon_rl.models.authority.isolated import AuthorityIsolatedActorCritic
    from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
    CK={
     "u50":ROOT/"runs/authority_isolated_h1-2026-09-25/model_50.pt",
     "u75":ROOT/"runs/authority_isolated_h1-2026-09-25/model_75.pt",
     "repair":ROOT/"runs/authority_isolated_projected_tail_descent-2026-09-25/model_10.pt"}
    OUT=ROOT/"runs/authority_isolated_residual_mechanism_classification-2026-09-25";OUT.mkdir(parents=True,exist_ok=True)
    CASES={
     "s2O_l2":(840003,[.1,.1,.7,.1],2),
     "s3S_l0":(840004,[.1,.1,.1,.7],0),
     "s3C_l0":(840004,[.25,.25,.25,.25],0)}
    NENV=8;H=24
    WINDOWS=[(0,5),(4,8),(6,10),(9,13),(11,15)]
    ALPHAS=[.1,.25,.5,.75,1.0]
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def quat_rp(q):
        r=torch.atan2(2*(q[:,0]*q[:,1]+q[:,2]*q[:,3]),1-2*(q[:,1]**2+q[:,2]**2))
        p=torch.asin(torch.clamp(2*(q[:,0]*q[:,2]-q[:,3]*q[:,1]),-1,1))
        return r,p
    def rollout(env,base,seed,wv,lane,donor=None,idx=None,window=None,alpha=1.0,trace=False):
        w=torch.tensor(wv,device="cuda").repeat(NENV,1)
        obs,_=env.reset(seed=seed);obs=ot(obs).cuda();ff=None;tr=[]
        for t in range(H):
            d=env.unwrapped.scene["robot"].data
            with torch.no_grad():
                ab=base.act_inference_with_preference(obs,w);a=ab.clone()
                if donor is not None:
                    ad=donor.act_inference_with_preference(obs,w)
                    active=(window is None or window[0]<=t<=window[1])
                    if active:
                        if idx is None:
                            a=(1-alpha)*ab+alpha*ad
                        else:
                            a[:,idx]=(1-alpha)*ab[:,idx]+alpha*ad[:,idx]
            if trace:
                r,p=quat_rp(d.root_quat_w)
                tr.append({"t":t,"height":float(d.root_pos_w[lane,2]),"roll":float(r[lane]),"pitch":float(p[lane]),
                 "lin_vel":d.root_lin_vel_b[lane].detach().cpu().tolist(),"ang_vel":d.root_ang_vel_b[lane].detach().cpu().tolist(),
                 "joint_vel":d.joint_vel[lane].detach().cpu().tolist(),"action":a[lane].detach().cpu().tolist()})
            nxt,_,te,tm,_=env.step(a);dd=(te|tm).detach().cpu().numpy().astype(bool)
            if dd[lane] and ff is None:ff=t
            obs=ot(nxt).cuda()
        return {"survived":ff is None,"first_fail":ff,"trace":tr if trace else None}
    def main():
        from isaaclab.app import AppLauncher
        sv=sys.argv[:];sys.argv=[sys.argv[0]]
        app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=sv;env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda()
            u50=AuthorityIsolatedActorCritic(o.shape[-1],12).cuda();u50.load_state_dict(torch.load(CK["u50"],map_location="cuda",weights_only=False)["model"]);u50.eval()
            u75=AuthorityIsolatedActorCritic(o.shape[-1],12).cuda();u75.load_state_dict(torch.load(CK["u75"],map_location="cuda",weights_only=False)["model"]);u75.eval()
            rep=AuthorityIsolatedWideCritic(o.shape[-1],12).cuda();rep.load_state_dict(torch.load(CK["repair"],map_location="cuda",weights_only=False)["model"]);rep.eval()
            donors={"u50":u50,"u75":u75};names=list(env.unwrapped.scene["robot"].data.joint_names)
            groups={"hips":[0,1,2,3],"front_left":[0,4,8],"front_right":[1,5,9],"rear_left":[2,6,10],"rear_right":[3,7,11],
                    "front":[0,1,4,5,8,9],"rear":[2,3,6,7,10,11],"all":list(range(12))}
            out={"joint_names":names,"cases":{}}
            for cname,(seed,wv,lane) in CASES.items():
                c={"baseline":{},"single":{},"groups":{},"phase":{},"strength":{},"traces":{}}
                for bl,m in [("u50",u50),("u75",u75),("repair",rep)]:
                    c["baseline"][bl]=rollout(env,m,seed,wv,lane,trace=True)
                    print(cname,"BASE",bl,c["baseline"][bl]["first_fail"],flush=True)
                rescuers=[]
                for dlab,dm in donors.items():
                    c["single"][dlab]={}
                    for j,n in enumerate(names):
                        q=rollout(env,rep,seed,wv,lane,dm,[j])
                        c["single"][dlab][n]=q
                        if q["survived"]:rescuers.append((dlab,j,n))
                        print(cname,"SINGLE",dlab,n,q["first_fail"],flush=True)
                    c["groups"][dlab]={}
                    for glab,idx in groups.items():
                        q=rollout(env,rep,seed,wv,lane,dm,idx)
                        c["groups"][dlab][glab]=q
                        print(cname,"GROUP",dlab,glab,q["first_fail"],flush=True)
                # phase + strength only for single-coordinate rescuers; always test full donor too
                phase_targets=rescuers+[("u50",None,"whole_u50"),("u75",None,"whole_u75")]
                for dlab,j,n in phase_targets:
                    key=f"{dlab}:{n}";c["phase"][key]={};c["strength"][key]={}
                    dm=donors[dlab];idx=None if j is None else [j]
                    for win in WINDOWS:
                        q=rollout(env,rep,seed,wv,lane,dm,idx,win,1.0)
                        c["phase"][key][f"{win[0]}-{win[1]}"]=q
                        print(cname,"PHASE",key,win,q["first_fail"],flush=True)
                    for a in ALPHAS:
                        q=rollout(env,rep,seed,wv,lane,dm,idx,None,a)
                        c["strength"][key][str(a)]=q
                        print(cname,"ALPHA",key,a,q["first_fail"],flush=True)
                out["cases"][cname]=c
            (OUT/"mechanism_classification.json").write_text(json.dumps(out,indent=2)+"\n")
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

STAGES = {
    "authority_isolated_postrepair_residual_audit": run_authority_isolated_postrepair_residual_audit,
    "authority_isolated_residual_lane_audit": run_authority_isolated_residual_lane_audit,
    "authority_isolated_residual_mechanism_classification": run_authority_isolated_residual_mechanism_classification,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
