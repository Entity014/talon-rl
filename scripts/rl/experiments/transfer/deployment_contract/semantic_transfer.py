"""Consolidated experiment stage module."""
from __future__ import annotations
import argparse

def run_phase1_d3c_command_generator_parity():
    """Run former phase1_d3c_command_generator_parity.py stage."""
    from pathlib import Path
    import json,math,numpy as np
    
    ROOT=Path(__file__).resolve().parents[4]
    SRC=ROOT/"runs/phase1_d3c_command_state/command_state.json"
    OUT=ROOT/"runs/phase1_d3c_command_generator_parity";OUT.mkdir(parents=True,exist_ok=True)
    
    def wrap_pi(x):
        return (x+np.pi)%(2*np.pi)-np.pi
    
    def main():
        d=json.load(open(SRC));rows=[];errs=[]
        for suite in d["rows"]:
            for lane in range(len(suite["lin_xy"])):
                lin=np.asarray(suite["lin_xy"][lane],float)
                init=np.asarray(suite["initial_command"][lane],float)
                standing=bool(suite["is_standing_env"][lane])
                htar=float(suite["heading_target"][lane]);h0=float(suite["initial_heading_w"][lane])
                err0=float(wrap_pi(htar-h0))
                target_mj=err0
                if standing:
                    got=np.zeros(3,float)
                else:
                    got=np.array([lin[0],lin[1],np.clip(.5*wrap_pi(target_mj-0.0),-1,1)],float)
                e=float(np.max(np.abs(got-init)))
                errs.append(e)
                rows.append({"suite":suite["suite"],"lane":lane,"standing":standing,
                             "source_initial":init.tolist(),"reconstructed":got.tolist(),
                             "source_heading_error":err0,"target_heading_mj":target_mj,
                             "max_abs_error":e})
        rep={"schema":"phase1_d3c_command_generator_parity_v1","n":len(rows),
             "max_abs_error":max(errs),"pass":bool(max(errs)<=1e-6),"rows":rows}
        (OUT/"command_generator_parity.json").write_text(json.dumps(rep,indent=2)+"\n")
        print(json.dumps({k:v for k,v in rep.items() if k!="rows"},indent=2))
    if True:main()

def run_phase1_d3c_extract_command_state():
    """Run former phase1_d3c_extract_command_state.py stage."""
    from pathlib import Path
    import json
    ROOT=Path(__file__).resolve().parents[4]
    from isaaclab.app import AppLauncher
    app=AppLauncher({"headless":True,"enable_cameras":False}).app
    env=None
    try:
        import gymnasium as gym,isaaclab_tasks
        from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
        cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=8;cfg.seed=0
        cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
        env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
        u=env.unwrapped;rows=[]
        for suite in range(4):
            seed=840001+suite;env.reset(seed=seed)
            term=u.command_manager._terms["base_velocity"]
            rows.append({
              "suite":suite,"seed":seed,
              "lin_xy":term.vel_command_b[:,:2].detach().cpu().tolist(),
              "initial_command":term.vel_command_b.detach().cpu().tolist(),
              "heading_target":term.heading_target.detach().cpu().tolist(),
              "is_heading_env":term.is_heading_env.detach().cpu().tolist(),
              "is_standing_env":term.is_standing_env.detach().cpu().tolist(),
              "initial_heading_w":u.scene["robot"].data.heading_w.detach().cpu().tolist(),
            })
        out=ROOT/"runs/phase1_d3c_command_state";out.mkdir(parents=True,exist_ok=True)
        (out/"command_state.json").write_text(json.dumps({"schema":"phase1_d3c_command_state_v1","rows":rows},indent=2)+"\n")
        print(json.dumps(rows,indent=2),flush=True)
    finally:
        if env is not None:env.close()
        app.close()

def run_phase1_d3c_objective_formula_parity():
    """Run former phase1_d3c_objective_formula_parity.py stage."""
    from pathlib import Path
    import json,sys,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    from talon_rl.rewards.objectives import raw_objective_vector
    
    from isaaclab.app import AppLauncher
    app=AppLauncher({"headless":True,"enable_cameras":False}).app
    env=None
    try:
        import gymnasium as gym,isaaclab_tasks
        from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
        cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=8;cfg.seed=0
        cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
        env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
        obs,_=env.reset(seed=260926)
        u=env.unwrapped;mgr=u.reward_manager;robot=u.scene["robot"]
        rng=np.random.default_rng(260926);errs=[];rows=[];comp_errs=[[],[],[],[]]
        for step in range(16):
            a=torch.tensor(rng.uniform(-.2,.2,size=(8,12)),device=u.device,dtype=torch.float32)
            cmd_before=u.command_manager.get_command("base_velocity").clone()
            _,_,te,tr,_=env.step(a)
            alive=(~(te|tr)).detach().cpu().numpy()
            raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
            wt={n:raw[:,i] for i,n in enumerate(names)}
            ref=raw_objective_vector(wt,shape=(8,))
            lin=robot.data.root_lin_vel_b;ang=robot.data.root_ang_vel_b;pg=robot.data.projected_gravity_b
            cmd=cmd_before
            cur=u.action_manager.action;prev=u.action_manager.prev_action
            track=1.5*torch.exp(-torch.sum((cmd[:,:2]-lin[:,:2])**2,dim=1)/(.5**2))
            track+=.75*torch.exp(-(cmd[:,2]-ang[:,2])**2/(.5**2))
            angular=-.05*torch.sum(ang[:,:2]**2,dim=1)
            orient=-2.5*torch.sum(pg[:,:2]**2,dim=1)
            smooth=-.01*torch.sum((cur-prev)**2,dim=1)
            manual=torch.stack([track,angular,orient,smooth],dim=1).detach().cpu().numpy()
            if alive.any():
                diff=np.abs(ref[alive]-manual[alive])
                e=float(diff.max());errs.append(e)
                ce=[float(diff[:,j].max()) for j in range(4)]
                for j,x in enumerate(ce):comp_errs[j].append(x)
            else:
                e=float("nan");ce=[float("nan")]*4
            rows.append({"step":step,"alive":int(alive.sum()),"max_abs":e,"component_max_abs":ce})
        comp_max=[max(x) if x else float("nan") for x in comp_errs]
        rep={"schema":"phase1_d3c_objective_formula_parity_v2","max_abs_error":max(errs),
             "component_max_abs":{"T":comp_max[0],"A":comp_max[1],"O":comp_max[2],"S":comp_max[3]},
             "mean_max_abs_error":float(np.mean(errs)),"rows":rows}
        rep["pass"]=bool(rep["max_abs_error"]<=1e-6)
        out=ROOT/"runs/phase1_d3c_objective_formula_parity";out.mkdir(parents=True,exist_ok=True)
        (out/"objective_formula_parity.json").write_text(json.dumps(rep,indent=2)+"\n")
        print(json.dumps(rep,indent=2),flush=True)
    finally:
        if env is not None:env.close()
        app.close()

def run_phase1_d3c_semantic_transfer():
    """Run former phase1_d3c_semantic_transfer.py stage."""
    from pathlib import Path
    import itertools,json,math,sys
    import mujoco
    import numpy as np
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path.insert(0,str(ROOT))
    from talon_rl.deployment.phase1 import Phase1EagerStateRuntime
    from talon_rl.deployment.phase1 import (
        CANONICAL_JOINT_ORDER,CANONICAL_DEFAULT_Q,
        build_canonical_obs,canonical_base_kinematics,
        root_com_velocity_b_from_freejoint,canonical_joint_target,
    )
    from talon_rl.rewards.objectives import NORMALIZATION_DIVISORS
    
    SCENE=ROOT/"talon_rl/assets/data/Robots/unitree_a1/mujoco/scene.xml"
    ART=ROOT/"artifacts/phase1_canonical_actor_state.pt"
    OUT=ROOT/"runs/phase1_d3c_semantic_transfer";OUT.mkdir(parents=True,exist_ok=True)
    
    ORDER=("T","A","O","S")
    PREFS={
     "T":np.array([.7,.1,.1,.1],np.float32),
     "A":np.array([.1,.7,.1,.1],np.float32),
     "O":np.array([.1,.1,.7,.1],np.float32),
     "S":np.array([.1,.1,.1,.7],np.float32),
     "C":np.array([.25,.25,.25,.25],np.float32),
    }
    COMMANDS=[
     ("forward",np.array([.5,0.,0.],np.float32)),
     ("turn_left",np.array([.3,0.,.3],np.float32)),
     ("turn_right",np.array([.3,0.,-.3],np.float32)),
     ("lateral",np.array([0.,.25,0.],np.float32)),
    ]
    PATHS=(("T","A"),("T","O"),("T","S"),("A","O"),("A","S"),("O","S"))
    ALPHAS=(0.,.25,.5,.75,1.)
    IDX={"T":0,"A":1,"O":2,"S":3}
    PHYS={"T":"tracking_error","A":"ang_vel_xy","O":"tilt_deg","S":"action_rate"}
    
    def tilt_deg(g):
        return math.degrees(math.acos(float(np.clip(-g[2],-1.,1.))))
    
    def monotonic_fraction(vals):
        vals=np.asarray(vals,float);d=np.diff(vals);target=np.sign(vals[-1]-vals[0])
        return 0.0 if target==0 else float(np.mean(np.sign(d)==target))
    
    def between_fraction(vals):
        vals=np.asarray(vals,float);lo=min(vals[0],vals[-1]);hi=max(vals[0],vals[-1])
        return float(np.mean((vals[1:-1]>=lo-1e-9)&(vals[1:-1]<=hi+1e-9)))
    
    def main():
        m=mujoco.MjModel.from_xml_path(str(SCENE));d=mujoco.MjData(m)
        rt=Phase1EagerStateRuntime(str(ART),device="cuda")
        jmap={}
        for j in range(m.njnt):
            n=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j)
            if n:jmap[n]=(int(m.jnt_qposadr[j]),int(m.jnt_dofadr[j]))
        act_joints=[]
        for a in range(m.nu):
            jid=int(m.actuator_trnid[a,0]);act_joints.append(mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,jid))
        trunk=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"trunk")
        com_b=m.body_ipos[trunk].copy()
        qa=int(m.jnt_qposadr[0]);va=int(m.jnt_dofadr[0]);hold=int(round(.02/m.opt.timestep))
    
        def reset():
            if m.nkey:mujoco.mj_resetDataKeyframe(m,d,0)
            else:mujoco.mj_resetData(m,d)
            d.qvel[:]=0
            for i,jn in enumerate(CANONICAL_JOINT_ORDER):d.qpos[jmap[jn][0]]=CANONICAL_DEFAULT_Q[i]
            for a,jn in enumerate(act_joints):d.ctrl[a]=d.qpos[jmap[jn][0]]
            mujoco.mj_forward(m,d)
    
        def state(command,prev):
            q=d.qpos[qa+3:qa+7].copy();qv=d.qvel[va:va+6].copy()
            _,ang_b,grav_b=canonical_base_kinematics(q,qv)
            lin_b=root_com_velocity_b_from_freejoint(q,qv[:3],qv[3:],com_b).astype(np.float32)
            jp=np.array([d.qpos[jmap[j][0]] for j in CANONICAL_JOINT_ORDER],np.float32)
            jv=np.array([d.qvel[jmap[j][1]] for j in CANONICAL_JOINT_ORDER],np.float32)
            obs=build_canonical_obs(lin_b,ang_b,grav_b,command,jp,jv,prev,CANONICAL_JOINT_ORDER)
            return obs,lin_b,ang_b,grav_b
    
        def trunk_contact():
            for k in range(d.ncon):
                c=d.contact[k];b1=int(m.geom_bodyid[c.geom1]);b2=int(m.geom_bodyid[c.geom2])
                if (b1==trunk and b2==0) or (b2==trunk and b1==0):return True
            return False
    
        def rollout(command,pref):
            reset();prev=np.zeros(12,np.float32)
            rt.last_action=np.zeros((1,12),np.float32);rt.estop_latched=False
            objs=[];phys=[];surv=True
            for step in range(64):
                obs,lin_b,ang_b,grav_b=state(command,prev)
                a=rt.act(obs[None,:],pref[None,:],age_ms=0.0)[0]
                # Exact frozen weighted objective terms.
                lin_err=float(np.sum((command[:2]-lin_b[:2])**2))
                yaw_err=float((command[2]-ang_b[2])**2)
                T=1.5*math.exp(-lin_err/.25)+.75*math.exp(-yaw_err/.25)
                A=-.05*float(np.sum(ang_b[:2]**2))
                O=-2.5*float(np.sum(grav_b[:2]**2))
                S=-.01*float(np.sum((a-prev)**2))
                obj=np.array([T,A,O,S],np.float32)/NORMALIZATION_DIVISORS
                objs.append(obj*.02)
                phys.append({
                  "tracking_error":float(abs(lin_b[0]-command[0])+abs(ang_b[2]-command[2])),
                  "vy_error":float(abs(lin_b[1]-command[1])),
                  "ang_vel_xy":float(np.linalg.norm(ang_b[:2])),
                  "tilt_deg":float(tilt_deg(grav_b)),
                  "action_rate":float(np.linalg.norm(a-prev)),
                })
                target=canonical_joint_target(a)
                d.ctrl[:]=np.array([target[CANONICAL_JOINT_ORDER.index(j)] for j in act_joints])
                for _ in range(hold):mujoco.mj_step(m,d)
                prev=a.astype(np.float32,copy=True)
                q=d.qpos[qa+3:qa+7].copy();qv=d.qvel[va:va+6].copy();_,_,g=canonical_base_kinematics(q,qv)
                if (not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all() or
                    d.qpos[qa+2]<.18 or tilt_deg(g)>75 or trunk_contact()):
                    surv=False;break
            arr=np.asarray(objs)
            return {
              "normalized_objective_mean":arr.mean(0).tolist(),
              "physical":{k:float(np.mean([x[k] for x in phys])) for k in phys[0]},
              "survival":1.0 if surv and len(objs)==64 else 0.0,
              "steps":len(objs),
            }
    
        endpoints=[]
        for suite,(cname,cmd) in enumerate(COMMANDS):
            for lab in ("T","A","O","S","C"):
                q=rollout(cmd,PREFS[lab]);q.update({"suite":suite,"command":cname,"label":lab});endpoints.append(q)
            print("ENDPOINT_SUITE",suite,cname,flush=True)
    
        endpoint={}
        for lab in ORDER:
            j=IDX[lab];pk=PHYS[lab];oo=[];pp=[];sv=[];do=[];dp=[]
            for suite in range(4):
                r=next(x for x in endpoints if x["suite"]==suite and x["label"]==lab)
                c=next(x for x in endpoints if x["suite"]==suite and x["label"]=="C")
                x=r["normalized_objective_mean"][j]-c["normalized_objective_mean"][j]
                y=r["physical"][pk]-c["physical"][pk]
                oo.append(x>0);pp.append(y<0);sv.append(r["survival"]);do.append(x);dp.append(y)
            endpoint[lab]={"objective_correct_fraction":float(np.mean(oo)),
                           "physical_correct_fraction":float(np.mean(pp)),
                           "mean_objective_delta_vs_center":float(np.mean(do)),
                           "mean_physical_delta_vs_center":float(np.mean(dp)),
                           "min_survival":float(min(sv))}
        endpoint_pass={lab:bool(endpoint[lab]["objective_correct_fraction"]>=.75 and
                                endpoint[lab]["physical_correct_fraction"]>=.75 and
                                endpoint[lab]["min_survival"]>=.75) for lab in ORDER}
    
        continuum=[]
        for pi,(pa,pb) in enumerate(PATHS):
            for suite,(cname,cmd) in enumerate(COMMANDS):
                for alpha in ALPHAS:
                    w=(1-alpha)*PREFS[pa]+alpha*PREFS[pb]
                    q=rollout(cmd,w.astype(np.float32));q.update({"path":f"{pa}-{pb}","suite":suite,"alpha":alpha});continuum.append(q)
            print("PATH",f"{pa}-{pb}",flush=True)
    
        cont=[]
        for pa,pb in PATHS:
            for suite in range(4):
                rr=sorted([x for x in continuum if x["path"]==f"{pa}-{pb}" and x["suite"]==suite],key=lambda x:x["alpha"])
                for lab in (pa,pb):
                    j=IDX[lab];pk=PHYS[lab]
                    ov=[x["normalized_objective_mean"][j] for x in rr]
                    pv=[x["physical"][pk] for x in rr]
                    cont.append({"path":f"{pa}-{pb}","suite":suite,"axis":lab,
                                 "objective_monotonic":monotonic_fraction(ov),
                                 "objective_between":between_fraction(ov),
                                 "physical_monotonic":monotonic_fraction(pv),
                                 "physical_between":between_fraction(pv)})
        mono=float(np.mean([(x["objective_monotonic"]+x["physical_monotonic"])/2 for x in cont]))
        between=float(np.mean([(x["objective_between"]+x["physical_between"])/2 for x in cont]))
    
        center_between=[]
        for suite in range(4):
            hs=[next(x for x in endpoints if x["suite"]==suite and x["label"]==lab) for lab in ORDER]
            cc=next(x for x in endpoints if x["suite"]==suite and x["label"]=="C")
            for j in range(4):
                vals=[x["normalized_objective_mean"][j] for x in hs];cv=cc["normalized_objective_mean"][j]
                center_between.append(min(vals)-1e-9<=cv<=max(vals)+1e-9)
        center=float(np.mean(center_between))
    
        min_surv=float(min([x["survival"] for x in endpoints+continuum]))
        criteria={
          "T":endpoint_pass["T"],"A":endpoint_pass["A"],"O":endpoint_pass["O"],
          "center":center>=.75,
          "continuum_monotonicity":mono>=.65,
          "continuum_between":between>=.65,
          "interpretable_survival":min_surv>=.75,
        }
        rep={"schema":"phase1_d3c_semantic_transfer_v1","endpoint":endpoint,"endpoint_pass":endpoint_pass,
             "continuum_summary":{"monotonicity_fraction":mono,"endpoint_between_fraction":between,"details":cont},
             "center_compromise_fraction":center,"min_survival_all":min_surv,
             "criteria":criteria,"semantic_pass":bool(all(criteria.values())),
             "endpoint_rows":endpoints,"continuum_rows":continuum}
        if endpoint_pass["S"]:sclass="S-valid"
        elif min_surv>=.75:sclass="S-semantic-inconsistent"
        else:sclass="S-engineering-confounded"
        rep["S_category"]=sclass
        (OUT/"semantic_transfer_report.json").write_text(json.dumps(rep,indent=2)+"\n")
        print("FINAL",json.dumps({"endpoint":endpoint,"endpoint_pass":endpoint_pass,
          "continuum":{"monotonicity":mono,"between":between},"center":center,
          "min_survival":min_surv,"S_category":sclass,"criteria":criteria,
          "semantic_pass":rep["semantic_pass"]},indent=2),flush=True)
    
    if True:main()

def run_phase1_d3c_source_command_timing_probe():
    """Run former phase1_d3c_source_command_timing_probe.py stage."""
    from pathlib import Path
    import json,math,torch
    ROOT=Path(__file__).resolve().parents[4]
    
    def wrap_pi(x):
        return (x+math.pi)%(2*math.pi)-math.pi
    
    from isaaclab.app import AppLauncher
    app=AppLauncher({"headless":True,"enable_cameras":False}).app
    env=None
    try:
        import gymnasium as gym,isaaclab_tasks
        from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
        cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=8;cfg.seed=0
        cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
        env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
        u=env.unwrapped;rows=[];errs0=[];errs1=[]
        for suite in range(4):
            seed=840001+suite
            obs,_=env.reset(seed=seed)
            term=u.command_manager._terms["base_velocity"]
            init=term.vel_command_b.detach().cpu().clone()
            target=term.heading_target.detach().cpu().clone()
            standing=term.is_standing_env.detach().cpu().clone()
            heading0=u.scene["robot"].data.heading_w.detach().cpu().clone()
            # step-0 manager command should be exactly the command visible at reset.
            cmd0=u.command_manager.get_command("base_velocity").detach().cpu().clone()
            errs0.append(float((cmd0-init).abs().max()))
            zero=torch.zeros((8,12),device=u.device)
            env.step(zero)
            cmd1=u.command_manager.get_command("base_velocity").detach().cpu()
            heading1=u.scene["robot"].data.heading_w.detach().cpu()
            expected=init.clone()
            for lane in range(8):
                if bool(standing[lane]):
                    expected[lane,:]=0
                else:
                    expected[lane,0:2]=init[lane,0:2]
                    ez=.5*wrap_pi(float(target[lane]-heading1[lane]))
                    expected[lane,2]=max(-1.0,min(1.0,ez))
            e1=float((cmd1-expected).abs().max());errs1.append(e1)
            rows.append({"suite":suite,"seed":seed,"step0_max_abs":errs0[-1],"step1_max_abs":e1})
        rep={"schema":"phase1_d3c_source_command_timing_v1",
             "step0_max_abs_error":max(errs0),"step1_max_abs_error":max(errs1),
             "pass":bool(max(errs0)<=1e-7 and max(errs1)<=1e-6),"rows":rows}
        out=ROOT/"runs/phase1_d3c_command_generator_parity";out.mkdir(parents=True,exist_ok=True)
        (out/"source_command_timing.json").write_text(json.dumps(rep,indent=2)+"\n")
        print(json.dumps(rep,indent=2),flush=True)
    finally:
        if env is not None:env.close()
        app.close()

STAGES = {
    "phase1_d3c_command_generator_parity": run_phase1_d3c_command_generator_parity,
    "phase1_d3c_extract_command_state": run_phase1_d3c_extract_command_state,
    "phase1_d3c_objective_formula_parity": run_phase1_d3c_objective_formula_parity,
    "phase1_d3c_semantic_transfer": run_phase1_d3c_semantic_transfer,
    "phase1_d3c_source_command_timing_probe": run_phase1_d3c_source_command_timing_probe,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
