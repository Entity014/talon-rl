"""Consolidated experiment stage module."""
from __future__ import annotations
import argparse

def run_phase1_d2_closed_loop():
    """Run former phase1_d2_closed_loop.py stage."""
    
    # scripts/ on sys.path so the absolute rl.experiments.* imports below
    # resolve when this file is run directly, as these scripts always are.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from pathlib import Path
    import json,os,sys,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"scripts/rl")]
    
    import rl.experiments.common.utilities.authority_isolated_h2a_u30_semantic_validity as h2
    from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
    from talon_rl.deployment.phase1 import Phase1DeploymentActor,Phase1DeploymentRuntime,Phase1EagerStateRuntime
    
    CK=ROOT/"runs/authority_isolated_simplex_edge_retain-2026-09-25/model_30.pt"
    ART=Path(os.environ.get("PHASE1_D2_ARTIFACT",str(ROOT/"artifacts/phase1_canonical_actor.pt")))
    BASE=ROOT/"runs/authority_isolated_h2a_u30_semantic_validity-2026-09-25/semantic_report.json"
    OUT=ROOT/"runs/phase1_d2_closed_loop";OUT.mkdir(parents=True,exist_ok=True)
    RUNTIME_DEVICE=os.environ.get("PHASE1_D2_RUNTIME_DEVICE","cpu")
    RUNTIME_MODE=os.environ.get("PHASE1_D2_RUNTIME_MODE","jit")
    
    def obs_tensor(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    
    def tilt_deg(q):
        _,x,y,_=[q[:,i] for i in range(4)]
        return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    
    class EagerRuntime:
        def __init__(self,source,device="cuda"):
            self.device=torch.device(device)
            self.policy=Phase1DeploymentActor(source).to(self.device).eval()
            self.last_action=np.zeros((1,12),dtype=np.float32)
            self.estop_latched=False
        def act(self,obs,w,age_ms=0.0):
            obs=np.asarray(obs,np.float32);w=np.asarray(w,np.float32)
            with torch.inference_mode():
                a=self.policy(torch.from_numpy(obs).to(self.device),torch.from_numpy(w).to(self.device))
            out=a.detach().cpu().numpy().astype(np.float32)
            self.last_action=out.copy();return out
    def evaluate_runtime(env,src,runtime,mgr,robot,w_np,seed):
        from talon_rl.rewards.objectives import normalized_objective_vector
        w_np_batch=np.repeat(np.asarray(w_np,np.float32)[None,:],h2.NENV,axis=0)
        w_gpu=torch.tensor(w_np_batch,device="cuda")
        cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda()
        R=[];phys=[];done_any=np.zeros(h2.NENV,bool)
        prev=torch.zeros((h2.NENV,env.unwrapped.action_manager.total_action_dim),device="cuda")
        max_online=0.0
        runtime.last_action=np.zeros((h2.NENV,12),dtype=np.float32)
        runtime.estop_latched=False
        with torch.no_grad():
            for _ in range(h2.STEPS):
                ref=src.act_inference_with_preference(cur,w_gpu)
                a_np=runtime.act(cur.detach().cpu().numpy().astype(np.float32),w_np_batch,age_ms=0.0)
                a=torch.from_numpy(a_np).to("cuda")
                max_online=max(max_online,float((a-ref).abs().max().cpu()))
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                R.append(normalized_objective_vector(terms(raw,names),shape=(h2.NENV,))*env.unwrapped.step_dt)
                dd=(te|tr).cpu().numpy();done_any|=dd
                data=robot.data;cmd=env.unwrapped.command_manager.get_command("base_velocity")
                vx=float((data.root_lin_vel_b[:,0]-cmd[:,0]).abs().mean())
                wz=float((data.root_ang_vel_b[:,2]-cmd[:,2]).abs().mean())
                phys.append({"tracking_error":vx+wz,
                             "ang_vel_xy":float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),
                             "tilt_deg":float(tilt_deg(data.root_quat_w).mean()),
                             "action_rate":float(torch.linalg.vector_norm(a-prev,dim=-1).mean())})
                prev=a;cur=obs_tensor(nxt).cuda()
        R=np.asarray(R)
        return {"normalized_objective_mean":R.mean((0,1)).tolist(),
                "physical":{k:float(np.mean([x[k] for x in phys])) for k in phys[0]},
                "survival":float(1-done_any.mean()),"max_online_action_error":max_online}
    def summarize(rows,continuum):
        endpoint={}
        for lab in h2.ORDER:
            j=h2.IDX[lab];pk=h2.PHYS[lab];oo=[];pp=[];sv=[];do=[];dp=[]
            for suite in range(h2.SUITES):
                r=next(x for x in rows if x["suite"]==suite and x["label"]==lab)
                c=next(x for x in rows if x["suite"]==suite and x["label"]=="C")
                x=r["normalized_objective_mean"][j]-c["normalized_objective_mean"][j]
                y=r["physical"][pk]-c["physical"][pk]
                oo.append(x>0);pp.append(y<0);sv.append(r["survival"]);do.append(x);dp.append(y)
            endpoint[lab]={"objective_correct_fraction":float(np.mean(oo)),
                           "physical_correct_fraction":float(np.mean(pp)),
                           "mean_objective_delta_vs_center":float(np.mean(do)),
                           "mean_physical_delta_vs_center":float(np.mean(dp)),
                           "min_survival":float(np.min(sv))}
        cont=[]
        for a,b in h2.PATHS:
            for suite in range(h2.SUITES):
                rr=sorted([x for x in continuum if x["path"]==f"{a}-{b}" and x["suite"]==suite],key=lambda x:x["alpha"])
                for lab in (a,b):
                    j=h2.IDX[lab];pk=h2.PHYS[lab]
                    ov=[x["normalized_objective_mean"][j] for x in rr];pv=[x["physical"][pk] for x in rr]
                    cont.append((h2.monotonic_fraction(ov),h2.between_fraction(ov),
                                 h2.monotonic_fraction(pv),h2.between_fraction(pv)))
        mono=float(np.mean([(a+c)/2 for a,b,c,d in cont]))
        between=float(np.mean([(b+d)/2 for a,b,c,d in cont]))
        center_between=[]
        for suite in range(h2.SUITES):
            hs=[next(x for x in rows if x["suite"]==suite and x["label"]==lab) for lab in h2.ORDER]
            cc=next(x for x in rows if x["suite"]==suite and x["label"]=="C")
            for j in range(4):
                vals=[x["normalized_objective_mean"][j] for x in hs];cv=cc["normalized_objective_mean"][j]
                center_between.append(min(vals)-1e-9<=cv<=max(vals)+1e-9)
        return endpoint,{"monotonicity_fraction":mono,"endpoint_between_fraction":between},float(np.mean(center_between))
    def main():
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]]
        app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved;env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=h2.NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=0);o=obs_tensor(o).cuda()
            mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
            src=AuthorityIsolatedWideCritic(48,12).cuda()
            src.load_state_dict(torch.load(CK,map_location="cuda",weights_only=False)["model"]);src.eval()
            if RUNTIME_MODE=="eager": runtime=EagerRuntime(src,device=RUNTIME_DEVICE)
            elif RUNTIME_MODE=="state": runtime=Phase1EagerStateRuntime(str(ART),device=RUNTIME_DEVICE)
            else: runtime=Phase1DeploymentRuntime(str(ART),device=RUNTIME_DEVICE)
            rows=[];continuum=[]
            for suite in range(h2.SUITES):
                seed=840001+suite
                for lab in ("T","A","O","S","C"):
                    q=evaluate_runtime(env,src,runtime,mgr,robot,h2.PREFS[lab],seed)
                    q.update({"suite":suite,"label":lab});rows.append(q)
            print("ENDPOINTS_DONE",flush=True)
            for pi,(a,b) in enumerate(h2.PATHS):
                for suite in range(h2.SUITES):
                    seed=850001+pi*1000+suite
                    for alpha in h2.ALPHAS:
                        w=h2.interp(a,b,alpha)
                        q=evaluate_runtime(env,src,runtime,mgr,robot,w,seed)
                        q.update({"path":f"{a}-{b}","suite":suite,"alpha":alpha});continuum.append(q)
                print("PATH_DONE",f"{a}-{b}",flush=True)
            endpoint,cont,center=summarize(rows,continuum)
            base=json.load(open(BASE))
            endpoint_pass={lab:(endpoint[lab]["objective_correct_fraction"]>=.75 and
                                endpoint[lab]["physical_correct_fraction"]>=.75 and
                                endpoint[lab]["min_survival"]>=.95) for lab in h2.ORDER}
            max_online=max([x["max_online_action_error"] for x in rows+continuum])
            min_surv=min([x["survival"] for x in rows+continuum])
            base_ep=base["endpoint"];deltas={}
            for lab in h2.ORDER:
                deltas[lab]={
                  "objective_correct_fraction_delta":endpoint[lab]["objective_correct_fraction"]-base_ep[lab]["objective_correct_fraction"],
                  "physical_correct_fraction_delta":endpoint[lab]["physical_correct_fraction"]-base_ep[lab]["physical_correct_fraction"],
                  "min_survival_delta":endpoint[lab]["min_survival"]-base_ep[lab]["min_survival"]}
            rep={"schema":"phase1_d2_closed_loop_v1",
                 "runtime_mode":RUNTIME_MODE,
                 "runtime_device":RUNTIME_DEVICE,
                 "max_online_action_error":max_online,
                 "endpoint":endpoint,"endpoint_pass":endpoint_pass,
                 "continuum_summary":cont,"center_compromise_fraction":center,
                 "min_survival_all":min_surv,
                 "canonical_endpoint_delta":deltas,
                 "canonical_continuum_delta":{
                   "monotonicity_fraction":cont["monotonicity_fraction"]-base["continuum_summary"]["monotonicity_fraction"],
                   "endpoint_between_fraction":cont["endpoint_between_fraction"]-base["continuum_summary"]["endpoint_between_fraction"]},
                 "canonical_center_delta":center-base["center_compromise"]["between_heavy_envelope_fraction"]}
            rep["pass"]=bool(max_online<=1e-5 and endpoint_pass["T"] and endpoint_pass["A"] and endpoint_pass["O"] and
                             cont["monotonicity_fraction"]>=.65 and cont["endpoint_between_fraction"]>=.65 and
                             center>=.75 and min_surv>=.875)
            artifact_tag=ART.stem.replace("phase1_canonical_actor","").strip("_") or "script"
            (OUT/f"closed_loop_report_{RUNTIME_MODE}_{artifact_tag}_{RUNTIME_DEVICE}.json").write_text(json.dumps(rep,indent=2)+"\n")
            print("FINAL",json.dumps({k:v for k,v in rep.items() if k not in ("endpoint",)},indent=2),flush=True)
            print("ENDPOINT",json.dumps(endpoint,indent=2),flush=True)
        finally:
            if env is not None:env.close()
            app.close()
    
    if True:main()

def run_phase1_d2_contract_probe():
    """Run former phase1_d2_contract_probe.py stage."""
    from pathlib import Path
    import json,sys
    ROOT=Path(__file__).resolve().parents[4]
    from isaaclab.app import AppLauncher
    app=AppLauncher({"headless":True,"enable_cameras":False}).app
    env=None
    try:
        import gymnasium as gym,isaaclab_tasks
        from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
        cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=1
        cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
        env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
        obs,_=env.reset(seed=123);u=env.unwrapped
        term=u.action_manager.get_term("joint_pos")
        rep={
          "step_dt":float(u.step_dt),
          "joint_names":list(u.scene["robot"].data.joint_names),
          "observation_terms":list(u.observation_manager.active_terms["policy"]),
          "observation_group_dim":str(u.observation_manager.group_obs_dim),
          "action_terms":list(u.action_manager.active_terms),
          "policy_enable_corruption":bool(cfg.observations.policy.enable_corruption),
          "policy_concatenate_terms":bool(cfg.observations.policy.concatenate_terms),
          "action_joint_ids":[int(x) for x in term._joint_ids] if hasattr(term,"_joint_ids") and not isinstance(term._joint_ids,slice) else str(getattr(term,"_joint_ids",None)),
          "action_joint_names":list(getattr(term,"_joint_names",[])),
          "action_scale":getattr(term,"_scale",None).detach().cpu().tolist() if hasattr(getattr(term,"_scale",None),"detach") else str(getattr(term,"_scale",None)),
          "action_offset":getattr(term,"_offset",None).detach().cpu().tolist() if hasattr(getattr(term,"_offset",None),"detach") else str(getattr(term,"_offset",None)),
        }
        out=ROOT/"runs/phase1_d2_contract_probe";out.mkdir(parents=True,exist_ok=True)
        (out/"runtime_contract.json").write_text(json.dumps(rep,indent=2)+"\n")
        print(json.dumps(rep,indent=2),flush=True)
    finally:
        if env is not None:env.close()
        app.close()

def run_phase1_d2_eager_runtime_guard():
    """Run former phase1_d2_eager_runtime_guard.py stage."""
    from pathlib import Path
    import json,time,numpy as np
    ROOT=Path(__file__).resolve().parents[4]
    from talon_rl.deployment.phase1 import Phase1EagerStateRuntime
    
    ART=ROOT/"artifacts/phase1_canonical_actor_state.pt"
    OUT=ROOT/"runs/phase1_d2_eager_runtime_guard";OUT.mkdir(parents=True,exist_ok=True)
    
    def run(device):
        rt=Phase1EagerStateRuntime(str(ART),device=device)
        obs=np.zeros((1,48),np.float32);w=np.array([[.25,.25,.25,.25]],np.float32)
        a0=rt.act(obs,w)
        repeat=all(np.array_equal(a0,rt.act(obs,w)) for _ in range(20))
        for _ in range(50):rt.act(obs,w)
        ts=[]
        for _ in range(1000):
            t=time.perf_counter_ns();rt.act(obs,w);ts.append((time.perf_counter_ns()-t)/1e6)
        ts=np.asarray(ts)
        tests={}
        cases=[("wrong_obs",lambda:rt.act(np.zeros((1,47),np.float32),w)),
               ("wrong_pref",lambda:rt.act(obs,np.zeros((1,3),np.float32))),
               ("nan_obs",lambda:rt.act(np.full((1,48),np.nan,np.float32),w)),
               ("bad_sum",lambda:rt.act(obs,np.array([[.2,.2,.2,.2]],np.float32)))]
        for name,fn in cases:
            try:fn();tests[name]=False
            except ValueError:tests[name]=True
        fresh=Phase1EagerStateRuntime(str(ART),device=device);base=fresh.act(obs,w)
        hold=np.array_equal(base,fresh.act(obs,w,age_ms=30.0))
        stopped=fresh.act(obs,w,age_ms=41.0)
        stop=np.array_equal(stopped,np.zeros_like(stopped)) and fresh.estop_latched
        return {"repeat_bitwise_identical":repeat,"guards":tests,"stale_hold_30ms":hold,"stale_stop_41ms":stop,
                "latency_ms":{"median":float(np.median(ts)),"p95":float(np.percentile(ts,95)),
                              "p99":float(np.percentile(ts,99)),"max":float(ts.max()),
                              "over_20ms":int((ts>20).sum())}}
    def main():
        rep={"schema":"phase1_d2_eager_runtime_guard_v1"}
        for dev in ("cpu","cuda"):rep[dev]=run(dev)
        for dev in ("cpu","cuda"):
            q=rep[dev];q["pass"]=bool(q["repeat_bitwise_identical"] and all(q["guards"].values()) and
                                      q["stale_hold_30ms"] and q["stale_stop_41ms"] and
                                      q["latency_ms"]["p99"]<10 and q["latency_ms"]["over_20ms"]==0)
        rep["pass"]=bool(rep["cpu"]["pass"] and rep["cuda"]["pass"])
        (OUT/"runtime_guard.json").write_text(json.dumps(rep,indent=2)+"\n")
        print(json.dumps(rep,indent=2),flush=True)
    
    if True:main()

def run_phase1_d2_eager_state_parity():
    """Run former phase1_d2_eager_state_parity.py stage."""
    from pathlib import Path
    import hashlib,json,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4]
    from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
    from talon_rl.deployment.phase1 import Phase1DeploymentActor,Phase1EagerStateRuntime
    
    CK=ROOT/"runs/authority_isolated_simplex_edge_retain-2026-09-25/model_30.pt"
    PROBE=ROOT/"runs/update_functional_effect_audit-2026-09-24/fixed_probe_states.npz"
    ART=ROOT/"artifacts/phase1_canonical_actor_state.pt"
    OUT=ROOT/"runs/phase1_d2_eager_state_parity";OUT.mkdir(parents=True,exist_ok=True)
    PREFS=np.array([[.7,.1,.1,.1],[.1,.7,.1,.1],[.1,.1,.7,.1],[.1,.1,.1,.7],[.25]*4,[.4,.3,.2,.1]],np.float32)
    
    def sha(p):
     h=hashlib.sha256()
     with open(p,"rb") as f:
      for b in iter(lambda:f.read(1<<20),b""):h.update(b)
     return h.hexdigest()
    def main():
     state=torch.load(CK,map_location="cpu",weights_only=False)["model"]
     src_cpu=AuthorityIsolatedWideCritic(48,12).cpu();src_cpu.load_state_dict(state);src_cpu.eval()
     dep=Phase1DeploymentActor(src_cpu).cpu().eval()
     torch.save({"schema":"phase1_actor_state_v1","canonical_checkpoint_sha256":sha(CK),
                 "actor_state":dep.state_dict(),"obs_dim":48,"pref_dim":4,"action_dim":12},ART)
     obs=np.load(PROBE)["obs"].astype(np.float32)[:256]
     rep={"schema":"phase1_d2_eager_state_parity_v1","artifact_sha256":sha(ART)}
     for device in ("cpu","cuda"):
      src=AuthorityIsolatedWideCritic(48,12).to(device);src.load_state_dict(state);src.eval()
      rt=Phase1EagerStateRuntime(str(ART),device=device)
      mx=0.0
      with torch.inference_mode():
       for p in PREFS:
        w_np=np.repeat(p[None,:],len(obs),axis=0)
        ref=src.act_inference_with_preference(torch.from_numpy(obs).to(device),torch.from_numpy(w_np).to(device)).cpu().numpy()
        out=rt.act(obs,w_np)
        mx=max(mx,float(np.max(np.abs(ref-out))))
      rep[device]={"max_action_abs_error":mx,"pass":bool(mx==0.0)}
     rep["pass"]=bool(rep["cpu"]["pass"] and rep["cuda"]["pass"])
     (OUT/"eager_state_parity.json").write_text(json.dumps(rep,indent=2)+"\n")
     print(json.dumps(rep,indent=2),flush=True)
    
    if True:main()

def run_phase1_d2_obs_action_parity():
    """Run former phase1_d2_obs_action_parity.py stage."""
    from pathlib import Path
    import json,sys,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4]
    from isaaclab.app import AppLauncher
    app=AppLauncher({"headless":True,"enable_cameras":False}).app
    env=None
    try:
        import gymnasium as gym,isaaclab_tasks
        from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
        cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=4
        cfg.observations.policy.enable_corruption=False
        cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
        env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
        obs,_=env.reset(seed=260926);u=env.unwrapped
        robot=u.scene["robot"];mgr=u.observation_manager;actmgr=u.action_manager
        term=actmgr.get_term("joint_pos")
        rng=np.random.default_rng(260926)
        max_obs=0.0;max_target=0.0;rows=[]
        for step in range(32):
            raw=torch.tensor(rng.uniform(-1,1,size=(4,12)),device=u.device,dtype=torch.float32)
            obs_dict,_,_,_,_=env.step(raw)
            manager_obs=obs_dict["policy"]
            manual=torch.cat([
                robot.data.root_lin_vel_b,
                robot.data.root_ang_vel_b,
                robot.data.projected_gravity_b,
                u.command_manager.get_command("base_velocity"),
                robot.data.joint_pos-robot.data.default_joint_pos,
                robot.data.joint_vel-robot.data.default_joint_vel,
                actmgr.action,
            ],dim=-1)
            oe=float((manual-manager_obs).abs().max().cpu());max_obs=max(max_obs,oe)
            expected=robot.data.default_joint_pos+0.25*actmgr.action
            actual=term.processed_actions
            te=float((expected-actual).abs().max().cpu());max_target=max(max_target,te)
            rows.append({"step":step,"obs_max_abs":oe,"target_max_abs":te})
        # Structural checks.
        report={
          "schema":"phase1_d2_observation_action_contract_v1",
          "clean_observation_reconstruction_max_abs_error":max_obs,
          "action_target_mapping_max_abs_error":max_target,
          "clean_observation_gate":bool(max_obs<=1e-7),
          "action_mapping_gate":bool(max_target<=1e-7),
          "observation_dim":int(manager_obs.shape[-1]),
          "action_dim":int(actmgr.action.shape[-1]),
          "joint_names":list(robot.data.joint_names),
          "step_dt":float(u.step_dt),
          "corruption_disabled_for_reconstruction_test":True,
          "canonical_training_corruption_enabled":True,
          "rows":rows,
        }
        report["pass"]=report["clean_observation_gate"] and report["action_mapping_gate"]
        out=ROOT/"runs/phase1_d2_obs_action_parity";out.mkdir(parents=True,exist_ok=True)
        (out/"obs_action_parity.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps({k:v for k,v in report.items() if k!="rows"},indent=2),flush=True)
    finally:
        if env is not None:env.close()
        app.close()

def run_phase1_d2_runtime_parity():
    """Run former phase1_d2_runtime_parity.py stage."""
    from pathlib import Path
    import hashlib,json,time
    import numpy as np,torch
    
    ROOT=Path(__file__).resolve().parents[4]
    CK=ROOT/"runs/authority_isolated_simplex_edge_retain-2026-09-25/model_30.pt"
    PROBE=ROOT/"runs/update_functional_effect_audit-2026-09-24/fixed_probe_states.npz"
    OUT=ROOT/"runs/phase1_d2_runtime_parity";OUT.mkdir(parents=True,exist_ok=True)
    ART=ROOT/"artifacts/phase1_canonical_actor.pt";ART.parent.mkdir(parents=True,exist_ok=True)
    
    from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
    from talon_rl.deployment.phase1 import Phase1DeploymentActor,Phase1DeploymentRuntime
    
    PREFS=np.array([
     [.7,.1,.1,.1],[.1,.7,.1,.1],[.1,.1,.7,.1],[.1,.1,.1,.7],[.25,.25,.25,.25],
     [.4,.3,.2,.1],[.1,.2,.3,.4],[.55,.15,.15,.15],[.15,.55,.15,.15]],np.float32)
    
    def sha(p):
     h=hashlib.sha256()
     with open(p,"rb") as f:
      for b in iter(lambda:f.read(1<<20),b""):h.update(b)
     return h.hexdigest()
    def main():
     state=torch.load(CK,map_location="cpu",weights_only=False)["model"]
     src=AuthorityIsolatedWideCritic(48,12).cpu();src.load_state_dict(state);src.eval()
     dep=Phase1DeploymentActor(src).cpu().eval()
     scripted=torch.jit.script(dep);scripted.save(str(ART))
     obs=np.load(PROBE)["obs"].astype(np.float32)
     if len(obs)>2048:obs=obs[:2048]
     max_pre=0.0;max_act=0.0;rms_pre=[];rms_act=[]
     with torch.inference_mode():
      x=torch.from_numpy(obs)
      for p in PREFS:
       w=torch.from_numpy(np.repeat(p[None,:],len(obs),axis=0))
       a0=src.act_inference_with_preference(x,w);u0=src._actor_mean_with_preference(x,w)
       a1=scripted(x,w);u1=scripted.pre_tanh(x,w)
       du=(u1-u0).abs();da=(a1-a0).abs()
       max_pre=max(max_pre,float(du.max()));max_act=max(max_act,float(da.max()))
       rms_pre.append(float(torch.sqrt((du*du).mean())));rms_act.append(float(torch.sqrt((da*da).mean())))
     runtime=Phase1DeploymentRuntime(str(ART))
     x=obs[:1];w=PREFS[4:5]
     a_first=runtime.act(x,w)
     repeat_equal=all(np.array_equal(a_first,runtime.act(x,w)) for _ in range(20))
     # Offline latency: single-batch deployment call.
     samples=[]
     for _ in range(50):runtime.act(x,w)
     for _ in range(1000):
      t=time.perf_counter_ns();runtime.act(x,w);samples.append((time.perf_counter_ns()-t)/1e6)
     tests={}
     cases=[
      ("wrong_obs",lambda:runtime.act(np.zeros((1,47),np.float32),w)),
      ("wrong_pref",lambda:runtime.act(x,np.zeros((1,3),np.float32))),
      ("nan_obs",lambda:runtime.act(np.full((1,48),np.nan,np.float32),w)),
      ("nan_pref",lambda:runtime.act(x,np.full((1,4),np.nan,np.float32))),
      ("negative_pref",lambda:runtime.act(x,np.array([[1.1,-.1,0,0]],np.float32))),
      ("bad_sum",lambda:runtime.act(x,np.array([[.2,.2,.2,.2]],np.float32))),
     ]
     for name,fn in cases:
      try:fn();tests[name]=False
      except ValueError:tests[name]=True
     fresh=Phase1DeploymentRuntime(str(ART));fresh_action=fresh.act(x,w)
     held=fresh.act(x,w,age_ms=30.0)
     hold_ok=np.array_equal(fresh_action,held)
     stopped=fresh.act(x,w,age_ms=41.0)
     stop_ok=np.array_equal(stopped,np.zeros_like(stopped)) and fresh.estop_latched
     lat=np.asarray(samples)
     rep={
      "schema":"phase1_d2_runtime_parity_v1",
      "canonical_checkpoint_sha256":sha(CK),
      "artifact_sha256":sha(ART),
      "probe_count":int(len(obs)),
      "preference_count":int(len(PREFS)),
      "max_pre_tanh_abs_error":max_pre,
      "max_action_abs_error":max_act,
      "rms_pre_tanh_error_max":max(rms_pre),
      "rms_action_error_max":max(rms_act),
      "pre_tanh_gate":bool(max_pre<=1e-6),
      "action_gate":bool(max_act<=1e-6),
      "repeat_bitwise_identical":bool(repeat_equal),
      "input_guard_tests":tests,
      "stale_hold_30ms":bool(hold_ok),
      "stale_stop_41ms":bool(stop_ok),
      "latency_ms":{"median":float(np.median(lat)),"p95":float(np.percentile(lat,95)),
                    "p99":float(np.percentile(lat,99)),"max":float(lat.max()),
                    "over_20ms":int((lat>20).sum())},
     }
     rep["pass"]=bool(rep["pre_tanh_gate"] and rep["action_gate"] and repeat_equal and
                      all(tests.values()) and hold_ok and stop_ok and rep["latency_ms"]["p99"]<10 and
                      rep["latency_ms"]["over_20ms"]==0)
     (OUT/"runtime_parity.json").write_text(json.dumps(rep,indent=2)+"\n")
     print(json.dumps(rep,indent=2),flush=True)
    
    if True:main()

def run_phase1_d2_trace_parity():
    """Run former phase1_d2_trace_parity.py stage."""
    from pathlib import Path
    import json,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4]
    from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
    from talon_rl.deployment.phase1 import Phase1DeploymentActor
    
    CK=ROOT/"runs/authority_isolated_simplex_edge_retain-2026-09-25/model_30.pt"
    PROBE=ROOT/"runs/update_functional_effect_audit-2026-09-24/fixed_probe_states.npz"
    ART=ROOT/"artifacts/phase1_canonical_actor_trace.pt"
    OUT=ROOT/"runs/phase1_d2_trace_parity";OUT.mkdir(parents=True,exist_ok=True)
    PREFS=np.array([[.7,.1,.1,.1],[.1,.7,.1,.1],[.1,.1,.7,.1],[.1,.1,.1,.7],[.25]*4,
                    [.4,.3,.2,.1],[.1,.2,.3,.4]],np.float32)
    
    def main():
        state=torch.load(CK,map_location="cpu",weights_only=False)["model"]
        src=AuthorityIsolatedWideCritic(48,12).cpu();src.load_state_dict(state);src.eval()
        dep=Phase1DeploymentActor(src).cpu().eval()
        ex_obs=torch.zeros((8,48),dtype=torch.float32)
        ex_w=torch.full((8,4),.25,dtype=torch.float32)
        traced=torch.jit.trace(dep,(ex_obs,ex_w),check_trace=True)
        traced.save(str(ART))
        obs=np.load(PROBE)["obs"].astype(np.float32)
        if len(obs)>256:obs=obs[:256]
        max_act=0.0
        with torch.inference_mode():
            x=torch.from_numpy(obs)
            for p in PREFS:
                w=torch.from_numpy(np.repeat(p[None,:],len(obs),axis=0))
                a0=src.act_inference_with_preference(x,w)
                a1=traced(x,w)
                max_act=max(max_act,float((a1-a0).abs().max()))
        rep={"schema":"phase1_d2_trace_parity_v1","max_action_abs_error":max_act,
             "action_gate":max_act<=1e-6,"pre_tanh_not_traced":True}
        rep["pass"]=bool(rep["action_gate"])
        (OUT/"trace_parity.json").write_text(json.dumps(rep,indent=2)+"\n")
        print(json.dumps(rep,indent=2),flush=True)
    
    if True:main()

STAGES = {
    "phase1_d2_closed_loop": run_phase1_d2_closed_loop,
    "phase1_d2_contract_probe": run_phase1_d2_contract_probe,
    "phase1_d2_eager_runtime_guard": run_phase1_d2_eager_runtime_guard,
    "phase1_d2_eager_state_parity": run_phase1_d2_eager_state_parity,
    "phase1_d2_obs_action_parity": run_phase1_d2_obs_action_parity,
    "phase1_d2_runtime_parity": run_phase1_d2_runtime_parity,
    "phase1_d2_trace_parity": run_phase1_d2_trace_parity,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
