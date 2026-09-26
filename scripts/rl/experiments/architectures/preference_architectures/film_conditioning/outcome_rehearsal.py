"""Consolidated thematic experiment stages."""
from __future__ import annotations
import argparse

def run_v2b_outcome_estimator_centered_audit():
    """Run former v2b_outcome_estimator_centered_audit.py stage."""
    from pathlib import Path
    import sys,json,copy
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    BASE=ROOT/"runs/v2b_adam_continuous-2026-09-24/model_75.pt"
    RUN=ROOT/"runs/v2b_semantic_outcome_rehearsal_gate_v2-2026-09-24"
    OUT=RUN/"outcome_gradient_crn_centered_audit.json"
    ORDER=("T","A","O","S")
    PREFS={
    "T":np.array([.7,.1,.1,.1],np.float32),
    "A":np.array([.1,.7,.1,.1],np.float32),
    "O":np.array([.1,.1,.7,.1],np.float32),
    "S":np.array([.1,.1,.1,.7],np.float32),
    "C":np.array([.25,.25,.25,.25],np.float32)}
    IDX={"T":0,"A":1,"O":2,"S":3}
    PHYS={"T":"tracking_error","A":"ang_vel_xy","O":"tilt_deg","S":"action_rate"}
    H=32;NENV=8;WINDOW=8;KAPPA=.5
    REF_PHASES=(0,8,16,24)
    REF_SEEDS={"T":[911001,911101,911201,911301],
               "A":[912001,912101,912201,912301],
               "O":[913001,913101,913201,913301],
               "S":[914001,914101,914201,914301]}
    REPS=4
    CHECKPOINTS={"u4":RUN/"semantic_outcome_u4.pt","u7":RUN/"semantic_outcome_u7.pt"}
    
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def tilt_deg(q):
        _,x,y,_=[q[:,i] for i in range(4)]
        return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def actor_named_params(m):
        return [(n,p) for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std" or n.startswith("preference_embedding") or n.startswith("preference_film")]
    def flat_grad(loss,nps):
        gs=torch.autograd.grad(loss,[p for _,p in nps],allow_unused=True)
        return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for (_,p),g in zip(nps,gs)])
    def cos(a,b):
        den=float(a.norm()*b.norm())
        return float(torch.dot(a,b)/den) if den>1e-12 else 0.0
    
    def phys_vector(robot,env,a,prev):
        data=robot.data;cmd=env.unwrapped.command_manager.get_command("base_velocity")
        return {
          "tracking_error":(data.root_lin_vel_b[:,0]-cmd[:,0]).abs()+(data.root_ang_vel_b[:,2]-cmd[:,2]).abs(),
          "ang_vel_xy":torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1),
          "tilt_deg":tilt_deg(data.root_quat_w),
          "action_rate":torch.linalg.vector_norm(a-prev,dim=-1)}
    
    def collect(env,m,mgr,robot,seed,lab,noise_rep):
        from talon_rl.rewards.objectives import normalized_objective_vector
        w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1)
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        torch.manual_seed(seed+1234+noise_rep*100003)  # CRN: same action-noise stream for heavy and center
        obs=[];u=[];obj=[];phys=[];done=[]
        prev=torch.zeros((NENV,env.unwrapped.action_manager.total_action_dim),device="cuda")
        with torch.no_grad():
            for _ in range(H):
                a,lp,uu=m.act_with_preference_latent(cur,w)
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
                obs.append(cur.detach().clone());u.append(uu.detach().clone());obj.append(torch.tensor(vec,device="cuda"))
                pv=phys_vector(robot,env,a,prev);phys.append({k:v.detach().clone() for k,v in pv.items()})
                done.append((te|tr).detach().clone());prev=a;cur=ot(nxt).cuda()
        return {"obs":torch.stack(obs),"u":torch.stack(u),"obj":torch.stack(obj),
                "phys":{k:torch.stack([x[k] for x in phys]) for k in phys[0]},
                "done":torch.stack(done)}
    
    def margins(h,c,lab):
        j=IDX[lab];pk=PHYS[lab];out={}
        for phase in REF_PHASES:
            sl=slice(phase,phase+WINDOW)
            out[phase]={
              "obj_margin":(h["obj"][sl,:,j].mean(0)-c["obj"][sl,:,j].mean(0)).detach(),
              "phys_margin":(c["phys"][pk][sl].mean(0)-h["phys"][pk][sl].mean(0)).detach()}
        return out
    
    def capture_reference(env,m,mgr,robot,lab):
        entries=[]
        for seed in REF_SEEDS[lab]:
            h=collect(env,m,mgr,robot,seed,lab,0);c=collect(env,m,mgr,robot,seed,"C",0)
            mm=margins(h,c,lab)
            for phase,w in mm.items():
                for e in range(NENV):
                    entries.append({"seed":seed,"phase":phase,"env":e,
                                    "obj_ref":float(w["obj_margin"][e].cpu()),
                                    "phys_ref":float(w["phys_margin"][e].cpu())})
        return entries
    
    def outcome_grad(env,m,mgr,robot,refs,noise_rep,only_lab=None):
        nps=actor_named_params(m);losses=[];active=0;total=0
        labs=[only_lab] if only_lab else list(refs)
        for lab in labs:
            entries=refs[lab];byseed={}
            for x in entries:byseed.setdefault(x["seed"],[]).append(x)
            for seed,ents in byseed.items():
                h=collect(env,m,mgr,robot,seed,lab,noise_rep);c=collect(env,m,mgr,robot,seed,"C",noise_rep)
                mm=margins(h,c,lab)
                oh=h["obs"].reshape(H*NENV,-1);uh=h["u"].reshape(H*NENV,-1)
                oc=c["obs"].reshape(H*NENV,-1);uc=c["u"].reshape(H*NENV,-1)
                wh=torch.tensor(PREFS[lab],device="cuda").repeat(H*NENV,1)
                wc=torch.tensor(PREFS["C"],device="cuda").repeat(H*NENV,1)
                lph=m.logp_from_pre_tanh_with_preference(oh,wh,uh).reshape(H,NENV)
                lpc=m.logp_from_pre_tanh_with_preference(oc,wc,uc).reshape(H,NENV)
                for x in ents:
                    total+=1;phase=x["phase"];e=x["env"];cur=mm[phase]
                    mo=float(cur["obj_margin"][e].cpu());mp=float(cur["phys_margin"][e].cpu())
                    ov=x["obj_ref"]>1e-8;pv=x["phys_ref"]>1e-8
                    fo=KAPPA*x["obj_ref"] if ov else None;fp=KAPPA*x["phys_ref"] if pv else None
                    do=max(0.,fo-mo) if ov else 0.;dp=max(0.,fp-mp) if pv else 0.
                    if do<=0 and dp<=0:continue
                    active+=1;sl=slice(phase,phase+WINDOW)
                    oh_all=h["obj"][sl,:,IDX[lab]].mean(0);oc_all=c["obj"][sl,:,IDX[lab]].mean(0)
                    ph_all=h["phys"][PHYS[lab]][sl].mean(0);pc_all=c["phys"][PHYS[lab]][sl].mean(0)
                    def centered(v): return float((v[e]-(v.sum()-v[e])/(NENV-1)).cpu())
                    ohv=centered(oh_all);ocv=centered(oc_all);phv=centered(ph_all);pcv=centered(pc_all)
                    logh=lph[sl,e].mean();logc=lpc[sl,e].mean();loss=torch.tensor(0.,device="cuda")
                    if do>0:loss=loss+(do/(abs(fo)+1e-6))*(-ohv*logh+ocv*logc)
                    if dp>0:loss=loss+(dp/(abs(fp)+1e-6))*(-pcv*logc+phv*logh)
                    losses.append(loss)
        if not losses:
            return torch.zeros(sum(p.numel() for _,p in nps),device="cuda"),{"active":0,"total":total}
        loss=sum(losses)/len(losses);g=flat_grad(loss,nps).detach()
        return g,{"active":active,"total":total,"loss":float(loss.detach().cpu())}
    
    def stats(gs):
        G=torch.stack(gs);mean=G.mean(0)
        resid=G-mean
        noise=float(torch.sqrt(torch.mean(torch.sum(resid*resid,dim=1))).cpu())
        snr=float(mean.norm().cpu())/(noise+1e-12)
        norms=np.array([float(g.norm().cpu()) for g in gs])
        pair=[]
        for i in range(len(gs)):
            for j in range(i+1,len(gs)):pair.append(cos(gs[i],gs[j]))
        cmean=[cos(g,mean) for g in gs]
        return {"mean_norm":float(mean.norm().cpu()),"norm_mean":float(norms.mean()),
                "norm_cv":float(norms.std()/(norms.mean()+1e-12)),
                "pairwise_cos_mean":float(np.mean(pair)),"pairwise_cos_min":float(np.min(pair)),
                "cos_to_mean_mean":float(np.mean(cmean)),"cos_to_mean_min":float(np.min(cmean)),
                "snr":snr}
    
    def load_model(cls,path,obsdim,ad):
        m=cls(obsdim,ad).cuda();m.load_state_dict(torch.load(path,map_location="cuda",weights_only=False)["model"]);m.eval();return m
    
    def main():
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.conditioning.single_site_film import V2BSingleSiteFiLMActorCritic as AC
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=0);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim
            mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
            # reconstruct memories from acquisition checkpoints
            mS=load_model(AC,BASE,o.shape[-1],ad)
            mO=load_model(AC,RUN/"semantic_outcome_u2.pt",o.shape[-1],ad)
            mT=load_model(AC,RUN/"semantic_outcome_u3.pt",o.shape[-1],ad)
            refs={"S":capture_reference(env,mS,mgr,robot,"S"),
                  "O":capture_reference(env,mO,mgr,robot,"O"),
                  "T":capture_reference(env,mT,mgr,robot,"T")}
            out={"reps":REPS,"checkpoints":{}}
            for ck,path in CHECKPOINTS.items():
                m=load_model(AC,path,o.shape[-1],ad)
                combined=[];metas=[];per={"O":[]}
                for r in range(REPS):
                    g,meta=outcome_grad(env,m,mgr,robot,refs,r);combined.append(g);metas.append(meta)
                    for lab in per:
                        gg,_=outcome_grad(env,m,mgr,robot,refs,r,only_lab=lab);per[lab].append(gg)
                out["checkpoints"][ck]={"combined":stats(combined),
                                         "per_axis":{lab:stats(gs) for lab,gs in per.items()},
                                         "rep_meta":metas}
                OUT.write_text(json.dumps(out,indent=2)+"\n")
                print(ck,json.dumps(out["checkpoints"][ck],indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_v2b_outcome_estimator_repair_audit():
    """Run former v2b_outcome_estimator_repair_audit.py stage."""
    from pathlib import Path
    import sys,json,copy
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    BASE=ROOT/"runs/v2b_adam_continuous-2026-09-24/model_75.pt"
    RUN=ROOT/"runs/v2b_semantic_outcome_rehearsal_gate_v2-2026-09-24"
    OUT=RUN/"outcome_gradient_crn_audit.json"
    ORDER=("T","A","O","S")
    PREFS={
    "T":np.array([.7,.1,.1,.1],np.float32),
    "A":np.array([.1,.7,.1,.1],np.float32),
    "O":np.array([.1,.1,.7,.1],np.float32),
    "S":np.array([.1,.1,.1,.7],np.float32),
    "C":np.array([.25,.25,.25,.25],np.float32)}
    IDX={"T":0,"A":1,"O":2,"S":3}
    PHYS={"T":"tracking_error","A":"ang_vel_xy","O":"tilt_deg","S":"action_rate"}
    H=32;NENV=8;WINDOW=8;KAPPA=.5
    REF_PHASES=(0,8,16,24)
    REF_SEEDS={"T":[911001,911101,911201,911301],
               "A":[912001,912101,912201,912301],
               "O":[913001,913101,913201,913301],
               "S":[914001,914101,914201,914301]}
    REPS=4
    CHECKPOINTS={"u4":RUN/"semantic_outcome_u4.pt","u7":RUN/"semantic_outcome_u7.pt"}
    
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def tilt_deg(q):
        _,x,y,_=[q[:,i] for i in range(4)]
        return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def actor_named_params(m):
        return [(n,p) for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std" or n.startswith("preference_embedding") or n.startswith("preference_film")]
    def flat_grad(loss,nps):
        gs=torch.autograd.grad(loss,[p for _,p in nps],allow_unused=True)
        return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for (_,p),g in zip(nps,gs)])
    def cos(a,b):
        den=float(a.norm()*b.norm())
        return float(torch.dot(a,b)/den) if den>1e-12 else 0.0
    
    def phys_vector(robot,env,a,prev):
        data=robot.data;cmd=env.unwrapped.command_manager.get_command("base_velocity")
        return {
          "tracking_error":(data.root_lin_vel_b[:,0]-cmd[:,0]).abs()+(data.root_ang_vel_b[:,2]-cmd[:,2]).abs(),
          "ang_vel_xy":torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1),
          "tilt_deg":tilt_deg(data.root_quat_w),
          "action_rate":torch.linalg.vector_norm(a-prev,dim=-1)}
    
    def collect(env,m,mgr,robot,seed,lab,noise_rep):
        from talon_rl.rewards.objectives import normalized_objective_vector
        w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1)
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        torch.manual_seed(seed+1234+noise_rep*100003)  # CRN: same action-noise stream for heavy and center
        obs=[];u=[];obj=[];phys=[];done=[]
        prev=torch.zeros((NENV,env.unwrapped.action_manager.total_action_dim),device="cuda")
        with torch.no_grad():
            for _ in range(H):
                a,lp,uu=m.act_with_preference_latent(cur,w)
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
                obs.append(cur.detach().clone());u.append(uu.detach().clone());obj.append(torch.tensor(vec,device="cuda"))
                pv=phys_vector(robot,env,a,prev);phys.append({k:v.detach().clone() for k,v in pv.items()})
                done.append((te|tr).detach().clone());prev=a;cur=ot(nxt).cuda()
        return {"obs":torch.stack(obs),"u":torch.stack(u),"obj":torch.stack(obj),
                "phys":{k:torch.stack([x[k] for x in phys]) for k in phys[0]},
                "done":torch.stack(done)}
    
    def margins(h,c,lab):
        j=IDX[lab];pk=PHYS[lab];out={}
        for phase in REF_PHASES:
            sl=slice(phase,phase+WINDOW)
            out[phase]={
              "obj_margin":(h["obj"][sl,:,j].mean(0)-c["obj"][sl,:,j].mean(0)).detach(),
              "phys_margin":(c["phys"][pk][sl].mean(0)-h["phys"][pk][sl].mean(0)).detach()}
        return out
    
    def capture_reference(env,m,mgr,robot,lab):
        entries=[]
        for seed in REF_SEEDS[lab]:
            h=collect(env,m,mgr,robot,seed,lab,0);c=collect(env,m,mgr,robot,seed,"C",0)
            mm=margins(h,c,lab)
            for phase,w in mm.items():
                for e in range(NENV):
                    entries.append({"seed":seed,"phase":phase,"env":e,
                                    "obj_ref":float(w["obj_margin"][e].cpu()),
                                    "phys_ref":float(w["phys_margin"][e].cpu())})
        return entries
    
    def outcome_grad(env,m,mgr,robot,refs,noise_rep,only_lab=None):
        nps=actor_named_params(m);losses=[];active=0;total=0
        labs=[only_lab] if only_lab else list(refs)
        for lab in labs:
            entries=refs[lab];byseed={}
            for x in entries:byseed.setdefault(x["seed"],[]).append(x)
            for seed,ents in byseed.items():
                h=collect(env,m,mgr,robot,seed,lab,noise_rep);c=collect(env,m,mgr,robot,seed,"C",noise_rep)
                mm=margins(h,c,lab)
                oh=h["obs"].reshape(H*NENV,-1);uh=h["u"].reshape(H*NENV,-1)
                oc=c["obs"].reshape(H*NENV,-1);uc=c["u"].reshape(H*NENV,-1)
                wh=torch.tensor(PREFS[lab],device="cuda").repeat(H*NENV,1)
                wc=torch.tensor(PREFS["C"],device="cuda").repeat(H*NENV,1)
                lph=m.logp_from_pre_tanh_with_preference(oh,wh,uh).reshape(H,NENV)
                lpc=m.logp_from_pre_tanh_with_preference(oc,wc,uc).reshape(H,NENV)
                for x in ents:
                    total+=1;phase=x["phase"];e=x["env"];cur=mm[phase]
                    mo=float(cur["obj_margin"][e].cpu());mp=float(cur["phys_margin"][e].cpu())
                    ov=x["obj_ref"]>1e-8;pv=x["phys_ref"]>1e-8
                    fo=KAPPA*x["obj_ref"] if ov else None;fp=KAPPA*x["phys_ref"] if pv else None
                    do=max(0.,fo-mo) if ov else 0.;dp=max(0.,fp-mp) if pv else 0.
                    if do<=0 and dp<=0:continue
                    active+=1;sl=slice(phase,phase+WINDOW)
                    ohv=float(h["obj"][sl,e,IDX[lab]].mean().cpu());ocv=float(c["obj"][sl,e,IDX[lab]].mean().cpu())
                    phv=float(h["phys"][PHYS[lab]][sl,e].mean().cpu());pcv=float(c["phys"][PHYS[lab]][sl,e].mean().cpu())
                    logh=lph[sl,e].mean();logc=lpc[sl,e].mean();loss=torch.tensor(0.,device="cuda")
                    if do>0:loss=loss+(do/(abs(fo)+1e-6))*(-ohv*logh+ocv*logc)
                    if dp>0:loss=loss+(dp/(abs(fp)+1e-6))*(-pcv*logc+phv*logh)
                    losses.append(loss)
        if not losses:
            return torch.zeros(sum(p.numel() for _,p in nps),device="cuda"),{"active":0,"total":total}
        loss=sum(losses)/len(losses);g=flat_grad(loss,nps).detach()
        return g,{"active":active,"total":total,"loss":float(loss.detach().cpu())}
    
    def stats(gs):
        G=torch.stack(gs);mean=G.mean(0)
        resid=G-mean
        noise=float(torch.sqrt(torch.mean(torch.sum(resid*resid,dim=1))).cpu())
        snr=float(mean.norm().cpu())/(noise+1e-12)
        norms=np.array([float(g.norm().cpu()) for g in gs])
        pair=[]
        for i in range(len(gs)):
            for j in range(i+1,len(gs)):pair.append(cos(gs[i],gs[j]))
        cmean=[cos(g,mean) for g in gs]
        return {"mean_norm":float(mean.norm().cpu()),"norm_mean":float(norms.mean()),
                "norm_cv":float(norms.std()/(norms.mean()+1e-12)),
                "pairwise_cos_mean":float(np.mean(pair)),"pairwise_cos_min":float(np.min(pair)),
                "cos_to_mean_mean":float(np.mean(cmean)),"cos_to_mean_min":float(np.min(cmean)),
                "snr":snr}
    
    def load_model(cls,path,obsdim,ad):
        m=cls(obsdim,ad).cuda();m.load_state_dict(torch.load(path,map_location="cuda",weights_only=False)["model"]);m.eval();return m
    
    def main():
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.conditioning.single_site_film import V2BSingleSiteFiLMActorCritic as AC
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=0);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim
            mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
            # reconstruct memories from acquisition checkpoints
            mS=load_model(AC,BASE,o.shape[-1],ad)
            mO=load_model(AC,RUN/"semantic_outcome_u2.pt",o.shape[-1],ad)
            mT=load_model(AC,RUN/"semantic_outcome_u3.pt",o.shape[-1],ad)
            refs={"S":capture_reference(env,mS,mgr,robot,"S"),
                  "O":capture_reference(env,mO,mgr,robot,"O"),
                  "T":capture_reference(env,mT,mgr,robot,"T")}
            out={"reps":REPS,"checkpoints":{}}
            for ck,path in CHECKPOINTS.items():
                m=load_model(AC,path,o.shape[-1],ad)
                combined=[];metas=[];per={"O":[]}
                for r in range(REPS):
                    g,meta=outcome_grad(env,m,mgr,robot,refs,r);combined.append(g);metas.append(meta)
                    for lab in per:
                        gg,_=outcome_grad(env,m,mgr,robot,refs,r,only_lab=lab);per[lab].append(gg)
                out["checkpoints"][ck]={"combined":stats(combined),
                                         "per_axis":{lab:stats(gs) for lab,gs in per.items()},
                                         "rep_meta":metas}
                OUT.write_text(json.dumps(out,indent=2)+"\n")
                print(ck,json.dumps(out["checkpoints"][ck],indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_v2b_outcome_estimator_replica_avg_audit():
    """Run former v2b_outcome_estimator_replica_avg_audit.py stage."""
    from pathlib import Path
    import sys,json,copy
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    BASE=ROOT/"runs/v2b_adam_continuous-2026-09-24/model_75.pt"
    RUN=ROOT/"runs/v2b_semantic_outcome_rehearsal_gate_v2-2026-09-24"
    OUT=RUN/"outcome_gradient_replica_avg_audit.json"
    ORDER=("T","A","O","S")
    PREFS={
    "T":np.array([.7,.1,.1,.1],np.float32),
    "A":np.array([.1,.7,.1,.1],np.float32),
    "O":np.array([.1,.1,.7,.1],np.float32),
    "S":np.array([.1,.1,.1,.7],np.float32),
    "C":np.array([.25,.25,.25,.25],np.float32)}
    IDX={"T":0,"A":1,"O":2,"S":3}
    PHYS={"T":"tracking_error","A":"ang_vel_xy","O":"tilt_deg","S":"action_rate"}
    H=32;NENV=8;WINDOW=8;KAPPA=.5
    REF_PHASES=(0,8,16,24)
    REF_SEEDS={"T":[911001,911101,911201,911301],
               "A":[912001,912101,912201,912301],
               "O":[913001,913101,913201,913301],
               "S":[914001,914101,914201,914301]}
    REPS=8
    CHECKPOINTS={"u4":RUN/"semantic_outcome_u4.pt","u7":RUN/"semantic_outcome_u7.pt"}
    
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def tilt_deg(q):
        _,x,y,_=[q[:,i] for i in range(4)]
        return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def actor_named_params(m):
        return [(n,p) for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std" or n.startswith("preference_embedding") or n.startswith("preference_film")]
    def flat_grad(loss,nps):
        gs=torch.autograd.grad(loss,[p for _,p in nps],allow_unused=True)
        return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for (_,p),g in zip(nps,gs)])
    def cos(a,b):
        den=float(a.norm()*b.norm())
        return float(torch.dot(a,b)/den) if den>1e-12 else 0.0
    
    def phys_vector(robot,env,a,prev):
        data=robot.data;cmd=env.unwrapped.command_manager.get_command("base_velocity")
        return {
          "tracking_error":(data.root_lin_vel_b[:,0]-cmd[:,0]).abs()+(data.root_ang_vel_b[:,2]-cmd[:,2]).abs(),
          "ang_vel_xy":torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1),
          "tilt_deg":tilt_deg(data.root_quat_w),
          "action_rate":torch.linalg.vector_norm(a-prev,dim=-1)}
    
    def collect(env,m,mgr,robot,seed,lab,noise_rep):
        from talon_rl.rewards.objectives import normalized_objective_vector
        w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1)
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        torch.manual_seed(seed+1234+noise_rep*100003)  # CRN: same action-noise stream for heavy and center
        obs=[];u=[];obj=[];phys=[];done=[]
        prev=torch.zeros((NENV,env.unwrapped.action_manager.total_action_dim),device="cuda")
        with torch.no_grad():
            for _ in range(H):
                a,lp,uu=m.act_with_preference_latent(cur,w)
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
                obs.append(cur.detach().clone());u.append(uu.detach().clone());obj.append(torch.tensor(vec,device="cuda"))
                pv=phys_vector(robot,env,a,prev);phys.append({k:v.detach().clone() for k,v in pv.items()})
                done.append((te|tr).detach().clone());prev=a;cur=ot(nxt).cuda()
        return {"obs":torch.stack(obs),"u":torch.stack(u),"obj":torch.stack(obj),
                "phys":{k:torch.stack([x[k] for x in phys]) for k in phys[0]},
                "done":torch.stack(done)}
    
    def margins(h,c,lab):
        j=IDX[lab];pk=PHYS[lab];out={}
        for phase in REF_PHASES:
            sl=slice(phase,phase+WINDOW)
            out[phase]={
              "obj_margin":(h["obj"][sl,:,j].mean(0)-c["obj"][sl,:,j].mean(0)).detach(),
              "phys_margin":(c["phys"][pk][sl].mean(0)-h["phys"][pk][sl].mean(0)).detach()}
        return out
    
    def capture_reference(env,m,mgr,robot,lab):
        entries=[]
        for seed in REF_SEEDS[lab]:
            h=collect(env,m,mgr,robot,seed,lab,0);c=collect(env,m,mgr,robot,seed,"C",0)
            mm=margins(h,c,lab)
            for phase,w in mm.items():
                for e in range(NENV):
                    entries.append({"seed":seed,"phase":phase,"env":e,
                                    "obj_ref":float(w["obj_margin"][e].cpu()),
                                    "phys_ref":float(w["phys_margin"][e].cpu())})
        return entries
    
    def outcome_grad(env,m,mgr,robot,refs,noise_rep,only_lab=None):
        nps=actor_named_params(m);losses=[];active=0;total=0
        labs=[only_lab] if only_lab else list(refs)
        for lab in labs:
            entries=refs[lab];byseed={}
            for x in entries:byseed.setdefault(x["seed"],[]).append(x)
            for seed,ents in byseed.items():
                h=collect(env,m,mgr,robot,seed,lab,noise_rep);c=collect(env,m,mgr,robot,seed,"C",noise_rep)
                mm=margins(h,c,lab)
                oh=h["obs"].reshape(H*NENV,-1);uh=h["u"].reshape(H*NENV,-1)
                oc=c["obs"].reshape(H*NENV,-1);uc=c["u"].reshape(H*NENV,-1)
                wh=torch.tensor(PREFS[lab],device="cuda").repeat(H*NENV,1)
                wc=torch.tensor(PREFS["C"],device="cuda").repeat(H*NENV,1)
                lph=m.logp_from_pre_tanh_with_preference(oh,wh,uh).reshape(H,NENV)
                lpc=m.logp_from_pre_tanh_with_preference(oc,wc,uc).reshape(H,NENV)
                for x in ents:
                    total+=1;phase=x["phase"];e=x["env"];cur=mm[phase]
                    mo=float(cur["obj_margin"][e].cpu());mp=float(cur["phys_margin"][e].cpu())
                    ov=x["obj_ref"]>1e-8;pv=x["phys_ref"]>1e-8
                    fo=KAPPA*x["obj_ref"] if ov else None;fp=KAPPA*x["phys_ref"] if pv else None
                    do=max(0.,fo-mo) if ov else 0.;dp=max(0.,fp-mp) if pv else 0.
                    if do<=0 and dp<=0:continue
                    active+=1;sl=slice(phase,phase+WINDOW)
                    oh_all=h["obj"][sl,:,IDX[lab]].mean(0);oc_all=c["obj"][sl,:,IDX[lab]].mean(0)
                    ph_all=h["phys"][PHYS[lab]][sl].mean(0);pc_all=c["phys"][PHYS[lab]][sl].mean(0)
                    def centered(v): return float((v[e]-(v.sum()-v[e])/(NENV-1)).cpu())
                    ohv=centered(oh_all);ocv=centered(oc_all);phv=centered(ph_all);pcv=centered(pc_all)
                    logh=lph[sl,e].mean();logc=lpc[sl,e].mean();loss=torch.tensor(0.,device="cuda")
                    if do>0:loss=loss+(do/(abs(fo)+1e-6))*(-ohv*logh+ocv*logc)
                    if dp>0:loss=loss+(dp/(abs(fp)+1e-6))*(-pcv*logc+phv*logh)
                    losses.append(loss)
        if not losses:
            return torch.zeros(sum(p.numel() for _,p in nps),device="cuda"),{"active":0,"total":total}
        loss=sum(losses)/len(losses);g=flat_grad(loss,nps).detach()
        return g,{"active":active,"total":total,"loss":float(loss.detach().cpu())}
    
    def stats(gs):
        G=torch.stack(gs);mean=G.mean(0)
        resid=G-mean
        noise=float(torch.sqrt(torch.mean(torch.sum(resid*resid,dim=1))).cpu())
        snr=float(mean.norm().cpu())/(noise+1e-12)
        norms=np.array([float(g.norm().cpu()) for g in gs])
        pair=[]
        for i in range(len(gs)):
            for j in range(i+1,len(gs)):pair.append(cos(gs[i],gs[j]))
        cmean=[cos(g,mean) for g in gs]
        return {"mean_norm":float(mean.norm().cpu()),"norm_mean":float(norms.mean()),
                "norm_cv":float(norms.std()/(norms.mean()+1e-12)),
                "pairwise_cos_mean":float(np.mean(pair)),"pairwise_cos_min":float(np.min(pair)),
                "cos_to_mean_mean":float(np.mean(cmean)),"cos_to_mean_min":float(np.min(cmean)),
                "snr":snr}
    
    def grouped_stats(gs,n):
        groups=[torch.stack(gs[i:i+n]).mean(0) for i in range(0,len(gs),n)]
        return stats(groups)
    
    def load_model(cls,path,obsdim,ad):
        m=cls(obsdim,ad).cuda();m.load_state_dict(torch.load(path,map_location="cuda",weights_only=False)["model"]);m.eval();return m
    
    def main():
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.conditioning.single_site_film import V2BSingleSiteFiLMActorCritic as AC
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=0);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim
            mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
            # reconstruct memories from acquisition checkpoints
            mS=load_model(AC,BASE,o.shape[-1],ad)
            mO=load_model(AC,RUN/"semantic_outcome_u2.pt",o.shape[-1],ad)
            mT=load_model(AC,RUN/"semantic_outcome_u3.pt",o.shape[-1],ad)
            refs={"S":capture_reference(env,mS,mgr,robot,"S"),
                  "O":capture_reference(env,mO,mgr,robot,"O"),
                  "T":capture_reference(env,mT,mgr,robot,"T")}
            out={"reps":REPS,"checkpoints":{}}
            for ck,path in CHECKPOINTS.items():
                m=load_model(AC,path,o.shape[-1],ad)
                combined=[];metas=[];per={"O":[]}
                for r in range(REPS):
                    g,meta=outcome_grad(env,m,mgr,robot,refs,r);combined.append(g);metas.append(meta)
                    for lab in per:
                        gg,_=outcome_grad(env,m,mgr,robot,refs,r,only_lab=lab);per[lab].append(gg)
                out["checkpoints"][ck]={"single_replica":{"combined":stats(combined),"per_axis":{lab:stats(gs) for lab,gs in per.items()}},
                                         "avg2":{"combined":grouped_stats(combined,2),"per_axis":{lab:grouped_stats(gs,2) for lab,gs in per.items()}},
                                         "avg4":{"combined":grouped_stats(combined,4),"per_axis":{lab:grouped_stats(gs,4) for lab,gs in per.items()}},
                                         "rep_meta":metas}
                OUT.write_text(json.dumps(out,indent=2)+"\n")
                print(ck,json.dumps(out["checkpoints"][ck],indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_v2b_outcome_pathwise_feasibility_audit():
    """Run former v2b_outcome_pathwise_feasibility_audit.py stage."""
    from pathlib import Path
    import sys,json,copy
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    BASE=ROOT/"runs/v2b_adam_continuous-2026-09-24/model_75.pt"
    RUN=ROOT/"runs/v2b_semantic_outcome_rehearsal_gate_v2-2026-09-24"
    OUT=RUN/"outcome_pathwise_feasibility_audit.json"
    ORDER=("T","A","O","S")
    PREFS={
    "T":np.array([.7,.1,.1,.1],np.float32),
    "A":np.array([.1,.7,.1,.1],np.float32),
    "O":np.array([.1,.1,.7,.1],np.float32),
    "S":np.array([.1,.1,.1,.7],np.float32),
    "C":np.array([.25,.25,.25,.25],np.float32)}
    IDX={"T":0,"A":1,"O":2,"S":3}
    PHYS={"T":"tracking_error","A":"ang_vel_xy","O":"tilt_deg","S":"action_rate"}
    H=32;NENV=8;WINDOW=8;KAPPA=.5;HFD=.03
    REF_PHASES=(0,8,16,24)
    REF_SEEDS={"T":[911001,911101,911201,911301],
               "A":[912001,912101,912201,912301],
               "O":[913001,913101,913201,913301],
               "S":[914001,914101,914201,914301]}
    REPS=4
    CHECKPOINTS={"u4":RUN/"semantic_outcome_u4.pt","u7":RUN/"semantic_outcome_u7.pt"}
    
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def tilt_deg(q):
        _,x,y,_=[q[:,i] for i in range(4)]
        return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def actor_named_params(m):
        return [(n,p) for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std" or n.startswith("preference_embedding") or n.startswith("preference_film")]
    def flat_grad(loss,nps):
        gs=torch.autograd.grad(loss,[p for _,p in nps],allow_unused=True)
        return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for (_,p),g in zip(nps,gs)])
    def cos(a,b):
        den=float(a.norm()*b.norm())
        return float(torch.dot(a,b)/den) if den>1e-12 else 0.0
    
    def phys_vector(robot,env,a,prev):
        data=robot.data;cmd=env.unwrapped.command_manager.get_command("base_velocity")
        return {
          "tracking_error":(data.root_lin_vel_b[:,0]-cmd[:,0]).abs()+(data.root_ang_vel_b[:,2]-cmd[:,2]).abs(),
          "ang_vel_xy":torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1),
          "tilt_deg":tilt_deg(data.root_quat_w),
          "action_rate":torch.linalg.vector_norm(a-prev,dim=-1)}
    
    def collect(env,m,mgr,robot,seed,lab,noise_rep,delta=None,ds=None,sign=0):
        from talon_rl.rewards.objectives import normalized_objective_vector
        w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1)
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        torch.manual_seed(seed+1234+noise_rep*100003)
        obs=[];u=[];obj=[];phys=[];done=[]
        prev=torch.zeros((NENV,env.unwrapped.action_manager.total_action_dim),device="cuda")
        with torch.no_grad():
            for t in range(H):
                dist=m._pre_tanh_dist_with_preference(cur,w);loc=dist.loc;std=dist.scale
                z=torch.randn_like(loc)
                if sign and delta is not None:
                    std_eff=std*torch.exp(sign*HFD*ds) if ds is not None else std
                    uu=loc+std_eff*z+sign*HFD*delta[t]
                else:
                    uu=loc+std*z
                a=torch.tanh(uu)*m.ACTION_CLIP
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
                obs.append(cur.detach().clone());u.append(uu.detach().clone());obj.append(torch.tensor(vec,device="cuda"))
                pv=phys_vector(robot,env,a,prev);phys.append({k:v.detach().clone() for k,v in pv.items()})
                done.append((te|tr).detach().clone());prev=a;cur=ot(nxt).cuda()
        return {"obs":torch.stack(obs),"u":torch.stack(u),"obj":torch.stack(obj),
                "phys":{k:torch.stack([x[k] for x in phys]) for k in phys[0]},
                "done":torch.stack(done)}
    
    def margins(h,c,lab):
        j=IDX[lab];pk=PHYS[lab];out={}
        for phase in REF_PHASES:
            sl=slice(phase,phase+WINDOW)
            out[phase]={
              "obj_margin":(h["obj"][sl,:,j].mean(0)-c["obj"][sl,:,j].mean(0)).detach(),
              "phys_margin":(c["phys"][pk][sl].mean(0)-h["phys"][pk][sl].mean(0)).detach()}
        return out
    
    def capture_reference(env,m,mgr,robot,lab):
        entries=[]
        for seed in REF_SEEDS[lab]:
            h=collect(env,m,mgr,robot,seed,lab,0);c=collect(env,m,mgr,robot,seed,"C",0)
            mm=margins(h,c,lab)
            for phase,w in mm.items():
                for e in range(NENV):
                    entries.append({"seed":seed,"phase":phase,"env":e,
                                    "obj_ref":float(w["obj_margin"][e].cpu()),
                                    "phys_ref":float(w["phys_margin"][e].cpu())})
        return entries
    
    def outcome_grad(env,m,mgr,robot,refs,noise_rep,only_lab=None):
        nps=actor_named_params(m);losses=[];active=0;total=0
        labs=[only_lab] if only_lab else list(refs)
        for lab in labs:
            entries=refs[lab];byseed={}
            for x in entries:byseed.setdefault(x["seed"],[]).append(x)
            for seed,ents in byseed.items():
                h=collect(env,m,mgr,robot,seed,lab,noise_rep);c=collect(env,m,mgr,robot,seed,"C",noise_rep)
                mm=margins(h,c,lab)
                oh=h["obs"].reshape(H*NENV,-1);uh=h["u"].reshape(H*NENV,-1)
                oc=c["obs"].reshape(H*NENV,-1);uc=c["u"].reshape(H*NENV,-1)
                wh=torch.tensor(PREFS[lab],device="cuda").repeat(H*NENV,1)
                wc=torch.tensor(PREFS["C"],device="cuda").repeat(H*NENV,1)
                lph=m.logp_from_pre_tanh_with_preference(oh,wh,uh).reshape(H,NENV)
                lpc=m.logp_from_pre_tanh_with_preference(oc,wc,uc).reshape(H,NENV)
                for x in ents:
                    total+=1;phase=x["phase"];e=x["env"];cur=mm[phase]
                    mo=float(cur["obj_margin"][e].cpu());mp=float(cur["phys_margin"][e].cpu())
                    ov=x["obj_ref"]>1e-8;pv=x["phys_ref"]>1e-8
                    fo=KAPPA*x["obj_ref"] if ov else None;fp=KAPPA*x["phys_ref"] if pv else None
                    do=max(0.,fo-mo) if ov else 0.;dp=max(0.,fp-mp) if pv else 0.
                    if do<=0 and dp<=0:continue
                    active+=1;sl=slice(phase,phase+WINDOW)
                    ohv=float(h["obj"][sl,e,IDX[lab]].mean().cpu());ocv=float(c["obj"][sl,e,IDX[lab]].mean().cpu())
                    phv=float(h["phys"][PHYS[lab]][sl,e].mean().cpu());pcv=float(c["phys"][PHYS[lab]][sl,e].mean().cpu())
                    logh=lph[sl,e].mean();logc=lpc[sl,e].mean();loss=torch.tensor(0.,device="cuda")
                    if do>0:loss=loss+(do/(abs(fo)+1e-6))*(-ohv*logh+ocv*logc)
                    if dp>0:loss=loss+(dp/(abs(fp)+1e-6))*(-pcv*logc+phv*logh)
                    losses.append(loss)
        if not losses:
            return torch.zeros(sum(p.numel() for _,p in nps),device="cuda"),{"active":0,"total":total}
        loss=sum(losses)/len(losses);g=flat_grad(loss,nps).detach()
        return g,{"active":active,"total":total,"loss":float(loss.detach().cpu())}
    
    def semantic_score_from_active(h,c,lab,active):
        mm=margins(h,c,lab);vals=[]
        for x in active:
            cur=mm[x["phase"]];e=x["env"];v=0.0
            if x["sev_obj"]>0: v+=x["sev_obj"]*float(cur["obj_margin"][e].cpu())
            if x["sev_phys"]>0: v+=x["sev_phys"]*float(cur["phys_margin"][e].cpu())
            vals.append(v)
        return float(np.mean(vals)) if vals else 0.0
    
    def pathwise_grad(env,m,mgr,robot,refs,noise_rep,only_lab=None):
        nps=actor_named_params(m); nonstd=[(n,p) for n,p in nps if n!="log_std"]
        labs=[only_lab] if only_lab else list(refs); axis_grads=[];meta={}
        ad=env.unwrapped.action_manager.total_action_dim
        for lab in labs:
            seed_grads=[];active_total=0
            byseed={}
            for x in refs[lab]: byseed.setdefault(x["seed"],[]).append(x)
            for seed,ents in byseed.items():
                h0=collect(env,m,mgr,robot,seed,lab,noise_rep);c0=collect(env,m,mgr,robot,seed,"C",noise_rep)
                mm0=margins(h0,c0,lab);active=[]
                for x in ents:
                    cur=mm0[x["phase"]];e=x["env"];mo=float(cur["obj_margin"][e]);mp=float(cur["phys_margin"][e])
                    ov=x["obj_ref"]>1e-8;pv=x["phys_ref"]>1e-8
                    fo=KAPPA*x["obj_ref"] if ov else None;fp=KAPPA*x["phys_ref"] if pv else None
                    do=max(0.,fo-mo) if ov else 0.;dp=max(0.,fp-mp) if pv else 0.
                    if do<=0 and dp<=0: continue
                    active.append({"phase":x["phase"],"env":e,
                                   "sev_obj":do/(abs(fo)+1e-6) if do>0 else 0.0,
                                   "sev_phys":dp/(abs(fp)+1e-6) if dp>0 else 0.0})
                active_total+=len(active)
                if not active: continue
                gen=torch.Generator(device="cuda");gen.manual_seed(15000000+noise_rep*100003+seed+IDX[lab]*997)
                dh=(torch.randint(0,2,(H,NENV,ad),device="cuda",generator=gen,dtype=torch.int64)*2-1).float()
                dc=(torch.randint(0,2,(H,NENV,ad),device="cuda",generator=gen,dtype=torch.int64)*2-1).float()
                ds=(torch.randint(0,2,(ad,),device="cuda",generator=gen,dtype=torch.int64)*2-1).float()
                hp=collect(env,m,mgr,robot,seed,lab,noise_rep,dh,ds,+1);cp=collect(env,m,mgr,robot,seed,"C",noise_rep,dc,ds,+1)
                hm=collect(env,m,mgr,robot,seed,lab,noise_rep,dh,ds,-1);cm=collect(env,m,mgr,robot,seed,"C",noise_rep,dc,ds,-1)
                fd=(semantic_score_from_active(hp,cp,lab,active)-semantic_score_from_active(hm,cm,lab,active))/(2*HFD)
                oh=h0["obs"].reshape(H*NENV,-1);oc=c0["obs"].reshape(H*NENV,-1)
                wh=torch.tensor(PREFS[lab],device="cuda").repeat(H*NENV,1);wc=torch.tensor(PREFS["C"],device="cuda").repeat(H*NENV,1)
                lh=m._pre_tanh_dist_with_preference(oh,wh).loc.reshape(H,NENV,ad)
                lc=m._pre_tanh_dist_with_preference(oc,wc).loc.reshape(H,NENV,ad)
                sur=-(fd*((lh*dh).sum()+(lc*dc).sum())/(2*H*NENV))
                ga=flat_grad(sur,nonstd).detach();parts=[];off=0
                for n,p in nps:
                    if n=="log_std": parts.append((-fd*ds).reshape(-1))
                    else:
                        nn=p.numel();parts.append(ga[off:off+nn]);off+=nn
                seed_grads.append(torch.cat(parts))
            z=torch.zeros(sum(p.numel() for _,p in nps),device="cuda")
            g=torch.stack(seed_grads).mean(0) if seed_grads else z
            axis_grads.append(g);meta[lab]={"active":active_total,"seed_grads":len(seed_grads)}
        g=torch.stack(axis_grads).mean(0) if axis_grads else torch.zeros(sum(p.numel() for _,p in nps),device="cuda")
        axis_map={lab:gg for lab,gg in zip(labs,axis_grads)}
        return g,meta,axis_map
    
    def stats(gs):
        G=torch.stack(gs);mean=G.mean(0)
        resid=G-mean
        noise=float(torch.sqrt(torch.mean(torch.sum(resid*resid,dim=1))).cpu())
        snr=float(mean.norm().cpu())/(noise+1e-12)
        norms=np.array([float(g.norm().cpu()) for g in gs])
        pair=[]
        for i in range(len(gs)):
            for j in range(i+1,len(gs)):pair.append(cos(gs[i],gs[j]))
        cmean=[cos(g,mean) for g in gs]
        return {"mean_norm":float(mean.norm().cpu()),"norm_mean":float(norms.mean()),
                "norm_cv":float(norms.std()/(norms.mean()+1e-12)),
                "pairwise_cos_mean":float(np.mean(pair)),"pairwise_cos_min":float(np.min(pair)),
                "cos_to_mean_mean":float(np.mean(cmean)),"cos_to_mean_min":float(np.min(cmean)),
                "snr":snr}
    
    def load_model(cls,path,obsdim,ad):
        m=cls(obsdim,ad).cuda();m.load_state_dict(torch.load(path,map_location="cuda",weights_only=False)["model"]);m.eval();return m
    
    def main():
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.conditioning.single_site_film import V2BSingleSiteFiLMActorCritic as AC
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=0);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim
            mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
            # reconstruct memories from acquisition checkpoints
            mS=load_model(AC,BASE,o.shape[-1],ad)
            mO=load_model(AC,RUN/"semantic_outcome_u2.pt",o.shape[-1],ad)
            mT=load_model(AC,RUN/"semantic_outcome_u3.pt",o.shape[-1],ad)
            refs={"S":capture_reference(env,mS,mgr,robot,"S"),
                  "O":capture_reference(env,mO,mgr,robot,"O"),
                  "T":capture_reference(env,mT,mgr,robot,"T")}
            out={"reps":REPS,"checkpoints":{}}
            for ck,path in CHECKPOINTS.items():
                m=load_model(AC,path,o.shape[-1],ad)
                combined=[];metas=[];per={lab:[] for lab in refs}
                for r in range(REPS):
                    g,meta,axis_map=pathwise_grad(env,m,mgr,robot,refs,r);combined.append(g);metas.append(meta)
                    for lab in refs: per[lab].append(axis_map[lab])
                out["checkpoints"][ck]={"combined":stats(combined),
                                         "per_axis":{lab:stats(gs) for lab,gs in per.items()},
                                         "rep_meta":metas}
                OUT.write_text(json.dumps(out,indent=2)+"\n")
                print(ck,json.dumps(out["checkpoints"][ck],indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_v2b_outcome_rehearsal_gradient_snr_audit():
    """Run former v2b_outcome_rehearsal_gradient_snr_audit.py stage."""
    from pathlib import Path
    import sys,json,copy
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    BASE=ROOT/"runs/v2b_adam_continuous-2026-09-24/model_75.pt"
    RUN=ROOT/"runs/v2b_semantic_outcome_rehearsal_gate_v2-2026-09-24"
    OUT=RUN/"outcome_gradient_snr_audit.json"
    ORDER=("T","A","O","S")
    PREFS={
    "T":np.array([.7,.1,.1,.1],np.float32),
    "A":np.array([.1,.7,.1,.1],np.float32),
    "O":np.array([.1,.1,.7,.1],np.float32),
    "S":np.array([.1,.1,.1,.7],np.float32),
    "C":np.array([.25,.25,.25,.25],np.float32)}
    IDX={"T":0,"A":1,"O":2,"S":3}
    PHYS={"T":"tracking_error","A":"ang_vel_xy","O":"tilt_deg","S":"action_rate"}
    H=32;NENV=8;WINDOW=8;KAPPA=.5
    REF_PHASES=(0,8,16,24)
    REF_SEEDS={"T":[911001,911101,911201,911301],
               "A":[912001,912101,912201,912301],
               "O":[913001,913101,913201,913301],
               "S":[914001,914101,914201,914301]}
    REPS=4
    CHECKPOINTS={"u4":RUN/"semantic_outcome_u4.pt","u7":RUN/"semantic_outcome_u7.pt"}
    
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def tilt_deg(q):
        _,x,y,_=[q[:,i] for i in range(4)]
        return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def actor_named_params(m):
        return [(n,p) for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std" or n.startswith("preference_embedding") or n.startswith("preference_film")]
    def flat_grad(loss,nps):
        gs=torch.autograd.grad(loss,[p for _,p in nps],allow_unused=True)
        return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for (_,p),g in zip(nps,gs)])
    def cos(a,b):
        den=float(a.norm()*b.norm())
        return float(torch.dot(a,b)/den) if den>1e-12 else 0.0
    
    def phys_vector(robot,env,a,prev):
        data=robot.data;cmd=env.unwrapped.command_manager.get_command("base_velocity")
        return {
          "tracking_error":(data.root_lin_vel_b[:,0]-cmd[:,0]).abs()+(data.root_ang_vel_b[:,2]-cmd[:,2]).abs(),
          "ang_vel_xy":torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1),
          "tilt_deg":tilt_deg(data.root_quat_w),
          "action_rate":torch.linalg.vector_norm(a-prev,dim=-1)}
    
    def collect(env,m,mgr,robot,seed,lab,noise_rep):
        from talon_rl.rewards.objectives import normalized_objective_vector
        w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1)
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        torch.manual_seed(seed+1234+(0 if lab=="C" else IDX.get(lab,0))*17+noise_rep*100003)
        obs=[];u=[];obj=[];phys=[];done=[]
        prev=torch.zeros((NENV,env.unwrapped.action_manager.total_action_dim),device="cuda")
        with torch.no_grad():
            for _ in range(H):
                a,lp,uu=m.act_with_preference_latent(cur,w)
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
                obs.append(cur.detach().clone());u.append(uu.detach().clone());obj.append(torch.tensor(vec,device="cuda"))
                pv=phys_vector(robot,env,a,prev);phys.append({k:v.detach().clone() for k,v in pv.items()})
                done.append((te|tr).detach().clone());prev=a;cur=ot(nxt).cuda()
        return {"obs":torch.stack(obs),"u":torch.stack(u),"obj":torch.stack(obj),
                "phys":{k:torch.stack([x[k] for x in phys]) for k in phys[0]},
                "done":torch.stack(done)}
    
    def margins(h,c,lab):
        j=IDX[lab];pk=PHYS[lab];out={}
        for phase in REF_PHASES:
            sl=slice(phase,phase+WINDOW)
            out[phase]={
              "obj_margin":(h["obj"][sl,:,j].mean(0)-c["obj"][sl,:,j].mean(0)).detach(),
              "phys_margin":(c["phys"][pk][sl].mean(0)-h["phys"][pk][sl].mean(0)).detach()}
        return out
    
    def capture_reference(env,m,mgr,robot,lab):
        entries=[]
        for seed in REF_SEEDS[lab]:
            h=collect(env,m,mgr,robot,seed,lab,0);c=collect(env,m,mgr,robot,seed,"C",0)
            mm=margins(h,c,lab)
            for phase,w in mm.items():
                for e in range(NENV):
                    entries.append({"seed":seed,"phase":phase,"env":e,
                                    "obj_ref":float(w["obj_margin"][e].cpu()),
                                    "phys_ref":float(w["phys_margin"][e].cpu())})
        return entries
    
    def outcome_grad(env,m,mgr,robot,refs,noise_rep,only_lab=None):
        nps=actor_named_params(m);losses=[];active=0;total=0
        labs=[only_lab] if only_lab else list(refs)
        for lab in labs:
            entries=refs[lab];byseed={}
            for x in entries:byseed.setdefault(x["seed"],[]).append(x)
            for seed,ents in byseed.items():
                h=collect(env,m,mgr,robot,seed,lab,noise_rep);c=collect(env,m,mgr,robot,seed,"C",noise_rep)
                mm=margins(h,c,lab)
                oh=h["obs"].reshape(H*NENV,-1);uh=h["u"].reshape(H*NENV,-1)
                oc=c["obs"].reshape(H*NENV,-1);uc=c["u"].reshape(H*NENV,-1)
                wh=torch.tensor(PREFS[lab],device="cuda").repeat(H*NENV,1)
                wc=torch.tensor(PREFS["C"],device="cuda").repeat(H*NENV,1)
                lph=m.logp_from_pre_tanh_with_preference(oh,wh,uh).reshape(H,NENV)
                lpc=m.logp_from_pre_tanh_with_preference(oc,wc,uc).reshape(H,NENV)
                for x in ents:
                    total+=1;phase=x["phase"];e=x["env"];cur=mm[phase]
                    mo=float(cur["obj_margin"][e].cpu());mp=float(cur["phys_margin"][e].cpu())
                    ov=x["obj_ref"]>1e-8;pv=x["phys_ref"]>1e-8
                    fo=KAPPA*x["obj_ref"] if ov else None;fp=KAPPA*x["phys_ref"] if pv else None
                    do=max(0.,fo-mo) if ov else 0.;dp=max(0.,fp-mp) if pv else 0.
                    if do<=0 and dp<=0:continue
                    active+=1;sl=slice(phase,phase+WINDOW)
                    ohv=float(h["obj"][sl,e,IDX[lab]].mean().cpu());ocv=float(c["obj"][sl,e,IDX[lab]].mean().cpu())
                    phv=float(h["phys"][PHYS[lab]][sl,e].mean().cpu());pcv=float(c["phys"][PHYS[lab]][sl,e].mean().cpu())
                    logh=lph[sl,e].mean();logc=lpc[sl,e].mean();loss=torch.tensor(0.,device="cuda")
                    if do>0:loss=loss+(do/(abs(fo)+1e-6))*(-ohv*logh+ocv*logc)
                    if dp>0:loss=loss+(dp/(abs(fp)+1e-6))*(-pcv*logc+phv*logh)
                    losses.append(loss)
        if not losses:
            return torch.zeros(sum(p.numel() for _,p in nps),device="cuda"),{"active":0,"total":total}
        loss=sum(losses)/len(losses);g=flat_grad(loss,nps).detach()
        return g,{"active":active,"total":total,"loss":float(loss.detach().cpu())}
    
    def stats(gs):
        G=torch.stack(gs);mean=G.mean(0)
        resid=G-mean
        noise=float(torch.sqrt(torch.mean(torch.sum(resid*resid,dim=1))).cpu())
        snr=float(mean.norm().cpu())/(noise+1e-12)
        norms=np.array([float(g.norm().cpu()) for g in gs])
        pair=[]
        for i in range(len(gs)):
            for j in range(i+1,len(gs)):pair.append(cos(gs[i],gs[j]))
        cmean=[cos(g,mean) for g in gs]
        return {"mean_norm":float(mean.norm().cpu()),"norm_mean":float(norms.mean()),
                "norm_cv":float(norms.std()/(norms.mean()+1e-12)),
                "pairwise_cos_mean":float(np.mean(pair)),"pairwise_cos_min":float(np.min(pair)),
                "cos_to_mean_mean":float(np.mean(cmean)),"cos_to_mean_min":float(np.min(cmean)),
                "snr":snr}
    
    def load_model(cls,path,obsdim,ad):
        m=cls(obsdim,ad).cuda();m.load_state_dict(torch.load(path,map_location="cuda",weights_only=False)["model"]);m.eval();return m
    
    def main():
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.conditioning.single_site_film import V2BSingleSiteFiLMActorCritic as AC
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=0);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim
            mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
            # reconstruct memories from acquisition checkpoints
            mS=load_model(AC,BASE,o.shape[-1],ad)
            mO=load_model(AC,RUN/"semantic_outcome_u2.pt",o.shape[-1],ad)
            mT=load_model(AC,RUN/"semantic_outcome_u3.pt",o.shape[-1],ad)
            refs={"S":capture_reference(env,mS,mgr,robot,"S"),
                  "O":capture_reference(env,mO,mgr,robot,"O"),
                  "T":capture_reference(env,mT,mgr,robot,"T")}
            out={"reps":REPS,"checkpoints":{}}
            for ck,path in CHECKPOINTS.items():
                m=load_model(AC,path,o.shape[-1],ad)
                combined=[];metas=[];per={lab:[] for lab in refs}
                for r in range(REPS):
                    g,meta=outcome_grad(env,m,mgr,robot,refs,r);combined.append(g);metas.append(meta)
                    for lab in refs:
                        gg,_=outcome_grad(env,m,mgr,robot,refs,r,only_lab=lab);per[lab].append(gg)
                out["checkpoints"][ck]={"combined":stats(combined),
                                         "per_axis":{lab:stats(gs) for lab,gs in per.items()},
                                         "rep_meta":metas}
                OUT.write_text(json.dumps(out,indent=2)+"\n")
                print(ck,json.dumps(out["checkpoints"][ck],indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_v2b_outcome_surrogate_gate():
    """Run former v2b_outcome_surrogate_gate.py stage."""
    from pathlib import Path
    import sys,json,copy
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    BASE=ROOT/"runs/v2b_adam_continuous-2026-09-24/model_75.pt"
    RUN=ROOT/"runs/v2b_semantic_outcome_rehearsal_gate_v2-2026-09-24"
    OUT=RUN/"outcome_surrogate_gate.json"
    ORDER=("T","A","O","S")
    PREFS={
    "T":np.array([.7,.1,.1,.1],np.float32),
    "A":np.array([.1,.7,.1,.1],np.float32),
    "O":np.array([.1,.1,.7,.1],np.float32),
    "S":np.array([.1,.1,.1,.7],np.float32),
    "C":np.array([.25,.25,.25,.25],np.float32)}
    IDX={"T":0,"A":1,"O":2,"S":3}
    PHYS={"T":"tracking_error","A":"ang_vel_xy","O":"tilt_deg","S":"action_rate"}
    H=32;NENV=8;WINDOW=8;KAPPA=.5
    REF_PHASES=(0,8,16,24)
    REF_SEEDS={"T":[911001,911101,911201,911301],
               "A":[912001,912101,912201,912301],
               "O":[913001,913101,913201,913301],
               "S":[914001,914101,914201,914301]}
    REPS=4
    CHECKPOINTS={"u4":RUN/"semantic_outcome_u4.pt","u7":RUN/"semantic_outcome_u7.pt"}
    
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def tilt_deg(q):
        _,x,y,_=[q[:,i] for i in range(4)]
        return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def actor_named_params(m):
        return [(n,p) for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std" or n.startswith("preference_embedding") or n.startswith("preference_film")]
    def flat_grad(loss,nps):
        gs=torch.autograd.grad(loss,[p for _,p in nps],allow_unused=True)
        return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for (_,p),g in zip(nps,gs)])
    def cos(a,b):
        den=float(a.norm()*b.norm())
        return float(torch.dot(a,b)/den) if den>1e-12 else 0.0
    
    def phys_vector(robot,env,a,prev):
        data=robot.data;cmd=env.unwrapped.command_manager.get_command("base_velocity")
        return {
          "tracking_error":(data.root_lin_vel_b[:,0]-cmd[:,0]).abs()+(data.root_ang_vel_b[:,2]-cmd[:,2]).abs(),
          "ang_vel_xy":torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1),
          "tilt_deg":tilt_deg(data.root_quat_w),
          "action_rate":torch.linalg.vector_norm(a-prev,dim=-1)}
    
    def collect(env,m,mgr,robot,seed,lab,noise_rep):
        from talon_rl.rewards.objectives import normalized_objective_vector
        w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1)
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        torch.manual_seed(seed+1234+(0 if lab=="C" else IDX.get(lab,0))*17+noise_rep*100003)
        obs=[];u=[];obj=[];phys=[];done=[]
        prev=torch.zeros((NENV,env.unwrapped.action_manager.total_action_dim),device="cuda")
        with torch.no_grad():
            for _ in range(H):
                a,lp,uu=m.act_with_preference_latent(cur,w)
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
                obs.append(cur.detach().clone());u.append(uu.detach().clone());obj.append(torch.tensor(vec,device="cuda"))
                pv=phys_vector(robot,env,a,prev);phys.append({k:v.detach().clone() for k,v in pv.items()})
                done.append((te|tr).detach().clone());prev=a;cur=ot(nxt).cuda()
        return {"obs":torch.stack(obs),"u":torch.stack(u),"obj":torch.stack(obj),
                "phys":{k:torch.stack([x[k] for x in phys]) for k in phys[0]},
                "done":torch.stack(done)}
    
    def margins(h,c,lab):
        j=IDX[lab];pk=PHYS[lab];out={}
        for phase in REF_PHASES:
            sl=slice(phase,phase+WINDOW)
            out[phase]={
              "obj_margin":(h["obj"][sl,:,j].mean(0)-c["obj"][sl,:,j].mean(0)).detach(),
              "phys_margin":(c["phys"][pk][sl].mean(0)-h["phys"][pk][sl].mean(0)).detach()}
        return out
    
    def capture_reference(env,m,mgr,robot,lab):
        entries=[]
        for seed in REF_SEEDS[lab]:
            h=collect(env,m,mgr,robot,seed,lab,0);c=collect(env,m,mgr,robot,seed,"C",0)
            mm=margins(h,c,lab)
            for phase,w in mm.items():
                for e in range(NENV):
                    entries.append({"seed":seed,"phase":phase,"env":e,
                                    "obj_ref":float(w["obj_margin"][e].cpu()),
                                    "phys_ref":float(w["phys_margin"][e].cpu())})
        return entries
    
    def ridge_fit(X,Y,l2=1.0):
        X=np.asarray(X,np.float64);Y=np.asarray(Y,np.float64)
        A=np.c_[X,np.ones(len(X))];I=np.eye(A.shape[1]);I[-1,-1]=0
        sol=np.linalg.solve(A.T@A+l2*I,A.T@Y)
        return sol[:-1],sol[-1]
    
    def pred_metrics(y,p):
        y=np.asarray(y,float);p=np.asarray(p,float)
        out={}
        for j,name in enumerate(("obj","phys")):
            den=np.var(y[:,j])+1e-12
            out[name]={"mae":float(np.mean(np.abs(p[:,j]-y[:,j]))),
                       "rmse":float(np.sqrt(np.mean((p[:,j]-y[:,j])**2))),
                       "ev":float(1-np.var(y[:,j]-p[:,j])/den),
                       "corr":float(np.corrcoef(y[:,j],p[:,j])[0,1]) if np.std(y[:,j])>1e-12 and np.std(p[:,j])>1e-12 else 0.0}
        return out
    
    def dataset_for_lab(env,m,mgr,robot,lab,noise_rep=0):
        rows=[];ad=env.unwrapped.action_manager.total_action_dim
        for seed in REF_SEEDS[lab]:
            h=collect(env,m,mgr,robot,seed,lab,noise_rep);c=collect(env,m,mgr,robot,seed,"C",noise_rep)
            mm=margins(h,c,lab)
            wh=torch.tensor(PREFS[lab],device="cuda").repeat(H*NENV,1)
            wc=torch.tensor(PREFS["C"],device="cuda").repeat(H*NENV,1)
            with torch.no_grad():
                lh=m._pre_tanh_dist_with_preference(h["obs"].reshape(H*NENV,-1),wh).loc.reshape(H,NENV,ad)
                lc=m._pre_tanh_dist_with_preference(c["obs"].reshape(H*NENV,-1),wc).loc.reshape(H,NENV,ad)
            for phase in REF_PHASES:
                sl=slice(phase,phase+WINDOW)
                for e in range(NENV):
                    # Detached local context: first 12 physical/command observations from heavy and center.
                    ch=h["obs"][sl,e,:12].mean(0);cc=c["obs"][sl,e,:12].mean(0)
                    ah=lh[sl,e].mean(0);ac=lc[sl,e].mean(0)
                    x=torch.cat((ch,cc,ah,ac)).cpu().numpy()
                    rows.append({"seed":seed,"phase":phase,"env":e,"x":x,
                                 "y":np.array([float(mm[phase]["obj_margin"][e]),float(mm[phase]["phys_margin"][e])],np.float64),
                                 "obs_h":h["obs"][sl,e].detach().clone(),"obs_c":c["obs"][sl,e].detach().clone()})
        return rows
    
    def heldout_fidelity(rows,l2=1.0):
        Xall=np.stack([r["x"] for r in rows]);Yall=np.stack([r["y"] for r in rows]);Wa,ba=ridge_fit(Xall,Yall,l2)
        train_metrics=pred_metrics(Yall,Xall@Wa+ba)
        preds=[];ys=[];folds=[]
        seeds=sorted({r["seed"] for r in rows})
        for held in seeds:
            tr=[r for r in rows if r["seed"]!=held];te=[r for r in rows if r["seed"]==held]
            X=np.stack([r["x"] for r in tr]);Y=np.stack([r["y"] for r in tr]);W,b=ridge_fit(X,Y,l2)
            Xt=np.stack([r["x"] for r in te]);Yt=np.stack([r["y"] for r in te]);P=Xt@W+b
            preds.append(P);ys.append(Yt);folds.append({"held_seed":held,"metrics":pred_metrics(Yt,P)})
        return {"train":train_metrics,"aggregate":pred_metrics(np.concatenate(ys),np.concatenate(preds)),"folds":folds}
    
    def heldout_mlp_fidelity(rows,seed_base):
        preds=[];ys=[];folds=[]
        seeds=sorted({r["seed"] for r in rows})
        for fi,held in enumerate(seeds):
            tr=[r for r in rows if r["seed"]!=held];te=[r for r in rows if r["seed"]==held]
            X=np.stack([r["x"] for r in tr]).astype(np.float32);Y=np.stack([r["y"] for r in tr]).astype(np.float32)
            Xt=np.stack([r["x"] for r in te]).astype(np.float32);Yt=np.stack([r["y"] for r in te]).astype(np.float32)
            xm=X.mean(0);xs=X.std(0)+1e-6;ym=Y.mean(0);ysd=Y.std(0)+1e-6
            torch.manual_seed(seed_base+fi)
            net=torch.nn.Sequential(torch.nn.Linear(X.shape[1],64),torch.nn.ELU(),torch.nn.Linear(64,64),torch.nn.ELU(),torch.nn.Linear(64,2)).cuda()
            opt=torch.optim.AdamW(net.parameters(),lr=3e-3,weight_decay=1e-3)
            xx=torch.tensor((X-xm)/xs,device="cuda");yy=torch.tensor((Y-ym)/ysd,device="cuda")
            for _ in range(400):
                pr=net(xx);loss=((pr-yy)**2).mean();opt.zero_grad();loss.backward();opt.step()
            with torch.no_grad():
                pp=net(torch.tensor((Xt-xm)/xs,device="cuda")).cpu().numpy()*ysd+ym
            preds.append(pp);ys.append(Yt);folds.append({"held_seed":held,"metrics":pred_metrics(Yt,pp)})
        return {"aggregate":pred_metrics(np.concatenate(ys),np.concatenate(preds)),"folds":folds}
    
    def surrogate_gradient(m,rows,refs_lab,W,b,bootstrap_idx=None):
        nps=actor_named_params(m);device="cuda";ad=m.actor_mean.out_features
        if bootstrap_idx is not None:
            fitrows=[rows[int(i)] for i in bootstrap_idx]
            X=np.stack([r["x"] for r in fitrows]);Y=np.stack([r["y"] for r in fitrows]);W,b=ridge_fit(X,Y,1.0)
        Wt=torch.tensor(W,device=device,dtype=torch.float32);bt=torch.tensor(b,device=device,dtype=torch.float32)
        refmap={(x["seed"],x["phase"],x["env"]):x for x in refs_lab};losses=[]
        for r in rows:
            key=(r["seed"],r["phase"],r["env"]);ref=refmap[key]
            oh=r["obs_h"].to(device);oc=r["obs_c"].to(device)
            wh=torch.tensor(PREFS[next(k for k,v in REF_SEEDS.items() if r["seed"] in v)],device=device).repeat(len(oh),1)
            wc=torch.tensor(PREFS["C"],device=device).repeat(len(oc),1)
            lh=m._pre_tanh_dist_with_preference(oh,wh).loc.mean(0);lc=m._pre_tanh_dist_with_preference(oc,wc).loc.mean(0)
            ctx=torch.tensor(r["x"][:24],device=device,dtype=torch.float32)
            xg=torch.cat((ctx,lh,lc));pred=xg@Wt+bt
            terms=[]
            if ref["obj_ref"]>1e-8:
                floor=KAPPA*ref["obj_ref"];terms.append(torch.relu(torch.tensor(floor,device=device)-pred[0])/(abs(floor)+1e-6))
            if ref["phys_ref"]>1e-8:
                floor=KAPPA*ref["phys_ref"];terms.append(torch.relu(torch.tensor(floor,device=device)-pred[1])/(abs(floor)+1e-6))
            if terms:losses.append(sum(terms))
        if not losses:return torch.zeros(sum(p.numel() for _,p in nps),device=device)
        return flat_grad(sum(losses)/len(losses),nps).detach()
    
    def grad_stats(gs):
        G=torch.stack(gs);mean=G.mean(0);res=G-mean;noise=float(torch.sqrt(torch.mean(torch.sum(res*res,dim=1))))
        norms=np.array([float(g.norm()) for g in gs]);pairs=[cos(gs[i],gs[j]) for i in range(len(gs)) for j in range(i+1,len(gs))]
        cm=[cos(g,mean) for g in gs]
        return {"pairwise_cos_mean":float(np.mean(pairs)),"pairwise_cos_min":float(np.min(pairs)),
                "cos_to_mean_mean":float(np.mean(cm)),"cos_to_mean_min":float(np.min(cm)),
                "norm_cv":float(norms.std()/(norms.mean()+1e-12)),"snr":float(mean.norm())/(noise+1e-12),
                "mean_norm":float(mean.norm())}
    
    def surrogate_gate_for_lab(m,rows,refs_lab,seed_base):
        X=np.stack([r["x"] for r in rows]);Y=np.stack([r["y"] for r in rows]);W,b=ridge_fit(X,Y,1.0)
        fidelity=heldout_fidelity(rows,1.0)
        mlp_fidelity=heldout_mlp_fidelity(rows,seed_base+50000)
        rng=np.random.default_rng(seed_base);grads=[]
        # Stratified bootstrap by seed: resample within each seed to retain reference-distribution support.
        seed_groups={s:np.array([i for i,r in enumerate(rows) if r["seed"]==s]) for s in sorted({r["seed"] for r in rows})}
        for rep in range(6):
            idx=np.concatenate([rng.choice(ii,size=len(ii),replace=True) for ii in seed_groups.values()])
            grads.append(surrogate_gradient(m,rows,refs_lab,W,b,idx))
        return {"fidelity":fidelity,"mlp_fidelity":mlp_fidelity,"gradient":grad_stats(grads),"fit":{"n":len(rows),"feature_dim":X.shape[1]}}
    
    def load_model(cls,path,obsdim,ad):
        m=cls(obsdim,ad).cuda();m.load_state_dict(torch.load(path,map_location="cuda",weights_only=False)["model"]);m.eval();return m
    
    def main():
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.conditioning.single_site_film import V2BSingleSiteFiLMActorCritic as AC
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=0);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim
            mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
            mS=load_model(AC,BASE,o.shape[-1],ad);mO=load_model(AC,RUN/"semantic_outcome_u2.pt",o.shape[-1],ad);mT=load_model(AC,RUN/"semantic_outcome_u3.pt",o.shape[-1],ad)
            refs={"S":capture_reference(env,mS,mgr,robot,"S"),"O":capture_reference(env,mO,mgr,robot,"O"),"T":capture_reference(env,mT,mgr,robot,"T")}
            out={"schema":"v2b_semantic_outcome_surrogate_gate_v1","kappa":KAPPA,"rho":0.25,"checkpoints":{}}
            for ci,(ck,path) in enumerate(CHECKPOINTS.items()):
                m=load_model(AC,path,o.shape[-1],ad);out["checkpoints"][ck]={}
                axis_grads=[]
                for li,lab in enumerate(("S","O","T")):
                    rows=dataset_for_lab(env,m,mgr,robot,lab,noise_rep=0)
                    res=surrogate_gate_for_lab(m,rows,refs[lab],17000000+ci*1000+li*100)
                    out["checkpoints"][ck][lab]=res
                    # mean gradient from same 6 bootstrap fits for combined check is reconstructed deterministically below
                    print(ck,lab,json.dumps(res,indent=2),flush=True)
                    OUT.write_text(json.dumps(out,indent=2)+"\n")
            print("WROTE",OUT,flush=True)
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_v2b_outcome_surrogate_rich_gate():
    """Run former v2b_outcome_surrogate_rich_gate.py stage."""
    from pathlib import Path
    import sys,json,copy
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    BASE=ROOT/"runs/v2b_adam_continuous-2026-09-24/model_75.pt"
    RUN=ROOT/"runs/v2b_semantic_outcome_rehearsal_gate_v2-2026-09-24"
    OUT=RUN/"outcome_surrogate_rich_gate.json"
    ORDER=("T","A","O","S")
    PREFS={
    "T":np.array([.7,.1,.1,.1],np.float32),
    "A":np.array([.1,.7,.1,.1],np.float32),
    "O":np.array([.1,.1,.7,.1],np.float32),
    "S":np.array([.1,.1,.1,.7],np.float32),
    "C":np.array([.25,.25,.25,.25],np.float32)}
    IDX={"T":0,"A":1,"O":2,"S":3}
    PHYS={"T":"tracking_error","A":"ang_vel_xy","O":"tilt_deg","S":"action_rate"}
    H=32;NENV=8;WINDOW=8;KAPPA=.5
    REF_PHASES=(0,8,16,24)
    REF_SEEDS={"T":[911001,911101,911201,911301],
               "A":[912001,912101,912201,912301],
               "O":[913001,913101,913201,913301],
               "S":[914001,914101,914201,914301]}
    REPS=4
    CHECKPOINTS={"u4":RUN/"semantic_outcome_u4.pt","u7":RUN/"semantic_outcome_u7.pt"}
    
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def tilt_deg(q):
        _,x,y,_=[q[:,i] for i in range(4)]
        return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def actor_named_params(m):
        return [(n,p) for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std" or n.startswith("preference_embedding") or n.startswith("preference_film")]
    def flat_grad(loss,nps):
        gs=torch.autograd.grad(loss,[p for _,p in nps],allow_unused=True)
        return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for (_,p),g in zip(nps,gs)])
    def cos(a,b):
        den=float(a.norm()*b.norm())
        return float(torch.dot(a,b)/den) if den>1e-12 else 0.0
    
    def phys_vector(robot,env,a,prev):
        data=robot.data;cmd=env.unwrapped.command_manager.get_command("base_velocity")
        return {
          "tracking_error":(data.root_lin_vel_b[:,0]-cmd[:,0]).abs()+(data.root_ang_vel_b[:,2]-cmd[:,2]).abs(),
          "ang_vel_xy":torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1),
          "tilt_deg":tilt_deg(data.root_quat_w),
          "action_rate":torch.linalg.vector_norm(a-prev,dim=-1)}
    
    def collect(env,m,mgr,robot,seed,lab,noise_rep):
        from talon_rl.rewards.objectives import normalized_objective_vector
        w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1)
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        torch.manual_seed(seed+1234+(0 if lab=="C" else IDX.get(lab,0))*17+noise_rep*100003)
        obs=[];u=[];obj=[];phys=[];done=[]
        prev=torch.zeros((NENV,env.unwrapped.action_manager.total_action_dim),device="cuda")
        with torch.no_grad():
            for _ in range(H):
                a,lp,uu=m.act_with_preference_latent(cur,w)
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
                obs.append(cur.detach().clone());u.append(uu.detach().clone());obj.append(torch.tensor(vec,device="cuda"))
                pv=phys_vector(robot,env,a,prev);phys.append({k:v.detach().clone() for k,v in pv.items()})
                done.append((te|tr).detach().clone());prev=a;cur=ot(nxt).cuda()
        return {"obs":torch.stack(obs),"u":torch.stack(u),"obj":torch.stack(obj),
                "phys":{k:torch.stack([x[k] for x in phys]) for k in phys[0]},
                "done":torch.stack(done)}
    
    def margins(h,c,lab):
        j=IDX[lab];pk=PHYS[lab];out={}
        for phase in REF_PHASES:
            sl=slice(phase,phase+WINDOW)
            out[phase]={
              "obj_margin":(h["obj"][sl,:,j].mean(0)-c["obj"][sl,:,j].mean(0)).detach(),
              "phys_margin":(c["phys"][pk][sl].mean(0)-h["phys"][pk][sl].mean(0)).detach()}
        return out
    
    def capture_reference(env,m,mgr,robot,lab):
        entries=[]
        for seed in REF_SEEDS[lab]:
            h=collect(env,m,mgr,robot,seed,lab,0);c=collect(env,m,mgr,robot,seed,"C",0)
            mm=margins(h,c,lab)
            for phase,w in mm.items():
                for e in range(NENV):
                    entries.append({"seed":seed,"phase":phase,"env":e,
                                    "obj_ref":float(w["obj_margin"][e].cpu()),
                                    "phys_ref":float(w["phys_margin"][e].cpu())})
        return entries
    
    def ridge_fit(X,Y,l2=1.0):
        X=np.asarray(X,np.float64);Y=np.asarray(Y,np.float64)
        A=np.c_[X,np.ones(len(X))];I=np.eye(A.shape[1]);I[-1,-1]=0
        sol=np.linalg.solve(A.T@A+l2*I,A.T@Y)
        return sol[:-1],sol[-1]
    
    def pred_metrics(y,p):
        y=np.asarray(y,float);p=np.asarray(p,float)
        out={}
        for j,name in enumerate(("obj","phys")):
            den=np.var(y[:,j])+1e-12
            out[name]={"mae":float(np.mean(np.abs(p[:,j]-y[:,j]))),
                       "rmse":float(np.sqrt(np.mean((p[:,j]-y[:,j])**2))),
                       "ev":float(1-np.var(y[:,j]-p[:,j])/den),
                       "corr":float(np.corrcoef(y[:,j],p[:,j])[0,1]) if np.std(y[:,j])>1e-12 and np.std(p[:,j])>1e-12 else 0.0}
        return out
    
    def dataset_for_lab(env,m,mgr,robot,lab,noise_rep=0):
        rows=[];ad=env.unwrapped.action_manager.total_action_dim
        for seed in REF_SEEDS[lab]:
            h=collect(env,m,mgr,robot,seed,lab,noise_rep);c=collect(env,m,mgr,robot,seed,"C",noise_rep)
            mm=margins(h,c,lab)
            wh=torch.tensor(PREFS[lab],device="cuda").repeat(H*NENV,1)
            wc=torch.tensor(PREFS["C"],device="cuda").repeat(H*NENV,1)
            with torch.no_grad():
                lh=m._pre_tanh_dist_with_preference(h["obs"].reshape(H*NENV,-1),wh).loc.reshape(H,NENV,ad)
                lc=m._pre_tanh_dist_with_preference(c["obs"].reshape(H*NENV,-1),wc).loc.reshape(H,NENV,ad)
            for phase in REF_PHASES:
                sl=slice(phase,phase+WINDOW)
                for e in range(NENV):
                    # Detached local context: first 12 physical/command observations from heavy and center.
                    hs=h["obs"][sl,e,:12];cs=c["obs"][sl,e,:12]
                    context=torch.cat((hs.mean(0),cs.mean(0),hs.std(0),cs.std(0),hs[-1]-hs[0],cs[-1]-cs[0]))
                    ah=lh[sl,e].mean(0);ac=lc[sl,e].mean(0)
                    x=torch.cat((context,ah,ac)).cpu().numpy()
                    rows.append({"seed":seed,"phase":phase,"env":e,"x":x,
                                 "y":np.array([float(mm[phase]["obj_margin"][e]),float(mm[phase]["phys_margin"][e])],np.float64),
                                 "obs_h":h["obs"][sl,e].detach().clone(),"obs_c":c["obs"][sl,e].detach().clone()})
        return rows
    
    def heldout_fidelity(rows,l2=1.0):
        Xall=np.stack([r["x"] for r in rows]);Yall=np.stack([r["y"] for r in rows]);Wa,ba=ridge_fit(Xall,Yall,l2)
        train_metrics=pred_metrics(Yall,Xall@Wa+ba)
        preds=[];ys=[];folds=[]
        seeds=sorted({r["seed"] for r in rows})
        for held in seeds:
            tr=[r for r in rows if r["seed"]!=held];te=[r for r in rows if r["seed"]==held]
            X=np.stack([r["x"] for r in tr]);Y=np.stack([r["y"] for r in tr]);W,b=ridge_fit(X,Y,l2)
            Xt=np.stack([r["x"] for r in te]);Yt=np.stack([r["y"] for r in te]);P=Xt@W+b
            preds.append(P);ys.append(Yt);folds.append({"held_seed":held,"metrics":pred_metrics(Yt,P)})
        return {"train":train_metrics,"aggregate":pred_metrics(np.concatenate(ys),np.concatenate(preds)),"folds":folds}
    
    def heldout_mlp_fidelity(rows,seed_base):
        preds=[];ys=[];folds=[]
        seeds=sorted({r["seed"] for r in rows})
        for fi,held in enumerate(seeds):
            tr=[r for r in rows if r["seed"]!=held];te=[r for r in rows if r["seed"]==held]
            X=np.stack([r["x"] for r in tr]).astype(np.float32);Y=np.stack([r["y"] for r in tr]).astype(np.float32)
            Xt=np.stack([r["x"] for r in te]).astype(np.float32);Yt=np.stack([r["y"] for r in te]).astype(np.float32)
            xm=X.mean(0);xs=X.std(0)+1e-6;ym=Y.mean(0);ysd=Y.std(0)+1e-6
            torch.manual_seed(seed_base+fi)
            net=torch.nn.Sequential(torch.nn.Linear(X.shape[1],64),torch.nn.ELU(),torch.nn.Linear(64,64),torch.nn.ELU(),torch.nn.Linear(64,2)).cuda()
            opt=torch.optim.AdamW(net.parameters(),lr=3e-3,weight_decay=1e-3)
            xx=torch.tensor((X-xm)/xs,device="cuda");yy=torch.tensor((Y-ym)/ysd,device="cuda")
            for _ in range(400):
                pr=net(xx);loss=((pr-yy)**2).mean();opt.zero_grad();loss.backward();opt.step()
            with torch.no_grad():
                pp=net(torch.tensor((Xt-xm)/xs,device="cuda")).cpu().numpy()*ysd+ym
            preds.append(pp);ys.append(Yt);folds.append({"held_seed":held,"metrics":pred_metrics(Yt,pp)})
        return {"aggregate":pred_metrics(np.concatenate(ys),np.concatenate(preds)),"folds":folds}
    
    def surrogate_gradient(m,rows,refs_lab,W,b,bootstrap_idx=None):
        nps=actor_named_params(m);device="cuda";ad=m.actor_mean.out_features
        if bootstrap_idx is not None:
            fitrows=[rows[int(i)] for i in bootstrap_idx]
            X=np.stack([r["x"] for r in fitrows]);Y=np.stack([r["y"] for r in fitrows]);W,b=ridge_fit(X,Y,1.0)
        Wt=torch.tensor(W,device=device,dtype=torch.float32);bt=torch.tensor(b,device=device,dtype=torch.float32)
        refmap={(x["seed"],x["phase"],x["env"]):x for x in refs_lab};losses=[]
        for r in rows:
            key=(r["seed"],r["phase"],r["env"]);ref=refmap[key]
            oh=r["obs_h"].to(device);oc=r["obs_c"].to(device)
            wh=torch.tensor(PREFS[next(k for k,v in REF_SEEDS.items() if r["seed"] in v)],device=device).repeat(len(oh),1)
            wc=torch.tensor(PREFS["C"],device=device).repeat(len(oc),1)
            lh=m._pre_tanh_dist_with_preference(oh,wh).loc.mean(0);lc=m._pre_tanh_dist_with_preference(oc,wc).loc.mean(0)
            ctx=torch.tensor(r["x"][:-2*ad],device=device,dtype=torch.float32)
            xg=torch.cat((ctx,lh,lc));pred=xg@Wt+bt
            terms=[]
            if ref["obj_ref"]>1e-8:
                floor=KAPPA*ref["obj_ref"];terms.append(torch.relu(torch.tensor(floor,device=device)-pred[0])/(abs(floor)+1e-6))
            if ref["phys_ref"]>1e-8:
                floor=KAPPA*ref["phys_ref"];terms.append(torch.relu(torch.tensor(floor,device=device)-pred[1])/(abs(floor)+1e-6))
            if terms:losses.append(sum(terms))
        if not losses:return torch.zeros(sum(p.numel() for _,p in nps),device=device)
        return flat_grad(sum(losses)/len(losses),nps).detach()
    
    def grad_stats(gs):
        G=torch.stack(gs);mean=G.mean(0);res=G-mean;noise=float(torch.sqrt(torch.mean(torch.sum(res*res,dim=1))))
        norms=np.array([float(g.norm()) for g in gs]);pairs=[cos(gs[i],gs[j]) for i in range(len(gs)) for j in range(i+1,len(gs))]
        cm=[cos(g,mean) for g in gs]
        return {"pairwise_cos_mean":float(np.mean(pairs)),"pairwise_cos_min":float(np.min(pairs)),
                "cos_to_mean_mean":float(np.mean(cm)),"cos_to_mean_min":float(np.min(cm)),
                "norm_cv":float(norms.std()/(norms.mean()+1e-12)),"snr":float(mean.norm())/(noise+1e-12),
                "mean_norm":float(mean.norm())}
    
    def surrogate_gate_for_lab(m,rows,refs_lab,seed_base):
        X=np.stack([r["x"] for r in rows]);Y=np.stack([r["y"] for r in rows]);W,b=ridge_fit(X,Y,1.0)
        fidelity=heldout_fidelity(rows,1.0)
        mlp_fidelity=heldout_mlp_fidelity(rows,seed_base+50000)
        rng=np.random.default_rng(seed_base);grads=[]
        # Stratified bootstrap by seed: resample within each seed to retain reference-distribution support.
        seed_groups={s:np.array([i for i,r in enumerate(rows) if r["seed"]==s]) for s in sorted({r["seed"] for r in rows})}
        for rep in range(6):
            idx=np.concatenate([rng.choice(ii,size=len(ii),replace=True) for ii in seed_groups.values()])
            grads.append(surrogate_gradient(m,rows,refs_lab,W,b,idx))
        return {"fidelity":fidelity,"mlp_fidelity":mlp_fidelity,"gradient":grad_stats(grads),"fit":{"n":len(rows),"feature_dim":X.shape[1]}}
    
    def load_model(cls,path,obsdim,ad):
        m=cls(obsdim,ad).cuda();m.load_state_dict(torch.load(path,map_location="cuda",weights_only=False)["model"]);m.eval();return m
    
    def main():
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.conditioning.single_site_film import V2BSingleSiteFiLMActorCritic as AC
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=0);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim
            mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
            mS=load_model(AC,BASE,o.shape[-1],ad);mO=load_model(AC,RUN/"semantic_outcome_u2.pt",o.shape[-1],ad);mT=load_model(AC,RUN/"semantic_outcome_u3.pt",o.shape[-1],ad)
            refs={"S":capture_reference(env,mS,mgr,robot,"S"),"O":capture_reference(env,mO,mgr,robot,"O"),"T":capture_reference(env,mT,mgr,robot,"T")}
            out={"schema":"v2b_semantic_outcome_surrogate_rich_gate_v1","kappa":KAPPA,"rho":0.25,"checkpoints":{}}
            for ci,(ck,path) in enumerate(CHECKPOINTS.items()):
                m=load_model(AC,path,o.shape[-1],ad);out["checkpoints"][ck]={}
                axis_grads=[]
                for li,lab in enumerate(("S","O","T")):
                    rows=dataset_for_lab(env,m,mgr,robot,lab,noise_rep=0)
                    res=surrogate_gate_for_lab(m,rows,refs[lab],17000000+ci*1000+li*100)
                    out["checkpoints"][ck][lab]=res
                    # mean gradient from same 6 bootstrap fits for combined check is reconstructed deterministically below
                    print(ck,lab,json.dumps(res,indent=2),flush=True)
                    OUT.write_text(json.dumps(out,indent=2)+"\n")
            print("WROTE",OUT,flush=True)
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_v2b_semantic_outcome_rehearsal_gate():
    """Run former v2b_semantic_outcome_rehearsal_gate.py stage."""
    from pathlib import Path
    import argparse,json,sys,hashlib,copy
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    CKPT=ROOT/"runs/v2b_adam_continuous-2026-09-24/model_75.pt"
    OPTLOG=ROOT/"runs/v2b_adam_continuous-2026-09-24/adam_history_report.json"
    OUT=ROOT/"runs/v2b_semantic_outcome_rehearsal_gate-2026-09-24"
    ORDER=("T","A","O","S")
    PREFS={
    "T":np.array([.7,.1,.1,.1],np.float32),
    "A":np.array([.1,.7,.1,.1],np.float32),
    "O":np.array([.1,.1,.7,.1],np.float32),
    "S":np.array([.1,.1,.1,.7],np.float32),
    "C":np.array([.25,.25,.25,.25],np.float32)}
    IDX={"T":0,"A":1,"O":2,"S":3}
    PHYS={"T":"tracking_error","A":"ang_vel_xy","O":"tilt_deg","S":"action_rate"}
    G=.99;LAM=1.0;H=32;NENV=8;K=8;STEP_SCALE=.5
    FULL_STEPS=64;ENDPOINT_SUITES=4;BASE_SEED=980001
    REF_SEEDS={"T":[911001,911101,911201,911301],
               "A":[912001,912101,912201,912301],
               "O":[913001,913101,913201,913301],
               "S":[914001,914101,914201,914301]}
    REF_PHASES=(0,8,16,24)
    WINDOW=8
    RHO=0.25
    BETA0=2.497041993384243
    KAPPA=0.50
    EPS=1e-12
    
    def ot(x):
        if isinstance(x,dict): x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names): return {n:raw[:,i] for i,n in enumerate(names)}
    def tilt_deg(q):
        _,x,y,_=[q[:,i] for i in range(4)]
        return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def actor_named_params(m):
        return [(n,p) for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std" or n.startswith("preference_embedding") or n.startswith("preference_film")]
    def flat_grad(loss,nps,retain=False):
        gs=torch.autograd.grad(loss,[p for _,p in nps],retain_graph=retain,allow_unused=True)
        return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for (_,p),g in zip(nps,gs)])
    def apply_flat_step(m,direction,step_norm):
        off=0
        with torch.no_grad():
            for _,p in actor_named_params(m):
                n=p.numel();p.add_(direction[off:off+n].view_as(p)*step_norm);off+=n
            m.log_std.clamp_(m.LOG_STD_MIN,m.LOG_STD_MAX)
    def cos(a,b):
        den=float(a.norm()*b.norm())
        return float(torch.dot(a,b)/den) if den>1e-12 else 0.0
    def nominal_step():
        d=json.load(open(OPTLOG))
        return float(np.mean([r["actual_param_delta_norm"] for r in d["rows"] if 51<=r["update"]<=75]))
    
    def collect_batch(env,m,mgr,seed,w_env):
        from talon_rl.rewards.objectives import normalized_objective_vector
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda();torch.manual_seed(seed+777)
        ob=[];u=[];old=[];R=[];D=[]
        with torch.no_grad():
            for _ in range(H):
                a,lp,uu=m.act_with_preference_latent(cur,w_env)
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
                ob.append(cur);u.append(uu);old.append(lp)
                R.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt)
                D.append((te|tr).cuda());cur=ot(nxt).cuda()
        return {"obs":torch.cat(ob),"u":torch.cat(u),"old":torch.cat(old),
                "rt":torch.stack(R),"dt":torch.stack(D).bool(),"next_obs":cur,
                "w":w_env.repeat(H,1),"w_env":w_env}
    def gae(reward,value,nextv,done):
        adv=torch.zeros_like(reward);last=torch.zeros_like(nextv)
        for t in range(H-1,-1,-1):
            boot=nextv if t==H-1 else value[t+1]
            nt=(~done[t]).to(reward.dtype).unsqueeze(-1)
            delta=reward[t]+G*boot*nt-value[t]
            last=delta+G*LAM*nt*last;adv[t]=last
        return adv
    def objective_losses(m,b):
        with torch.no_grad():
            V=m.value_with_preference(b["obs"],b["w"]).reshape(H,NENV,4)
            NV=m.value_with_preference(b["next_obs"],b["w_env"])
            A=gae(b["rt"],V,NV,b["dt"]).reshape(-1,4).detach()
        logp=m.logp_from_pre_tanh_with_preference(b["obs"],b["w"],b["u"])
        ratio=torch.exp(logp-b["old"].detach());clip=ratio.clamp(.8,1.2)
        weighted={}
        for j,lab in enumerate(ORDER):
            po=torch.minimum(ratio*A[:,j],clip*A[:,j])
            weighted[lab]=-(4.0*b["w"][:,j]*po).mean()
        return weighted,ratio
    def mixed_w(device,k):
        labs=[ORDER[(k+i)%4] for i in range(NENV)]
        return labs,torch.tensor(np.stack([PREFS[x] for x in labs]),device=device)
    
    def phys_vector(robot,env,a,prev):
        data=robot.data;cmd=env.unwrapped.command_manager.get_command("base_velocity")
        vx=(data.root_lin_vel_b[:,0]-cmd[:,0]).abs()
        wz=(data.root_ang_vel_b[:,2]-cmd[:,2]).abs()
        return {
            "tracking_error":vx+wz,
            "ang_vel_xy":torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1),
            "tilt_deg":tilt_deg(data.root_quat_w),
            "action_rate":torch.linalg.vector_norm(a-prev,dim=-1)
        }
    
    def collect_semantic_rollout(env,m,mgr,robot,seed,lab):
        from talon_rl.rewards.objectives import normalized_objective_vector
        w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1)
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda();torch.manual_seed(seed+1234+(0 if lab=="C" else IDX.get(lab,0))*17)
        obs=[];u=[];obj=[];phys=[];done=[]
        prev=torch.zeros((NENV,env.unwrapped.action_manager.total_action_dim),device="cuda")
        with torch.no_grad():
            for _ in range(H):
                a,lp,uu=m.act_with_preference_latent(cur,w)
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
                obs.append(cur.detach().clone());u.append(uu.detach().clone())
                obj.append(torch.tensor(vec,device="cuda"))
                pv=phys_vector(robot,env,a,prev)
                phys.append({k:v.detach().clone() for k,v in pv.items()})
                done.append((te|tr).detach().clone())
                prev=a;cur=ot(nxt).cuda()
        return {"obs":torch.stack(obs),"u":torch.stack(u),"obj":torch.stack(obj),
                "phys":{k:torch.stack([x[k] for x in phys]) for k in phys[0]},
                "done":torch.stack(done),"w":w}
    
    def margin_windows(h,c,lab):
        j=IDX[lab];pk=PHYS[lab]
        out=[]
        for phase in REF_PHASES:
            sl=slice(phase,phase+WINDOW)
            # each env is one matched sample; positive means heavy preference is semantically better.
            obj_h=h["obj"][sl,:,j].mean(0);obj_c=c["obj"][sl,:,j].mean(0)
            ph_h=h["phys"][pk][sl].mean(0);ph_c=c["phys"][pk][sl].mean(0)
            surv_h=(~h["done"][sl]).float().mean(0);surv_c=(~c["done"][sl]).float().mean(0)
            out.append({"phase":phase,
                        "obj_margin":(obj_h-obj_c).detach(),
                        "phys_margin":(ph_c-ph_h).detach(),
                        "survival":torch.minimum(surv_h,surv_c).detach()})
        return out
    
    def capture_outcome_reference(env,m,mgr,robot,lab):
        entries=[]
        for seed in REF_SEEDS[lab]:
            h=collect_semantic_rollout(env,m,mgr,robot,seed,lab)
            c=collect_semantic_rollout(env,m,mgr,robot,seed,"C")
            wins=margin_windows(h,c,lab)
            for w in wins:
                for e in range(NENV):
                    entries.append({
                        "seed":seed,"phase":w["phase"],"env":e,
                        "obj_ref":float(w["obj_margin"][e].cpu()),
                        "phys_ref":float(w["phys_margin"][e].cpu()),
                        "survival_ref":float(w["survival"][e].cpu())})
        return entries
    
    def semantic_outcome_rehearsal_gradient(env,m,mgr,robot,refs):
        nps=actor_named_params(m)
        if not refs:
            z=torch.zeros(sum(p.numel() for _,p in nps),device="cuda")
            return z,{"active_windows":0,"total_windows":0,"mean_obj_deficit":0.0,"mean_phys_deficit":0.0}
        losses=[];objdefs=[];phydefs=[];active=0;total=0
        for lab,entries in refs.items():
            byseed={}
            for x in entries: byseed.setdefault(x["seed"],[]).append(x)
            for seed,ents in byseed.items():
                h=collect_semantic_rollout(env,m,mgr,robot,seed,lab)
                c=collect_semantic_rollout(env,m,mgr,robot,seed,"C")
                wins={w["phase"]:w for w in margin_windows(h,c,lab)}
                # differentiable current-policy log-probs for sampled actions.
                oh=h["obs"].reshape(H*NENV,-1); uh=h["u"].reshape(H*NENV,-1)
                oc=c["obs"].reshape(H*NENV,-1); uc=c["u"].reshape(H*NENV,-1)
                wh=torch.tensor(PREFS[lab],device="cuda").repeat(H*NENV,1)
                wc=torch.tensor(PREFS["C"],device="cuda").repeat(H*NENV,1)
                lph=m.logp_from_pre_tanh_with_preference(oh,wh,uh).reshape(H,NENV)
                lpc=m.logp_from_pre_tanh_with_preference(oc,wc,uc).reshape(H,NENV)
                for x in ents:
                    total+=1;phase=x["phase"];e=x["env"];cur=wins[phase]
                    mobj=float(cur["obj_margin"][e].cpu()); mphys=float(cur["phys_margin"][e].cpu())
                    obj_valid=x["obj_ref"]>1e-8
                    phys_valid=x["phys_ref"]>1e-8
                    floor_obj=KAPPA*x["obj_ref"] if obj_valid else None
                    floor_phys=KAPPA*x["phys_ref"] if phys_valid else None
                    dobj=max(0.0,floor_obj-mobj) if obj_valid else 0.0
                    dphys=max(0.0,floor_phys-mphys) if phys_valid else 0.0
                    if dobj<=0 and dphys<=0: continue
                    active+=1;objdefs.append(dobj);phydefs.append(dphys)
                    sl=slice(phase,phase+WINDOW)
                    # score-function surrogate for increasing heavy-center semantic margin.
                    # Objective return higher is better; physical metric lower is better.
                    obj_h=float(h["obj"][sl,e,IDX[lab]].mean().cpu())
                    obj_c=float(c["obj"][sl,e,IDX[lab]].mean().cpu())
                    ph_h=float(h["phys"][PHYS[lab]][sl,e].mean().cpu())
                    ph_c=float(c["phys"][PHYS[lab]][sl,e].mean().cpu())
                    logh=lph[sl,e].mean(); logc=lpc[sl,e].mean()
                    sample_loss=torch.tensor(0.0,device="cuda")
                    if dobj>0:
                        sev=dobj/(abs(floor_obj)+1e-6)
                        sample_loss=sample_loss+sev*(-obj_h*logh + obj_c*logc)
                    if dphys>0:
                        sev=dphys/(abs(floor_phys)+1e-6)
                        # maximize phys margin = M_center - M_heavy
                        sample_loss=sample_loss+sev*(-ph_c*logc + ph_h*logh)
                    losses.append(sample_loss)
        if not losses:
            z=torch.zeros(sum(p.numel() for _,p in nps),device="cuda")
            return z,{"active_windows":0,"total_windows":total,"mean_obj_deficit":0.0,"mean_phys_deficit":0.0}
        loss=sum(losses)/len(losses)
        g=flat_grad(loss,nps).detach()
        return g,{"active_windows":active,"total_windows":total,
                  "active_fraction":active/max(total,1),
                  "mean_obj_deficit":float(np.mean(objdefs)) if objdefs else 0.0,
                  "mean_phys_deficit":float(np.mean(phydefs)) if phydefs else 0.0,
                  "loss":float(loss.detach().cpu())}
    
    def action_response_reference(env,m,lab):
        states=[];meta=[]
        for seed in REF_SEEDS[lab]:
            for plab in (lab,"C"):
                w=torch.tensor(PREFS[plab],device="cuda").repeat(NENV,1)
                cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
                with torch.no_grad():
                    for t in range(H):
                        if t in REF_PHASES:
                            states.append(cur.detach().clone());meta.extend([{"seed":seed,"phase":t,"source_pref":plab} for _ in range(NENV)])
                        a=m.act_inference_with_preference(cur,w)
                        nxt,_,_,_,_=env.step(a);cur=ot(nxt).cuda()
        s=torch.cat(states,0);wi=torch.tensor(PREFS[lab],device="cuda").repeat(len(s),1);wc=torch.tensor(PREFS["C"],device="cuda").repeat(len(s),1)
        with torch.no_grad():
            delta=(m.act_inference_with_preference(s,wi)-m.act_inference_with_preference(s,wc)).detach().clone()
        scale=float(torch.sqrt(torch.mean(delta*delta)).cpu())
        return {"states":s,"wi":wi,"wc":wc,"delta_ref":delta,"scale":max(scale,1e-4),"n_states":len(s)}
    def action_rehearsal_gradient(m,refs):
        nps=actor_named_params(m)
        if not refs: return torch.zeros(sum(p.numel() for _,p in nps),device="cuda"),{"loss":0.0}
        ls=[]
        for r in refs.values():
            d=m.act_inference_with_preference(r["states"],r["wi"])-m.act_inference_with_preference(r["states"],r["wc"])
            ls.append(torch.mean((d-r["delta_ref"])**2)/(r["scale"]**2+1e-12))
        loss=sum(ls)/len(ls);g=flat_grad(loss,nps).detach()
        return g,{"loss":float(loss.detach().cpu())}
    
    def mixed_gradient(m,b):
        weighted,ratio=objective_losses(m,b);loss=sum(weighted.values())
        g=flat_grad(loss,actor_named_params(m)).detach()
        return g,{"mixed_loss":float(loss.detach().cpu()),"ratio_maxerr":float((ratio-1).abs().max().detach().cpu())}
    def combine_bounded(gm,gr):
        gmnorm=float(gm.norm().cpu());grnorm=float(gr.norm().cpu())
        alpha=min(BETA0,RHO*gmnorm/(grnorm+EPS)) if grnorm>EPS else 0.0
        gt=gm+alpha*gr
        d=-gt/(gt.norm()+EPS);dm=-gm/(gm.norm()+EPS)
        return d,{"alpha_t":float(alpha),"mixed_grad_norm":gmnorm,"rehearsal_grad_norm":grnorm,
                  "raw_rehearsal_to_mixed_grad_ratio":grnorm/(gmnorm+EPS),
                  "weighted_rehearsal_to_mixed_grad_ratio":float((alpha*gr).norm().cpu()/(gmnorm+EPS)),
                  "cos_mixed_total":cos(dm,d)}
    
    def rollout(env,m,mgr,robot,w_np,seed):
        from talon_rl.rewards.objectives import normalized_objective_vector
        w=torch.tensor(w_np,device="cuda").repeat(NENV,1);cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        R=[];phys=[];done_any=np.zeros(NENV,bool);prev=torch.zeros((NENV,env.unwrapped.action_manager.total_action_dim),device="cuda")
        with torch.no_grad():
            for _ in range(FULL_STEPS):
                a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                R.append(normalized_objective_vector(terms(raw,names),shape=(NENV,))*env.unwrapped.step_dt)
                done_any|=(te|tr).cpu().numpy()
                pv=phys_vector(robot,env,a,prev);phys.append({k:float(v.mean().cpu()) for k,v in pv.items()})
                prev=a;cur=ot(nxt).cuda()
        R=np.asarray(R)
        return {"objective_mean":R.mean((0,1)).tolist(),
                "physical":{k:float(np.mean([x[k] for x in phys])) for k in phys[0]},
                "survival":float(1-done_any.mean())}
    def endpoint_eval(env,m,mgr,robot):
        rows=[]
        for suite in range(ENDPOINT_SUITES):
            seed=840001+suite
            for lab in ("T","A","O","S","C"):
                q=rollout(env,m,mgr,robot,PREFS[lab],seed);q.update({"suite":suite,"label":lab});rows.append(q)
        ep={}
        for lab in ORDER:
            j=IDX[lab];pk=PHYS[lab];obj=[];phy=[];surv=[]
            for suite in range(ENDPOINT_SUITES):
                r=next(x for x in rows if x["suite"]==suite and x["label"]==lab);c=next(x for x in rows if x["suite"]==suite and x["label"]=="C")
                obj.append((r["objective_mean"][j]-c["objective_mean"][j])>0)
                phy.append((r["physical"][pk]-c["physical"][pk])<0);surv.append(r["survival"])
            of=float(np.mean(obj));pf=float(np.mean(phy))
            ep[lab]={"objective_correct_fraction":of,"physical_correct_fraction":pf,"semantic_score":0.5*(of+pf),
                     "pass":bool(of>=.75 and pf>=.75 and min(surv)>=.95),"min_survival":float(min(surv))}
        return ep
    def preference_separation(m,probe):
        acts={}
        with torch.no_grad():
            for lab in ORDER:
                w=torch.tensor(PREFS[lab],device="cuda").repeat(len(probe),1);acts[lab]=m.act_inference_with_preference(probe,w)
        vals=[]
        for i,a in enumerate(ORDER):
            for j,b in enumerate(ORDER):
                if j>i: vals.append(float(torch.linalg.vector_norm(acts[a]-acts[b],dim=-1).mean().cpu()))
        return float(np.mean(vals))
    def retention_stats(base,rows):
        out={}
        for lab in ORDER:
            scores=[base[lab]["semantic_score"]]+[r["semantic_scores"][lab] for r in rows]
            passes=[base[lab]["pass"]]+[r["endpoint"][lab]["pass"] for r in rows]
            rb=np.maximum.accumulate(scores);fg=[float(rb[i]-scores[i]) for i in range(len(scores))]
            fp=next((i for i,v in enumerate(passes) if v),None)
            ret=None if fp is None else (1.0 if fp==len(passes)-1 else float(np.mean(passes[fp+1:])))
            out[lab]={"max_forgetting":float(max(fg)),"best_score":float(max(scores)),"final_score":float(scores[-1]),
                      "first_pass_checkpoint":fp,"retained_pass_fraction_after_first_pass":ret}
        return out
    
    def main():
        ap=argparse.ArgumentParser();ap.add_argument("--output-dir",type=Path,default=OUT);args=ap.parse_args()
        if not args.output_dir.is_absolute():args.output_dir=(ROOT/args.output_dir).resolve()
        args.output_dir.mkdir(parents=True,exist_ok=True)
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]]
        app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.conditioning.single_site_film import V2BSingleSiteFiLMActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=0);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim
            mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
            base=V2BSingleSiteFiLMActorCritic(o.shape[-1],ad).cuda();base.load_state_dict(torch.load(CKPT,map_location="cuda",weights_only=False)["model"]);base.eval()
            init=copy.deepcopy(base.state_dict());step_norm=STEP_SCALE*nominal_step()
            wb=torch.tensor(PREFS["A"],device="cuda").repeat(NENV,1);probe=collect_batch(env,base,mgr,BASE_SEED,wb)["obs"][:64].detach().clone()
            base_ep=endpoint_eval(env,base,mgr,robot)
            arms=("control","action_response","semantic_outcome")
            models={};rows={}
            act_refs={};out_refs={}
            activation={"action_response":{lab:None for lab in ORDER},
                        "semantic_outcome":{lab:None for lab in ORDER}}
            for arm in arms:
                m=V2BSingleSiteFiLMActorCritic(o.shape[-1],ad).cuda();m.load_state_dict(init);m.eval();models[arm]=m;rows[arm]=[]
            for lab in ORDER:
                if base_ep[lab]["pass"]:
                    act_refs[lab]=action_response_reference(env,models["action_response"],lab)
                    out_refs[lab]=capture_outcome_reference(env,models["semantic_outcome"],mgr,robot,lab)
                    activation["action_response"][lab]=0
                    activation["semantic_outcome"][lab]=0
            prev={a:None for a in arms}
            for k in range(1,K+1):
                seed=BASE_SEED+k*313
                for arm in arms:
                    m=models[arm];_,w=mixed_w(torch.device("cuda"),k);b=collect_batch(env,m,mgr,seed,w)
                    gm,mmeta=mixed_gradient(m,b)
                    if arm=="control":
                        gr=torch.zeros_like(gm);rmeta={"type":"none"}
                    elif arm=="action_response":
                        gr,rmeta=action_rehearsal_gradient(m,act_refs);rmeta["type"]="action_response"
                    else:
                        gr,rmeta=semantic_outcome_rehearsal_gradient(env,m,mgr,robot,out_refs);rmeta["type"]="semantic_outcome"
                    d,bmeta=combine_bounded(gm,gr)
                    cp=None if prev[arm] is None else cos(d,prev[arm]);prev[arm]=d.detach().clone()
                    apply_flat_step(m,d,step_norm)
                    ep=endpoint_eval(env,m,mgr,robot)
                    new=[]
                    if arm=="action_response":
                        for lab in ORDER:
                            if ep[lab]["pass"] and lab not in act_refs:
                                act_refs[lab]=action_response_reference(env,m,lab)
                                activation["action_response"][lab]=k;new.append(lab)
                        active_before=sorted(set(act_refs)-set(new))
                    elif arm=="semantic_outcome":
                        for lab in ORDER:
                            if ep[lab]["pass"] and lab not in out_refs:
                                out_refs[lab]=capture_outcome_reference(env,m,mgr,robot,lab)
                                activation["semantic_outcome"][lab]=k;new.append(lab)
                        active_before=sorted(set(out_refs)-set(new))
                    else:
                        active_before=[]
                    rows[arm].append({"update":k,"active_refs_before_update":active_before,
                        "new_axes_activated":new,"step_norm":step_norm,"mixed_meta":mmeta,"rehearsal_meta":rmeta,"budget_meta":bmeta,
                        "update_cos_prev":cp,"preference_separation":preference_separation(m,probe),"endpoint":ep,
                        "semantic_scores":{lab:ep[lab]["semantic_score"] for lab in ORDER}})
                    torch.save({"model":m.state_dict(),"arm":arm,"update":k},args.output_dir/f"{arm}_u{k}.pt")
                (args.output_dir/"semantic_outcome_partial.json").write_text(json.dumps({"activation":activation,"active":{"action_response":sorted(act_refs),"semantic_outcome":sorted(out_refs)},"arms":rows},indent=2)+"\n")
            report={"schema":"v2b_semantic_outcome_rehearsal_gate_v1","base_checkpoint":str(CKPT.relative_to(ROOT)),
                    "rho":RHO,"beta_cap":BETA0,"kappa":KAPPA,"window":WINDOW,"reference_phases":list(REF_PHASES),
                    "reference_seeds":REF_SEEDS,"step_norm":step_norm,"updates":K,"baseline_endpoint":base_ep,"activation":activation,
                    "arms":rows,"retention_stats":{a:retention_stats(base_ep,r) for a,r in rows.items()}}
            out=args.output_dir/"semantic_outcome_rehearsal_report.json";out.write_text(json.dumps(report,indent=2)+"\n")
            sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
            (args.output_dir/"PROVENANCE_MANIFEST.json").write_text(json.dumps({"status":"FROZEN_BY_HASH","report_sha256":sha(out),
              "script_sha256":sha(Path(__file__).resolve()),"base_checkpoint_sha256":sha(CKPT)},indent=2)+"\n")
            print(json.dumps({"activation":activation,"retention_stats":report["retention_stats"]},indent=2))
        finally:
            if env is not None: env.close()
            app.close()
    if True: main()

def run_v2b_soft_semantic_rehearsal_gate():
    """Run former v2b_soft_semantic_rehearsal_gate.py stage."""
    from pathlib import Path
    import argparse,json,sys,hashlib,copy
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    CKPT=ROOT/"runs/v2b_adam_continuous-2026-09-24/model_75.pt"
    OPTLOG=ROOT/"runs/v2b_adam_continuous-2026-09-24/adam_history_report.json"
    OUT=ROOT/"runs/v2b_soft_semantic_rehearsal_gate-2026-09-24"
    ORDER=("T","A","O","S")
    PREFS={
    "T":np.array([.7,.1,.1,.1],np.float32),
    "A":np.array([.1,.7,.1,.1],np.float32),
    "O":np.array([.1,.1,.7,.1],np.float32),
    "S":np.array([.1,.1,.1,.7],np.float32),
    "C":np.array([.25,.25,.25,.25],np.float32)}
    IDX={"T":0,"A":1,"O":2,"S":3}
    PHYS={"T":"tracking_error","A":"ang_vel_xy","O":"tilt_deg","S":"action_rate"}
    G=.99;LAM=1.0;H=32;NENV=8;K=8;STEP_SCALE=.5
    FULL_STEPS=64;ENDPOINT_SUITES=4;BASE_SEED=980001
    REF_SEEDS={"T":[911001,911101,911201,911301],
               "A":[912001,912101,912201,912301],
               "O":[913001,913101,913201,913301],
               "S":[914001,914101,914201,914301]}
    REF_PHASES=(0,8,16,24)
    BETA=0.25
    
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def tilt_deg(q):
        _,x,y,_=[q[:,i] for i in range(4)]
        return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def actor_named_params(m):
        return [(n,p) for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std" or n.startswith("preference_embedding") or n.startswith("preference_film")]
    def flat_grad(loss,nps):
        gs=torch.autograd.grad(loss,[p for _,p in nps],retain_graph=False,allow_unused=True)
        return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for (_,p),g in zip(nps,gs)])
    def apply_flat_step(m,direction,step_norm):
        off=0
        with torch.no_grad():
            for _,p in actor_named_params(m):
                n=p.numel();p.add_(direction[off:off+n].view_as(p)*step_norm);off+=n
            m.log_std.clamp_(m.LOG_STD_MIN,m.LOG_STD_MAX)
    def cos(a,b):
        den=float(a.norm()*b.norm())
        return float(torch.dot(a,b)/den) if den>1e-12 else 0.0
    def nominal_step():
        d=json.load(open(OPTLOG))
        return float(np.mean([r["actual_param_delta_norm"] for r in d["rows"] if 51<=r["update"]<=75]))
    
    def collect_batch(env,m,mgr,seed,w_env):
        from talon_rl.rewards.objectives import normalized_objective_vector
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda();torch.manual_seed(seed+777)
        ob=[];u=[];old=[];R=[];D=[]
        with torch.no_grad():
            for _ in range(H):
                a,lp,uu=m.act_with_preference_latent(cur,w_env)
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
                ob.append(cur);u.append(uu);old.append(lp)
                R.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt)
                D.append((te|tr).cuda());cur=ot(nxt).cuda()
        return {"obs":torch.cat(ob),"u":torch.cat(u),"old":torch.cat(old),
                "rt":torch.stack(R),"dt":torch.stack(D).bool(),"next_obs":cur,
                "w":w_env.repeat(H,1),"w_env":w_env}
    
    def gae(reward,value,nextv,done):
        adv=torch.zeros_like(reward);last=torch.zeros_like(nextv)
        for t in range(H-1,-1,-1):
            boot=nextv if t==H-1 else value[t+1]
            nt=(~done[t]).to(reward.dtype).unsqueeze(-1)
            delta=reward[t]+G*boot*nt-value[t]
            last=delta+G*LAM*nt*last;adv[t]=last
        return adv
    
    def objective_losses(m,b):
        with torch.no_grad():
            V=m.value_with_preference(b["obs"],b["w"]).reshape(H,NENV,4)
            NV=m.value_with_preference(b["next_obs"],b["w_env"])
            A=gae(b["rt"],V,NV,b["dt"]).reshape(-1,4).detach()
        logp=m.logp_from_pre_tanh_with_preference(b["obs"],b["w"],b["u"])
        ratio=torch.exp(logp-b["old"].detach());clip=ratio.clamp(.8,1.2)
        weighted={}
        for j,lab in enumerate(ORDER):
            po=torch.minimum(ratio*A[:,j],clip*A[:,j])
            weighted[lab]=-(4.0*b["w"][:,j]*po).mean()
        return weighted,ratio
    
    def mixed_w(device,k):
        labs=[ORDER[(k+i)%4] for i in range(NENV)]
        return labs,torch.tensor(np.stack([PREFS[x] for x in labs]),device=device)
    
    def collect_reference_distribution(env,m,lab):
        states=[];meta=[]
        prefs=[lab,"C"]
        for seed in REF_SEEDS[lab]:
            for plab in prefs:
                w=torch.tensor(PREFS[plab],device="cuda").repeat(NENV,1)
                cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
                with torch.no_grad():
                    for t in range(H):
                        if t in REF_PHASES:
                            states.append(cur.detach().clone())
                            meta.extend([{"seed":seed,"phase":t,"source_pref":plab} for _ in range(NENV)])
                        a=m.act_inference_with_preference(cur,w)
                        nxt,_,_,_,_=env.step(a);cur=ot(nxt).cuda()
        s=torch.cat(states,0)
        wi=torch.tensor(PREFS[lab],device="cuda").repeat(len(s),1)
        wc=torch.tensor(PREFS["C"],device="cuda").repeat(len(s),1)
        with torch.no_grad():
            ai=m.act_inference_with_preference(s,wi)
            ac=m.act_inference_with_preference(s,wc)
            delta=(ai-ac).detach().clone()
        scale=float(torch.sqrt(torch.mean(delta*delta)).cpu())
        return {"states":s,"wi":wi,"wc":wc,"delta_ref":delta,
                "scale":max(scale,1e-4),"meta":meta,"n_states":len(s)}
    
    def rehearsal_loss(m,refs):
        if not refs:return torch.tensor(0.0,device="cuda"),{}
        ls=[];diag={}
        for lab,r in refs.items():
            ai=m.act_inference_with_preference(r["states"],r["wi"])
            ac=m.act_inference_with_preference(r["states"],r["wc"])
            delta=ai-ac
            mse=torch.mean((delta-r["delta_ref"])**2)/(r["scale"]**2+1e-12)
            ls.append(mse);diag[lab]=mse.detach()
        return sum(ls)/len(ls),diag
    
    def update_direction(m,b,refs,beta):
        weighted,ratio=objective_losses(m,b)
        mixed=sum(weighted.values())
        reh,per=rehearsal_loss(m,refs)
        total=mixed+beta*reh
        nps=actor_named_params(m)
        gm=flat_grad(mixed,nps).detach()
        # recompute because first autograd consumed graph
        weighted2,_=objective_losses(m,b);mixed2=sum(weighted2.values())
        reh2,per2=rehearsal_loss(m,refs);total2=mixed2+beta*reh2
        gt=flat_grad(total2,nps).detach()
        d=-gt/(gt.norm()+1e-12)
        dm=-gm/(gm.norm()+1e-12)
        return d,{"mixed_loss":float(mixed.detach().cpu()),"rehearsal_loss":float(reh.detach().cpu()),
                  "total_loss":float(total.detach().cpu()),"mixed_grad_norm":float(gm.norm().cpu()),
                  "total_grad_norm":float(gt.norm().cpu()),"cos_mixed_total":cos(dm,d),
                  "ratio_maxerr":float((ratio-1).abs().max().detach().cpu()),
                  "per_axis_rehearsal":{k:float(v.cpu()) for k,v in per.items()}}
    
    def rehearsal_drift(m,refs):
        out={}
        with torch.no_grad():
            for lab,r in refs.items():
                ai=m.act_inference_with_preference(r["states"],r["wi"])
                ac=m.act_inference_with_preference(r["states"],r["wc"])
                delta=ai-ac
                rmse=float(torch.sqrt(torch.mean((delta-r["delta_ref"])**2)).cpu())
                out[lab]={"rmse":rmse,"normalized_rmse":rmse/(r["scale"]+1e-12),
                          "reference_scale":r["scale"],"n_states":r["n_states"]}
        return out
    
    def rollout(env,m,mgr,robot,w_np,seed):
        from talon_rl.rewards.objectives import normalized_objective_vector
        w=torch.tensor(w_np,device="cuda").repeat(NENV,1)
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        R=[];phys=[];done_any=np.zeros(NENV,bool);prev=torch.zeros((NENV,env.unwrapped.action_manager.total_action_dim),device="cuda")
        with torch.no_grad():
            for _ in range(FULL_STEPS):
                a=m.act_inference_with_preference(cur,w)
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                R.append(normalized_objective_vector(terms(raw,names),shape=(NENV,))*env.unwrapped.step_dt)
                done_any|=(te|tr).cpu().numpy()
                data=robot.data;cmd=env.unwrapped.command_manager.get_command("base_velocity")
                vx=float((data.root_lin_vel_b[:,0]-cmd[:,0]).abs().mean())
                wz=float((data.root_ang_vel_b[:,2]-cmd[:,2]).abs().mean())
                phys.append({"tracking_error":vx+wz,
                             "ang_vel_xy":float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),
                             "tilt_deg":float(tilt_deg(data.root_quat_w).mean()),
                             "action_rate":float(torch.linalg.vector_norm(a-prev,dim=-1).mean())})
                prev=a;cur=ot(nxt).cuda()
        R=np.asarray(R)
        return {"objective_mean":R.mean((0,1)).tolist(),
                "physical":{k:float(np.mean([x[k] for x in phys])) for k in phys[0]},
                "survival":float(1-done_any.mean())}
    
    def endpoint_eval(env,m,mgr,robot):
        rows=[]
        for suite in range(ENDPOINT_SUITES):
            seed=840001+suite
            for lab in ("T","A","O","S","C"):
                q=rollout(env,m,mgr,robot,PREFS[lab],seed);q.update({"suite":suite,"label":lab});rows.append(q)
        ep={}
        for lab in ORDER:
            j=IDX[lab];pk=PHYS[lab];obj=[];phy=[];surv=[]
            for suite in range(ENDPOINT_SUITES):
                r=next(x for x in rows if x["suite"]==suite and x["label"]==lab)
                c=next(x for x in rows if x["suite"]==suite and x["label"]=="C")
                obj.append((r["objective_mean"][j]-c["objective_mean"][j])>0)
                phy.append((r["physical"][pk]-c["physical"][pk])<0);surv.append(r["survival"])
            of=float(np.mean(obj));pf=float(np.mean(phy))
            ep[lab]={"objective_correct_fraction":of,"physical_correct_fraction":pf,
                     "semantic_score":0.5*(of+pf),"pass":bool(of>=.75 and pf>=.75 and min(surv)>=.95),
                     "min_survival":float(min(surv))}
        return ep
    
    def preference_separation(m,probe):
        acts={}
        with torch.no_grad():
            for lab in ORDER:
                w=torch.tensor(PREFS[lab],device="cuda").repeat(len(probe),1)
                acts[lab]=m.act_inference_with_preference(probe,w)
        vals=[]
        for i,a in enumerate(ORDER):
            for j,b in enumerate(ORDER):
                if j>i:vals.append(float(torch.linalg.vector_norm(acts[a]-acts[b],dim=-1).mean().cpu()))
        return float(np.mean(vals))
    
    def retention_stats(base,rows):
        out={}
        for lab in ORDER:
            scores=[base[lab]["semantic_score"]]+[r["semantic_scores"][lab] for r in rows]
            passes=[base[lab]["pass"]]+[r["endpoint"][lab]["pass"] for r in rows]
            rb=np.maximum.accumulate(scores);fg=[float(rb[i]-scores[i]) for i in range(len(scores))]
            fp=next((i for i,v in enumerate(passes) if v),None)
            ret=None if fp is None else (1.0 if fp==len(passes)-1 else float(np.mean(passes[fp+1:])))
            out[lab]={"max_forgetting":float(max(fg)),"best_score":float(max(scores)),
                      "final_score":float(scores[-1]),"first_pass_checkpoint":fp,
                      "retained_pass_fraction_after_first_pass":ret}
        return out
    
    def main():
        ap=argparse.ArgumentParser();ap.add_argument("--output-dir",type=Path,default=OUT);args=ap.parse_args()
        if not args.output_dir.is_absolute():args.output_dir=(ROOT/args.output_dir).resolve()
        args.output_dir.mkdir(parents=True,exist_ok=True)
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]]
        app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.conditioning.single_site_film import V2BSingleSiteFiLMActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=0);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim
            mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
            base=V2BSingleSiteFiLMActorCritic(o.shape[-1],ad).cuda()
            base.load_state_dict(torch.load(CKPT,map_location="cuda",weights_only=False)["model"]);base.eval()
            init=copy.deepcopy(base.state_dict());step_norm=STEP_SCALE*nominal_step()
            wb=torch.tensor(PREFS["A"],device="cuda").repeat(NENV,1)
            probe=collect_batch(env,base,mgr,BASE_SEED,wb)["obs"][:64].detach().clone()
            base_ep=endpoint_eval(env,base,mgr,robot)
            models={};rows={};refs={};activation={}
            for arm in ("control","rehearsal"):
                m=V2BSingleSiteFiLMActorCritic(o.shape[-1],ad).cuda();m.load_state_dict(init);m.eval()
                models[arm]=m;rows[arm]=[]
            for lab in ORDER:
                activation[lab]=None
                if base_ep[lab]["pass"]:
                    refs[lab]=collect_reference_distribution(env,models["rehearsal"],lab);activation[lab]=0
            prev_dirs={a:None for a in models}
            for k in range(1,K+1):
                seed=BASE_SEED+k*313
                # treatment
                mt=models["rehearsal"];_,wt=mixed_w(torch.device("cuda"),k)
                bt=collect_batch(env,mt,mgr,seed,wt)
                d_t,meta_t=update_direction(mt,bt,refs,BETA)
                cosprev_t=None if prev_dirs["rehearsal"] is None else cos(d_t,prev_dirs["rehearsal"])
                apply_flat_step(mt,d_t,step_norm);prev_dirs["rehearsal"]=d_t.detach().clone()
                ep_t=endpoint_eval(env,mt,mgr,robot)
                new_axes=[]
                for lab in ORDER:
                    if ep_t[lab]["pass"] and lab not in refs:
                        refs[lab]=collect_reference_distribution(env,mt,lab);activation[lab]=k;new_axes.append(lab)
                rows["rehearsal"].append({"update":k,"active_refs_before_update":sorted([x for x in refs if x not in new_axes]),
                    "new_axes_activated":new_axes,"step_norm":step_norm,"update_meta":meta_t,
                    "update_cos_prev":cosprev_t,"preference_separation":preference_separation(mt,probe),
                    "rehearsal_drift":rehearsal_drift(mt,refs),"endpoint":ep_t,
                    "semantic_scores":{lab:ep_t[lab]["semantic_score"] for lab in ORDER}})
                torch.save({"model":mt.state_dict(),"arm":"rehearsal","update":k},args.output_dir/f"rehearsal_u{k}.pt")
                # control - compute rehearsal diagnostic for matched compute, beta=0
                mc=models["control"];_,wc=mixed_w(torch.device("cuda"),k)
                bc=collect_batch(env,mc,mgr,seed,wc)
                d_c,meta_c=update_direction(mc,bc,refs,0.0)
                cosprev_c=None if prev_dirs["control"] is None else cos(d_c,prev_dirs["control"])
                apply_flat_step(mc,d_c,step_norm);prev_dirs["control"]=d_c.detach().clone()
                ep_c=endpoint_eval(env,mc,mgr,robot)
                rows["control"].append({"update":k,"active_refs_before_update":sorted(refs),
                    "step_norm":step_norm,"update_meta":meta_c,"update_cos_prev":cosprev_c,
                    "preference_separation":preference_separation(mc,probe),
                    "rehearsal_drift":rehearsal_drift(mc,refs),"endpoint":ep_c,
                    "semantic_scores":{lab:ep_c[lab]["semantic_score"] for lab in ORDER}})
                torch.save({"model":mc.state_dict(),"arm":"control","update":k},args.output_dir/f"control_u{k}.pt")
                # lightweight partial, omit reference tensors
                ref_meta={lab:{"n_states":r["n_states"],"scale":r["scale"],
                               "phase_counts":{str(p):sum(1 for x in r["meta"] if x["phase"]==p) for p in REF_PHASES}}
                          for lab,r in refs.items()}
                (args.output_dir/"soft_rehearsal_partial.json").write_text(json.dumps({
                    "activation":activation,"active":sorted(refs),"reference_meta":ref_meta,"arms":rows},indent=2)+"\n")
            ref_meta={lab:{"n_states":r["n_states"],"scale":r["scale"],
                           "phase_counts":{str(p):sum(1 for x in r["meta"] if x["phase"]==p) for p in REF_PHASES}}
                      for lab,r in refs.items()}
            report={"schema":"v2b_soft_semantic_rehearsal_gate_v1","base_checkpoint":str(CKPT.relative_to(ROOT)),
                    "beta":BETA,"rehearsal_target":"preference-conditioned action response delta pi(s,w_i)-pi(s,w_center)",
                    "reference_seeds":REF_SEEDS,"reference_phases":list(REF_PHASES),"reference_meta":ref_meta,
                    "step_norm":step_norm,"updates":K,"baseline_endpoint":base_ep,"activation":activation,
                    "arms":rows,"retention_stats":{a:retention_stats(base_ep,r) for a,r in rows.items()}}
            out=args.output_dir/"soft_semantic_rehearsal_report.json";out.write_text(json.dumps(report,indent=2)+"\n")
            sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
            (args.output_dir/"PROVENANCE_MANIFEST.json").write_text(json.dumps({"status":"FROZEN_BY_HASH",
              "report_sha256":sha(out),"script_sha256":sha(Path(__file__).resolve()),"base_checkpoint_sha256":sha(CKPT)},indent=2)+"\n")
            print(json.dumps({"activation":activation,"retention_stats":report["retention_stats"]},indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_v2b_soft_semantic_rehearsal_gate_v2():
    """Run former v2b_soft_semantic_rehearsal_gate_v2.py stage."""
    from pathlib import Path
    import argparse,json,sys,hashlib,copy
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    CKPT=ROOT/"runs/v2b_adam_continuous-2026-09-24/model_75.pt"
    OPTLOG=ROOT/"runs/v2b_adam_continuous-2026-09-24/adam_history_report.json"
    OUT=ROOT/"runs/v2b_soft_semantic_rehearsal_gate_v2-2026-09-24"
    ORDER=("T","A","O","S")
    PREFS={
    "T":np.array([.7,.1,.1,.1],np.float32),
    "A":np.array([.1,.7,.1,.1],np.float32),
    "O":np.array([.1,.1,.7,.1],np.float32),
    "S":np.array([.1,.1,.1,.7],np.float32),
    "C":np.array([.25,.25,.25,.25],np.float32)}
    IDX={"T":0,"A":1,"O":2,"S":3}
    PHYS={"T":"tracking_error","A":"ang_vel_xy","O":"tilt_deg","S":"action_rate"}
    G=.99;LAM=1.0;H=32;NENV=8;K=8;STEP_SCALE=.5
    FULL_STEPS=64;ENDPOINT_SUITES=4;BASE_SEED=980001
    REF_SEEDS={"T":[911001,911101,911201,911301],
               "A":[912001,912101,912201,912301],
               "O":[913001,913101,913201,913301],
               "S":[914001,914101,914201,914301]}
    REF_PHASES=(0,8,16,24)
    TARGET_REHEARSAL_GRAD_FRAC=0.25
    
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def tilt_deg(q):
        _,x,y,_=[q[:,i] for i in range(4)]
        return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def actor_named_params(m):
        return [(n,p) for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std" or n.startswith("preference_embedding") or n.startswith("preference_film")]
    def flat_grad(loss,nps):
        gs=torch.autograd.grad(loss,[p for _,p in nps],retain_graph=False,allow_unused=True)
        return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for (_,p),g in zip(nps,gs)])
    def apply_flat_step(m,direction,step_norm):
        off=0
        with torch.no_grad():
            for _,p in actor_named_params(m):
                n=p.numel();p.add_(direction[off:off+n].view_as(p)*step_norm);off+=n
            m.log_std.clamp_(m.LOG_STD_MIN,m.LOG_STD_MAX)
    def cos(a,b):
        den=float(a.norm()*b.norm())
        return float(torch.dot(a,b)/den) if den>1e-12 else 0.0
    def nominal_step():
        d=json.load(open(OPTLOG))
        return float(np.mean([r["actual_param_delta_norm"] for r in d["rows"] if 51<=r["update"]<=75]))
    
    def collect_batch(env,m,mgr,seed,w_env):
        from talon_rl.rewards.objectives import normalized_objective_vector
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda();torch.manual_seed(seed+777)
        ob=[];u=[];old=[];R=[];D=[]
        with torch.no_grad():
            for _ in range(H):
                a,lp,uu=m.act_with_preference_latent(cur,w_env)
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
                ob.append(cur);u.append(uu);old.append(lp)
                R.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt)
                D.append((te|tr).cuda());cur=ot(nxt).cuda()
        return {"obs":torch.cat(ob),"u":torch.cat(u),"old":torch.cat(old),
                "rt":torch.stack(R),"dt":torch.stack(D).bool(),"next_obs":cur,
                "w":w_env.repeat(H,1),"w_env":w_env}
    
    def gae(reward,value,nextv,done):
        adv=torch.zeros_like(reward);last=torch.zeros_like(nextv)
        for t in range(H-1,-1,-1):
            boot=nextv if t==H-1 else value[t+1]
            nt=(~done[t]).to(reward.dtype).unsqueeze(-1)
            delta=reward[t]+G*boot*nt-value[t]
            last=delta+G*LAM*nt*last;adv[t]=last
        return adv
    
    def objective_losses(m,b):
        with torch.no_grad():
            V=m.value_with_preference(b["obs"],b["w"]).reshape(H,NENV,4)
            NV=m.value_with_preference(b["next_obs"],b["w_env"])
            A=gae(b["rt"],V,NV,b["dt"]).reshape(-1,4).detach()
        logp=m.logp_from_pre_tanh_with_preference(b["obs"],b["w"],b["u"])
        ratio=torch.exp(logp-b["old"].detach());clip=ratio.clamp(.8,1.2)
        weighted={}
        for j,lab in enumerate(ORDER):
            po=torch.minimum(ratio*A[:,j],clip*A[:,j])
            weighted[lab]=-(4.0*b["w"][:,j]*po).mean()
        return weighted,ratio
    
    def mixed_w(device,k):
        labs=[ORDER[(k+i)%4] for i in range(NENV)]
        return labs,torch.tensor(np.stack([PREFS[x] for x in labs]),device=device)
    
    def collect_reference_distribution(env,m,lab):
        states=[];meta=[]
        prefs=[lab,"C"]
        for seed in REF_SEEDS[lab]:
            for plab in prefs:
                w=torch.tensor(PREFS[plab],device="cuda").repeat(NENV,1)
                cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
                with torch.no_grad():
                    for t in range(H):
                        if t in REF_PHASES:
                            states.append(cur.detach().clone())
                            meta.extend([{"seed":seed,"phase":t,"source_pref":plab} for _ in range(NENV)])
                        a=m.act_inference_with_preference(cur,w)
                        nxt,_,_,_,_=env.step(a);cur=ot(nxt).cuda()
        s=torch.cat(states,0)
        wi=torch.tensor(PREFS[lab],device="cuda").repeat(len(s),1)
        wc=torch.tensor(PREFS["C"],device="cuda").repeat(len(s),1)
        with torch.no_grad():
            ai=m.act_inference_with_preference(s,wi)
            ac=m.act_inference_with_preference(s,wc)
            delta=(ai-ac).detach().clone()
        scale=float(torch.sqrt(torch.mean(delta*delta)).cpu())
        return {"states":s,"wi":wi,"wc":wc,"delta_ref":delta,
                "scale":max(scale,1e-4),"meta":meta,"n_states":len(s)}
    
    def rehearsal_loss(m,refs):
        if not refs:return torch.tensor(0.0,device="cuda"),{}
        ls=[];diag={}
        for lab,r in refs.items():
            ai=m.act_inference_with_preference(r["states"],r["wi"])
            ac=m.act_inference_with_preference(r["states"],r["wc"])
            delta=ai-ac
            mse=torch.mean((delta-r["delta_ref"])**2)/(r["scale"]**2+1e-12)
            ls.append(mse);diag[lab]=mse.detach()
        return sum(ls)/len(ls),diag
    
    def gradient_components(m,b,refs):
        nps=actor_named_params(m)
        weighted,ratio=objective_losses(m,b)
        mixed=sum(weighted.values())
        gm=flat_grad(mixed,nps).detach()
        reh,per=rehearsal_loss(m,refs)
        if refs:
            gr=flat_grad(reh,nps).detach()
        else:
            gr=torch.zeros_like(gm)
        return gm,gr,{
            "mixed_loss":float(mixed.detach().cpu()),
            "rehearsal_loss":float(reh.detach().cpu()),
            "mixed_grad_norm":float(gm.norm().cpu()),
            "rehearsal_grad_norm":float(gr.norm().cpu()),
            "ratio_maxerr":float((ratio-1).abs().max().detach().cpu()),
            "per_axis_rehearsal":{k:float(v.cpu()) for k,v in per.items()}
        }
    
    def update_direction(m,b,refs,beta):
        gm,gr,meta=gradient_components(m,b,refs)
        gt=gm+beta*gr
        d=-gt/(gt.norm()+1e-12)
        dm=-gm/(gm.norm()+1e-12)
        meta.update({
            "beta":float(beta),
            "total_grad_norm":float(gt.norm().cpu()),
            "weighted_rehearsal_grad_norm":float((beta*gr).norm().cpu()),
            "weighted_rehearsal_to_mixed_grad_ratio":float((beta*gr).norm().cpu()/(gm.norm().cpu()+1e-12)),
            "cos_mixed_total":cos(dm,d)
        })
        return d,meta
    
    def rehearsal_drift(m,refs):
        out={}
        with torch.no_grad():
            for lab,r in refs.items():
                ai=m.act_inference_with_preference(r["states"],r["wi"])
                ac=m.act_inference_with_preference(r["states"],r["wc"])
                delta=ai-ac
                rmse=float(torch.sqrt(torch.mean((delta-r["delta_ref"])**2)).cpu())
                out[lab]={"rmse":rmse,"normalized_rmse":rmse/(r["scale"]+1e-12),
                          "reference_scale":r["scale"],"n_states":r["n_states"]}
        return out
    
    def rollout(env,m,mgr,robot,w_np,seed):
        from talon_rl.rewards.objectives import normalized_objective_vector
        w=torch.tensor(w_np,device="cuda").repeat(NENV,1)
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        R=[];phys=[];done_any=np.zeros(NENV,bool);prev=torch.zeros((NENV,env.unwrapped.action_manager.total_action_dim),device="cuda")
        with torch.no_grad():
            for _ in range(FULL_STEPS):
                a=m.act_inference_with_preference(cur,w)
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                R.append(normalized_objective_vector(terms(raw,names),shape=(NENV,))*env.unwrapped.step_dt)
                done_any|=(te|tr).cpu().numpy()
                data=robot.data;cmd=env.unwrapped.command_manager.get_command("base_velocity")
                vx=float((data.root_lin_vel_b[:,0]-cmd[:,0]).abs().mean())
                wz=float((data.root_ang_vel_b[:,2]-cmd[:,2]).abs().mean())
                phys.append({"tracking_error":vx+wz,
                             "ang_vel_xy":float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),
                             "tilt_deg":float(tilt_deg(data.root_quat_w).mean()),
                             "action_rate":float(torch.linalg.vector_norm(a-prev,dim=-1).mean())})
                prev=a;cur=ot(nxt).cuda()
        R=np.asarray(R)
        return {"objective_mean":R.mean((0,1)).tolist(),
                "physical":{k:float(np.mean([x[k] for x in phys])) for k in phys[0]},
                "survival":float(1-done_any.mean())}
    
    def endpoint_eval(env,m,mgr,robot):
        rows=[]
        for suite in range(ENDPOINT_SUITES):
            seed=840001+suite
            for lab in ("T","A","O","S","C"):
                q=rollout(env,m,mgr,robot,PREFS[lab],seed);q.update({"suite":suite,"label":lab});rows.append(q)
        ep={}
        for lab in ORDER:
            j=IDX[lab];pk=PHYS[lab];obj=[];phy=[];surv=[]
            for suite in range(ENDPOINT_SUITES):
                r=next(x for x in rows if x["suite"]==suite and x["label"]==lab)
                c=next(x for x in rows if x["suite"]==suite and x["label"]=="C")
                obj.append((r["objective_mean"][j]-c["objective_mean"][j])>0)
                phy.append((r["physical"][pk]-c["physical"][pk])<0);surv.append(r["survival"])
            of=float(np.mean(obj));pf=float(np.mean(phy))
            ep[lab]={"objective_correct_fraction":of,"physical_correct_fraction":pf,
                     "semantic_score":0.5*(of+pf),"pass":bool(of>=.75 and pf>=.75 and min(surv)>=.95),
                     "min_survival":float(min(surv))}
        return ep
    
    def preference_separation(m,probe):
        acts={}
        with torch.no_grad():
            for lab in ORDER:
                w=torch.tensor(PREFS[lab],device="cuda").repeat(len(probe),1)
                acts[lab]=m.act_inference_with_preference(probe,w)
        vals=[]
        for i,a in enumerate(ORDER):
            for j,b in enumerate(ORDER):
                if j>i:vals.append(float(torch.linalg.vector_norm(acts[a]-acts[b],dim=-1).mean().cpu()))
        return float(np.mean(vals))
    
    def retention_stats(base,rows):
        out={}
        for lab in ORDER:
            scores=[base[lab]["semantic_score"]]+[r["semantic_scores"][lab] for r in rows]
            passes=[base[lab]["pass"]]+[r["endpoint"][lab]["pass"] for r in rows]
            rb=np.maximum.accumulate(scores);fg=[float(rb[i]-scores[i]) for i in range(len(scores))]
            fp=next((i for i,v in enumerate(passes) if v),None)
            ret=None if fp is None else (1.0 if fp==len(passes)-1 else float(np.mean(passes[fp+1:])))
            out[lab]={"max_forgetting":float(max(fg)),"best_score":float(max(scores)),
                      "final_score":float(scores[-1]),"first_pass_checkpoint":fp,
                      "retained_pass_fraction_after_first_pass":ret}
        return out
    
    def main():
        ap=argparse.ArgumentParser();ap.add_argument("--output-dir",type=Path,default=OUT);args=ap.parse_args()
        if not args.output_dir.is_absolute():args.output_dir=(ROOT/args.output_dir).resolve()
        args.output_dir.mkdir(parents=True,exist_ok=True)
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]]
        app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.conditioning.single_site_film import V2BSingleSiteFiLMActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=0);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim
            mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
            base=V2BSingleSiteFiLMActorCritic(o.shape[-1],ad).cuda()
            base.load_state_dict(torch.load(CKPT,map_location="cuda",weights_only=False)["model"]);base.eval()
            init=copy.deepcopy(base.state_dict());step_norm=STEP_SCALE*nominal_step()
            wb=torch.tensor(PREFS["A"],device="cuda").repeat(NENV,1)
            probe=collect_batch(env,base,mgr,BASE_SEED,wb)["obs"][:64].detach().clone()
            base_ep=endpoint_eval(env,base,mgr,robot)
            models={};rows={};refs={};activation={}
            for arm in ("control","rehearsal"):
                m=V2BSingleSiteFiLMActorCritic(o.shape[-1],ad).cuda();m.load_state_dict(init);m.eval()
                models[arm]=m;rows[arm]=[]
            for lab in ORDER:
                activation[lab]=None
                if base_ep[lab]["pass"]:
                    refs[lab]=collect_reference_distribution(env,models["rehearsal"],lab);activation[lab]=0
            prev_dirs={a:None for a in models}
            beta=0.0
            beta_calibration=None
            for k in range(1,K+1):
                seed=BASE_SEED+k*313
                # treatment
                mt=models["rehearsal"];_,wt=mixed_w(torch.device("cuda"),k)
                bt=collect_batch(env,mt,mgr,seed,wt)
                if k==2 and beta_calibration is None and refs:
                    gm_cal,gr_cal,cal_meta=gradient_components(mt,bt,refs)
                    if float(gr_cal.norm())>1e-12:
                        beta=TARGET_REHEARSAL_GRAD_FRAC*float(gm_cal.norm().cpu())/float(gr_cal.norm().cpu())
                    else:
                        beta=0.0
                    beta_calibration={
                        "update":k,
                        "target_weighted_rehearsal_to_mixed_grad_ratio":TARGET_REHEARSAL_GRAD_FRAC,
                        "mixed_grad_norm":float(gm_cal.norm().cpu()),
                        "rehearsal_grad_norm":float(gr_cal.norm().cpu()),
                        "beta":float(beta)
                    }
                d_t,meta_t=update_direction(mt,bt,refs,beta)
                cosprev_t=None if prev_dirs["rehearsal"] is None else cos(d_t,prev_dirs["rehearsal"])
                apply_flat_step(mt,d_t,step_norm);prev_dirs["rehearsal"]=d_t.detach().clone()
                ep_t=endpoint_eval(env,mt,mgr,robot)
                new_axes=[]
                for lab in ORDER:
                    if ep_t[lab]["pass"] and lab not in refs:
                        refs[lab]=collect_reference_distribution(env,mt,lab);activation[lab]=k;new_axes.append(lab)
                rows["rehearsal"].append({"update":k,"active_refs_before_update":sorted([x for x in refs if x not in new_axes]),
                    "new_axes_activated":new_axes,"step_norm":step_norm,"update_meta":meta_t,
                    "update_cos_prev":cosprev_t,"preference_separation":preference_separation(mt,probe),
                    "rehearsal_drift":rehearsal_drift(mt,refs),"endpoint":ep_t,
                    "semantic_scores":{lab:ep_t[lab]["semantic_score"] for lab in ORDER}})
                torch.save({"model":mt.state_dict(),"arm":"rehearsal","update":k},args.output_dir/f"rehearsal_u{k}.pt")
                # control - compute rehearsal diagnostic for matched compute, beta=0
                mc=models["control"];_,wc=mixed_w(torch.device("cuda"),k)
                bc=collect_batch(env,mc,mgr,seed,wc)
                d_c,meta_c=update_direction(mc,bc,refs,0.0)
                cosprev_c=None if prev_dirs["control"] is None else cos(d_c,prev_dirs["control"])
                apply_flat_step(mc,d_c,step_norm);prev_dirs["control"]=d_c.detach().clone()
                ep_c=endpoint_eval(env,mc,mgr,robot)
                rows["control"].append({"update":k,"active_refs_before_update":sorted(refs),
                    "step_norm":step_norm,"update_meta":meta_c,"update_cos_prev":cosprev_c,
                    "preference_separation":preference_separation(mc,probe),
                    "rehearsal_drift":rehearsal_drift(mc,refs),"endpoint":ep_c,
                    "semantic_scores":{lab:ep_c[lab]["semantic_score"] for lab in ORDER}})
                torch.save({"model":mc.state_dict(),"arm":"control","update":k},args.output_dir/f"control_u{k}.pt")
                # lightweight partial, omit reference tensors
                ref_meta={lab:{"n_states":r["n_states"],"scale":r["scale"],
                               "phase_counts":{str(p):sum(1 for x in r["meta"] if x["phase"]==p) for p in REF_PHASES}}
                          for lab,r in refs.items()}
                (args.output_dir/"soft_rehearsal_partial.json").write_text(json.dumps({
                    "activation":activation,"active":sorted(refs),"beta":beta,"beta_calibration":beta_calibration,"reference_meta":ref_meta,"arms":rows},indent=2)+"\n")
            ref_meta={lab:{"n_states":r["n_states"],"scale":r["scale"],
                           "phase_counts":{str(p):sum(1 for x in r["meta"] if x["phase"]==p) for p in REF_PHASES}}
                      for lab,r in refs.items()}
            report={"schema":"v2b_soft_semantic_rehearsal_gate_v2","base_checkpoint":str(CKPT.relative_to(ROOT)),
                    "beta":beta,"beta_calibration":beta_calibration,"target_rehearsal_grad_fraction":TARGET_REHEARSAL_GRAD_FRAC,"rehearsal_target":"preference-conditioned action response delta pi(s,w_i)-pi(s,w_center)",
                    "reference_seeds":REF_SEEDS,"reference_phases":list(REF_PHASES),"reference_meta":ref_meta,
                    "step_norm":step_norm,"updates":K,"baseline_endpoint":base_ep,"activation":activation,
                    "arms":rows,"retention_stats":{a:retention_stats(base_ep,r) for a,r in rows.items()}}
            out=args.output_dir/"soft_semantic_rehearsal_report.json";out.write_text(json.dumps(report,indent=2)+"\n")
            sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
            (args.output_dir/"PROVENANCE_MANIFEST.json").write_text(json.dumps({"status":"FROZEN_BY_HASH",
              "report_sha256":sha(out),"script_sha256":sha(Path(__file__).resolve()),"base_checkpoint_sha256":sha(CKPT)},indent=2)+"\n")
            print(json.dumps({"activation":activation,"retention_stats":report["retention_stats"]},indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_v2b_soft_semantic_rehearsal_gate_v3():
    """Run former v2b_soft_semantic_rehearsal_gate_v3.py stage."""
    from pathlib import Path
    import argparse,json,sys,hashlib,copy
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    CKPT=ROOT/"runs/v2b_adam_continuous-2026-09-24/model_75.pt"
    OPTLOG=ROOT/"runs/v2b_adam_continuous-2026-09-24/adam_history_report.json"
    OUT=ROOT/"runs/v2b_soft_semantic_rehearsal_gate_v3-2026-09-24"
    ORDER=("T","A","O","S")
    PREFS={
    "T":np.array([.7,.1,.1,.1],np.float32),
    "A":np.array([.1,.7,.1,.1],np.float32),
    "O":np.array([.1,.1,.7,.1],np.float32),
    "S":np.array([.1,.1,.1,.7],np.float32),
    "C":np.array([.25,.25,.25,.25],np.float32)}
    IDX={"T":0,"A":1,"O":2,"S":3}
    PHYS={"T":"tracking_error","A":"ang_vel_xy","O":"tilt_deg","S":"action_rate"}
    G=.99;LAM=1.0;H=32;NENV=8;K=8;STEP_SCALE=.5
    FULL_STEPS=64;ENDPOINT_SUITES=4;BASE_SEED=980001
    REF_SEEDS={"T":[911001,911101,911201,911301],
               "A":[912001,912101,912201,912301],
               "O":[913001,913101,913201,913301],
               "S":[914001,914101,914201,914301]}
    REF_PHASES=(0,8,16,24)
    TARGET_REHEARSAL_GRAD_FRAC=0.25
    BETA0=2.497041993384243
    EPS=1e-12
    
    def ot(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def tilt_deg(q):
        _,x,y,_=[q[:,i] for i in range(4)]
        return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def actor_named_params(m):
        return [(n,p) for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std" or n.startswith("preference_embedding") or n.startswith("preference_film")]
    def flat_grad(loss,nps):
        gs=torch.autograd.grad(loss,[p for _,p in nps],retain_graph=False,allow_unused=True)
        return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for (_,p),g in zip(nps,gs)])
    def apply_flat_step(m,direction,step_norm):
        off=0
        with torch.no_grad():
            for _,p in actor_named_params(m):
                n=p.numel();p.add_(direction[off:off+n].view_as(p)*step_norm);off+=n
            m.log_std.clamp_(m.LOG_STD_MIN,m.LOG_STD_MAX)
    def cos(a,b):
        den=float(a.norm()*b.norm())
        return float(torch.dot(a,b)/den) if den>1e-12 else 0.0
    def nominal_step():
        d=json.load(open(OPTLOG))
        return float(np.mean([r["actual_param_delta_norm"] for r in d["rows"] if 51<=r["update"]<=75]))
    
    def collect_batch(env,m,mgr,seed,w_env):
        from talon_rl.rewards.objectives import normalized_objective_vector
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda();torch.manual_seed(seed+777)
        ob=[];u=[];old=[];R=[];D=[]
        with torch.no_grad():
            for _ in range(H):
                a,lp,uu=m.act_with_preference_latent(cur,w_env)
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
                ob.append(cur);u.append(uu);old.append(lp)
                R.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt)
                D.append((te|tr).cuda());cur=ot(nxt).cuda()
        return {"obs":torch.cat(ob),"u":torch.cat(u),"old":torch.cat(old),
                "rt":torch.stack(R),"dt":torch.stack(D).bool(),"next_obs":cur,
                "w":w_env.repeat(H,1),"w_env":w_env}
    
    def gae(reward,value,nextv,done):
        adv=torch.zeros_like(reward);last=torch.zeros_like(nextv)
        for t in range(H-1,-1,-1):
            boot=nextv if t==H-1 else value[t+1]
            nt=(~done[t]).to(reward.dtype).unsqueeze(-1)
            delta=reward[t]+G*boot*nt-value[t]
            last=delta+G*LAM*nt*last;adv[t]=last
        return adv
    
    def objective_losses(m,b):
        with torch.no_grad():
            V=m.value_with_preference(b["obs"],b["w"]).reshape(H,NENV,4)
            NV=m.value_with_preference(b["next_obs"],b["w_env"])
            A=gae(b["rt"],V,NV,b["dt"]).reshape(-1,4).detach()
        logp=m.logp_from_pre_tanh_with_preference(b["obs"],b["w"],b["u"])
        ratio=torch.exp(logp-b["old"].detach());clip=ratio.clamp(.8,1.2)
        weighted={}
        for j,lab in enumerate(ORDER):
            po=torch.minimum(ratio*A[:,j],clip*A[:,j])
            weighted[lab]=-(4.0*b["w"][:,j]*po).mean()
        return weighted,ratio
    
    def mixed_w(device,k):
        labs=[ORDER[(k+i)%4] for i in range(NENV)]
        return labs,torch.tensor(np.stack([PREFS[x] for x in labs]),device=device)
    
    def collect_reference_distribution(env,m,lab):
        states=[];meta=[]
        prefs=[lab,"C"]
        for seed in REF_SEEDS[lab]:
            for plab in prefs:
                w=torch.tensor(PREFS[plab],device="cuda").repeat(NENV,1)
                cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
                with torch.no_grad():
                    for t in range(H):
                        if t in REF_PHASES:
                            states.append(cur.detach().clone())
                            meta.extend([{"seed":seed,"phase":t,"source_pref":plab} for _ in range(NENV)])
                        a=m.act_inference_with_preference(cur,w)
                        nxt,_,_,_,_=env.step(a);cur=ot(nxt).cuda()
        s=torch.cat(states,0)
        wi=torch.tensor(PREFS[lab],device="cuda").repeat(len(s),1)
        wc=torch.tensor(PREFS["C"],device="cuda").repeat(len(s),1)
        with torch.no_grad():
            ai=m.act_inference_with_preference(s,wi)
            ac=m.act_inference_with_preference(s,wc)
            delta=(ai-ac).detach().clone()
        scale=float(torch.sqrt(torch.mean(delta*delta)).cpu())
        return {"states":s,"wi":wi,"wc":wc,"delta_ref":delta,
                "scale":max(scale,1e-4),"meta":meta,"n_states":len(s)}
    
    def rehearsal_loss(m,refs):
        if not refs:return torch.tensor(0.0,device="cuda"),{}
        ls=[];diag={}
        for lab,r in refs.items():
            ai=m.act_inference_with_preference(r["states"],r["wi"])
            ac=m.act_inference_with_preference(r["states"],r["wc"])
            delta=ai-ac
            mse=torch.mean((delta-r["delta_ref"])**2)/(r["scale"]**2+1e-12)
            ls.append(mse);diag[lab]=mse.detach()
        return sum(ls)/len(ls),diag
    
    def gradient_components(m,b,refs):
        nps=actor_named_params(m)
        weighted,ratio=objective_losses(m,b)
        mixed=sum(weighted.values())
        gm=flat_grad(mixed,nps).detach()
        reh,per=rehearsal_loss(m,refs)
        if refs:
            gr=flat_grad(reh,nps).detach()
        else:
            gr=torch.zeros_like(gm)
        return gm,gr,{
            "mixed_loss":float(mixed.detach().cpu()),
            "rehearsal_loss":float(reh.detach().cpu()),
            "mixed_grad_norm":float(gm.norm().cpu()),
            "rehearsal_grad_norm":float(gr.norm().cpu()),
            "ratio_maxerr":float((ratio-1).abs().max().detach().cpu()),
            "per_axis_rehearsal":{k:float(v.cpu()) for k,v in per.items()}
        }
    
    def update_direction(m,b,refs,beta_cap):
        gm,gr,meta=gradient_components(m,b,refs)
        gm_norm=float(gm.norm().cpu())
        gr_norm=float(gr.norm().cpu())
        if gr_norm>EPS and refs:
            alpha=min(float(beta_cap), TARGET_REHEARSAL_GRAD_FRAC*gm_norm/(gr_norm+EPS))
        else:
            alpha=0.0
        gt=gm+alpha*gr
        d=-gt/(gt.norm()+1e-12)
        dm=-gm/(gm.norm()+1e-12)
        raw_ratio=gr_norm/(gm_norm+EPS)
        weighted_ratio=float((alpha*gr).norm().cpu()/(gm_norm+EPS))
        meta.update({
            "alpha_t":float(alpha),
            "beta_cap":float(beta_cap),
            "raw_rehearsal_to_mixed_grad_ratio":float(raw_ratio),
            "weighted_rehearsal_to_mixed_grad_ratio":weighted_ratio,
            "total_grad_norm":float(gt.norm().cpu()),
            "weighted_rehearsal_grad_norm":float((alpha*gr).norm().cpu()),
            "cos_mixed_total":cos(dm,d)
        })
        return d,meta
    
    def rehearsal_drift(m,refs):
        out={}
        with torch.no_grad():
            for lab,r in refs.items():
                ai=m.act_inference_with_preference(r["states"],r["wi"])
                ac=m.act_inference_with_preference(r["states"],r["wc"])
                delta=ai-ac
                rmse=float(torch.sqrt(torch.mean((delta-r["delta_ref"])**2)).cpu())
                out[lab]={"rmse":rmse,"normalized_rmse":rmse/(r["scale"]+1e-12),
                          "reference_scale":r["scale"],"n_states":r["n_states"]}
        return out
    
    def rollout(env,m,mgr,robot,w_np,seed):
        from talon_rl.rewards.objectives import normalized_objective_vector
        w=torch.tensor(w_np,device="cuda").repeat(NENV,1)
        cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
        R=[];phys=[];done_any=np.zeros(NENV,bool);prev=torch.zeros((NENV,env.unwrapped.action_manager.total_action_dim),device="cuda")
        with torch.no_grad():
            for _ in range(FULL_STEPS):
                a=m.act_inference_with_preference(cur,w)
                nxt,_,te,tr,_=env.step(a)
                raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                R.append(normalized_objective_vector(terms(raw,names),shape=(NENV,))*env.unwrapped.step_dt)
                done_any|=(te|tr).cpu().numpy()
                data=robot.data;cmd=env.unwrapped.command_manager.get_command("base_velocity")
                vx=float((data.root_lin_vel_b[:,0]-cmd[:,0]).abs().mean())
                wz=float((data.root_ang_vel_b[:,2]-cmd[:,2]).abs().mean())
                phys.append({"tracking_error":vx+wz,
                             "ang_vel_xy":float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),
                             "tilt_deg":float(tilt_deg(data.root_quat_w).mean()),
                             "action_rate":float(torch.linalg.vector_norm(a-prev,dim=-1).mean())})
                prev=a;cur=ot(nxt).cuda()
        R=np.asarray(R)
        return {"objective_mean":R.mean((0,1)).tolist(),
                "physical":{k:float(np.mean([x[k] for x in phys])) for k in phys[0]},
                "survival":float(1-done_any.mean())}
    
    def endpoint_eval(env,m,mgr,robot):
        rows=[]
        for suite in range(ENDPOINT_SUITES):
            seed=840001+suite
            for lab in ("T","A","O","S","C"):
                q=rollout(env,m,mgr,robot,PREFS[lab],seed);q.update({"suite":suite,"label":lab});rows.append(q)
        ep={}
        for lab in ORDER:
            j=IDX[lab];pk=PHYS[lab];obj=[];phy=[];surv=[]
            for suite in range(ENDPOINT_SUITES):
                r=next(x for x in rows if x["suite"]==suite and x["label"]==lab)
                c=next(x for x in rows if x["suite"]==suite and x["label"]=="C")
                obj.append((r["objective_mean"][j]-c["objective_mean"][j])>0)
                phy.append((r["physical"][pk]-c["physical"][pk])<0);surv.append(r["survival"])
            of=float(np.mean(obj));pf=float(np.mean(phy))
            ep[lab]={"objective_correct_fraction":of,"physical_correct_fraction":pf,
                     "semantic_score":0.5*(of+pf),"pass":bool(of>=.75 and pf>=.75 and min(surv)>=.95),
                     "min_survival":float(min(surv))}
        return ep
    
    def preference_separation(m,probe):
        acts={}
        with torch.no_grad():
            for lab in ORDER:
                w=torch.tensor(PREFS[lab],device="cuda").repeat(len(probe),1)
                acts[lab]=m.act_inference_with_preference(probe,w)
        vals=[]
        for i,a in enumerate(ORDER):
            for j,b in enumerate(ORDER):
                if j>i:vals.append(float(torch.linalg.vector_norm(acts[a]-acts[b],dim=-1).mean().cpu()))
        return float(np.mean(vals))
    
    def retention_stats(base,rows):
        out={}
        for lab in ORDER:
            scores=[base[lab]["semantic_score"]]+[r["semantic_scores"][lab] for r in rows]
            passes=[base[lab]["pass"]]+[r["endpoint"][lab]["pass"] for r in rows]
            rb=np.maximum.accumulate(scores);fg=[float(rb[i]-scores[i]) for i in range(len(scores))]
            fp=next((i for i,v in enumerate(passes) if v),None)
            ret=None if fp is None else (1.0 if fp==len(passes)-1 else float(np.mean(passes[fp+1:])))
            out[lab]={"max_forgetting":float(max(fg)),"best_score":float(max(scores)),
                      "final_score":float(scores[-1]),"first_pass_checkpoint":fp,
                      "retained_pass_fraction_after_first_pass":ret}
        return out
    
    def main():
        ap=argparse.ArgumentParser();ap.add_argument("--output-dir",type=Path,default=OUT);args=ap.parse_args()
        if not args.output_dir.is_absolute():args.output_dir=(ROOT/args.output_dir).resolve()
        args.output_dir.mkdir(parents=True,exist_ok=True)
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]]
        app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.conditioning.single_site_film import V2BSingleSiteFiLMActorCritic
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=0);o=ot(o).cuda();ad=env.unwrapped.action_manager.total_action_dim
            mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
            base=V2BSingleSiteFiLMActorCritic(o.shape[-1],ad).cuda()
            base.load_state_dict(torch.load(CKPT,map_location="cuda",weights_only=False)["model"]);base.eval()
            init=copy.deepcopy(base.state_dict());step_norm=STEP_SCALE*nominal_step()
            wb=torch.tensor(PREFS["A"],device="cuda").repeat(NENV,1)
            probe=collect_batch(env,base,mgr,BASE_SEED,wb)["obs"][:64].detach().clone()
            base_ep=endpoint_eval(env,base,mgr,robot)
            models={};rows={};refs={};activation={}
            for arm in ("control","rehearsal"):
                m=V2BSingleSiteFiLMActorCritic(o.shape[-1],ad).cuda();m.load_state_dict(init);m.eval()
                models[arm]=m;rows[arm]=[]
            for lab in ORDER:
                activation[lab]=None
                if base_ep[lab]["pass"]:
                    refs[lab]=collect_reference_distribution(env,models["rehearsal"],lab);activation[lab]=0
            prev_dirs={a:None for a in models}
            beta_cap=BETA0
            for k in range(1,K+1):
                seed=BASE_SEED+k*313
                # treatment
                mt=models["rehearsal"];_,wt=mixed_w(torch.device("cuda"),k)
                bt=collect_batch(env,mt,mgr,seed,wt)
                d_t,meta_t=update_direction(mt,bt,refs,beta_cap)
                cosprev_t=None if prev_dirs["rehearsal"] is None else cos(d_t,prev_dirs["rehearsal"])
                apply_flat_step(mt,d_t,step_norm);prev_dirs["rehearsal"]=d_t.detach().clone()
                ep_t=endpoint_eval(env,mt,mgr,robot)
                new_axes=[]
                for lab in ORDER:
                    if ep_t[lab]["pass"] and lab not in refs:
                        refs[lab]=collect_reference_distribution(env,mt,lab);activation[lab]=k;new_axes.append(lab)
                rows["rehearsal"].append({"update":k,"active_refs_before_update":sorted([x for x in refs if x not in new_axes]),
                    "new_axes_activated":new_axes,"step_norm":step_norm,"update_meta":meta_t,
                    "update_cos_prev":cosprev_t,"preference_separation":preference_separation(mt,probe),
                    "rehearsal_drift":rehearsal_drift(mt,refs),"endpoint":ep_t,
                    "semantic_scores":{lab:ep_t[lab]["semantic_score"] for lab in ORDER}})
                torch.save({"model":mt.state_dict(),"arm":"rehearsal","update":k},args.output_dir/f"rehearsal_u{k}.pt")
                # control - compute rehearsal diagnostic for matched compute, beta=0
                mc=models["control"];_,wc=mixed_w(torch.device("cuda"),k)
                bc=collect_batch(env,mc,mgr,seed,wc)
                d_c,meta_c=update_direction(mc,bc,refs,0.0)
                cosprev_c=None if prev_dirs["control"] is None else cos(d_c,prev_dirs["control"])
                apply_flat_step(mc,d_c,step_norm);prev_dirs["control"]=d_c.detach().clone()
                ep_c=endpoint_eval(env,mc,mgr,robot)
                rows["control"].append({"update":k,"active_refs_before_update":sorted(refs),
                    "step_norm":step_norm,"update_meta":meta_c,"update_cos_prev":cosprev_c,
                    "preference_separation":preference_separation(mc,probe),
                    "rehearsal_drift":rehearsal_drift(mc,refs),"endpoint":ep_c,
                    "semantic_scores":{lab:ep_c[lab]["semantic_score"] for lab in ORDER}})
                torch.save({"model":mc.state_dict(),"arm":"control","update":k},args.output_dir/f"control_u{k}.pt")
                # lightweight partial, omit reference tensors
                ref_meta={lab:{"n_states":r["n_states"],"scale":r["scale"],
                               "phase_counts":{str(p):sum(1 for x in r["meta"] if x["phase"]==p) for p in REF_PHASES}}
                          for lab,r in refs.items()}
                (args.output_dir/"soft_rehearsal_partial.json").write_text(json.dumps({
                    "activation":activation,"active":sorted(refs),"beta_cap":beta_cap,"rho":TARGET_REHEARSAL_GRAD_FRAC,"reference_meta":ref_meta,"arms":rows},indent=2)+"\n")
            ref_meta={lab:{"n_states":r["n_states"],"scale":r["scale"],
                           "phase_counts":{str(p):sum(1 for x in r["meta"] if x["phase"]==p) for p in REF_PHASES}}
                      for lab,r in refs.items()}
            report={"schema":"v2b_soft_semantic_rehearsal_gate_v3","base_checkpoint":str(CKPT.relative_to(ROOT)),
                    "beta_cap":beta_cap,"rho":TARGET_REHEARSAL_GRAD_FRAC,"rehearsal_target":"preference-conditioned action response delta pi(s,w_i)-pi(s,w_center)",
                    "reference_seeds":REF_SEEDS,"reference_phases":list(REF_PHASES),"reference_meta":ref_meta,
                    "step_norm":step_norm,"updates":K,"baseline_endpoint":base_ep,"activation":activation,
                    "arms":rows,"retention_stats":{a:retention_stats(base_ep,r) for a,r in rows.items()}}
            out=args.output_dir/"soft_semantic_rehearsal_report.json";out.write_text(json.dumps(report,indent=2)+"\n")
            sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
            (args.output_dir/"PROVENANCE_MANIFEST.json").write_text(json.dumps({"status":"FROZEN_BY_HASH",
              "report_sha256":sha(out),"script_sha256":sha(Path(__file__).resolve()),"base_checkpoint_sha256":sha(CKPT)},indent=2)+"\n")
            print(json.dumps({"activation":activation,"retention_stats":report["retention_stats"]},indent=2))
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

STAGES = {
    "v2b_outcome_estimator_centered_audit": run_v2b_outcome_estimator_centered_audit,
    "v2b_outcome_estimator_repair_audit": run_v2b_outcome_estimator_repair_audit,
    "v2b_outcome_estimator_replica_avg_audit": run_v2b_outcome_estimator_replica_avg_audit,
    "v2b_outcome_pathwise_feasibility_audit": run_v2b_outcome_pathwise_feasibility_audit,
    "v2b_outcome_rehearsal_gradient_snr_audit": run_v2b_outcome_rehearsal_gradient_snr_audit,
    "v2b_outcome_surrogate_gate": run_v2b_outcome_surrogate_gate,
    "v2b_outcome_surrogate_rich_gate": run_v2b_outcome_surrogate_rich_gate,
    "v2b_semantic_outcome_rehearsal_gate": run_v2b_semantic_outcome_rehearsal_gate,
    "v2b_soft_semantic_rehearsal_gate": run_v2b_soft_semantic_rehearsal_gate,
    "v2b_soft_semantic_rehearsal_gate_v2": run_v2b_soft_semantic_rehearsal_gate_v2,
    "v2b_soft_semantic_rehearsal_gate_v3": run_v2b_soft_semantic_rehearsal_gate_v3,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
