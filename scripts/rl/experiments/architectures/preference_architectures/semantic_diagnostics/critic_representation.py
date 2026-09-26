"""Consolidated thematic experiment stages."""
from __future__ import annotations
import argparse

def run_post_v2_t5_c10_endpoint_compare():
    """Run former post_v2_t5_c10_endpoint_compare.py stage."""
    from pathlib import Path
    import sys,json,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUNS={"control":ROOT/"runs/post_v2_t5_c10_h32_control-2026-09-23","mc":ROOT/"runs/post_v2_t5_c10_h32_mc-2026-09-23"}
    ORDER=("T","A","O","S")
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    def obs_tensor(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def tilt(q):
     _,x,y,_=[q[:,i] for i in range(4)]
     return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=8;cfg.seed=0;env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
      obs,_=env.reset(seed=0);obs=obs_tensor(obs).cuda();ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
      out={}
      for tag,run in RUNS.items():
       models={}
       for lab in ORDER:
        m=T4SharedActorCritic(obs.shape[-1],ad).cuda();m.load_state_dict(torch.load(run/f"{lab}_snap_25.pt",map_location="cuda",weights_only=False)["model"]);m.eval();models[lab]=m
       rows=[]
       for suite in range(4):
        seed=940001+suite
        for lab in ORDER:
         m=models[lab];w=torch.tensor(PREFS[lab],device="cuda").repeat(8,1);cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda();prev=torch.zeros((8,ad),device="cuda");norm=[];phys=[];done=np.zeros(8,bool)
         with torch.no_grad():
          for _ in range(64):
           a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);norm.append(normalized_objective_vector(terms(raw,names),shape=(8,)).mean(0));data=robot.data;cmd=env.unwrapped.command_manager.get_command("base_velocity")
           phys.append({"vx_error":float((data.root_lin_vel_b[:,0]-cmd[:,0]).abs().mean()),"wz_error":float((data.root_ang_vel_b[:,2]-cmd[:,2]).abs().mean()),"ang_vel_xy":float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),"tilt_deg":float(tilt(data.root_quat_w).mean()),"action_rate":float(torch.linalg.vector_norm(a-prev,dim=-1).mean()),"abs_lin_vel_z":float(data.root_lin_vel_b[:,2].abs().mean())})
           done|=(te|tr).cpu().numpy();prev=a;cur=obs_tensor(nxt).cuda()
         n=np.asarray(norm).mean(0);pm={k:float(np.mean([x[k] for x in phys])) for k in phys[0]}
         rows.append({"suite":suite,"policy":lab,"normalized_objective_mean":n.tolist(),"scalarized":{e:float(n@PREFS[e]) for e in ORDER},"physical":pm,"survival":float(1-done.mean())})
       ow={};pw={};sw={};pk={"A":"ang_vel_xy","O":"tilt_deg","S":"action_rate"}
       for i,lab in enumerate(ORDER):
        a=[];b=[];c=[]
        for suite in range(4):
         rr=[r for r in rows if r["suite"]==suite]
         a.append(max(rr,key=lambda r:r["normalized_objective_mean"][i])["policy"]==lab)
         b.append((min(rr,key=lambda r:r["physical"]["vx_error"]+r["physical"]["wz_error"])["policy"]==lab) if lab=="T" else (min(rr,key=lambda r:r["physical"][pk[lab]])["policy"]==lab))
         c.append(max(rr,key=lambda r:r["scalarized"][lab])["policy"]==lab)
        ow[lab]=float(np.mean(a));pw[lab]=float(np.mean(b));sw[lab]=float(np.mean(c))
       vb={p:float(np.mean([r["physical"]["abs_lin_vel_z"] for r in rows if r["policy"]==p])) for p in ORDER};vr={p:vb[p]/(vb["T"]+1e-12) for p in ORDER};mins=min(r["survival"] for r in rows)
       out[tag]={"rows":rows,"gates":{"objective_winner_fraction":ow,"physical_winner_fraction":pw,"diagonal_scalarized_winner_fraction":sw,"min_survival":mins,"vertical_ratio_to_tracking_policy":vr,"safety_pass":bool(mins>=.95 and max(vr.values())<=2)}}
      p=ROOT/"runs/post_v2_t5_c10_h32_mc-2026-09-23/endpoint_compare.json";p.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps({k:v["gates"] for k,v in out.items()},indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c10_exact_h32_target_gap():
    """Run former post_v2_t5_c10_exact_h32_target_gap.py stage."""
    from pathlib import Path
    import sys,json,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUNS={"control":ROOT/"runs/post_v2_t5_c10_h32_control-2026-09-23","mc":ROOT/"runs/post_v2_t5_c10_h32_mc-2026-09-23"}
    ORDER=("T","A","O","S");SNAPS=(10,25);G=.99;H=32
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    def obs_tensor(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def mc(R,D):
     out=np.zeros_like(R);run=np.zeros_like(R[0])
     for t in range(len(R)-1,-1,-1):run=R[t]+G*run*(~D[t])[:,None];out[t]=run
     return out
    def trunc32(R,D):
     out=np.zeros_like(R)
     for st in (0,32):
      run=np.zeros_like(R[0])
      for t in range(st+31,st-1,-1):
       run=R[t]+G*run*(~D[t])[:,None];out[t]=run
     return out
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic,vector_gae
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=32;cfg.seed=0;env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
      o,_=env.reset(seed=0);o=obs_tensor(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
      out={}
      for tag,run in RUNS.items():
       out[tag]={}
       for bi,lab in enumerate(ORDER):
        w=torch.tensor(PREFS[lab],device="cuda").repeat(32,1);out[tag][lab]={}
        for snap in SNAPS:
         m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(run/f"{lab}_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
         cur,_=env.reset(seed=930000+bi*1000+snap);cur=obs_tensor(cur).cuda();R=[];D=[];V=[];NV=[]
         with torch.no_grad():
          for _ in range(64):
           V.append(m.value_with_preference(cur,w).cpu().numpy());a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a);nxt=obs_tensor(nxt).cuda()
           raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);R.append(normalized_objective_vector(terms(raw,names),shape=(32,))*env.unwrapped.step_dt);D.append((te|tr).cpu().numpy());NV.append(m.value_with_preference(nxt,w).cpu().numpy());cur=nxt
         R=np.asarray(R);D=np.asarray(D,bool);V=np.asarray(V);NV=np.asarray(NV);MC=mc(R,D)
         if tag=="mc":
          T=trunc32(R,D)
         else:
          T=np.zeros_like(R)
          for st in (0,32):
           en=st+32
           rt=torch.tensor(R[st:en],dtype=torch.float32,device="cuda");vt=torch.tensor(V[st:en],dtype=torch.float32,device="cuda");dt=torch.tensor(D[st:en],device="cuda");nxt=torch.tensor(NV[en-1],dtype=torch.float32,device="cuda")
           _,ret=vector_gae(rt,vt,nxt,dt,lam=.95);T[st:en]=ret.cpu().numpy()
         out[tag][lab][str(snap)]={"target_mc_mae":[float(np.mean(np.abs(T[:,:,j]-MC[:,:,j]))) for j in range(4)],"target_mc_bias":[float(np.mean(T[:,:,j]-MC[:,:,j])) for j in range(4)]}
      p=ROOT/"runs/post_v2_t5_c10_h32_mc-2026-09-23/exact_h32_target_gap.json";p.write_text(json.dumps(out,indent=2)+"\n")
      for s in SNAPS:
       for tag in RUNS:
        vals=[]
        for lab in ORDER:vals+=out[tag][lab][str(s)]["target_mc_mae"]
        print(s,tag,float(np.mean(vals)))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c10_value_compare():
    """Run former post_v2_t5_c10_value_compare.py stage."""
    from pathlib import Path
    import sys,json,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUNS={
     "control":ROOT/"runs/post_v2_t5_c10_h32_control-2026-09-23",
     "mc":ROOT/"runs/post_v2_t5_c10_h32_mc-2026-09-23",
    }
    ORDER=("T","A","O","S");SNAPS=(0,10,25);PERT=(10,25);G=.99
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    IDX={"T":0,"A":1,"O":2,"S":3};H=(1,2,4,8,16,32)
    def obs_tensor(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def ev(y,p):
     y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
     return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def flat(gs,ps):return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,ps)])
    def setflat(ps,base,delta):
     o=0
     with torch.no_grad():
      for p,b in zip(ps,base):
       n=p.numel();p.copy_(b+delta[o:o+n].view_as(p));o+=n
    def cloneps(ps):return [p.detach().clone() for p in ps]
    def tilt(q):
     _,x,y,_=[q[:,i] for i in range(4)]
     return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def mc64(R,D):
     out=np.zeros_like(R);run=np.zeros_like(R[0])
     for t in range(len(R)-1,-1,-1):run=R[t]+G*run*(~D[t])[:,None];out[t]=run
     return out
    def rollout_metric(env,m,w,seed):
     cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda();robot=env.unwrapped.scene["robot"];prev=torch.zeros((len(cur),env.unwrapped.action_manager.total_action_dim),device="cuda")
     rows=[];done=np.zeros(len(cur),bool)
     with torch.no_grad():
      for _ in range(32):
       a=m.act_inference_with_preference(cur,w);nxt,_,term,trunc,_=env.step(a);data=robot.data
       rows.append({"ang_vel_xy":float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),"tilt_deg":float(tilt(data.root_quat_w).mean()),"action_rate":float(torch.linalg.vector_norm(a-prev,dim=-1).mean())})
       done|=(term|trunc).cpu().numpy();prev=a;cur=obs_tensor(nxt).cuda()
     return {str(h):{k:float(np.mean([r[k] for r in rows[:h]])) for k in rows[0]} for h in H},float(1-done.mean())
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic,vector_gae
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=32;cfg.seed=0;env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
      obs,_=env.reset(seed=0);obs=obs_tensor(obs).cuda();od=obs.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
      out={"critic":{},"fit_gap":{},"perturbations":{}}
      for tag,run in RUNS.items():
       out["critic"][tag]={};out["fit_gap"][tag]={};out["perturbations"][tag]=[]
       for bi,lab in enumerate(ORDER):
        w=torch.tensor(PREFS[lab],device="cuda").repeat(32,1);out["critic"][tag][lab]={}
        for snap in SNAPS:
         m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(run/f"{lab}_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
         cur,_=env.reset(seed=900000+bi*1000+snap);cur=obs_tensor(cur).cuda();O=[];R=[];D=[];V=[]
         with torch.no_grad():
          for _ in range(64):
           O.append(cur.cpu());V.append(m.value_with_preference(cur,w).cpu().numpy());a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a)
           raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);R.append(normalized_objective_vector(terms(raw,names),shape=(32,))*env.unwrapped.step_dt);D.append((te|tr).cpu().numpy());cur=obs_tensor(nxt).cuda()
         O=torch.stack(O);R=np.asarray(R);D=np.asarray(D,bool);V=np.asarray(V);MC=mc64(R,D)
         out["critic"][tag][lab][str(snap)]={"ev":[ev(MC[:,:,j],V[:,:,j]) for j in range(4)],"bias":[float(np.mean(V[:,:,j]-MC[:,:,j])) for j in range(4)]}
         if snap==25:
          X=O.reshape(-1,od).cuda();W=torch.tensor(PREFS[lab],device="cuda").repeat(len(X),1);Y=torch.tensor(MC.reshape(-1,4),device="cuda",dtype=torch.float32)
          with torch.no_grad():
           feats=m.critic_body(m._with_w(X,W));A=torch.cat([feats,torch.ones((len(feats),1),device="cuda")],1);sol=torch.linalg.lstsq(A,Y).solution;P=A@sol
          out["fit_gap"][tag][lab]={"online_ev":[ev(Y[:,j].cpu(),torch.tensor(V.reshape(-1,4)[:,j])) for j in range(4)],"frozen_body_best_linear_ev":[ev(Y[:,j].cpu(),P[:,j].cpu()) for j in range(4)]}
        if lab in ("A","O"):
         j=IDX[lab]
         for snap in PERT:
          m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(run/f"{lab}_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"])
          cur,_=env.reset(seed=910000+bi*1000+snap);cur=obs_tensor(cur).cuda();ob=[];pre=[];old=[];rw=[];val=[];dn=[]
          for _ in range(32):
           with torch.no_grad():a,lp,u=m.act_with_preference_latent(cur,w);v=m.value_with_preference(cur,w)
           nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);vec=normalized_objective_vector(terms(raw,names),shape=(32,))
           ob.append(cur);pre.append(u);old.append(lp);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt);val.append(v);dn.append((te|tr).cuda());cur=obs_tensor(nxt).cuda()
          with torch.no_grad():nv=m.value_with_preference(cur,w)
          rt=torch.stack(rw);vt=torch.stack(val);dt=torch.stack(dn).bool();adv,_=vector_gae(rt,vt,nv,dt,lam=.95)
          fo=torch.cat(ob);fu=torch.cat(pre);fold=torch.cat(old);fw=w.repeat(32,1);ratio=torch.exp(m.logp_from_pre_tanh_with_preference(fo,fw,fu)-fold.detach())
          aps=[p for n,p in m.named_parameters() if n.startswith("actor_")];loss=-(ratio*adv.reshape(-1,4)[:,j].detach()).mean();gg=flat(torch.autograd.grad(loss,aps,allow_unused=True),aps).detach()
          basevec=torch.cat([p.detach().reshape(-1) for p in aps]);delta=-gg/(gg.norm()+1e-12)*(1e-4*(basevec.norm()+1e-12))
          pm=T4SharedActorCritic(od,ad).cuda();pm.load_state_dict(torch.load(run/f"{lab}_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"]);paps=[p for n,p in pm.named_parameters() if n.startswith("actor_")];setflat(paps,cloneps(paps),delta);pm.eval();m.eval()
          metric="ang_vel_xy" if lab=="A" else "tilt_deg";suites=[]
          for ss in range(2):
           seed=920000+bi*1000+snap*10+ss;b,_=rollout_metric(env,m,w,seed);p,_=rollout_metric(env,pm,w,seed);suites.append({"suite":ss,"baseline":b,"perturbed":p,"metric":metric})
          out["perturbations"][tag].append({"branch":lab,"snapshot":snap,"suites":suites})
      p=ROOT/"runs/post_v2_t5_c10_h32_mc-2026-09-23/value_compare.json";p.write_text(json.dumps(out,indent=2)+"\n")
      print(json.dumps({"critic":out["critic"],"fit_gap":out["fit_gap"],"pert_n":{k:len(v) for k,v in out["perturbations"].items()}},indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c11_critic_loss_geometry_audit():
    """Run former post_v2_t5_c11_critic_loss_geometry_audit.py stage."""
    from pathlib import Path
    import sys,json,copy,hashlib,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c10_h32_mc-2026-09-23"
    OUT=ROOT/"runs/post_v2_t5_c11_loss_geometry-2026-09-23"
    ORDER=("T","A","O","S");SNAPS=(10,25);G=.99;EPS=1e-6
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    def obs_tensor(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def trunc_mc(R,D):
     out=torch.zeros_like(R);run=torch.zeros_like(R[-1])
     for t in range(len(R)-1,-1,-1):
      run=R[t]+G*run*(~D[t]).unsqueeze(-1);out[t]=run
     return out
    def ev(y,p):
     y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
     return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def flat_grads(gs,ps):return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,ps)])
    def pvec(ps):return torch.cat([p.detach().reshape(-1) for p in ps])
    def cos(a,b):return float((a@b)/(a.norm()*b.norm()+1e-12))
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic
      from talon_rl.rewards.objectives import normalized_objective_vector
      OUT.mkdir(parents=True,exist_ok=True)
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=32;cfg.seed=0;env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
      o,_=env.reset(seed=0);o=obs_tensor(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
      report={"schema":"t5_c11_critic_loss_geometry_audit_v1","run_source":str(RUN),"snapshots":SNAPS,"specialists":{}}
      for bi,lab in enumerate(ORDER):
       report["specialists"][lab]={}
       w=torch.tensor(PREFS[lab],device="cuda").repeat(32,1)
       for snap in SNAPS:
        cp=RUN/f"{lab}_snap_{snap}.pt"
        model=T4SharedActorCritic(od,ad).cuda();model.load_state_dict(torch.load(cp,map_location="cuda",weights_only=False)["model"]);model.eval()
        cur,_=env.reset(seed=950000+bi*1000+snap);cur=obs_tensor(cur).cuda();OBS=[];RW=[];DN=[]
        with torch.no_grad():
         for _ in range(32):
          OBS.append(cur);a=model.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a)
          raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);vec=normalized_objective_vector(terms(raw,names),shape=(32,))
          RW.append(torch.tensor(vec,dtype=torch.float32,device="cuda")*env.unwrapped.step_dt);DN.append((te|tr).cuda());cur=obs_tensor(nxt).cuda()
        X=torch.cat(OBS);W=w.repeat(32,1);R=torch.stack(RW);D=torch.stack(DN).bool();Y=trunc_mc(R,D).reshape(-1,4).detach()
        with torch.no_grad():P=model.value_with_preference(X,W)
        mu=Y.mean(0);std=Y.std(0,unbiased=False).clamp_min(EPS);var=std.pow(2);rng=Y.max(0).values-Y.min(0).values
        err=(P-Y).pow(2);head_loss=err.mean(0);raw_share=head_loss/head_loss.sum()
        critic_params=[p for n,p in model.named_parameters() if n.startswith("critic_")]
        body_params=[p for n,p in model.named_parameters() if n.startswith("critic_body")]
        head_params=[p for n,p in model.named_parameters() if n.startswith("critic_head")]
        per_head_grad=[]
        for j in range(4):
         gs=torch.autograd.grad((model.value_with_preference(X,W)[:,j]-Y[:,j]).pow(2).mean(),critic_params,retain_graph=False,allow_unused=True)
         gb=torch.autograd.grad((model.value_with_preference(X,W)[:,j]-Y[:,j]).pow(2).mean(),body_params,retain_graph=False,allow_unused=True)
         gh=torch.autograd.grad((model.value_with_preference(X,W)[:,j]-Y[:,j]).pow(2).mean(),head_params,retain_graph=False,allow_unused=True)
         per_head_grad.append({"all":float(flat_grads(gs,critic_params).norm()),"body":float(flat_grads(gb,body_params).norm()),"head":float(flat_grads(gh,head_params).norm())})
        modes={}
        updates={}
        for mode in ("raw","variance_norm","standardized"):
         m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(cp,map_location="cuda",weights_only=False)["model"]);m.train()
         ps=[p for n,p in m.named_parameters() if n.startswith("critic_")]
         before=pvec(ps).clone()
         opt=torch.optim.Adam(ps,lr=1e-4)
         pred=m.value_with_preference(X,W)
         if mode=="raw":
          per=(pred-Y).pow(2).mean(0);loss=per.mean()
         elif mode=="variance_norm":
          per=((pred-Y).pow(2)/(var+EPS)).mean(0);loss=per.mean()
         else:
          # algebraically equivalent to variance-normalized regression when both pred/target share affine transform
          per=(((pred-mu)/(std+EPS)-(Y-mu)/(std+EPS)).pow(2)).mean(0);loss=per.mean()
         opt.zero_grad(set_to_none=True);loss.backward();opt.step()
         after=pvec(ps).clone();updates[mode]=after-before
         step_by_name={}
         with torch.no_grad():
          for n,p in m.named_parameters():
           if n.startswith("critic_"):
            p0=torch.load(cp,map_location="cuda",weights_only=False)["model"][n].to(p.device)
            step_by_name[n]=float((p-p0).norm())
         with torch.no_grad():
          pred2=m.value_with_preference(X,W);pre=((P-Y).pow(2).mean(0)).cpu().numpy();post=((pred2-Y).pow(2).mean(0)).cpu().numpy()
          ev0=[ev(Y[:,j].cpu(),P[:,j].cpu()) for j in range(4)];ev1=[ev(Y[:,j].cpu(),pred2[:,j].cpu()) for j in range(4)]
         modes[mode]={
          "optimization_loss":float(loss.detach()),"optimization_per_head":per.detach().cpu().tolist(),
          "parameter_step_norm":float((after-before).norm()),"relative_step_norm":float((after-before).norm()/(before.norm()+1e-12)),
          "step_by_parameter":step_by_name,
          "raw_mse_pre":pre.tolist(),"raw_mse_post":post.tolist(),
          "raw_mse_reduction_fraction":((pre-post)/(pre+1e-12)).tolist(),
          "ev_pre":ev0,"ev_post":ev1,"ev_delta":(np.array(ev1)-np.array(ev0)).tolist(),
          "heads_mse_improved":int(np.sum(post<pre)),"heads_ev_improved":int(np.sum(np.array(ev1)>np.array(ev0)))
         }
        modes["variance_norm"]["update_cosine_vs_raw"]=cos(updates["variance_norm"],updates["raw"])
        modes["standardized"]["update_cosine_vs_variance_norm"]=cos(updates["standardized"],updates["variance_norm"])
        # fixed-batch 10-step convergence for the two genuinely distinct objectives
        for mode in ("raw","variance_norm"):
         m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(cp,map_location="cuda",weights_only=False)["model"]);m.train()
         ps=[p for n,p in m.named_parameters() if n.startswith("critic_")];opt=torch.optim.Adam(ps,lr=1e-4);trace=[]
         for kk in range(10):
          pred=m.value_with_preference(X,W)
          if mode=="raw":loss=(pred-Y).pow(2).mean()
          else:loss=((pred-Y).pow(2)/(var+EPS)).mean()
          opt.zero_grad(set_to_none=True);loss.backward();opt.step()
          with torch.no_grad():
           pred2=m.value_with_preference(X,W);mse=((pred2-Y).pow(2).mean(0)).cpu().numpy();evs=[ev(Y[:,j].cpu(),pred2[:,j].cpu()) for j in range(4)]
          trace.append({"step":kk+1,"raw_mse":mse.tolist(),"ev":evs})
         modes[mode]["fixed_batch_10step_trace"]=trace
        report["specialists"][lab][str(snap)]={
         "target_stats":{"mean":mu.cpu().tolist(),"std":std.cpu().tolist(),"variance":var.cpu().tolist(),"range":rng.cpu().tolist(),
                         "nonzero_fraction":[float((Y[:,j].abs()>1e-12).float().mean()) for j in range(4)]},
         "raw_loss":{"per_head":head_loss.cpu().tolist(),"share":raw_share.cpu().tolist(),"total_mean":float(head_loss.mean())},
         "per_head_gradient_norm":per_head_grad,
         "counterfactual":modes
        }
      # aggregate
      agg={}
      for snap in map(str,SNAPS):
       agg[snap]={}
       for mode in ("raw","variance_norm","standardized"):
        mse=[];evd=[];hmi=[];hei=[];steps=[]
        for lab in ORDER:
         x=report["specialists"][lab][snap]["counterfactual"][mode]
         mse+=x["raw_mse_reduction_fraction"];evd+=x["ev_delta"];hmi.append(x["heads_mse_improved"]);hei.append(x["heads_ev_improved"]);steps.append(x["relative_step_norm"])
        agg[snap][mode]={"mean_mse_reduction_fraction":float(np.mean(mse)),"mean_ev_delta":float(np.mean(evd)),
                         "head_mse_improve_fraction":float(np.sum(hmi)/(4*len(ORDER))),"head_ev_improve_fraction":float(np.sum(hei)/(4*len(ORDER))),
                         "mean_relative_step_norm":float(np.mean(steps))}
       cs=[];eq=[]
       for lab in ORDER:
        x=report["specialists"][lab][snap]["counterfactual"]
        cs.append(x["variance_norm"]["update_cosine_vs_raw"]);eq.append(x["standardized"]["update_cosine_vs_variance_norm"])
       agg[snap]["update_geometry"]={"variance_norm_vs_raw_cosine_mean":float(np.mean(cs)),"standardized_vs_variance_norm_cosine_mean":float(np.mean(eq))}
      report["aggregate"]=agg
      out=OUT/"audit.json";out.write_text(json.dumps(report,indent=2)+"\n")
      print(json.dumps({"aggregate":agg},indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c12_moving_batch_audit():
    """Run former post_v2_t5_c12_moving_batch_audit.py stage."""
    from pathlib import Path
    import sys,json,hashlib,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c10_h32_mc-2026-09-23"
    OUT=ROOT/"runs/post_v2_t5_c12_moving_batch-2026-09-23"
    ORDER=("T","A","O","S");NB=6;H=32;G=.99
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    def obs_tensor(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def trunc_mc(R,D):
        out=torch.zeros_like(R);run=torch.zeros_like(R[-1])
        for t in range(len(R)-1,-1,-1):
            run=R[t]+G*run*(~D[t]).unsqueeze(-1);out[t]=run
        return out
    def ev(y,p):
        y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
        return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def flat(gs,ps):return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,ps)])
    def cos(a,b):return float((a@b)/(a.norm()*b.norm()+1e-12))
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def solve_head(model,X,W,Y):
        with torch.no_grad():
            F=model.critic_body(model._with_w(X,W));A=torch.cat([F,torch.ones((len(F),1),device=F.device)],1)
            sol=torch.linalg.lstsq(A,Y).solution
        return sol[:-1,:].T.contiguous(),sol[-1,:].contiguous()
    def set_head(model,W,b):
        with torch.no_grad():model.critic_head.weight.copy_(W);model.critic_head.bias.copy_(b)
    def eval_mse_ev(model,X,W,Y):
        with torch.no_grad():P=model.value_with_preference(X,W);m=((P-Y)**2).mean(0).cpu().numpy();e=[ev(Y[:,j].cpu(),P[:,j].cpu()) for j in range(4)]
        return m,e
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic
      from talon_rl.rewards.objectives import normalized_objective_vector
      OUT.mkdir(parents=True,exist_ok=True)
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=32;cfg.seed=0;env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
      o,_=env.reset(seed=0);o=obs_tensor(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
      report={"schema":"t5_c12_moving_batch_v1","checkpoint_snapshot":25,"batch_horizon":H,"num_batches":NB,"specialists":{}}
      for bi,lab in enumerate(ORDER):
        cp=RUN/f"{lab}_snap_25.pt";base=T4SharedActorCritic(od,ad).cuda();base.load_state_dict(torch.load(cp,map_location="cuda",weights_only=False)["model"]);base.eval()
        w=torch.tensor(PREFS[lab],device="cuda").repeat(32,1)
        # one continuous reset seed, then consecutive batches with no manual resets between batches
        cur,_=env.reset(seed=970000+bi*1000);cur=obs_tensor(cur).cuda()
        batches=[]
        for k in range(NB):
          obs=[];rw=[];dn=[]
          with torch.no_grad():
            for _ in range(H):
              obs.append(cur);a=base.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a)
              raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);vec=normalized_objective_vector(terms(raw,names),shape=(32,))
              rw.append(torch.tensor(vec,dtype=torch.float32,device="cuda")*env.unwrapped.step_dt);dn.append((te|tr).cuda());cur=obs_tensor(nxt).cuda()
          X=torch.cat(obs);W=w.repeat(H,1);R=torch.stack(rw);D=torch.stack(dn).bool();Y=trunc_mc(R,D).reshape(-1,4).detach()
          batches.append((X,W,Y))
        # gradients + stats + optimal heads
        critic_params=[p for n,p in base.named_parameters() if n.startswith("critic_")]
        grads=[];stats=[];heads=[]
        for k,(X,W,Y) in enumerate(batches):
          pred=base.value_with_preference(X,W);loss=((pred-Y)**2).mean()
          gs=torch.autograd.grad(loss,critic_params,retain_graph=False,allow_unused=True);g=flat(gs,critic_params).detach();grads.append(g)
          Wh,bh=solve_head(base,X,W,Y);heads.append((Wh,bh))
          obs_np=X.detach().cpu().numpy();stats.append({
            "batch":k,
            "target_mean":Y.mean(0).cpu().tolist(),"target_std":Y.std(0,unbiased=False).cpu().tolist(),
            "obs_mean":obs_np.mean(0).tolist(),"obs_std":obs_np.std(0).tolist(),
            "grad_norm":float(g.norm())
          })
        pairs=[]
        for k in range(NB-1):
          Wh,bh=heads[k];Wn,bn=heads[k+1]
          hd=float(torch.sqrt((Wh-Wn).pow(2).sum()+(bh-bn).pow(2).sum()))
          pairs.append({
            "pair":f"{k}->{k+1}",
            "gradient_cosine":cos(grads[k],grads[k+1]),
            "gradient_norm_ratio_next_over_current":float(grads[k+1].norm()/(grads[k].norm()+1e-12)),
            "optimal_head_l2_drift":hd,
            "target_mean_shift_l2":float(np.linalg.norm(np.array(stats[k+1]["target_mean"])-np.array(stats[k]["target_mean"]))),
            "target_std_shift_l2":float(np.linalg.norm(np.array(stats[k+1]["target_std"])-np.array(stats[k]["target_std"]))),
            "obs_mean_shift_l2":float(np.linalg.norm(np.array(stats[k+1]["obs_mean"])-np.array(stats[k]["obs_mean"]))),
            "obs_std_shift_l2":float(np.linalg.norm(np.array(stats[k+1]["obs_std"])-np.array(stats[k]["obs_std"])))
          })
        # transfer: fit 10 Adam steps on B_t, evaluate B_t and B_t+1 before/after
        transfer=[]
        for k in range(NB-1):
          Xt,Wt,Yt=batches[k];Xn,Wn,Yn=batches[k+1]
          m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(cp,map_location="cuda",weights_only=False)["model"]);ps=[p for n,p in m.named_parameters() if n.startswith("critic_")]
          before_t=eval_mse_ev(m,Xt,Wt,Yt);before_n=eval_mse_ev(m,Xn,Wn,Yn)
          opt=torch.optim.Adam(ps,lr=1e-4)
          for _ in range(10):
            pred=m.value_with_preference(Xt,Wt);loss=((pred-Yt)**2).mean();opt.zero_grad(set_to_none=True);loss.backward();opt.step()
          after_t=eval_mse_ev(m,Xt,Wt,Yt);after_n=eval_mse_ev(m,Xn,Wn,Yn)
          transfer.append({
            "pair":f"{k}->{k+1}",
            "current_mse_before":before_t[0].tolist(),"current_mse_after":after_t[0].tolist(),
            "next_mse_before":before_n[0].tolist(),"next_mse_after":after_n[0].tolist(),
            "current_ev_before":before_t[1],"current_ev_after":after_t[1],
            "next_ev_before":before_n[1],"next_ev_after":after_n[1],
            "current_mse_change_fraction":((before_t[0]-after_t[0])/(before_t[0]+1e-12)).tolist(),
            "next_mse_change_fraction":((before_n[0]-after_n[0])/(before_n[0]+1e-12)).tolist(),
            "next_ev_delta":(np.array(after_n[1])-np.array(before_n[1])).tolist()
          })
        report["specialists"][lab]={"batch_stats":stats,"pairwise":pairs,"transfer":transfer}
      # aggregate
      pcs=[];hds=[];tmean=[];omean=[];transfer_next=[];transfer_cur=[];transfer_next_evd=[]
      for lab in ORDER:
        for p in report["specialists"][lab]["pairwise"]:
          pcs.append(p["gradient_cosine"]);hds.append(p["optimal_head_l2_drift"]);tmean.append(p["target_mean_shift_l2"]);omean.append(p["obs_mean_shift_l2"])
        for t in report["specialists"][lab]["transfer"]:
          transfer_cur+=t["current_mse_change_fraction"];transfer_next+=t["next_mse_change_fraction"];transfer_next_evd+=t["next_ev_delta"]
      report["aggregate"]={
        "gradient_cosine_mean":float(np.mean(pcs)),"gradient_cosine_median":float(np.median(pcs)),
        "gradient_cosine_negative_fraction":float(np.mean(np.array(pcs)<0)),
        "gradient_cosine_lt_0p25_fraction":float(np.mean(np.array(pcs)<.25)),
        "optimal_head_l2_drift_mean":float(np.mean(hds)),
        "target_mean_shift_l2_mean":float(np.mean(tmean)),
        "obs_mean_shift_l2_mean":float(np.mean(omean)),
        "fit_current_mse_improvement_mean":float(np.mean(transfer_cur)),
        "transfer_next_mse_improvement_mean":float(np.mean(transfer_next)),
        "transfer_next_mse_worsen_fraction":float(np.mean(np.array(transfer_next)<0)),
        "transfer_next_ev_delta_mean":float(np.mean(transfer_next_evd)),
        "transfer_next_ev_worsen_fraction":float(np.mean(np.array(transfer_next_evd)<0))
      }
      (OUT/"audit.json").write_text(json.dumps(report,indent=2)+"\n")
      print(json.dumps(report["aggregate"],indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c12_optimal_head_transfer():
    """Run former post_v2_t5_c12_optimal_head_transfer.py stage."""
    from pathlib import Path
    import sys,json,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c10_h32_mc-2026-09-23";OUT=ROOT/"runs/post_v2_t5_c12_moving_batch-2026-09-23"
    ORDER=("T","A","O","S");NB=6;H=32;G=.99
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    def obs_tensor(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def trunc_mc(R,D):
     out=torch.zeros_like(R);run=torch.zeros_like(R[-1])
     for t in range(len(R)-1,-1,-1):
      run=R[t]+G*run*(~D[t]).unsqueeze(-1);out[t]=run
     return out
    def ev(y,p):
     y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
     return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=32;cfg.seed=0;env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
      o,_=env.reset(seed=0);o=obs_tensor(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
      out={"specialists":{}}
      for bi,lab in enumerate(ORDER):
       m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"{lab}_snap_25.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
       w=torch.tensor(PREFS[lab],device="cuda").repeat(32,1);cur,_=env.reset(seed=980000+bi*1000);cur=obs_tensor(cur).cuda();B=[]
       for k in range(NB):
        obs=[];rw=[];dn=[]
        with torch.no_grad():
         for _ in range(H):
          obs.append(cur);a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
          rw.append(torch.tensor(normalized_objective_vector(terms(raw,names),shape=(32,)),device="cuda",dtype=torch.float32)*env.unwrapped.step_dt);dn.append((te|tr).cuda());cur=obs_tensor(nxt).cuda()
        X=torch.cat(obs);W=w.repeat(H,1);Y=trunc_mc(torch.stack(rw),torch.stack(dn).bool()).reshape(-1,4)
        with torch.no_grad():
          F=m.critic_body(m._with_w(X,W));A=torch.cat([F,torch.ones((len(F),1),device="cuda")],1)
          sol=torch.linalg.lstsq(A,Y).solution;pred=A@sol;s=torch.linalg.svdvals(A)
        B.append({"A":A,"Y":Y,"sol":sol,"self_ev":[ev(Y[:,j].cpu(),pred[:,j].cpu()) for j in range(4)],
                  "cond":float(s.max()/(s.min()+1e-12)),"effective_rank":int((s>1e-6*s.max()).sum())})
       pairs=[]
       for k in range(NB-1):
        curB,nxt=B[k],B[k+1]
        with torch.no_grad():
          p_cur_on_next=nxt["A"]@curB["sol"];p_next_on_next=nxt["A"]@nxt["sol"]
        pairs.append({"pair":f"{k}->{k+1}",
          "cond_current":curB["cond"],"cond_next":nxt["cond"],"rank_current":curB["effective_rank"],"rank_next":nxt["effective_rank"],
          "self_ev_current":curB["self_ev"],"self_ev_next":nxt["self_ev"],
          "current_optimum_on_next_ev":[ev(nxt["Y"][:,j].cpu(),p_cur_on_next[:,j].cpu()) for j in range(4)],
          "next_optimum_on_next_ev":[ev(nxt["Y"][:,j].cpu(),p_next_on_next[:,j].cpu()) for j in range(4)],
          "prediction_disagreement_rmse":[float(torch.sqrt(((p_cur_on_next[:,j]-p_next_on_next[:,j])**2).mean())) for j in range(4)]
        })
       out["specialists"][lab]=pairs
      vals=[];gaps=[];conds=[]
      for lab in ORDER:
       for p in out["specialists"][lab]:
        vals+=p["current_optimum_on_next_ev"];gaps+=(np.array(p["next_optimum_on_next_ev"])-np.array(p["current_optimum_on_next_ev"])).tolist();conds += [p["cond_current"],p["cond_next"]]
      out["aggregate"]={"prior_batch_optimum_on_next_ev_mean":float(np.mean(vals)),"next_batch_optimum_ev_advantage_mean":float(np.mean(gaps)),
                        "prior_optimum_next_ev_negative_fraction":float(np.mean(np.array(vals)<0)),
                        "design_condition_number_median":float(np.median(conds)),"design_condition_number_max":float(np.max(conds))}
      (OUT/"optimal_head_transfer.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out["aggregate"],indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c12_ridge_head_transfer():
    """Run former post_v2_t5_c12_ridge_head_transfer.py stage."""
    from pathlib import Path
    import sys,json,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c10_h32_mc-2026-09-23";OUT=ROOT/"runs/post_v2_t5_c12_moving_batch-2026-09-23"
    ORDER=("T","A","O","S");NB=6;H=32;G=.99;RIDGES=(0.0,1e-6,1e-4,1e-3,1e-2,1e-1)
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    def obs_tensor(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def trunc_mc(R,D):
     out=torch.zeros_like(R);run=torch.zeros_like(R[-1])
     for t in range(len(R)-1,-1,-1):run=R[t]+G*run*(~D[t]).unsqueeze(-1);out[t]=run
     return out
    def ev(y,p):
     y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
     return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def solve(A,Y,l2):
     if l2==0:
      return torch.linalg.lstsq(A,Y).solution
     I=torch.eye(A.shape[1],device=A.device,dtype=A.dtype);I[-1,-1]=0
     return torch.linalg.solve(A.T@A+l2*I,A.T@Y)
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=32;cfg.seed=0;env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
      o,_=env.reset(seed=0);o=obs_tensor(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
      out={"ridge_values":RIDGES,"specialists":{}}
      for bi,lab in enumerate(ORDER):
       m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"{lab}_snap_25.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
       w=torch.tensor(PREFS[lab],device="cuda").repeat(32,1);cur,_=env.reset(seed=990000+bi*1000);cur=obs_tensor(cur).cuda();B=[]
       for k in range(NB):
        obs=[];rw=[];dn=[]
        with torch.no_grad():
         for _ in range(H):
          obs.append(cur);a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
          rw.append(torch.tensor(normalized_objective_vector(terms(raw,names),shape=(32,)),device="cuda",dtype=torch.float32)*env.unwrapped.step_dt);dn.append((te|tr).cuda());cur=obs_tensor(nxt).cuda()
        X=torch.cat(obs);W=w.repeat(H,1);Y=trunc_mc(torch.stack(rw),torch.stack(dn).bool()).reshape(-1,4)
        with torch.no_grad():
         F=m.critic_body(m._with_w(X,W));A=torch.cat([F,torch.ones((len(F),1),device="cuda")],1)
        B.append((A,Y))
       labout={}
       for l2 in RIDGES:
        pairs=[]
        for k in range(NB-1):
         A,Y=B[k];An,Yn=B[k+1];sol=solve(A,Y,l2);soln=solve(An,Yn,l2)
         with torch.no_grad():
          pc=A@sol;pn=An@sol;pnopt=An@soln
         pairs.append({
          "pair":f"{k}->{k+1}",
          "self_ev":[ev(Y[:,j].cpu(),pc[:,j].cpu()) for j in range(4)],
          "prior_on_next_ev":[ev(Yn[:,j].cpu(),pn[:,j].cpu()) for j in range(4)],
          "next_opt_ev":[ev(Yn[:,j].cpu(),pnopt[:,j].cpu()) for j in range(4)],
          "solution_norm":float(sol.norm()),"solution_drift":float((soln-sol).norm())
         })
        labout[str(l2)]=pairs
       out["specialists"][lab]=labout
      agg={}
      for l2 in RIDGES:
       selfe=[];nextv=[];opte=[];norm=[];drift=[]
       for lab in ORDER:
        for p in out["specialists"][lab][str(l2)]:
         selfe+=p["self_ev"];nextv+=p["prior_on_next_ev"];opte+=p["next_opt_ev"];norm.append(p["solution_norm"]);drift.append(p["solution_drift"])
       agg[str(l2)]={"self_ev_mean":float(np.mean(selfe)),"prior_on_next_ev_mean":float(np.mean(nextv)),
                      "prior_on_next_negative_fraction":float(np.mean(np.array(nextv)<0)),
                      "next_opt_ev_mean":float(np.mean(opte)),"solution_norm_mean":float(np.mean(norm)),"solution_drift_mean":float(np.mean(drift))}
      out["aggregate"]=agg;(OUT/"ridge_head_transfer.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(agg,indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c13_collect_features():
    """Run former post_v2_t5_c13_collect_features.py stage."""
    from pathlib import Path
    import sys,json,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c10_h32_mc-2026-09-23";OUT=ROOT/"runs/post_v2_t5_c13_conditioning-2026-09-23"
    ORDER=("T","A","O","S");NB=6;H=32;G=.99
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    def obs_tensor(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def trunc_mc(R,D):
     out=torch.zeros_like(R);run=torch.zeros_like(R[-1])
     for t in range(len(R)-1,-1,-1):run=R[t]+G*run*(~D[t]).unsqueeze(-1);out[t]=run
     return out
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic
      from talon_rl.rewards.objectives import normalized_objective_vector
      OUT.mkdir(parents=True,exist_ok=True)
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=32;cfg.seed=0;env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
      o,_=env.reset(seed=0);o=obs_tensor(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
      for bi,lab in enumerate(ORDER):
       m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"{lab}_snap_25.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
       w=torch.tensor(PREFS[lab],device="cuda").repeat(32,1);cur,_=env.reset(seed=996000+bi*1000);cur=obs_tensor(cur).cuda()
       data={}
       for k in range(NB):
        obs=[];rw=[];dn=[]
        with torch.no_grad():
         for _ in range(H):
          obs.append(cur);a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a)
          raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);vec=normalized_objective_vector(terms(raw,names),shape=(32,))
          rw.append(torch.tensor(vec,dtype=torch.float32,device="cuda")*env.unwrapped.step_dt);dn.append((te|tr).cuda());cur=obs_tensor(nxt).cuda()
        X=torch.cat(obs);Wp=w.repeat(H,1);Y=trunc_mc(torch.stack(rw),torch.stack(dn).bool()).reshape(-1,4)
        with torch.no_grad():F=m.critic_body(m._with_w(X,Wp))
        data[f"F{k}"]=F.cpu().numpy();data[f"Y{k}"]=Y.cpu().numpy()
       data["head_weight"]=m.critic_head.weight.detach().cpu().numpy();data["head_bias"]=m.critic_head.bias.detach().cpu().numpy()
       np.savez_compressed(OUT/f"{lab}_features_targets.npz",**data)
       print(lab,"saved",data["F0"].shape,data["Y0"].shape,flush=True)
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c13_conditioning_regularization_audit():
    """Run former post_v2_t5_c13_conditioning_regularization_audit.py stage."""
    from pathlib import Path
    import sys,json,hashlib,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c10_h32_mc-2026-09-23"
    OUT=ROOT/"runs/post_v2_t5_c13_conditioning-2026-09-23"
    ORDER=("T","A","O","S"); NB=6; H=32; G=.99
    RIDGES=(0.0,1e-4,1e-3,1e-2,1e-1,1.0)
    SEQ_L2=(0.0,1e-4,1e-3,1e-2,1e-1)
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    def obs_tensor(x):
        if isinstance(x,dict): x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names): return {n:raw[:,i] for i,n in enumerate(names)}
    def trunc_mc(R,D):
        out=torch.zeros_like(R); run=torch.zeros_like(R[-1])
        for t in range(len(R)-1,-1,-1):
            run=R[t]+G*run*(~D[t]).unsqueeze(-1); out[t]=run
        return out
    def ev(y,p):
        y=np.asarray(y,float).reshape(-1); p=np.asarray(p,float).reshape(-1)
        return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def solve_ridge(A,Y,l2):
        if l2==0: return torch.linalg.lstsq(A,Y).solution
        I=torch.eye(A.shape[1],device=A.device,dtype=A.dtype); I[-1,-1]=0
        return torch.linalg.solve(A.T@A+l2*I,A.T@Y)
    def metrics(Y,P):
        return {"ev":[ev(Y[:,j].cpu(),P[:,j].cpu()) for j in range(4)],
                "mse":[float(((P[:,j]-Y[:,j])**2).mean()) for j in range(4)]}
    def make_whitener(F, rel_floor=1e-4):
        mu=F.mean(0,keepdim=True); X=F-mu
        C=(X.T@X)/max(1,len(F)-1)
        evals,evecs=torch.linalg.eigh(C)
        mx=evals.max().clamp_min(1e-12); floor=rel_floor*mx
        inv=torch.rsqrt(torch.clamp(evals,min=floor))
        W=evecs@torch.diag(inv)@evecs.T
        return mu,W,evals
    def apply_whiten(F,mu,W):
        return (F-mu)@W
    def head_predict(F,W,b): return F@W.T+b
    def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def main():
        from isaaclab.app import AppLauncher
        app=AppLauncher({"headless":True,"enable_cameras":False}).app; env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            from talon_rl.models.foundations.four_objective import T4SharedActorCritic
            from talon_rl.rewards.objectives import normalized_objective_vector
            OUT.mkdir(parents=True,exist_ok=True)
            cfg=UnitreeA1FlatEnvCfg(); cfg.scene.num_envs=32; cfg.seed=0
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            o,_=env.reset(seed=0); o=obs_tensor(o).cuda(); od=o.shape[-1]; ad=env.unwrapped.action_manager.total_action_dim; mgr=env.unwrapped.reward_manager
            report={"schema":"t5_c13_conditioning_regularization_v1","source_checkpoint":"C10 H32 MC u25","specialists":{}}
            for bi,lab in enumerate(ORDER):
                cp=RUN/f"{lab}_snap_25.pt"
                m=T4SharedActorCritic(od,ad).cuda(); m.load_state_dict(torch.load(cp,map_location="cuda",weights_only=False)["model"]); m.eval()
                w=torch.tensor(PREFS[lab],device="cuda").repeat(32,1)
                cur,_=env.reset(seed=995000+bi*1000); cur=obs_tensor(cur).cuda()
                batches=[]
                for k in range(NB):
                    obs=[]; rw=[]; dn=[]
                    with torch.no_grad():
                        for _ in range(H):
                            obs.append(cur)
                            a=m.act_inference_with_preference(cur,w)
                            nxt,_,te,tr,_=env.step(a)
                            raw=mgr._step_reward.detach().cpu().numpy(); names=list(mgr.active_terms)
                            vec=normalized_objective_vector(terms(raw,names),shape=(32,))
                            rw.append(torch.tensor(vec,dtype=torch.float32,device="cuda")*env.unwrapped.step_dt)
                            dn.append((te|tr).cuda()); cur=obs_tensor(nxt).cuda()
                    X=torch.cat(obs); Wpref=w.repeat(H,1); Y=trunc_mc(torch.stack(rw),torch.stack(dn).bool()).reshape(-1,4)
                    with torch.no_grad(): F=m.critic_body(m._with_w(X,Wpref))
                    batches.append({"Y":Y.cpu(),"F":F.cpu()})
                # 1) singular spectrum and target projection onto left singular directions
                spectra=[]
                for k,b in enumerate(batches):
                    Fc=b["F"]-b["F"].mean(0,keepdim=True)
                    U,S,Vh=torch.linalg.svd(Fc,full_matrices=False)
                    Yc=b["Y"]-b["Y"].mean(0,keepdim=True)
                    coeff=U.T@Yc
                    energy=(coeff**2).sum(1); total=energy.sum().clamp_min(1e-12)
                    small=S < (1e-2*S.max())
                    spectra.append({
                        "batch":k,"singular_values":S.cpu().tolist(),
                        "condition_number":float(S.max()/(S.min()+1e-12)),
                        "effective_rank_1e-3":int((S>1e-3*S.max()).sum()),
                        "effective_rank_1e-2":int((S>1e-2*S.max()).sum()),
                        "target_energy_fraction_small_sv_lt_1e-2":[float((coeff[small,j]**2).sum()/((coeff[:,j]**2).sum()+1e-12)) for j in range(4)],
                        "total_target_energy_fraction_small_sv_lt_1e-2":float(energy[small].sum()/total)
                    })
                # 2) ridge sweep: self and next-batch transfer
                ridge={}
                for l2 in RIDGES:
                    rows=[]
                    for k in range(NB-1):
                        b,n=batches[k],batches[k+1]
                        A=torch.cat([b["F"],torch.ones((len(b["F"]),1),device=b["F"].device)],1)
                        An=torch.cat([n["F"],torch.ones((len(n["F"]),1),device=b["F"].device)],1)
                        sol=solve_ridge(A,b["Y"],l2); soln=solve_ridge(An,n["Y"],l2)
                        Ps=A@sol; Pn=An@sol; Pnopt=An@soln
                        rows.append({
                            "pair":f"{k}->{k+1}",
                            "self":metrics(b["Y"],Ps),
                            "prior_on_next":metrics(n["Y"],Pn),
                            "next_opt":metrics(n["Y"],Pnopt),
                            "solution_norm":float(sol[:-1].norm()),
                            "solution_drift":float((soln-sol).norm())
                        })
                    ridge[str(l2)]=rows
                # 3) sequential head-only Adam with explicit L2 on weights only
                seq={}
                baseW=m.critic_head.weight.detach().cpu().clone(); baseb=m.critic_head.bias.detach().cpu().clone()
                for l2 in SEQ_L2:
                    W=torch.nn.Parameter(baseW.clone()); bpar=torch.nn.Parameter(baseb.clone())
                    opt=torch.optim.Adam([W,bpar],lr=1e-4)
                    trace=[]
                    for k,b in enumerate(batches):
                        # before current
                        Pb=head_predict(b["F"],W,bpar); before=metrics(b["Y"],Pb)
                        # exactly one online-like head update on current batch
                        pred=head_predict(b["F"],W,bpar)
                        loss=((pred-b["Y"])**2).mean()+l2*(W**2).mean()
                        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                        Pa=head_predict(b["F"],W,bpar); after=metrics(b["Y"],Pa)
                        nextm=None
                        if k+1<NB:
                            n=batches[k+1]; Pnext=head_predict(n["F"],W,bpar); nextm=metrics(n["Y"],Pnext)
                        trace.append({"batch":k,"loss":float(loss.detach()),"current_before":before,"current_after":after,"next_after":nextm,
                                      "weight_norm":float(W.norm()),"bias_norm":float(bpar.norm())})
                    # final generalization across all batches
                    all_ev=[]; all_mse=[]
                    for bb in batches:
                        mm=metrics(bb["Y"],head_predict(bb["F"],W,bpar)); all_ev+=mm["ev"]; all_mse+=mm["mse"]
                    seq[str(l2)]={"trace":trace,"final_all_batches_ev_mean":float(np.mean(all_ev)),"final_all_batches_mse_mean":float(np.mean(all_mse)),
                                  "final_weight_norm":float(W.norm())}
                # 4) whitening / PCA-truncated whitening counterfactual
                white={}
                for rel_floor in (1e-6,1e-4,1e-3,1e-2):
                    rows=[]
                    for k in range(NB-1):
                        b,n=batches[k],batches[k+1]
                        mu,Wht,evals=make_whitener(b["F"],rel_floor)
                        Fb=apply_whiten(b["F"],mu,Wht); Fn=apply_whiten(n["F"],mu,Wht)
                        A=torch.cat([Fb,torch.ones((len(Fb),1),device=b["F"].device)],1)
                        An=torch.cat([Fn,torch.ones((len(Fn),1),device=b["F"].device)],1)
                        sol=torch.linalg.lstsq(A,b["Y"]).solution
                        Pb=A@sol; Pn=An@sol
                        s=torch.linalg.svdvals(Fb-Fb.mean(0,keepdim=True))
                        rows.append({"pair":f"{k}->{k+1}","self":metrics(b["Y"],Pb),"prior_on_next":metrics(n["Y"],Pn),
                                     "whitened_condition_number":float(s.max()/(s.min()+1e-12)),
                                     "cov_eigen_min":float(evals.min()),"cov_eigen_max":float(evals.max())})
                    white[str(rel_floor)]=rows
                report["specialists"][lab]={"spectrum":spectra,"ridge":ridge,"sequential_head_l2":seq,"whitening":white}
            # aggregate
            agg={"ridge":{},"sequential_head_l2":{},"whitening":{}}
            for l2 in RIDGES:
                selfe=[]; nexte=[]; dr=[]; norm=[]
                for lab in ORDER:
                    for row in report["specialists"][lab]["ridge"][str(l2)]:
                        selfe+=row["self"]["ev"]; nexte+=row["prior_on_next"]["ev"]; dr.append(row["solution_drift"]); norm.append(row["solution_norm"])
                agg["ridge"][str(l2)]={"self_ev_mean":float(np.mean(selfe)),"next_ev_mean":float(np.mean(nexte)),
                                       "next_ev_negative_fraction":float(np.mean(np.array(nexte)<0)),
                                       "solution_drift_mean":float(np.mean(dr)),"solution_norm_mean":float(np.mean(norm))}
            for l2 in SEQ_L2:
                cur=[]; nxt=[]; final=[]; norms=[]
                for lab in ORDER:
                    q=report["specialists"][lab]["sequential_head_l2"][str(l2)]
                    final.append(q["final_all_batches_ev_mean"]); norms.append(q["final_weight_norm"])
                    for tr in q["trace"]:
                        cur+=tr["current_after"]["ev"]
                        if tr["next_after"] is not None: nxt+=tr["next_after"]["ev"]
                agg["sequential_head_l2"][str(l2)]={"current_after_ev_mean":float(np.mean(cur)),"next_after_ev_mean":float(np.mean(nxt)),
                                                    "next_after_ev_negative_fraction":float(np.mean(np.array(nxt)<0)),
                                                    "final_all_batches_ev_mean":float(np.mean(final)),"final_weight_norm_mean":float(np.mean(norms))}
            for rf in (1e-6,1e-4,1e-3,1e-2):
                selfe=[]; nexte=[]; cond=[]
                for lab in ORDER:
                    for row in report["specialists"][lab]["whitening"][str(rf)]:
                        selfe+=row["self"]["ev"]; nexte+=row["prior_on_next"]["ev"]; cond.append(row["whitened_condition_number"])
                agg["whitening"][str(rf)]={"self_ev_mean":float(np.mean(selfe)),"next_ev_mean":float(np.mean(nexte)),
                                           "next_ev_negative_fraction":float(np.mean(np.array(nexte)<0)),
                                           "condition_number_median":float(np.median(cond))}
            conds=[]; er=[]; small=[]
            for lab in ORDER:
                for s in report["specialists"][lab]["spectrum"]:
                    conds.append(s["condition_number"]); er.append(s["effective_rank_1e-2"]); small.append(s["total_target_energy_fraction_small_sv_lt_1e-2"])
            agg["spectrum"]={"condition_number_median":float(np.median(conds)),"condition_number_max":float(np.max(conds)),
                             "effective_rank_1e-2_mean":float(np.mean(er)),"small_sv_target_energy_fraction_mean":float(np.mean(small))}
            report["aggregate"]=agg
            (OUT/"audit.json").write_text(json.dumps(report,indent=2)+"\n")
            print(json.dumps(agg,indent=2))
        finally:
            if env is not None: env.close()
            app.close()
    if True: main()

def run_post_v2_t5_c13_offline_analyze():
    """Run former post_v2_t5_c13_offline_analyze.py stage."""
    from pathlib import Path
    import json,hashlib,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4]
    OUT=ROOT/"runs/post_v2_t5_c13_conditioning-2026-09-23"
    ORDER=("T","A","O","S");NB=6
    RIDGES=(0.0,1e-4,1e-3,1e-2,1e-1,1.0)
    SEQ_L2=(0.0,1e-4,1e-3,1e-2,1e-1,1.0)
    def ev(y,p):
        y=np.asarray(y,float).reshape(-1); p=np.asarray(p,float).reshape(-1)
        return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def metrics(Y,P):
        return {"ev":[ev(Y[:,j],P[:,j]) for j in range(4)],"mse":[float(np.mean((P[:,j]-Y[:,j])**2)) for j in range(4)]}
    def solve_ridge(A,Y,l2):
        if l2==0: return np.linalg.lstsq(A,Y,rcond=None)[0]
        I=np.eye(A.shape[1]);I[-1,-1]=0
        return np.linalg.solve(A.T@A+l2*I,A.T@Y)
    def whiten_fit(F,Y,Fn,rel_floor):
        mu=F.mean(0,keepdims=True);X=F-mu
        C=(X.T@X)/max(1,len(F)-1)
        evals,evecs=np.linalg.eigh(C);mx=max(float(evals.max()),1e-12);floor=rel_floor*mx
        inv=1/np.sqrt(np.maximum(evals,floor));W=evecs@np.diag(inv)@evecs.T
        Fb=(F-mu)@W;Fnext=(Fn-mu)@W
        A=np.c_[Fb,np.ones(len(Fb))];An=np.c_[Fnext,np.ones(len(Fnext))]
        sol=np.linalg.lstsq(A,Y,rcond=None)[0]
        s=np.linalg.svd(Fb-Fb.mean(0),compute_uv=False)
        return metrics(Y,A@sol),An@sol,float(s.max()/(s.min()+1e-12))
    def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    report={"schema":"t5_c13_conditioning_regularization_v2","specialists":{}}
    for lab in ORDER:
        z=np.load(OUT/f"{lab}_features_targets.npz")
        batches=[(z[f"F{k}"].astype(np.float64),z[f"Y{k}"].astype(np.float64)) for k in range(NB)]
        baseW=z["head_weight"].astype(np.float32);baseb=z["head_bias"].astype(np.float32)
        spectra=[]
        for k,(F,Y) in enumerate(batches):
            X=F-F.mean(0);U,S,Vh=np.linalg.svd(X,full_matrices=False);Yc=Y-Y.mean(0);coef=U.T@Yc
            small=S<1e-2*S.max()
            spectra.append({"batch":k,"condition_number":float(S.max()/(S.min()+1e-12)),
                            "effective_rank_1e-3":int(np.sum(S>1e-3*S.max())),"effective_rank_1e-2":int(np.sum(S>1e-2*S.max())),
                            "target_energy_fraction_small_sv_lt_1e-2":[float(np.sum(coef[small,j]**2)/(np.sum(coef[:,j]**2)+1e-12)) for j in range(4)],
                            "singular_values":S.tolist()})
        ridge={}
        for l2 in RIDGES:
            rows=[]
            for k in range(NB-1):
                F,Y=batches[k];Fn,Yn=batches[k+1]
                A=np.c_[F,np.ones(len(F))];An=np.c_[Fn,np.ones(len(Fn))]
                sol=solve_ridge(A,Y,l2);soln=solve_ridge(An,Yn,l2)
                rows.append({"pair":f"{k}->{k+1}","self":metrics(Y,A@sol),"prior_on_next":metrics(Yn,An@sol),
                             "next_opt":metrics(Yn,An@soln),"solution_norm":float(np.linalg.norm(sol[:-1])),
                             "solution_drift":float(np.linalg.norm(soln-sol))})
            ridge[str(l2)]=rows
        seq={}
        for l2 in SEQ_L2:
            W=torch.nn.Parameter(torch.tensor(baseW));b=torch.nn.Parameter(torch.tensor(baseb));opt=torch.optim.Adam([W,b],lr=1e-4)
            trace=[]
            for k,(F,Y) in enumerate(batches):
                Ft=torch.tensor(F,dtype=torch.float32);Yt=torch.tensor(Y,dtype=torch.float32)
                with torch.no_grad():pb=Ft@W.T+b
                before=metrics(Y,pb.numpy())
                pred=Ft@W.T+b;loss=((pred-Yt)**2).mean()+float(l2)*(W**2).mean()
                opt.zero_grad();loss.backward();opt.step()
                with torch.no_grad():pa=Ft@W.T+b
                after=metrics(Y,pa.numpy());nextm=None
                if k+1<NB:
                    Fn,Yn=batches[k+1];Pnext=torch.tensor(Fn,dtype=torch.float32)@W.T+b;nextm=metrics(Yn,Pnext.detach().numpy())
                trace.append({"batch":k,"loss":float(loss.detach()),"current_before":before,"current_after":after,"next_after":nextm,
                              "weight_norm":float(W.norm()),"bias_norm":float(b.norm())})
            all_ev=[];all_mse=[]
            with torch.no_grad():
                for F,Y in batches:
                    P=torch.tensor(F,dtype=torch.float32)@W.T+b;m=metrics(Y,P.numpy());all_ev+=m["ev"];all_mse+=m["mse"]
            seq[str(l2)]={"trace":trace,"final_all_batches_ev_mean":float(np.mean(all_ev)),"final_all_batches_mse_mean":float(np.mean(all_mse)),
                          "final_weight_norm":float(W.norm())}
        white={}
        for rf in (1e-6,1e-4,1e-3,1e-2):
            rows=[]
            for k in range(NB-1):
                F,Y=batches[k];Fn,Yn=batches[k+1]
                selfm,Pnext,cond=whiten_fit(F,Y,Fn,rf)
                rows.append({"pair":f"{k}->{k+1}","self":selfm,"prior_on_next":metrics(Yn,Pnext),"whitened_condition_number":cond})
            white[str(rf)]=rows
        report["specialists"][lab]={"spectrum":spectra,"ridge":ridge,"sequential_head_l2":seq,"whitening":white}
    agg={"ridge":{},"sequential_head_l2":{},"whitening":{}}
    for l2 in RIDGES:
        se=[];ne=[];dr=[];norm=[]
        for lab in ORDER:
            for r in report["specialists"][lab]["ridge"][str(l2)]:
                se+=r["self"]["ev"];ne+=r["prior_on_next"]["ev"];dr.append(r["solution_drift"]);norm.append(r["solution_norm"])
        agg["ridge"][str(l2)]={"self_ev_mean":float(np.mean(se)),"next_ev_mean":float(np.mean(ne)),"next_ev_negative_fraction":float(np.mean(np.array(ne)<0)),
                               "solution_drift_mean":float(np.mean(dr)),"solution_norm_mean":float(np.mean(norm))}
    for l2 in SEQ_L2:
        cur=[];nxt=[];final=[];norm=[]
        for lab in ORDER:
            q=report["specialists"][lab]["sequential_head_l2"][str(l2)];final.append(q["final_all_batches_ev_mean"]);norm.append(q["final_weight_norm"])
            for tr in q["trace"]:
                cur+=tr["current_after"]["ev"]
                if tr["next_after"] is not None:nxt+=tr["next_after"]["ev"]
        agg["sequential_head_l2"][str(l2)]={"current_after_ev_mean":float(np.mean(cur)),"next_after_ev_mean":float(np.mean(nxt)),
                                            "next_after_ev_negative_fraction":float(np.mean(np.array(nxt)<0)),
                                            "final_all_batches_ev_mean":float(np.mean(final)),"final_weight_norm_mean":float(np.mean(norm))}
    for rf in (1e-6,1e-4,1e-3,1e-2):
        se=[];ne=[];co=[]
        for lab in ORDER:
            for r in report["specialists"][lab]["whitening"][str(rf)]:
                se+=r["self"]["ev"];ne+=r["prior_on_next"]["ev"];co.append(r["whitened_condition_number"])
        agg["whitening"][str(rf)]={"self_ev_mean":float(np.mean(se)),"next_ev_mean":float(np.mean(ne)),"next_ev_negative_fraction":float(np.mean(np.array(ne)<0)),
                                   "condition_number_median":float(np.median(co))}
    conds=[];rank=[];small=[[],[],[],[]]
    for lab in ORDER:
        for s in report["specialists"][lab]["spectrum"]:
            conds.append(s["condition_number"]);rank.append(s["effective_rank_1e-2"])
            for j in range(4):small[j].append(s["target_energy_fraction_small_sv_lt_1e-2"][j])
    agg["spectrum"]={"condition_number_median":float(np.median(conds)),"condition_number_max":float(np.max(conds)),
                     "effective_rank_1e-2_mean":float(np.mean(rank)),"small_sv_target_energy_fraction_by_head":[float(np.mean(x)) for x in small]}
    report["aggregate"]=agg
    (OUT/"audit.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(agg,indent=2))

def run_post_v2_t5_c13_pcr_audit():
    """Run former post_v2_t5_c13_pcr_audit.py stage."""
    """Principal-component regression conditioning of the C13 critic features.
    
    Fits each feature block's targets in a truncated PCA basis, then applies that
    fit to the next block, to see how far a rank-limited solution carries.
    """
    import json
    import sys
    from pathlib import Path
    
    import numpy as np
    
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    
    from rl.core.diagnostics.offline_audit import OfflineAudit
    
    ORDER = ("T", "A", "O", "S")
    BLOCKS = 6
    RANKS = (16, 32, 48, 64, 96, 128)
    
    
    def ev(y, p):
        """Explained variance of a prediction against its target."""
        y = np.asarray(y, float).reshape(-1)
        p = np.asarray(p, float).reshape(-1)
        return float(1 - np.var(y - p) / (np.var(y) + 1e-12))
    
    
    def metrics(y, p):
        return {"ev": [ev(y[:, j], p[:, j]) for j in range(4)]}
    
    
    class PCRAudit(OfflineAudit):
        """Principal-component regression conditioning of the C13 critic features."""
    
        run = "post_v2_t5_c13_conditioning-2026-09-23"
        report = "pcr_audit.json"
    
        def analyze(self):
            out = {"ranks": RANKS, "specialists": {}}
            for lab in ORDER:
                z = self.load(f"{lab}_features_targets.npz")
                blocks = [(z[f"F{k}"].astype(np.float64), z[f"Y{k}"].astype(np.float64))
                          for k in range(BLOCKS)]
                labout = {}
                for rank in RANKS:
                    rows = []
                    for k in range(BLOCKS - 1):
                        f, y = blocks[k]
                        fn, yn = blocks[k + 1]
                        mu = f.mean(0)
                        x, xn = f - mu, fn - mu
                        _, s, vt = np.linalg.svd(x, full_matrices=False)
                        r = min(rank, len(s))
                        vr = vt[:r].T
                        a = np.c_[x @ vr, np.ones(len(x))]
                        an = np.c_[xn @ vr, np.ones(len(xn))]
                        sol = np.linalg.lstsq(a, y, rcond=None)[0]
                        rows.append({"pair": f"{k}->{k+1}",
                                     "self": metrics(y, a @ sol),
                                     "prior_on_next": metrics(yn, an @ sol),
                                     "retained_singular_energy": float(np.sum(s[:r] ** 2) / np.sum(s ** 2))})
                    labout[str(rank)] = rows
                out["specialists"][lab] = labout
            out["aggregate"] = self.aggregate(out)
            return out
    
        def aggregate(self, out):
            agg = {}
            for rank in RANKS:
                se, ne, energy = [], [], []
                for lab in ORDER:
                    for row in out["specialists"][lab][str(rank)]:
                        se += row["self"]["ev"]
                        ne += row["prior_on_next"]["ev"]
                        energy.append(row["retained_singular_energy"])
                agg[str(rank)] = {"self_ev_mean": float(np.mean(se)),
                                  "next_ev_mean": float(np.mean(ne)),
                                  "next_ev_negative_fraction": float(np.mean(np.array(ne) < 0)),
                                  "retained_singular_energy_mean": float(np.mean(energy))}
            return agg
    
        def summarize(self, report):
            print(json.dumps(report["aggregate"], indent=2))
    
    
    if True:
        PCRAudit.main()

def run_post_v2_t5_c14_collect_matched_features():
    """Run former post_v2_t5_c14_collect_matched_features.py stage."""
    from pathlib import Path
    import sys,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c10_h32_mc-2026-09-23";OUT=ROOT/"runs/post_v2_t5_c14_representation_drift-2026-09-23"
    ORDER=("T","A","O","S");SNAPS=(0,10,25);H=64;G=.99
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    def obs_tensor(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def mc(R,D):
     out=torch.zeros_like(R);run=torch.zeros_like(R[-1])
     for t in range(len(R)-1,-1,-1):run=R[t]+G*run*(~D[t]).unsqueeze(-1);out[t]=run
     return out
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic
      from talon_rl.rewards.objectives import normalized_objective_vector
      OUT.mkdir(parents=True,exist_ok=True)
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=32;cfg.seed=0;env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
      o,_=env.reset(seed=0);o=obs_tensor(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
      for bi,lab in enumerate(ORDER):
       models={}
       for s in SNAPS:
        m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"{lab}_snap_{s}.pt",map_location="cuda",weights_only=False)["model"]);m.eval();models[s]=m
       # fixed state distribution generated by terminal actor only
       actor=models[25];w=torch.tensor(PREFS[lab],device="cuda").repeat(32,1);cur,_=env.reset(seed=1010000+bi*1000);cur=obs_tensor(cur).cuda()
       obs=[];rw=[];dn=[]
       with torch.no_grad():
        for _ in range(H):
         obs.append(cur);a=actor.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a)
         raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);vec=normalized_objective_vector(terms(raw,names),shape=(32,))
         rw.append(torch.tensor(vec,dtype=torch.float32,device="cuda")*env.unwrapped.step_dt);dn.append((te|tr).cuda());cur=obs_tensor(nxt).cuda()
       X=torch.cat(obs);Wp=w.repeat(H,1);Y=mc(torch.stack(rw),torch.stack(dn).bool()).reshape(-1,4)
       data={"X":X.cpu().numpy(),"Y":Y.cpu().numpy()}
       with torch.no_grad():
        for s,m in models.items():
         F=m.critic_body(m._with_w(X,Wp))
         P=m.critic_head(F)
         data[f"F{s}"]=F.cpu().numpy();data[f"P{s}"]=P.cpu().numpy()
         data[f"W{s}"]=m.critic_head.weight.cpu().numpy();data[f"b{s}"]=m.critic_head.bias.cpu().numpy()
       np.savez_compressed(OUT/f"{lab}_matched.npz",**data)
       print(lab,"saved",X.shape,Y.shape,flush=True)
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c14_offline_analyze():
    """Run former post_v2_t5_c14_offline_analyze.py stage."""
    from pathlib import Path
    import json,hashlib,numpy as np
    ROOT=Path(__file__).resolve().parents[4];OUT=ROOT/"runs/post_v2_t5_c14_representation_drift-2026-09-23"
    ORDER=("T","A","O","S");SNAPS=(0,10,25);RANKS=(16,32,48)
    def ev(y,p):
     y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
     return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def cka(X,Y):
     X=X-X.mean(0);Y=Y-Y.mean(0)
     hs=np.linalg.norm(X.T@Y,'fro')**2
     return float(hs/(np.linalg.norm(X.T@X,'fro')*np.linalg.norm(Y.T@Y,'fro')+1e-12))
    def subspace_overlap(X,Y,r):
     X=X-X.mean(0);Y=Y-Y.mean(0)
     _,_,Vx=np.linalg.svd(X,full_matrices=False);_,_,Vy=np.linalg.svd(Y,full_matrices=False)
     Qx=Vx[:r].T;Qy=Vy[:r].T;s=np.linalg.svd(Qx.T@Qy,compute_uv=False)
     return {"mean_cos":float(np.mean(s)),"min_cos":float(np.min(s)),"mean_angle_deg":float(np.degrees(np.arccos(np.clip(s,-1,1))).mean())}
    def spectrum(F):
     X=F-F.mean(0);s=np.linalg.svd(X,compute_uv=False)
     return {"cond":float(s.max()/(s.min()+1e-12)),"rank_1e2":int(np.sum(s>1e-2*s.max())),"rank_1e3":int(np.sum(s>1e-3*s.max())),
             "top16_energy":float(np.sum(s[:16]**2)/np.sum(s**2)),"top32_energy":float(np.sum(s[:32]**2)/np.sum(s**2))}
    report={"schema":"t5_c14_representation_drift_v1","specialists":{}}
    for lab in ORDER:
     z=np.load(OUT/f"{lab}_matched.npz")
     Y=z["Y"].astype(np.float64)
     F={s:z[f"F{s}"].astype(np.float64) for s in SNAPS}
     W={s:z[f"W{s}"].astype(np.float64) for s in SNAPS}
     b={s:z[f"b{s}"].astype(np.float64) for s in SNAPS}
     P={s:z[f"P{s}"].astype(np.float64) for s in SNAPS}
     pair={}
     for i,j in ((0,10),(10,25),(0,25)):
      d={"cka":cka(F[i],F[j]),"feature_relative_l2":float(np.linalg.norm(F[j]-F[i])/(np.linalg.norm(F[i])+1e-12)),
         "mean_feature_cos":float(np.mean(np.sum(F[i]*F[j],1)/(np.linalg.norm(F[i],axis=1)*np.linalg.norm(F[j],axis=1)+1e-12)))}
      for r in RANKS:d[f"subspace_r{r}"]=subspace_overlap(F[i],F[j],r)
      pair[f"{i}->{j}"]=d
     # matched same-body head transplants
     trans={}
     for bs in SNAPS:
      trans[str(bs)]={}
      for hs in SNAPS:
       pred=F[bs]@W[hs].T+b[hs]
       trans[str(bs)][str(hs)]={"ev":[ev(Y[:,k],pred[:,k]) for k in range(4)],"mse":[float(np.mean((pred[:,k]-Y[:,k])**2)) for k in range(4)]}
     # best possible linear head on each frozen body
     best={}
     for s in SNAPS:
      A=np.c_[F[s],np.ones(len(F[s]))];sol=np.linalg.lstsq(A,Y,rcond=None)[0];pred=A@sol
      best[str(s)]={"ev":[ev(Y[:,k],pred[:,k]) for k in range(4)],"solution_norm":float(np.linalg.norm(sol[:-1]))}
     # actual predictions
     actual={str(s):{"ev":[ev(Y[:,k],P[s][:,k]) for k in range(4)]} for s in SNAPS}
     # head drift
     hd={}
     for i,j in ((0,10),(10,25),(0,25)):
      hd[f"{i}->{j}"]={"weight_l2":float(np.linalg.norm(W[j]-W[i])),"bias_l2":float(np.linalg.norm(b[j]-b[i])),
                       "relative_weight_l2":float(np.linalg.norm(W[j]-W[i])/(np.linalg.norm(W[i])+1e-12))}
     report["specialists"][lab]={"feature_pairwise":pair,"spectrum":{str(s):spectrum(F[s]) for s in SNAPS},
                                 "transplant":trans,"best_linear":best,"actual":actual,"head_drift":hd}
    # aggregate
    agg={}
    for key in ("0->10","10->25","0->25"):
     vals=[report["specialists"][lab]["feature_pairwise"][key] for lab in ORDER]
     agg.setdefault("feature_pairwise",{})[key]={
      "cka_mean":float(np.mean([x["cka"] for x in vals])),
      "feature_relative_l2_mean":float(np.mean([x["feature_relative_l2"] for x in vals])),
      "mean_feature_cos_mean":float(np.mean([x["mean_feature_cos"] for x in vals])),
      "subspace_r16_mean_cos":float(np.mean([x["subspace_r16"]["mean_cos"] for x in vals])),
      "subspace_r32_mean_cos":float(np.mean([x["subspace_r32"]["mean_cos"] for x in vals])),
      "subspace_r48_mean_cos":float(np.mean([x["subspace_r48"]["mean_cos"] for x in vals]))
     }
    # spectrum evolution
    for s in SNAPS:
     sp=[report["specialists"][lab]["spectrum"][str(s)] for lab in ORDER]
     agg.setdefault("spectrum",{})[str(s)]={"cond_median":float(np.median([x["cond"] for x in sp])),
      "rank_1e2_mean":float(np.mean([x["rank_1e2"] for x in sp])),"top16_energy_mean":float(np.mean([x["top16_energy"] for x in sp])),
      "top32_energy_mean":float(np.mean([x["top32_energy"] for x in sp]))}
    # actual vs transplants: aggregate all 16 head objectives
    for bs in SNAPS:
     for hs in SNAPS:
      vals=[]
      for lab in ORDER: vals+=report["specialists"][lab]["transplant"][str(bs)][str(hs)]["ev"]
      agg.setdefault("transplant_ev_mean",{}).setdefault(str(bs),{})[str(hs)]=float(np.mean(vals))
    # best and actual
    for s in SNAPS:
     be=[];ac=[]
     for lab in ORDER:
      be+=report["specialists"][lab]["best_linear"][str(s)]["ev"];ac+=report["specialists"][lab]["actual"][str(s)]["ev"]
     agg.setdefault("best_linear_ev_mean",{})[str(s)]=float(np.mean(be));agg.setdefault("actual_ev_mean",{})[str(s)]=float(np.mean(ac))
    # head drift
    for key in ("0->10","10->25","0->25"):
     hs=[report["specialists"][lab]["head_drift"][key] for lab in ORDER]
     agg.setdefault("head_drift",{})[key]={"weight_l2_mean":float(np.mean([x["weight_l2"] for x in hs])),
      "relative_weight_l2_mean":float(np.mean([x["relative_weight_l2"] for x in hs]))}
    report["aggregate"]=agg
    (OUT/"audit.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(agg,indent=2))

def run_post_v2_t5_c15_sequential_projection():
    """Run former post_v2_t5_c15_sequential_projection.py stage."""
    from pathlib import Path
    import json,numpy as np,torch
    
    ROOT=Path(__file__).resolve().parents[4]
    SRC=ROOT/"runs/post_v2_t5_c13_conditioning-2026-09-23"
    OUT=ROOT/"runs/post_v2_t5_c15_spectral_update-2026-09-23"
    ORDER=("T","A","O","S"); NB=6; LR=1e-4; THRESH=1e-2
    
    def ev(y,p):
        y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
        return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def metrics(Y,P):
        return {"ev":[ev(Y[:,j],P[:,j]) for j in range(4)],
                "mse":[float(np.mean((P[:,j]-Y[:,j])**2)) for j in range(4)]}
    def projector(F):
        X=F-F.mean(0,keepdims=True);_,S,Vt=np.linalg.svd(X.astype(np.float64),full_matrices=False)
        r=int(np.sum(S>=THRESH*S.max()));Vr=Vt[:r].T.astype(np.float32);Ps=Vr@Vr.T
        return Ps,np.eye(F.shape[1],dtype=np.float32)-Ps,r
    
    report={"schema":"t5_c15_sequential_projection_v1","specialists":{}}
    for lab in ORDER:
        z=np.load(SRC/f"{lab}_features_targets.npz")
        B=[(z[f"F{k}"].astype(np.float32),z[f"Y{k}"].astype(np.float32)) for k in range(NB)]
        W0=z["head_weight"].astype(np.float32);b0=z["head_bias"].astype(np.float32)
        branch={}
        for mode in ("full","strong_only","weak_only"):
            W=torch.nn.Parameter(torch.tensor(W0.copy()));b=torch.nn.Parameter(torch.tensor(b0.copy()))
            opt=torch.optim.Adam([W,b],lr=LR)
            trace=[]
            for k,(F,Y) in enumerate(B):
                Ft=torch.tensor(F);Yt=torch.tensor(Y)
                Ps,Pw,r=projector(F)
                preW=W.detach().clone();preb=b.detach().clone()
                pred=Ft@W.T+b;loss=((pred-Yt)**2).mean()
                opt.zero_grad();loss.backward();opt.step()
                postW=W.detach().clone();postb=b.detach().clone()
                dW=(postW-preW).numpy();db=(postb-preb).numpy()
                # project the APPLIED Adam weight delta; keep full bias step in all branches
                if mode=="strong_only": dW=dW@Ps
                elif mode=="weak_only": dW=dW@Pw
                with torch.no_grad():
                    W.copy_(preW+torch.tensor(dW));b.copy_(preb+torch.tensor(db))
                cur=metrics(Y,(F@W.detach().numpy().T+b.detach().numpy()))
                nxt=None
                if k+1<NB:
                    Fn,Yn=B[k+1];nxt=metrics(Yn,(Fn@W.detach().numpy().T+b.detach().numpy()))
                # retrospective average EV on all seen batches
                seen=[]
                for q in range(k+1):
                    Fq,Yq=B[q];seen+=metrics(Yq,Fq@W.detach().numpy().T+b.detach().numpy())["ev"]
                trace.append({"batch":k,"effective_rank":r,"current":cur,"next":nxt,"seen_ev_mean":float(np.mean(seen)),
                              "weight_norm":float(W.norm()),"bias_norm":float(b.norm())})
            all_ev=[];all_mse=[]
            for F,Y in B:
                mm=metrics(Y,F@W.detach().numpy().T+b.detach().numpy());all_ev+=mm["ev"];all_mse+=mm["mse"]
            branch[mode]={"trace":trace,"final_all_ev_mean":float(np.mean(all_ev)),"final_all_mse_mean":float(np.mean(all_mse)),
                          "final_weight_norm":float(W.norm())}
        report["specialists"][lab]=branch
    
    agg={}
    for mode in ("full","strong_only","weak_only"):
        cur=[];nxt=[];seen=[];final=[];norm=[]
        for lab in ORDER:
            q=report["specialists"][lab][mode];final.append(q["final_all_ev_mean"]);norm.append(q["final_weight_norm"])
            for tr in q["trace"]:
                cur+=tr["current"]["ev"];seen.append(tr["seen_ev_mean"])
                if tr["next"] is not None:nxt+=tr["next"]["ev"]
        agg[mode]={"current_ev_mean":float(np.mean(cur)),"next_ev_mean":float(np.mean(nxt)),
                   "next_ev_negative_fraction":float(np.mean(np.array(nxt)<0)),
                   "seen_ev_mean":float(np.mean(seen)),"final_all_ev_mean":float(np.mean(final)),
                   "final_weight_norm_mean":float(np.mean(norm))}
    report["aggregate"]=agg
    (OUT/"sequential_projection.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(agg,indent=2))

def run_post_v2_t5_c15_spectral_update_audit():
    """Run former post_v2_t5_c15_spectral_update_audit.py stage."""
    from pathlib import Path
    import json,hashlib,numpy as np,torch
    
    ROOT=Path(__file__).resolve().parents[4]
    SRC=ROOT/"runs/post_v2_t5_c13_conditioning-2026-09-23"
    OUT=ROOT/"runs/post_v2_t5_c15_spectral_update-2026-09-23"
    ORDER=("T","A","O","S"); NB=6; LR=1e-4; THRESH=1e-2
    
    def ev(y,p):
        y=np.asarray(y,float).reshape(-1); p=np.asarray(p,float).reshape(-1)
        return float(1-np.var(y-p)/(np.var(y)+1e-12))
    
    def metrics(Y,P):
        return {
            "ev":[ev(Y[:,j],P[:,j]) for j in range(4)],
            "mse":[float(np.mean((P[:,j]-Y[:,j])**2)) for j in range(4)]
        }
    
    def cos(a,b):
        a=np.asarray(a).reshape(-1); b=np.asarray(b).reshape(-1)
        return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
    
    def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    
    OUT.mkdir(parents=True,exist_ok=True)
    report={"schema":"t5_c15_spectral_update_audit_v1","threshold_relative_to_smax":THRESH,
            "adam_lr":LR,"bias_policy":"same full bias update in strong-only and weak-only; bias-only control reported","specialists":{}}
    
    for lab in ORDER:
        z=np.load(SRC/f"{lab}_features_targets.npz")
        batches=[(z[f"F{k}"].astype(np.float32),z[f"Y{k}"].astype(np.float32)) for k in range(NB)]
        W0=z["head_weight"].astype(np.float32); b0=z["head_bias"].astype(np.float32)
        rows=[]
        for k in range(NB-1):
            F,Y=batches[k]; Fn,Yn=batches[k+1]
            # spectral basis from current batch feature matrix
            X=F-F.mean(0,keepdims=True)
            _,S,Vt=np.linalg.svd(X.astype(np.float64),full_matrices=False)
            r=int(np.sum(S>=THRESH*S.max()))
            Vr=Vt[:r].T.astype(np.float32)
            Pstrong=Vr@Vr.T
            Pweak=np.eye(F.shape[1],dtype=np.float32)-Pstrong
    
            Ft=torch.tensor(F); Yt=torch.tensor(Y)
            W=torch.nn.Parameter(torch.tensor(W0.copy())); b=torch.nn.Parameter(torch.tensor(b0.copy()))
            opt=torch.optim.Adam([W,b],lr=LR)
            pred=Ft@W.T+b
            loss=((pred-Yt)**2).mean()
            opt.zero_grad(); loss.backward()
            gW=W.grad.detach().numpy().copy(); gb=b.grad.detach().numpy().copy()
            opt.step()
            dW=(W.detach().numpy()-W0); db=(b.detach().numpy()-b0)
    
            # decompose gradient and actual one-step Adam delta into feature singular subspaces
            gWs=gW@Pstrong; gWw=gW@Pweak
            dWs=dW@Pstrong; dWw=dW@Pweak
            # counterfactual states: keep same full bias update except bias-only
            variants={
                "base":(W0,b0),
                "full":(W0+dW,b0+db),
                "strong_only":(W0+dWs,b0+db),
                "weak_only":(W0+dWw,b0+db),
                "bias_only":(W0,b0+db),
                "weight_full_no_bias":(W0+dW,b0)
            }
            evals={}
            for name,(WV,bV) in variants.items():
                evals[name]={
                    "current":metrics(Y,F@WV.T+bV),
                    "next":metrics(Yn,Fn@WV.T+bV)
                }
    
            # per-head energy
            per_head=[]
            for j in range(4):
                gs=np.linalg.norm(gWs[j]); gw=np.linalg.norm(gWw[j])
                ds=np.linalg.norm(dWs[j]); dw=np.linalg.norm(dWw[j])
                per_head.append({
                    "head":j,
                    "gradient_strong_norm":float(gs),"gradient_weak_norm":float(gw),
                    "gradient_weak_energy_fraction":float(gw*gw/(gs*gs+gw*gw+1e-12)),
                    "update_strong_norm":float(ds),"update_weak_norm":float(dw),
                    "update_weak_energy_fraction":float(dw*dw/(ds*ds+dw*dw+1e-12)),
                    "strong_to_weak_update_norm_ratio":float(ds/(dw+1e-12))
                })
            rows.append({
                "pair":f"{k}->{k+1}","effective_rank":r,"singular_value_ratio_boundary":float(S[r-1]/S[0]) if r>0 else None,
                "gradient_weak_energy_fraction":float(np.linalg.norm(gWw)**2/(np.linalg.norm(gW)**2+1e-12)),
                "update_weak_energy_fraction":float(np.linalg.norm(dWw)**2/(np.linalg.norm(dW)**2+1e-12)),
                "gradient_strong_cosine_full":cos(gWs,gW),
                "update_strong_cosine_full":cos(dWs,dW),
                "gradient_weak_cosine_full":cos(gWw,gW),
                "update_weak_cosine_full":cos(dWw,dW),
                "bias_update_norm":float(np.linalg.norm(db)),
                "full_weight_update_norm":float(np.linalg.norm(dW)),
                "per_head":per_head,"eval":evals
            })
        report["specialists"][lab]=rows
    
    # aggregate deltas relative to base
    agg={}
    weakE=[];gradWeak=[];strongCos=[];weakCos=[];ranks=[]
    head_weak=[[] for _ in range(4)]
    variant_stats={v:{"cur_ev_delta":[],"next_ev_delta":[],"cur_mse_improve":[],"next_mse_improve":[]} for v in ("full","strong_only","weak_only","bias_only","weight_full_no_bias")}
    for lab in ORDER:
        for row in report["specialists"][lab]:
            weakE.append(row["update_weak_energy_fraction"]);gradWeak.append(row["gradient_weak_energy_fraction"])
            strongCos.append(row["update_strong_cosine_full"]);weakCos.append(row["update_weak_cosine_full"]);ranks.append(row["effective_rank"])
            for h in row["per_head"]:head_weak[h["head"]].append(h["update_weak_energy_fraction"])
            basec=row["eval"]["base"]["current"];basen=row["eval"]["base"]["next"]
            for v in variant_stats:
                vc=row["eval"][v]["current"];vn=row["eval"][v]["next"]
                variant_stats[v]["cur_ev_delta"] += (np.array(vc["ev"])-np.array(basec["ev"])).tolist()
                variant_stats[v]["next_ev_delta"] += (np.array(vn["ev"])-np.array(basen["ev"])).tolist()
                variant_stats[v]["cur_mse_improve"] += ((np.array(basec["mse"])-np.array(vc["mse"]))/(np.array(basec["mse"])+1e-12)).tolist()
                variant_stats[v]["next_mse_improve"] += ((np.array(basen["mse"])-np.array(vn["mse"]))/(np.array(basen["mse"])+1e-12)).tolist()
    
    agg["effective_rank_mean"]=float(np.mean(ranks))
    agg["gradient_weak_energy_fraction_mean"]=float(np.mean(gradWeak))
    agg["update_weak_energy_fraction_mean"]=float(np.mean(weakE))
    agg["update_strong_cosine_full_mean"]=float(np.mean(strongCos))
    agg["update_weak_cosine_full_mean"]=float(np.mean(weakCos))
    agg["per_head_update_weak_energy_fraction_mean"]=[float(np.mean(x)) for x in head_weak]
    agg["variants"]={}
    for v,x in variant_stats.items():
        agg["variants"][v]={
            "current_ev_delta_mean":float(np.mean(x["cur_ev_delta"])),
            "next_ev_delta_mean":float(np.mean(x["next_ev_delta"])),
            "current_mse_improvement_mean":float(np.mean(x["cur_mse_improve"])),
            "next_mse_improvement_mean":float(np.mean(x["next_mse_improve"])),
            "next_ev_improve_fraction":float(np.mean(np.array(x["next_ev_delta"])>0)),
            "next_mse_improve_fraction":float(np.mean(np.array(x["next_mse_improve"])>0))
        }
    # causal contrasts
    agg["contrasts"]={
        "strong_minus_weak_next_ev_delta":agg["variants"]["strong_only"]["next_ev_delta_mean"]-agg["variants"]["weak_only"]["next_ev_delta_mean"],
        "strong_minus_weak_current_ev_delta":agg["variants"]["strong_only"]["current_ev_delta_mean"]-agg["variants"]["weak_only"]["current_ev_delta_mean"],
        "full_minus_strong_next_ev_delta":agg["variants"]["full"]["next_ev_delta_mean"]-agg["variants"]["strong_only"]["next_ev_delta_mean"]
    }
    report["aggregate"]=agg
    (OUT/"audit.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(agg,indent=2))

def run_post_v2_t5_c16_function_space_progress():
    """Run former post_v2_t5_c16_function_space_progress.py stage."""
    from pathlib import Path
    import json,numpy as np
    ROOT=Path(__file__).resolve().parents[4]
    C14=ROOT/"runs/post_v2_t5_c14_representation_drift-2026-09-23"
    OUT=ROOT/"runs/post_v2_t5_c16_head_progress-2026-09-23"
    ORDER=("T","A","O","S");SNAPS=(0,10,25);RIDGE=1.0;PCR=32
    def cos(a,b):
     a=np.asarray(a).reshape(-1);b=np.asarray(b).reshape(-1)
     return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
    def ridge(F,Y,l2):
     A=np.c_[F,np.ones(len(F))];I=np.eye(A.shape[1]);I[-1,-1]=0;sol=np.linalg.solve(A.T@A+l2*I,A.T@Y);return A@sol
    def pcr(F,Y,r):
     mu=F.mean(0,keepdims=True);X=F-mu;U,S,Vt=np.linalg.svd(X,full_matrices=False);Vr=Vt[:r].T;A=np.c_[X@Vr,np.ones(len(X))]
     sol=np.linalg.lstsq(A,Y,rcond=None)[0];return A@sol
    def metric(p0,pt,pstar):
     d=pt-p0;ds=pstar-p0
     return {"cosine":cos(d,ds),"norm_progress_ratio":float(np.linalg.norm(d)/(np.linalg.norm(ds)+1e-12)),
             "projected_progress_ratio":float((d.reshape(-1)@ds.reshape(-1))/(np.linalg.norm(ds)**2+1e-12)),
             "orthogonal_fraction":float(np.linalg.norm(d-((d.reshape(-1)@ds.reshape(-1))/(np.linalg.norm(ds)**2+1e-12))*ds)/(np.linalg.norm(d)+1e-12))}
    report={"schema":"t5_c16_function_space_progress_v1","specialists":{}}
    for lab in ORDER:
     z=np.load(C14/f"{lab}_matched.npz");Y=z["Y"].astype(float);F=z["F25"].astype(float)
     W={s:z[f"W{s}"].astype(float) for s in SNAPS};b={s:z[f"b{s}"].astype(float) for s in SNAPS}
     P={s:F@W[s].T+b[s] for s in SNAPS}
     refs={"ridge":ridge(F,Y,RIDGE),"pcr":pcr(F,Y,PCR)}
     q={}
     for rn,ps in refs.items():
      q[rn]={}
      for t in (10,25):
       q[rn][str(t)]={"all":metric(P[0],P[t],ps),"per_head":[metric(P[0][:,j:j+1],P[t][:,j:j+1],ps[:,j:j+1]) for j in range(4)]}
     report["specialists"][lab]=q
    agg={}
    for rn in ("ridge","pcr"):
     agg[rn]={}
     for t in ("10","25"):
      vals=[report["specialists"][lab][rn][t]["all"] for lab in ORDER]
      agg[rn][t]={"cosine_mean":float(np.mean([x["cosine"] for x in vals])),"cosine_median":float(np.median([x["cosine"] for x in vals])),
                  "norm_progress_ratio_mean":float(np.mean([x["norm_progress_ratio"] for x in vals])),
                  "projected_progress_ratio_mean":float(np.mean([x["projected_progress_ratio"] for x in vals])),
                  "orthogonal_fraction_mean":float(np.mean([x["orthogonal_fraction"] for x in vals]))}
     report.setdefault("aggregate_per_head",{}).setdefault(rn,{})
     for j in range(4):
      for t in ("10","25"):
       vals=[report["specialists"][lab][rn][t]["per_head"][j] for lab in ORDER]
       report["aggregate_per_head"][rn].setdefault(str(j),{})[t]={"cosine_mean":float(np.mean([x["cosine"] for x in vals])),
        "projected_progress_ratio_mean":float(np.mean([x["projected_progress_ratio"] for x in vals]))}
    report["aggregate"]=agg
    (OUT/"function_space.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"aggregate":agg,"per_head":report["aggregate_per_head"]},indent=2))

def run_post_v2_t5_c16_head_progress_audit():
    """Run former post_v2_t5_c16_head_progress_audit.py stage."""
    """Is online head learning moving toward the solution a stable fit would pick?
    
    Freezes two reference solutions on the u25 body and matched targets — ridge and
    rank-32 PCR — then measures how far the online head has travelled from u0 in
    their direction, overall and per head. A negative cosine means it is moving
    away from what the data supports.
    """
    import json
    import sys
    from pathlib import Path
    
    import numpy as np
    
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    
    from rl.core.diagnostics.offline_audit import RUNS, OfflineAudit
    
    C14 = "post_v2_t5_c14_representation_drift-2026-09-23"
    ORDER = ("T", "A", "O", "S")
    SNAPS = (0, 10, 25)
    RIDGE = 1.0
    PCR_RANK = 32
    HEADS = 4
    
    
    def cos(a, b):
        a = np.asarray(a).reshape(-1)
        b = np.asarray(b).reshape(-1)
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    
    
    def ev(y, p):
        y = np.asarray(y, float).reshape(-1)
        p = np.asarray(p, float).reshape(-1)
        return float(1 - np.var(y - p) / (np.var(y) + 1e-12))
    
    
    def ridge_solution(f, y, l2):
        a = np.c_[f, np.ones(len(f))]
        i = np.eye(a.shape[1])
        i[-1, -1] = 0          # the intercept is not penalised
        sol = np.linalg.solve(a.T @ a + l2 * i, a.T @ y)
        return sol[:-1].T, sol[-1]
    
    
    def pcr_solution(f, y, rank):
        mu = f.mean(0, keepdims=True)
        x = f - mu
        _, s, vt = np.linalg.svd(x, full_matrices=False)
        r = min(rank, len(s))
        vr = vt[:r].T
        a = np.c_[x @ vr, np.ones(len(x))]
        sol = np.linalg.lstsq(a, y, rcond=None)[0]
        w_red = sol[:-1].T
        return w_red @ vr.T, sol[-1] - w_red @ (mu.reshape(-1) @ vr), s
    
    
    def displacement_metrics(w0, b0, wt, bt, wstar, bstar):
        """How the online step from u0 compares with the step to the stable fit."""
        d_on = np.r_[(wt - w0).reshape(-1), (bt - b0).reshape(-1)]
        d_st = np.r_[(wstar - w0).reshape(-1), (bstar - b0).reshape(-1)]
        n_on, n_st = np.linalg.norm(d_on), np.linalg.norm(d_st)
        rho_par = float((d_on @ d_st) / (d_st @ d_st + 1e-12))
        orth = np.linalg.norm(d_on - rho_par * d_st)
        return {"online_norm": float(n_on), "target_norm": float(n_st),
                "cosine": cos(d_on, d_st),
                "norm_progress_ratio": float(n_on / (n_st + 1e-12)),
                "projected_progress_ratio": rho_par,
                "orthogonal_fraction_of_online": float(orth / (n_on + 1e-12))}
    
    
    class HeadProgressAudit(OfflineAudit):
        """Online head progress against frozen ridge and PCR references."""
    
        run = "post_v2_t5_c16_head_progress-2026-09-23"
        report = "audit.json"
        schema = "t5_c16_head_progress_v1"
    
        def specialist(self, lab):
            z = np.load(RUNS / C14 / f"{lab}_matched.npz")
            y = z["Y"].astype(np.float64)
            f = {s: z[f"F{s}"].astype(np.float64) for s in SNAPS}
            w = {s: z[f"W{s}"].astype(np.float64) for s in SNAPS}
            b = {s: z[f"b{s}"].astype(np.float64) for s in SNAPS}
    
            # references frozen on the u25 body and matched target, robust classes only
            wr, br = ridge_solution(f[25], y, RIDGE)
            wp, bp, _ = pcr_solution(f[25], y, PCR_RANK)
    
            out = {"ridge_ref": {}, "pcr_ref": {}, "actual_ev": {}}
            for t in (10, 25):
                out["ridge_ref"][str(t)] = displacement_metrics(w[0], b[0], w[t], b[t], wr, br)
                out["pcr_ref"][str(t)] = displacement_metrics(w[0], b[0], w[t], b[t], wp, bp)
            for s in SNAPS:
                pred = f[s] @ w[s].T + b[s]
                out["actual_ev"][str(s)] = [ev(y[:, j], pred[:, j]) for j in range(HEADS)]
    
            per_head = {}
            for j in range(HEADS):
                sl = slice(j, j + 1)
                per_head[str(j)] = {
                    "ridge": {str(t): displacement_metrics(w[0][sl], b[0][sl], w[t][sl], b[t][sl],
                                                           wr[sl], br[sl]) for t in (10, 25)},
                    "pcr": {str(t): displacement_metrics(w[0][sl], b[0][sl], w[t][sl], b[t][sl],
                                                         wp[sl], bp[sl]) for t in (10, 25)}}
            out["per_head"] = per_head
            out["reference_ev"] = {
                "ridge_on_u25": [ev(y[:, j], (f[25] @ wr.T + br)[:, j]) for j in range(HEADS)],
                "pcr32_on_u25": [ev(y[:, j], (f[25] @ wp.T + bp)[:, j]) for j in range(HEADS)]}
            return out
    
        def analyze(self):
            report = {"schema": self.schema,
                      "stable_refs": {"ridge_lambda": RIDGE, "pcr_rank": PCR_RANK},
                      "specialists": {lab: self.specialist(lab) for lab in ORDER}}
            report["aggregate"] = self.aggregate(report["specialists"])
            return report
    
        def aggregate(self, spec):
            mean = lambda vals, k: float(np.mean([x[k] for x in vals]))  # noqa: E731
            agg = {"ridge": {}, "pcr": {}}
            for refkey, outkey in (("ridge_ref", "ridge"), ("pcr_ref", "pcr")):
                for t in ("10", "25"):
                    vals = [spec[lab][refkey][t] for lab in ORDER]
                    cosines = np.array([x["cosine"] for x in vals])
                    agg[outkey][t] = {
                        "cosine_mean": mean(vals, "cosine"),
                        "cosine_median": float(np.median(cosines)),
                        "cosine_negative_fraction": float(np.mean(cosines < 0)),
                        "norm_progress_ratio_mean": mean(vals, "norm_progress_ratio"),
                        "projected_progress_ratio_mean": mean(vals, "projected_progress_ratio"),
                        "projected_progress_ratio_median": float(
                            np.median([x["projected_progress_ratio"] for x in vals])),
                        "orthogonal_fraction_mean": mean(vals, "orthogonal_fraction_of_online")}
            agg["per_head"] = {}
            for refname in ("ridge", "pcr"):
                agg["per_head"][refname] = {}
                for j in range(HEADS):
                    for t in ("10", "25"):
                        vals = [spec[lab]["per_head"][str(j)][refname][t] for lab in ORDER]
                        agg["per_head"][refname].setdefault(str(j), {})[t] = {
                            "cosine_mean": mean(vals, "cosine"),
                            "projected_progress_ratio_mean": mean(vals, "projected_progress_ratio"),
                            "norm_progress_ratio_mean": mean(vals, "norm_progress_ratio")}
            return agg
    
        def summarize(self, report):
            print(json.dumps(report["aggregate"], indent=2))
    
    
    if True:
        HeadProgressAudit.main()

def run_post_v2_t5_c17_controlled_head_tracking_audit():
    """Run former post_v2_t5_c17_controlled_head_tracking_audit.py stage."""
    """Can a head fitted on a recent window predict the next batch?
    
    Controlled comparison from one initial head: Adam for a few steps, or a ridge
    closed form, each fitted on the last `window` batches and scored on the batch
    after. Progress is measured in function space against a ridge fit over all six
    batches, so a method that moves a long way without predicting better shows up.
    """
    import json
    import sys
    from pathlib import Path
    
    import numpy as np
    import torch
    
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    
    from rl.core.diagnostics.offline_audit import RUNS, OfflineAudit
    
    SRC = "post_v2_t5_c13_conditioning-2026-09-23"
    ORDER = ("T", "A", "O", "S")
    BATCHES = 6
    WINDOWS = (1, 2, 3, 4)
    RIDGES = (0.1, 1.0)
    ADAM_STEPS = (1, 5, 10, 25)
    LR = 1e-4
    HEADS = 4
    
    
    def ev(y, p):
        y = np.asarray(y, float).reshape(-1)
        p = np.asarray(p, float).reshape(-1)
        return float(1 - np.var(y - p) / (np.var(y) + 1e-12))
    
    
    def metrics(y, p):
        return {"ev": [ev(y[:, j], p[:, j]) for j in range(HEADS)],
                "mse": [float(np.mean((p[:, j] - y[:, j]) ** 2)) for j in range(HEADS)]}
    
    
    def ridge_solution(f, y, l2):
        a = np.c_[f, np.ones(len(f))]
        i = np.eye(a.shape[1])
        i[-1, -1] = 0          # the intercept is not penalised
        sol = np.linalg.solve(a.T @ a + l2 * i, a.T @ y)
        return sol[:-1].T, sol[-1]
    
    
    def cos(a, b):
        a, b = a.reshape(-1), b.reshape(-1)
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    
    
    def function_progress(f_ref, w0, b0, w, b, w_star, b_star):
        """Movement toward the stable fit, measured in predictions rather than weights."""
        p0 = f_ref @ w0.T + b0
        d = (f_ref @ w.T + b) - p0
        ds = (f_ref @ w_star.T + b_star) - p0
        return {"cosine": cos(d, ds),
                "norm_progress_ratio": float(np.linalg.norm(d) / (np.linalg.norm(ds) + 1e-12)),
                "projected_progress_ratio": float(
                    (d.reshape(-1) @ ds.reshape(-1)) / (np.linalg.norm(ds) ** 2 + 1e-12))}
    
    
    def adam_head(w0, b0, fw, yw, steps):
        w = torch.nn.Parameter(torch.tensor(w0.copy()))
        b = torch.nn.Parameter(torch.tensor(b0.copy()))
        opt = torch.optim.Adam([w, b], lr=LR)
        ft, yt = torch.tensor(fw), torch.tensor(yw)
        for _ in range(steps):
            loss = ((ft @ w.T + b - yt) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        return w.detach().numpy(), b.detach().numpy()
    
    
    class ControlledHeadTrackingAudit(OfflineAudit):
        """Window-fitted heads scored on the next batch, against a stable reference."""
    
        run = "post_v2_t5_c17_head_tracking-2026-09-23"
        report = "audit.json"
        schema = "t5_c17_controlled_head_tracking_v1"
    
        def specialist(self, lab):
            z = np.load(RUNS / SRC / f"{lab}_features_targets.npz")
            batches = [(z[f"F{k}"].astype(np.float32), z[f"Y{k}"].astype(np.float32))
                       for k in range(BATCHES)]
            w0 = z["head_weight"].astype(np.float32)
            b0 = z["head_bias"].astype(np.float32)
    
            f_all = np.concatenate([x[0] for x in batches], 0)
            y_all = np.concatenate([x[1] for x in batches], 0)
            w_star, b_star = ridge_solution(f_all.astype(np.float64), y_all.astype(np.float64), 1.0)
            # progress is scored on the whole feature support, not one batch's
            f_ref = f_all.astype(np.float64)
    
            out = {"reference": {"ridge_all6_lambda1": {"weight_norm": float(np.linalg.norm(w_star))}},
                   "windows": []}
            for end in range(1, BATCHES):
                next_f, next_y = batches[end]
                for win in WINDOWS:
                    sel = batches[max(0, end - win):end]
                    fw = np.concatenate([x[0] for x in sel], 0)
                    yw = np.concatenate([x[1] for x in sel], 0)
                    row = {"end_batch": end - 1, "next_batch": end, "window": win,
                           "actual_window_size": len(sel), "methods": {}}
                    row["methods"]["baseline"] = {
                        "window": metrics(yw, fw @ w0.T + b0),
                        "next": metrics(next_y, next_f @ w0.T + b0),
                        "progress": function_progress(f_ref, w0, b0, w0, b0, w_star, b_star)}
                    for steps in ADAM_STEPS:
                        wn, bn = adam_head(w0, b0, fw, yw, steps)
                        row["methods"][f"adam_{steps}"] = {
                            "window": metrics(yw, fw @ wn.T + bn),
                            "next": metrics(next_y, next_f @ wn.T + bn),
                            "progress": function_progress(f_ref, w0, b0, wn, bn, w_star, b_star),
                            "weight_norm": float(np.linalg.norm(wn))}
                    for l2 in RIDGES:
                        wr, br = ridge_solution(fw.astype(np.float64), yw.astype(np.float64), l2)
                        row["methods"][f"ridge_{l2}"] = {
                            "window": metrics(yw, fw @ wr.T + br),
                            "next": metrics(next_y, next_f @ wr.T + br),
                            "progress": function_progress(f_ref, w0, b0, wr, br, w_star, b_star),
                            "weight_norm": float(np.linalg.norm(wr))}
                    out["windows"].append(row)
            return out
    
        def analyze(self):
            report = {"schema": self.schema,
                      "specialists": {lab: self.specialist(lab) for lab in ORDER}}
            report["aggregate"] = self.aggregate(report["specialists"])
            return report
    
        def aggregate(self, spec):
            methods = ["baseline"] + [f"adam_{s}" for s in ADAM_STEPS] + [f"ridge_{r}" for r in RIDGES]
            agg = {}
            for win in WINDOWS:
                agg[str(win)] = {}
                for method in methods:
                    next_ev, window_ev, next_mse, prog, cosines = [], [], [], [], []
                    for lab in ORDER:
                        for row in spec[lab]["windows"]:
                            if row["window"] != win:
                                continue
                            m = row["methods"][method]
                            next_ev += m["next"]["ev"]
                            window_ev += m["window"]["ev"]
                            next_mse += m["next"]["mse"]
                            prog.append(m["progress"]["projected_progress_ratio"])
                            cosines.append(m["progress"]["cosine"])
                    agg[str(win)][method] = {
                        "window_ev_mean": float(np.mean(window_ev)),
                        "next_ev_mean": float(np.mean(next_ev)),
                        "next_ev_negative_fraction": float(np.mean(np.array(next_ev) < 0)),
                        "next_mse_mean": float(np.mean(next_mse)),
                        "projected_progress_mean": float(np.mean(prog)),
                        "function_cosine_mean": float(np.mean(cosines))}
            return agg
    
        def summarize(self, report):
            print(json.dumps(report["aggregate"], indent=2))
    
    
    if True:
        ControlledHeadTrackingAudit.main()

def run_post_v2_t5_c17_per_head_progress():
    """Run former post_v2_t5_c17_per_head_progress.py stage."""
    from pathlib import Path
    import json,numpy as np
    ROOT=Path(__file__).resolve().parents[4]
    SRC=ROOT/"runs/post_v2_t5_c13_conditioning-2026-09-23"
    OUT=ROOT/"runs/post_v2_t5_c17_head_tracking-2026-09-23"
    ORDER=("T","A","O","S");NB=6
    def ev(y,p):
     y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1);return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def ridge(F,Y,l2):
     A=np.c_[F,np.ones(len(F))];I=np.eye(A.shape[1]);I[-1,-1]=0;sol=np.linalg.solve(A.T@A+l2*I,A.T@Y);return sol[:-1].T,sol[-1]
    def prog(F,W0,b0,W,b,Ws,bs,j):
     p0=F@W0[j]+b0[j];pt=F@W[j]+b[j];ps=F@Ws[j]+bs[j];d=pt-p0;ds=ps-p0
     return float(d@ds/(ds@ds+1e-12))
    out={"schema":"c17_per_head_progress_v1","specialists":{}}
    for lab in ORDER:
     z=np.load(SRC/f"{lab}_features_targets.npz");B=[(z[f"F{k}"].astype(float),z[f"Y{k}"].astype(float)) for k in range(NB)]
     W0=z["head_weight"].astype(float);b0=z["head_bias"].astype(float)
     Fall=np.concatenate([x[0] for x in B]);Yall=np.concatenate([x[1] for x in B]);Ws,bs=ridge(Fall,Yall,1.0)
     rows=[]
     for win in (1,2,3,4):
      pe=[[],[],[],[]];nev=[[],[],[],[]]
      for end in range(1,NB):
       sel=B[max(0,end-win):end];Fw=np.concatenate([x[0] for x in sel]);Yw=np.concatenate([x[1] for x in sel]);Fn,Yn=B[end]
       W,b=ridge(Fw,Yw,1.0)
       for j in range(4):
        pe[j].append(prog(Fall,W0,b0,W,b,Ws,bs,j));nev[j].append(ev(Yn[:,j],Fn@W[j]+b[j]))
      rows.append({"window":win,"projected_progress_by_head":[float(np.mean(x)) for x in pe],"next_ev_by_head":[float(np.mean(x)) for x in nev]})
     out["specialists"][lab]=rows
    agg={}
    for win in (1,2,3,4):
     pe=[[],[],[],[]];ne=[[],[],[],[]]
     for lab in ORDER:
      r=next(x for x in out["specialists"][lab] if x["window"]==win)
      for j in range(4):pe[j].append(r["projected_progress_by_head"][j]);ne[j].append(r["next_ev_by_head"][j])
     agg[str(win)]={"projected_progress_by_head":[float(np.mean(x)) for x in pe],"next_ev_by_head":[float(np.mean(x)) for x in ne]}
    out["aggregate"]=agg
    (OUT/"per_head_progress.json").write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(agg,indent=2))

def run_post_v2_t5_c18_horizon_value_compare():
    """Run former post_v2_t5_c18_horizon_value_compare.py stage."""
    from pathlib import Path
    import sys,json,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUNS={"control":ROOT/"runs/post_v2_t5_c18_control-2026-09-23","ridge3":ROOT/"runs/post_v2_t5_c18_ridge3-2026-09-23"}
    ORDER=("T","A","O","S");G=.99
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    def obs_tensor(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def ev(y,p):
     y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1);return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def returns(R,D,segment=None):
     out=np.zeros_like(R)
     if segment is None:
      run=np.zeros_like(R[0])
      for t in range(len(R)-1,-1,-1):run=R[t]+G*run*(~D[t])[:,None];out[t]=run
     else:
      for st in range(0,len(R),segment):
       en=min(st+segment,len(R));run=np.zeros_like(R[0])
       for t in range(en-1,st-1,-1):run=R[t]+G*run*(~D[t])[:,None];out[t]=run
     return out
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=32;cfg.seed=0;env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
      o,_=env.reset(seed=0);o=obs_tensor(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
      out={}
      for tag,run in RUNS.items():
       out[tag]={}
       for bi,lab in enumerate(ORDER):
        m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(run/f"{lab}_snap_25.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
        w=torch.tensor(PREFS[lab],device="cuda").repeat(32,1);cur,_=env.reset(seed=1150000+bi*1000);cur=obs_tensor(cur).cuda();R=[];D=[];V=[]
        with torch.no_grad():
         for _ in range(64):
          V.append(m.value_with_preference(cur,w).cpu().numpy());a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
          R.append(normalized_objective_vector(terms(raw,names),shape=(32,))*env.unwrapped.step_dt);D.append((te|tr).cpu().numpy());cur=obs_tensor(nxt).cuda()
        R=np.asarray(R);D=np.asarray(D,bool);V=np.asarray(V);H32=returns(R,D,32);MC64=returns(R,D,None)
        out[tag][lab]={
          "h32_ev":[ev(H32[:,:,j],V[:,:,j]) for j in range(4)],
          "h32_bias":[float(np.mean(V[:,:,j]-H32[:,:,j])) for j in range(4)],
          "mc64_ev":[ev(MC64[:,:,j],V[:,:,j]) for j in range(4)],
          "mc64_bias":[float(np.mean(V[:,:,j]-MC64[:,:,j])) for j in range(4)]
        }
      p=RUNS["ridge3"]/"horizon_value_compare.json";p.write_text(json.dumps(out,indent=2)+"\n")
      for tag in out:
       h=[];m=[];hb=[];mb=[]
       for lab in ORDER:h+=out[tag][lab]["h32_ev"];m+=out[tag][lab]["mc64_ev"];hb+=out[tag][lab]["h32_bias"];mb+=out[tag][lab]["mc64_bias"]
       print(tag,"H32 EV",np.mean(h),"MC64 EV",np.mean(m),"|bias|",np.mean(np.abs(hb)),np.mean(np.abs(mb)))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c18_online_head_tracker():
    """Run former post_v2_t5_c18_online_head_tracker.py stage."""
    import argparse,json,sys,traceback
    from pathlib import Path
    from collections import deque
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    ORDER=("T","A","O","S");SNAPS=(0,1,5,10,25)
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    GAMMA=.99
    def obs_tensor(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def flat_grad(gs,params):
        return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,params)])
    def cos(a,b):return float((a@b)/(a.norm()*b.norm()+1e-12))
    def corrmat(x):return np.corrcoef(np.asarray(x,float),rowvar=False).tolist()
    def trunc_mc(rt,dt):
        out=torch.zeros_like(rt);running=torch.zeros_like(rt[-1])
        for t in range(len(rt)-1,-1,-1):
            running=rt[t]+GAMMA*running*(~dt[t]).unsqueeze(-1);out[t]=running
        return out
    def ev_np(y,p):
        y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
        return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def ridge_fit(F,Y,l2=1.0):
        # float64 CPU for stable solve; no regularization on bias
        Fc=F.detach().cpu().double().numpy();Yc=Y.detach().cpu().double().numpy()
        A=np.c_[Fc,np.ones(len(Fc))];I=np.eye(A.shape[1]);I[-1,-1]=0
        sol=np.linalg.solve(A.T@A+l2*I,A.T@Yc)
        return torch.tensor(sol[:-1].T,dtype=F.dtype,device=F.device),torch.tensor(sol[-1],dtype=F.dtype,device=F.device)
    def main():
        ap=argparse.ArgumentParser()
        ap.add_argument("--output",type=Path,required=True)
        ap.add_argument("--tracker",choices=("adam1","ridge3"),required=True)
        ap.add_argument("--updates",type=int,default=25);ap.add_argument("--num-envs",type=int,default=8);ap.add_argument("--horizon",type=int,default=32)
        ap.add_argument("--checkpoint",type=Path,default=Path("runs/m0_1_seed0_2026-09-22/model_299.pt"))
        ap.add_argument("--actor-lr",type=float,default=1e-3);ap.add_argument("--critic-lr",type=float,default=1e-4)
        ap.add_argument("--ridge-lambda",type=float,default=1.0);ap.add_argument("--recent-window",type=int,default=3)
        args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
          import gymnasium as gym,isaaclab_tasks
          from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
          from talon_rl.models.foundations.four_objective import T4SharedActorCritic,initialize_from_rsl_m01,vector_gae,scalarized_late_weighted_ppo
          from talon_rl.rewards.objectives import normalized_objective_vector
          cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.seed=0;env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
          obs,_=env.reset(seed=0);obs=obs_tensor(obs).cuda();ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
          all_logs={};snap_paths={}
          for idx,label in enumerate(ORDER):
            torch.manual_seed(41000+idx);np.random.seed(41000+idx)
            m=T4SharedActorCritic(obs.shape[-1],ad).cuda();initialize_from_rsl_m01(m,args.checkpoint,device="cpu",critic_head_init="zero")
            # freeze body; online repair changes head tracker only
            for n,p in m.named_parameters():
                if n.startswith("critic_body"):p.requires_grad_(False)
            actor_params=[p for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std"]
            head_params=[m.critic_head.weight,m.critic_head.bias]
            actor_opt=torch.optim.Adam(actor_params,lr=args.actor_lr)
            head_opt=torch.optim.Adam(head_params,lr=args.critic_lr)
            recent=deque(maxlen=args.recent_window)
            w=torch.as_tensor(np.repeat(PREFS[label][None,:],args.num_envs,axis=0),device="cuda")
            cur,_=env.reset(seed=410001+idx*1000);cur=obs_tensor(cur).cuda()
            logs=[]
            def save_snap(tag):
                p=args.output.parent/f"{label}_snap_{tag}.pt"
                torch.save({"model":m.state_dict(),"specialist":label,"snapshot":tag,"tracker":args.tracker},p);snap_paths[f"{label}:{tag}"]=str(p)
            save_snap(0)
            for update in range(1,args.updates+1):
                ob=[];pre=[];old=[];rw=[];dn=[];surv=np.ones(args.num_envs,bool)
                for _ in range(args.horizon):
                    with torch.no_grad():a,lp,u=m.act_with_preference_latent(cur,w)
                    nxt,_,term,trunc,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                    vec=normalized_objective_vector(terms(raw,names),shape=(args.num_envs,))
                    ob.append(cur);pre.append(u);old.append(lp);rw.append(torch.as_tensor(vec,device="cuda")*env.unwrapped.step_dt);dn.append((term|trunc).cuda())
                    surv &= ~(term|trunc).cpu().numpy();cur=obs_tensor(nxt).cuda()
                fo=torch.cat(ob);fu=torch.cat(pre);fold=torch.cat(old);fw=w.repeat(args.horizon,1)
                rt=torch.stack(rw);dt=torch.stack(dn).bool();target_ret=trunc_mc(rt,dt).reshape(-1,4).detach()
                # frozen body features for explicit head tracker
                with torch.no_grad():F=m.critic_body(m._with_w(fo,fw)).detach()
                recent.append((F.detach().cpu(),target_ret.detach().cpu()))
                # pre-refresh value quality
                with torch.no_grad():vpre=m.critic_head(F);pre_ev=[ev_np(target_ret[:,j].cpu(),vpre[:,j].cpu()) for j in range(4)]
                # controlled head update
                if args.tracker=="adam1":
                    head_opt.zero_grad(set_to_none=True);pred=m.critic_head(F);loss=((pred-target_ret)**2).mean();loss.backward();head_opt.step()
                else:
                    FF=torch.cat([x[0] for x in recent],0).cuda();YY=torch.cat([x[1] for x in recent],0).cuda()
                    Wfit,bfit=ridge_fit(FF,YY,args.ridge_lambda)
                    with torch.no_grad():m.critic_head.weight.copy_(Wfit);m.critic_head.bias.copy_(bfit)
                with torch.no_grad():
                    vpost=m.critic_head(F);post_ev=[ev_np(target_ret[:,j].cpu(),vpost[:,j].cpu()) for j in range(4)]
                    post_bias=[float((vpost[:,j]-target_ret[:,j]).mean()) for j in range(4)]
                # actor advantage uses refreshed value head; body remains frozen
                with torch.no_grad():
                    vt=m.value_with_preference(fo,fw).reshape(args.horizon,args.num_envs,4)
                    nv=m.value_with_preference(cur,w)
                    adv,_=vector_gae(rt,vt,nv,dt,lam=.95)
                logp=m.logp_from_pre_tanh_with_preference(fo,fw,fu);ratio=torch.exp(logp-fold.detach())
                ratio_maxerr=float((ratio-1).abs().max())
                if ratio_maxerr>1e-4 or not torch.isfinite(ratio).all():raise RuntimeError(f"ratio invariant failed {ratio_maxerr}")
                advflat=adv.reshape(-1,4).detach()
                # semantic actor gradient geometry BEFORE actor step
                gobj=[]
                for j in range(4):
                    lj=-(ratio*advflat[:,j]).mean()
                    gobj.append(flat_grad(torch.autograd.grad(lj,actor_params,retain_graph=True,allow_unused=True),actor_params).detach())
                al=scalarized_late_weighted_ppo(ratio,advflat,fw)
                actor_opt.zero_grad(set_to_none=True);al.backward();actor_opt.step()
                snap_tag=0 if update==1 else update-1
                if snap_tag in SNAPS:
                    logs.append({
                      "snapshot":snap_tag,"update":update,"pre_ev":pre_ev,"post_ev":post_ev,"post_bias":post_bias,
                      "adv_mean":advflat.mean(0).cpu().tolist(),"adv_std":advflat.std(0).cpu().tolist(),"adv_corr":corrmat(advflat.cpu().numpy()),
                      "objective_grad_norm":[float(g.norm()) for g in gobj],
                      "objective_grad_cosine":[[cos(gobj[i],gobj[j]) for j in range(4)] for i in range(4)],
                      "survival":float(surv.mean()),"recent_buffer_len":len(recent),
                      "head_weight_norm":float(m.critic_head.weight.norm()),"head_bias_norm":float(m.critic_head.bias.norm())
                    })
                if update in SNAPS:save_snap(update)
            all_logs[label]=logs
          report={"schema":"t5_c18_online_head_tracker_v1","status":"MEASUREMENT_COMPLETE","tracker":args.tracker,"critic_body_frozen":True,
                  "horizon":args.horizon,"updates":args.updates,"env_steps":args.horizon*args.updates,"actor_lr":args.actor_lr,"critic_lr":args.critic_lr,
                  "ridge_lambda":args.ridge_lambda,"recent_window":args.recent_window,"specialist_logs":all_logs,"snapshot_paths":snap_paths}
          args.output.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps({"status":report["status"],"tracker":args.tracker},indent=2))
        except BaseException as e:
          args.output.with_name(args.output.stem+".ERROR.json").write_text(json.dumps({"error":str(e),"traceback":traceback.format_exc()},indent=2));raise
        finally:
          if env is not None:env.close()
          app.close()
    if True:main()

def run_post_v2_t5_c18_terminal_compare():
    """Run former post_v2_t5_c18_terminal_compare.py stage."""
    from pathlib import Path
    import sys,json,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUNS={"control":ROOT/"runs/post_v2_t5_c18_control-2026-09-23","ridge3":ROOT/"runs/post_v2_t5_c18_ridge3-2026-09-23"}
    ORDER=("T","A","O","S");PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    IDX={"T":0,"A":1,"O":2,"S":3};H=(1,2,4,8,16,32);G=.99
    def obs_tensor(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def ev(y,p):
     y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1);return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def flat(gs,ps):return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,ps)])
    def setflat(ps,base,delta):
     o=0
     with torch.no_grad():
      for p,b in zip(ps,base):
       n=p.numel();p.copy_(b+delta[o:o+n].view_as(p));o+=n
    def cloneps(ps):return [p.detach().clone() for p in ps]
    def tilt(q):
     _,x,y,_=[q[:,i] for i in range(4)]
     return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def rollout_metric(env,m,w,seed):
     cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda();robot=env.unwrapped.scene["robot"];prev=torch.zeros((len(cur),env.unwrapped.action_manager.total_action_dim),device="cuda")
     rows=[];done=np.zeros(len(cur),bool)
     with torch.no_grad():
      for _ in range(32):
       a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a);data=robot.data
       rows.append({"ang_vel_xy":float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),"tilt_deg":float(tilt(data.root_quat_w).mean()),"action_rate":float(torch.linalg.vector_norm(a-prev,dim=-1).mean()),"abs_lin_vel_z":float(data.root_lin_vel_b[:,2].abs().mean())})
       done|=(te|tr).cpu().numpy();prev=a;cur=obs_tensor(nxt).cuda()
     return {str(h):{k:float(np.mean([r[k] for r in rows[:h]])) for k in rows[0]} for h in H},float(1-done.mean())
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic,vector_gae
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=32;cfg.seed=0;env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
      obs,_=env.reset(seed=0);obs=obs_tensor(obs).cuda();od=obs.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
      out={}
      for tag,run in RUNS.items():
       critic={};pert=[];endrows=[]
       models={}
       for lab in ORDER:
        m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(run/f"{lab}_snap_25.pt",map_location="cuda",weights_only=False)["model"]);m.eval();models[lab]=m
       # critic terminal EV on same fresh 64-step rollouts
       for bi,lab in enumerate(ORDER):
        m=models[lab];w=torch.tensor(PREFS[lab],device="cuda").repeat(32,1)
        cur,_=env.reset(seed=1110000+bi*1000);cur=obs_tensor(cur).cuda();R=[];D=[];V=[]
        with torch.no_grad():
         for _ in range(64):
          V.append(m.value_with_preference(cur,w).cpu().numpy());a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a)
          raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);R.append(normalized_objective_vector(terms(raw,names),shape=(32,))*env.unwrapped.step_dt);D.append((te|tr).cpu().numpy());cur=obs_tensor(nxt).cuda()
        R=np.asarray(R);D=np.asarray(D,bool);V=np.asarray(V);MC=np.zeros_like(R);rr=np.zeros_like(R[0])
        for t in range(63,-1,-1):rr=R[t]+G*rr*(~D[t])[:,None];MC[t]=rr
        critic[lab]={"ev":[ev(MC[:,:,j],V[:,:,j]) for j in range(4)],"bias":[float(np.mean(V[:,:,j]-MC[:,:,j])) for j in range(4)]}
       # A/O raw gradient perturbations at terminal
       for lab in ("A","O"):
        bi=ORDER.index(lab);m=models[lab];w=torch.tensor(PREFS[lab],device="cuda").repeat(32,1);j=IDX[lab]
        cur,_=env.reset(seed=1120000+bi*1000);cur=obs_tensor(cur).cuda();ob=[];pre=[];old=[];rw=[];val=[];dn=[]
        for _ in range(16):
         with torch.no_grad():a,lp,u=m.act_with_preference_latent(cur,w);v=m.value_with_preference(cur,w)
         nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);vec=normalized_objective_vector(terms(raw,names),shape=(32,))
         ob.append(cur);pre.append(u);old.append(lp);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt);val.append(v);dn.append((te|tr).cuda());cur=obs_tensor(nxt).cuda()
        with torch.no_grad():nv=m.value_with_preference(cur,w)
        rt=torch.stack(rw);vt=torch.stack(val);dt=torch.stack(dn).bool();adv,_=vector_gae(rt,vt,nv,dt,lam=.95)
        fo=torch.cat(ob);fu=torch.cat(pre);fold=torch.cat(old);fw=w.repeat(16,1);ratio=torch.exp(m.logp_from_pre_tanh_with_preference(fo,fw,fu)-fold.detach())
        aps=[p for n,p in m.named_parameters() if n.startswith("actor_")]
        loss=-(ratio*adv.reshape(-1,4)[:,j].detach()).mean();g=flat(torch.autograd.grad(loss,aps,allow_unused=True),aps).detach()
        basevec=torch.cat([p.detach().reshape(-1) for p in aps]);delta=-g/(g.norm()+1e-12)*(1e-4*(basevec.norm()+1e-12))
        pm=T4SharedActorCritic(od,ad).cuda();pm.load_state_dict(torch.load(run/f"{lab}_snap_25.pt",map_location="cuda",weights_only=False)["model"]);paps=[p for n,p in pm.named_parameters() if n.startswith("actor_")];setflat(paps,cloneps(paps),delta);pm.eval()
        metric="ang_vel_xy" if lab=="A" else "tilt_deg";suites=[]
        for ss in range(3):
         seed=1130000+bi*1000+ss;b,_=rollout_metric(env,m,w,seed);p,_=rollout_metric(env,pm,w,seed);suites.append({"suite":ss,"baseline":b,"perturbed":p,"metric":metric})
        pert.append({"branch":lab,"grad_norm":float(g.norm()),"suites":suites})
       # endpoint matched suites
       for suite in range(4):
        seed=1140000+suite
        for lab in ORDER:
         m=models[lab];w=torch.tensor(PREFS[lab],device="cuda").repeat(32,1);cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda();prev=torch.zeros((32,ad),device="cuda");norm=[];phys=[];done=np.zeros(32,bool)
         with torch.no_grad():
          for _ in range(64):
           a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);norm.append(normalized_objective_vector(terms(raw,names),shape=(32,)).mean(0));data=robot.data;cmd=env.unwrapped.command_manager.get_command("base_velocity")
           phys.append({"vx_error":float((data.root_lin_vel_b[:,0]-cmd[:,0]).abs().mean()),"wz_error":float((data.root_ang_vel_b[:,2]-cmd[:,2]).abs().mean()),"ang_vel_xy":float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),"tilt_deg":float(tilt(data.root_quat_w).mean()),"action_rate":float(torch.linalg.vector_norm(a-prev,dim=-1).mean()),"abs_lin_vel_z":float(data.root_lin_vel_b[:,2].abs().mean())})
           done|=(te|tr).cpu().numpy();prev=a;cur=obs_tensor(nxt).cuda()
         n=np.asarray(norm).mean(0);pm={k:float(np.mean([x[k] for x in phys])) for k in phys[0]}
         endrows.append({"suite":suite,"policy":lab,"normalized_objective_mean":n.tolist(),"scalarized":{e:float(n@PREFS[e]) for e in ORDER},"physical":pm,"survival":float(1-done.mean())})
       ow={};pw={};sw={};pk={"A":"ang_vel_xy","O":"tilt_deg","S":"action_rate"}
       for i,lab in enumerate(ORDER):
        aa=[];bb=[];cc=[]
        for suite in range(4):
         rr=[r for r in endrows if r["suite"]==suite]
         aa.append(max(rr,key=lambda r:r["normalized_objective_mean"][i])["policy"]==lab)
         bb.append((min(rr,key=lambda r:r["physical"]["vx_error"]+r["physical"]["wz_error"])["policy"]==lab) if lab=="T" else min(rr,key=lambda r:r["physical"][pk[lab]])["policy"]==lab)
         cc.append(max(rr,key=lambda r:r["scalarized"][lab])["policy"]==lab)
        ow[lab]=float(np.mean(aa));pw[lab]=float(np.mean(bb));sw[lab]=float(np.mean(cc))
       vb={p:float(np.mean([r["physical"]["abs_lin_vel_z"] for r in endrows if r["policy"]==p])) for p in ORDER};vr={p:vb[p]/(vb["T"]+1e-12) for p in ORDER}
       gates={"objective_winner_fraction":ow,"physical_winner_fraction":pw,"diagonal_scalarized_winner_fraction":sw,"min_survival":min(r["survival"] for r in endrows),"vertical_ratio_to_tracking_policy":vr}
       out[tag]={"critic":critic,"perturbations":pert,"endpoint_rows":endrows,"gates":gates}
      target=RUNS["ridge3"]/"terminal_compare.json";target.write_text(json.dumps(out,indent=2)+"\n")
      print(json.dumps({k:{"gates":v["gates"],"critic":v["critic"],"pert_n":len(v["perturbations"])} for k,v in out.items()},indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c19_actor_coupling_train():
    """Run former post_v2_t5_c19_actor_coupling_train.py stage."""
    import argparse,json,sys,traceback
    from pathlib import Path
    from collections import deque
    import numpy as np,torch
    
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    ORDER=("T","A","O","S")
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    G=.99
    def obs_tensor(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def trunc_mc(rt,dt):
        out=torch.zeros_like(rt);run=torch.zeros_like(rt[-1])
        for t in range(len(rt)-1,-1,-1):run=rt[t]+G*run*(~dt[t]).unsqueeze(-1);out[t]=run
        return out
    def ev_np(y,p):
        y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
        return float(1-np.var(y-p)/(np.var(y)+1e-12))
    def ridge_fit(F,Y,l2):
        Fc=F.detach().cpu().double().numpy();Yc=Y.detach().cpu().double().numpy()
        A=np.c_[Fc,np.ones(len(Fc))];I=np.eye(A.shape[1]);I[-1,-1]=0
        sol=np.linalg.solve(A.T@A+l2*I,A.T@Yc)
        return torch.tensor(sol[:-1].T,dtype=F.dtype,device=F.device),torch.tensor(sol[-1],dtype=F.dtype,device=F.device)
    def main():
        ap=argparse.ArgumentParser()
        ap.add_argument("--output",type=Path,required=True)
        ap.add_argument("--actor-update",choices=("on","off"),required=True)
        ap.add_argument("--updates",type=int,default=25);ap.add_argument("--num-envs",type=int,default=8);ap.add_argument("--horizon",type=int,default=32)
        ap.add_argument("--checkpoint",type=Path,default=Path("runs/m0_1_seed0_2026-09-22/model_299.pt"))
        ap.add_argument("--actor-lr",type=float,default=1e-3);ap.add_argument("--ridge-lambda",type=float,default=1.0);ap.add_argument("--recent-window",type=int,default=3)
        args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        env=None
        try:
          import gymnasium as gym,isaaclab_tasks
          from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
          from talon_rl.models.foundations.four_objective import T4SharedActorCritic,initialize_from_rsl_m01,vector_gae,scalarized_late_weighted_ppo
          from talon_rl.rewards.objectives import normalized_objective_vector
          cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.seed=0;env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
          o,_=env.reset(seed=0);o=obs_tensor(o).cuda();ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
          all_logs={}
          for idx,label in enumerate(ORDER):
            torch.manual_seed(51000+idx);np.random.seed(51000+idx)
            m=T4SharedActorCritic(o.shape[-1],ad).cuda();initialize_from_rsl_m01(m,args.checkpoint,device="cpu",critic_head_init="zero")
            for n,p in m.named_parameters():
                if n.startswith("critic_body"):p.requires_grad_(False)
            actor_params=[p for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std"]
            actor_opt=torch.optim.Adam(actor_params,lr=args.actor_lr)
            w=torch.tensor(np.repeat(PREFS[label][None,:],args.num_envs,axis=0),device="cuda")
            recent=deque(maxlen=args.recent_window)
            cur,_=env.reset(seed=510001+idx*1000);cur=obs_tensor(cur).cuda()
            logs=[]
            def save_snap(tag):
                torch.save({"model":m.state_dict(),"specialist":label,"snapshot":tag,"actor_update":args.actor_update},
                           args.output.parent/f"{label}_snap_{tag}.pt")
            save_snap(0)
            prev_F=None;prev_Y=None;prev_W=m.critic_head.weight.detach().clone();prev_b=m.critic_head.bias.detach().clone()
            prev_actor=torch.cat([p.detach().reshape(-1) for p in actor_params])
            for update in range(1,args.updates+1):
                ob=[];pre=[];old=[];rw=[];dn=[];acts=[]
                for _ in range(args.horizon):
                    with torch.no_grad():a,lp,u=m.act_with_preference_latent(cur,w)
                    nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
                    vec=normalized_objective_vector(terms(raw,names),shape=(args.num_envs,))
                    ob.append(cur);pre.append(u);old.append(lp);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt);dn.append((te|tr).cuda());acts.append(a)
                    cur=obs_tensor(nxt).cuda()
                fo=torch.cat(ob);fu=torch.cat(pre);fold=torch.cat(old);fw=w.repeat(args.horizon,1);fa=torch.cat(acts)
                rt=torch.stack(rw);dt=torch.stack(dn).bool();Y=trunc_mc(rt,dt).reshape(-1,4).detach()
                with torch.no_grad():F=m.critic_body(m._with_w(fo,fw)).detach()
                recent.append((F.detach().cpu(),Y.detach().cpu()))
                with torch.no_grad():pre=m.critic_head(F);pre_ev=[ev_np(Y[:,j].cpu(),pre[:,j].cpu()) for j in range(4)]
                FF=torch.cat([x[0] for x in recent],0).cuda();YY=torch.cat([x[1] for x in recent],0).cuda()
                Wfit,bfit=ridge_fit(FF,YY,args.ridge_lambda)
                with torch.no_grad():m.critic_head.weight.copy_(Wfit);m.critic_head.bias.copy_(bfit);post=m.critic_head(F)
                post_ev=[ev_np(Y[:,j].cpu(),post[:,j].cpu()) for j in range(4)]
                # drift metrics before optional actor update
                Wnow=m.critic_head.weight.detach();bnow=m.critic_head.bias.detach()
                feat_mean=F.mean(0);feat_std=F.std(0);tar_mean=Y.mean(0);tar_std=Y.std(0);act_mean=fa.mean(0);act_std=fa.std(0)
                row={"update":update,"pre_ev":pre_ev,"post_ev":post_ev,"buffer_len":len(recent),
                     "feature_mean_norm":float(feat_mean.norm()),"feature_std_norm":float(feat_std.norm()),
                     "target_mean":tar_mean.cpu().tolist(),"target_std":tar_std.cpu().tolist(),
                     "action_mean_norm":float(act_mean.norm()),"action_std_norm":float(act_std.norm()),
                     "head_weight_norm":float(Wnow.norm()),"head_solution_drift":float((Wnow-prev_W).norm()),
                     "head_bias_drift":float((bnow-prev_b).norm())}
                if prev_F is not None:
                    row["feature_mean_drift"]=float((feat_mean-prev_F[0]).norm())
                    row["feature_std_drift"]=float((feat_std-prev_F[1]).norm())
                    row["target_mean_drift"]=float((tar_mean-prev_Y[0]).norm())
                    row["target_std_drift"]=float((tar_std-prev_Y[1]).norm())
                # optional actor update from refreshed critic
                if args.actor_update=="on":
                    with torch.no_grad():
                        vt=m.value_with_preference(fo,fw).reshape(args.horizon,args.num_envs,4);nv=m.value_with_preference(cur,w)
                        adv,_=vector_gae(rt,vt,nv,dt,lam=.95)
                    ratio=torch.exp(m.logp_from_pre_tanh_with_preference(fo,fw,fu)-fold.detach())
                    if float((ratio-1).abs().max())>1e-4:raise RuntimeError("ratio invariant")
                    al=scalarized_late_weighted_ppo(ratio,adv.reshape(-1,4).detach(),fw)
                    actor_opt.zero_grad(set_to_none=True);al.backward();actor_opt.step()
                actor_now=torch.cat([p.detach().reshape(-1) for p in actor_params])
                row["actor_param_drift"]=float((actor_now-prev_actor).norm())
                prev_actor=actor_now.clone();prev_W=Wnow.clone();prev_b=bnow.clone();prev_F=(feat_mean.clone(),feat_std.clone());prev_Y=(tar_mean.clone(),tar_std.clone())
                logs.append(row);save_snap(update)
            all_logs[label]=logs
          out={"schema":"t5_c19_actor_coupling_train_v1","status":"COMPLETE","actor_update":args.actor_update,"recent_window":args.recent_window,
               "ridge_lambda":args.ridge_lambda,"horizon":args.horizon,"updates":args.updates,"specialist_logs":all_logs}
          args.output.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps({"status":"COMPLETE","actor_update":args.actor_update},indent=2))
        except BaseException as e:
          args.output.with_name(args.output.stem+".ERROR.json").write_text(json.dumps({"error":str(e),"traceback":traceback.format_exc()},indent=2));raise
        finally:
          if env is not None:env.close()
          app.close()
    if True:main()

def run_post_v2_t5_c19_replay_audit():
    """Run former post_v2_t5_c19_replay_audit.py stage."""
    from pathlib import Path
    import sys,json,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUNS={"frozen":ROOT/"runs/post_v2_t5_c19_actor_frozen-2026-09-23","moving":ROOT/"runs/post_v2_t5_c19_actor_moving-2026-09-23"}
    ORDER=("T","A","O","S");SNAPS=tuple(range(26));G=.99
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
    def obs_tensor(x):
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
      for t in range(len(R)-1,-1,-1):run=R[t]+G*run*(~D[t])[:,None];out[t]=run
     else:
      for st in range(0,len(R),seg):
       en=min(st+seg,len(R));run=np.zeros_like(R[0])
       for t in range(en-1,st-1,-1):run=R[t]+G*run*(~D[t])[:,None];out[t]=run
     return out
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=8;cfg.seed=0;env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
      o,_=env.reset(seed=0);o=obs_tensor(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
      out={"schema":"t5_c19_replay_audit_v1","snapshots":list(SNAPS),"arms":{}}
      for arm,run in RUNS.items():
       armout={}
       for bi,lab in enumerate(ORDER):
        w=torch.tensor(PREFS[lab],device="cuda").repeat(8,1);rows=[];base_pred=None;base_feat_mean=None;base_feat_std=None;base_action_mean=None;base_action_std=None
        for snap in SNAPS:
         m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(run/f"{lab}_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
         cur,_=env.reset(seed=610000+bi*1000);cur=obs_tensor(cur).cuda();R=[];D=[];V=[];F=[];A=[]
         with torch.no_grad():
          for _ in range(64):
           fw=w;mfeat=m.critic_body(m._with_w(cur,fw));v=m.critic_head(mfeat);a=m.act_inference_with_preference(cur,w)
           nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
           R.append(normalized_objective_vector(terms(raw,names),shape=(8,))*env.unwrapped.step_dt);D.append((te|tr).cpu().numpy());V.append(v.cpu().numpy());F.append(mfeat.cpu().numpy());A.append(a.cpu().numpy());cur=obs_tensor(nxt).cuda()
         R=np.asarray(R);D=np.asarray(D,bool);V=np.asarray(V);F=np.concatenate(F,0);A=np.concatenate(A,0);H32=ret(R,D,32);MC=ret(R,D,None)
         fm=F.mean(0);fs=F.std(0);am=A.mean(0);ast=A.std(0);pred=V.reshape(-1,4)
         if base_pred is None:base_pred=pred.copy();base_feat_mean=fm.copy();base_feat_std=fs.copy();base_action_mean=am.copy();base_action_std=ast.copy()
         rows.append({
          "snapshot":snap,
          "h32_ev":[ev(H32[:,:,j],V[:,:,j]) for j in range(4)],
          "mc64_ev":[ev(MC[:,:,j],V[:,:,j]) for j in range(4)],
          "h32_bias":[float(np.mean(V[:,:,j]-H32[:,:,j])) for j in range(4)],
          "mc64_bias":[float(np.mean(V[:,:,j]-MC[:,:,j])) for j in range(4)],
          "feature_mean_drift_from_u0":float(np.linalg.norm(fm-base_feat_mean)),
          "feature_std_drift_from_u0":float(np.linalg.norm(fs-base_feat_std)),
          "action_mean_drift_from_u0":float(np.linalg.norm(am-base_action_mean)),
          "action_std_drift_from_u0":float(np.linalg.norm(ast-base_action_std)),
          "prediction_drift_from_u0":float(np.linalg.norm(pred-base_pred)/(np.linalg.norm(base_pred)+1e-12)),
          "survival":float(1-D.any(0).mean())
         })
        armout[lab]=rows
       out["arms"][arm]=armout
      # paired aggregate trajectories
      agg={}
      for arm in RUNS:
       traj=[]
       for snap in SNAPS:
        h=[];mc=[];hb=[];mb=[];fd=[];fs=[];ad=[];asd=[];pd=[];sv=[]
        for lab in ORDER:
         r=out["arms"][arm][lab][snap];h+=r["h32_ev"];mc+=r["mc64_ev"];hb+=r["h32_bias"];mb+=r["mc64_bias"];fd.append(r["feature_mean_drift_from_u0"]);fs.append(r["feature_std_drift_from_u0"]);ad.append(r["action_mean_drift_from_u0"]);asd.append(r["action_std_drift_from_u0"]);pd.append(r["prediction_drift_from_u0"]);sv.append(r["survival"])
        traj.append({"snapshot":snap,"h32_ev_mean":float(np.mean(h)),"h32_negative_fraction":float(np.mean(np.array(h)<0)),
                     "mc64_ev_mean":float(np.mean(mc)),"mc64_negative_fraction":float(np.mean(np.array(mc)<0)),
                     "h32_mean_abs_bias":float(np.mean(np.abs(hb))),"mc64_mean_abs_bias":float(np.mean(np.abs(mb))),
                     "feature_mean_drift_mean":float(np.mean(fd)),"feature_std_drift_mean":float(np.mean(fs)),
                     "action_mean_drift_mean":float(np.mean(ad)),"action_std_drift_mean":float(np.mean(asd)),
                     "prediction_drift_mean":float(np.mean(pd)),"min_survival":float(np.min(sv))})
       agg[arm]=traj
      out["aggregate"]=agg
      target=RUNS["moving"]/"replay_audit.json";target.write_text(json.dumps(out,indent=2)+"\n")
      for arm in ("frozen","moving"):
       print("\\n",arm)
       for s in (0,1,2,3,5,10,15,20,25):
        r=agg[arm][s];print(s,"H32",round(r["h32_ev_mean"],3),"MC",round(r["mc64_ev_mean"],3),"feat",round(r["feature_mean_drift_mean"],3),"act",round(r["action_mean_drift_mean"],3),"pred",round(r["prediction_drift_mean"],3),"surv",round(r["min_survival"],3))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

STAGES = {
    "post_v2_t5_c10_endpoint_compare": run_post_v2_t5_c10_endpoint_compare,
    "post_v2_t5_c10_exact_h32_target_gap": run_post_v2_t5_c10_exact_h32_target_gap,
    "post_v2_t5_c10_value_compare": run_post_v2_t5_c10_value_compare,
    "post_v2_t5_c11_critic_loss_geometry_audit": run_post_v2_t5_c11_critic_loss_geometry_audit,
    "post_v2_t5_c12_moving_batch_audit": run_post_v2_t5_c12_moving_batch_audit,
    "post_v2_t5_c12_optimal_head_transfer": run_post_v2_t5_c12_optimal_head_transfer,
    "post_v2_t5_c12_ridge_head_transfer": run_post_v2_t5_c12_ridge_head_transfer,
    "post_v2_t5_c13_collect_features": run_post_v2_t5_c13_collect_features,
    "post_v2_t5_c13_conditioning_regularization_audit": run_post_v2_t5_c13_conditioning_regularization_audit,
    "post_v2_t5_c13_offline_analyze": run_post_v2_t5_c13_offline_analyze,
    "post_v2_t5_c13_pcr_audit": run_post_v2_t5_c13_pcr_audit,
    "post_v2_t5_c14_collect_matched_features": run_post_v2_t5_c14_collect_matched_features,
    "post_v2_t5_c14_offline_analyze": run_post_v2_t5_c14_offline_analyze,
    "post_v2_t5_c15_sequential_projection": run_post_v2_t5_c15_sequential_projection,
    "post_v2_t5_c15_spectral_update_audit": run_post_v2_t5_c15_spectral_update_audit,
    "post_v2_t5_c16_function_space_progress": run_post_v2_t5_c16_function_space_progress,
    "post_v2_t5_c16_head_progress_audit": run_post_v2_t5_c16_head_progress_audit,
    "post_v2_t5_c17_controlled_head_tracking_audit": run_post_v2_t5_c17_controlled_head_tracking_audit,
    "post_v2_t5_c17_per_head_progress": run_post_v2_t5_c17_per_head_progress,
    "post_v2_t5_c18_horizon_value_compare": run_post_v2_t5_c18_horizon_value_compare,
    "post_v2_t5_c18_online_head_tracker": run_post_v2_t5_c18_online_head_tracker,
    "post_v2_t5_c18_terminal_compare": run_post_v2_t5_c18_terminal_compare,
    "post_v2_t5_c19_actor_coupling_train": run_post_v2_t5_c19_actor_coupling_train,
    "post_v2_t5_c19_replay_audit": run_post_v2_t5_c19_replay_audit,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
