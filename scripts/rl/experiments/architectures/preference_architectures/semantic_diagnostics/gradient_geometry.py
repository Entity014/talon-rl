"""Consolidated thematic experiment stages."""
from __future__ import annotations
import argparse

def run_post_v2_t5_c30_timestep_window():
    """Run former post_v2_t5_c30_timestep_window.py stage."""
    from pathlib import Path
    import sys,json,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c25_actor_updating25-2026-09-23"
    OUT=ROOT/"runs/post_v2_t5_c30_angular_timestep_window-2026-09-23";OUT.mkdir(parents=True,exist_ok=True)
    SNAPS=(10,25); NENV=8; NB=8; G=.99; J=1; T=8
    WREF=np.array([.1,.7,.1,.1],np.float32)
    WINDOWS={"t0":[0],"t1":[1],"t2":[2],"t3":[3],"t4_7":[4,5,6,7]}
    def ot(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names): return {n:raw[:,i] for i,n in enumerate(names)}
    def flat(gs,ps): return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,ps)])
    def cosine(a,b): return float(torch.dot(a,b)/(a.norm()*b.norm()+1e-12))
    def tilt_deg(q):
     _,x,y,_=[q[:,i] for i in range(4)]
     return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def contact_frac(env):
     try:
      sensor=env.unwrapped.scene.sensors["contact_forces"]
      f=sensor.data.net_forces_w
      if f.ndim==4:f=f[:,-1]
      return (torch.linalg.vector_norm(f,dim=-1)>1.0).float().mean(-1)
     except Exception:
      return torch.zeros(NENV,device="cuda")
    def rollout(env,m,w,mgr,seed,steps=8):
     from talon_rl.rewards.objectives import normalized_objective_vector
     robot=env.unwrapped.scene["robot"];cur,_=env.reset(seed=seed);cur=ot(cur).cuda();rows=[]
     with torch.no_grad():
      for t in range(steps):
       data=robot.data
       pre={"ang":torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).cpu().numpy().copy(),
            "tilt":tilt_deg(data.root_quat_w).cpu().numpy().copy(),
            "vx":data.root_lin_vel_b[:,0].cpu().numpy().copy(),
            "vz":data.root_lin_vel_b[:,2].cpu().numpy().copy(),
            "joint_vel":torch.linalg.vector_norm(data.joint_vel,dim=-1).cpu().numpy().copy(),
            "contact":contact_frac(env).cpu().numpy().copy()}
       a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a)
       raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
       vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
       pre["obj"]=vec[:,1].copy();rows.append(pre);cur=ot(nxt).cuda()
     return rows
    def perturb(cls,od,ad,snap,delta):
     pm=cls(od,ad).cuda();pm.load_state_dict(torch.load(RUN/f"A_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"]);pm.eval()
     ps=[p for n,p in pm.named_parameters() if n.startswith("actor_")];o=0
     with torch.no_grad():
      for p in ps:
       n=p.numel();p.add_(delta[o:o+n].view_as(p));o+=n
     return pm
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda()
      od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
      report={"schema":"c30_timestep_window_causal_v1","snapshots":{}}
      for snap in SNAPS:
       m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"A_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
       aps=[p for n,p in m.named_parameters() if n.startswith("actor_")];w=torch.tensor(WREF,device="cuda").repeat(NENV,1)
       wg={k:[] for k in WINDOWS};state={k:[] for k in WINDOWS}
       # exact per-window reward-to-policy gradient, no GAE/critic
       for bi in range(NB):
        cur,_=env.reset(seed=5010000+snap*10000+bi*211);cur=ot(cur).cuda()
        ratios=[];rews=[];states=[]
        for t in range(T):
         data=robot.data
         states.append({"ang":float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),
                        "tilt":float(tilt_deg(data.root_quat_w).mean()),
                        "vx":float(data.root_lin_vel_b[:,0].mean()),
                        "vz_abs":float(data.root_lin_vel_b[:,2].abs().mean()),
                        "joint_vel":float(torch.linalg.vector_norm(data.joint_vel,dim=-1).mean()),
                        "contact":float(contact_frac(env).mean())})
         with torch.no_grad():a,lp,u=m.act_with_preference_latent(cur,w)
         nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
         vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
         ratio=torch.exp(m.logp_from_pre_tanh_with_preference(cur,w,u)-lp.detach())
         ratios.append(ratio);rews.append(torch.tensor(vec[:,J],device="cuda")*env.unwrapped.step_dt);cur=ot(nxt).cuda()
        for name,idxs in WINDOWS.items():
         # each timestep reward attributed to action at same timestep; window aggregates those local terms
         loss=0.
         for t in idxs: loss += -(ratios[t]*rews[t].detach()).mean()
         loss=loss/len(idxs)
         g=flat(torch.autograd.grad(loss,aps,retain_graph=True,allow_unused=True),aps).detach();wg[name].append(g)
         state[name].append({k:float(np.mean([states[t][k] for t in idxs])) for k in states[0]})
       # H2 local reference = mean(t0,t1)
       ref=(torch.stack(wg["t0"]).mean(0)+torch.stack(wg["t1"]).mean(0))/2
       basevec=torch.cat([p.detach().reshape(-1) for p in aps]);snapout={}
       for name,gs in wg.items():
        g=torch.stack(gs).mean(0);delta=-g/(g.norm()+1e-12)*(1e-4*(basevec.norm()+1e-12));pm=perturb(T4SharedActorCritic,od,ad,snap,delta)
        da=[];do=[];dyn={k:[] for k in ("tilt","vx","vz_abs","joint_vel","contact")}
        idxs=WINDOWS[name]
        for ss in range(8):
         seed=5110000+snap*10000+ss*149;b=rollout(env,m,w,mgr,seed,T);p=rollout(env,pm,w,mgr,seed,T)
         for t in idxs:
          da.append(float(np.mean(p[t]["ang"]-b[t]["ang"])));do.append(float(np.mean(p[t]["obj"]-b[t]["obj"])))
          for k in dyn:dyn[k].append(float(np.mean(p[t][k]-b[t][k])))
        cos_pairs=[cosine(gs[i],gs[j]) for i in range(len(gs)) for j in range(i+1,len(gs))]
        snapout[name]={
          "norm":float(g.norm()),"cos_to_H2_local_ref":cosine(g,ref),
          "pairwise_cos_mean":float(np.mean(cos_pairs)),"pairwise_negative_fraction":float(np.mean(np.array(cos_pairs)<0)),
          "delta_ang_mean":float(np.mean(da)),"physical_correct_fraction":float(np.mean(np.array(da)<0)),
          "delta_objective_mean":float(np.mean(do)),"objective_correct_fraction":float(np.mean(np.array(do)>0)),
          "state_baseline_mean":{k:float(np.mean([x[k] for x in state[name]])) for k in state[name][0]},
          "perturb_downstream_delta":{k:float(np.mean(v)) for k,v in dyn.items()}
        }
       report["snapshots"][str(snap)]=snapout
      (OUT/"audit.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c30_window_core():
    """Run former post_v2_t5_c30_window_core.py stage."""
    from pathlib import Path
    import sys,json,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c25_actor_updating25-2026-09-23";OUT=ROOT/"runs/post_v2_t5_c30_angular_timestep_window-2026-09-23";OUT.mkdir(parents=True,exist_ok=True)
    SNAPS=(10,25);NENV=8;NB=8;J=1;T=8;WREF=np.array([.1,.7,.1,.1],np.float32)
    WINDOWS={"t0":[0],"t1":[1],"t2":[2],"t3":[3],"t4_7":[4,5,6,7]}
    def ot(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def flat(gs,ps):return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,ps)])
    def cos(a,b):return float(torch.dot(a,b)/(a.norm()*b.norm()+1e-12))
    def tilt(q):
     _,x,y,_=[q[:,i] for i in range(4)]
     return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def contact(env):
     try:
      s=env.unwrapped.scene.sensors["contact_forces"].data.net_forces_w
      if s.ndim==4:s=s[:,-1]
      return (torch.linalg.vector_norm(s,dim=-1)>1.0).float().mean(-1)
     except:return torch.zeros(NENV,device="cuda")
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
      rep={"schema":"c30_window_core_v1","snapshots":{}}
      for snap in SNAPS:
       m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"A_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
       aps=[p for n,p in m.named_parameters() if n.startswith("actor_")];w=torch.tensor(WREF,device="cuda").repeat(NENV,1)
       gs={k:[] for k in WINDOWS};ss={k:[] for k in WINDOWS}
       for bi in range(NB):
        cur,_=env.reset(seed=5310000+snap*10000+bi*211);cur=ot(cur).cuda();rat=[];rew=[];st=[]
        for t in range(T):
         d=robot.data
         st.append({"ang":float(torch.linalg.vector_norm(d.root_ang_vel_b[:,:2],dim=-1).mean()),"tilt":float(tilt(d.root_quat_w).mean()),"vx":float(d.root_lin_vel_b[:,0].mean()),"vy":float(d.root_lin_vel_b[:,1].mean()),"vz_abs":float(d.root_lin_vel_b[:,2].abs().mean()),"joint_vel":float(torch.linalg.vector_norm(d.joint_vel,dim=-1).mean()),"contact":float(contact(env).mean())})
         with torch.no_grad():a,lp,u=m.act_with_preference_latent(cur,w)
         nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
         rat.append(torch.exp(m.logp_from_pre_tanh_with_preference(cur,w,u)-lp.detach()));rew.append(torch.tensor(vec[:,J],device="cuda")*env.unwrapped.step_dt);cur=ot(nxt).cuda()
        for name,idx in WINDOWS.items():
         loss=sum([-(rat[t]*rew[t].detach()).mean() for t in idx])/len(idx)
         gs[name].append(flat(torch.autograd.grad(loss,aps,retain_graph=True,allow_unused=True),aps).detach())
         ss[name].append({k:float(np.mean([st[t][k] for t in idx])) for k in st[0]})
       ref=(torch.stack(gs["t0"]).mean(0)+torch.stack(gs["t1"]).mean(0))/2
       out={}
       prev=None
       for name in WINDOWS:
        G=torch.stack(gs[name]);g=G.mean(0);pcs=[cos(G[i],G[j]) for i in range(NB) for j in range(i+1,NB)]
        row={"norm":float(g.norm()),"cos_to_H2_local_ref":cos(g,ref),"pairwise_cos_mean":float(np.mean(pcs)),"negative_fraction":float(np.mean(np.array(pcs)<0)),"state":{k:float(np.mean([x[k] for x in ss[name]])) for k in ss[name][0]}}
        if prev is not None:row["cos_to_prev_window"]=cos(g,prev)
        out[name]=row;prev=g
       rep["snapshots"][str(snap)]=out
      (OUT/"core.json").write_text(json.dumps(rep,indent=2)+"\n");print(json.dumps(rep,indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c31_combo_single():
    """Run former post_v2_t5_c31_combo_single.py stage."""
    from pathlib import Path
    import sys,json,argparse,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c25_actor_updating25-2026-09-23";OUT=ROOT/"runs/post_v2_t5_c31_dynamics_source-2026-09-23";OUT.mkdir(parents=True,exist_ok=True)
    NENV=8;NB=20;J=1;T=4;WREF=np.array([.1,.7,.1,.1],np.float32)
    PAIRS=(("vz_abs","vxy"),("vz_abs","wx_abs"),("vz_abs","tilt"),("vxy","wx_abs"))
    def ot(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def flat(gs,ps):return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,ps)])
    def cos(a,b):return float(torch.dot(a,b)/(a.norm()*b.norm()+1e-12))
    def tilt(q):
     _,x,y,_=[q[:,i] for i in range(4)]
     return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def main():
     ap=argparse.ArgumentParser();ap.add_argument("--snap",type=int,required=True);a=ap.parse_args();snap=a.snap
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
      m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"A_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"]);m.eval();aps=[p for n,p in m.named_parameters() if n.startswith("actor_")];w=torch.tensor(WREF,device="cuda").repeat(NENV,1)
      rows=[]
      for bi in range(NB):
       cur,_=env.reset(seed=6010000+snap*10000+bi*211);cur=ot(cur).cuda();rat=[];rw=[];ff=[]
       for t in range(T):
        d=robot.data;ff.append({"vz_abs":float(d.root_lin_vel_b[:,2].abs().mean()),"vxy":float(torch.linalg.vector_norm(d.root_lin_vel_b[:,:2],dim=-1).mean()),"wx_abs":float(d.root_ang_vel_b[:,0].abs().mean()),"tilt":float(tilt(d.root_quat_w).mean())})
        with torch.no_grad():aa,lp,u=m.act_with_preference_latent(cur,w)
        nxt,_,te,tr,_=env.step(aa);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
        rat.append(torch.exp(m.logp_from_pre_tanh_with_preference(cur,w,u)-lp.detach()));rw.append(torch.tensor(vec[:,J],device="cuda")*env.unwrapped.step_dt);cur=ot(nxt).cuda()
       gg=[]
       for t in range(T):gg.append(flat(torch.autograd.grad(-(rat[t]*rw[t].detach()).mean(),aps,retain_graph=True,allow_unused=True),aps).detach().cpu())
       rows.append({"g":gg,"f":ff})
      ref=torch.stack([(r["g"][0]+r["g"][1])/2 for r in rows]).mean(0);out={"schema":"c31_combo_v1","snap":snap,"windows":{}}
      for t in (2,3):
       gs=[r["g"][t] for r in rows];src={}
       for a1,a2 in PAIRS:
        x1=np.array([r["f"][t][a1] for r in rows]);x2=np.array([r["f"][t][a2] for r in rows]);m1=np.median(x1);m2=np.median(x2)
        low=[gs[i] for i in range(NB) if x1[i]<=m1 and x2[i]<=m2];high=[gs[i] for i in range(NB) if x1[i]>m1 and x2[i]>m2]
        mix=[gs[i] for i in range(NB) if not (x1[i]<=m1 and x2[i]<=m2) and not (x1[i]>m1 and x2[i]>m2)]
        def st(q):return None if len(q)<2 else {"n":len(q),"cos":cos(torch.stack(q).mean(0),ref),"norm":float(torch.stack(q).mean(0).norm())}
        src[f"{a1}+{a2}"]={"low_low":st(low),"high_high":st(high),"mixed":st(mix),"medians":[float(m1),float(m2)]}
       out["windows"][str(t)]={"all_cos":cos(torch.stack(gs).mean(0),ref),"pairs":src}
      (OUT/f"combo_u{snap}.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c31_dynamics_source():
    """Run former post_v2_t5_c31_dynamics_source.py stage."""
    from pathlib import Path
    import sys,json,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c25_actor_updating25-2026-09-23"
    OUT=ROOT/"runs/post_v2_t5_c31_dynamics_source-2026-09-23";OUT.mkdir(parents=True,exist_ok=True)
    SNAPS=(10,25); NENV=8; NB=16; J=1; T=4
    WREF=np.array([.1,.7,.1,.1],np.float32)
    FEATURES=("vz_abs","tilt","wx_abs","wy_abs","joint_pos_norm","joint_vel_norm","vxy")
    def ot(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def flat(gs,ps):return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,ps)])
    def cos(a,b):return float(torch.dot(a,b)/(a.norm()*b.norm()+1e-12))
    def tilt(q):
     _,x,y,_=[q[:,i] for i in range(4)]
     return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def zscore(x):
     x=np.asarray(x,float);return (x-x.mean())/(x.std()+1e-12)
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda()
      od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
      rep={"schema":"c31_dynamics_source_v1","snapshots":{}}
      for snap in SNAPS:
       m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"A_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
       aps=[p for n,p in m.named_parameters() if n.startswith("actor_")];w=torch.tensor(WREF,device="cuda").repeat(NENV,1)
       rows=[]
       for bi in range(NB):
        cur,_=env.reset(seed=5610000+snap*10000+bi*211);cur=ot(cur).cuda()
        ratios=[];rews=[];feats=[]
        for t in range(T):
         d=robot.data
         f={
          "vz_abs":float(d.root_lin_vel_b[:,2].abs().mean()),
          "tilt":float(tilt(d.root_quat_w).mean()),
          "wx_abs":float(d.root_ang_vel_b[:,0].abs().mean()),
          "wy_abs":float(d.root_ang_vel_b[:,1].abs().mean()),
          "joint_pos_norm":float(torch.linalg.vector_norm(d.joint_pos,dim=-1).mean()),
          "joint_vel_norm":float(torch.linalg.vector_norm(d.joint_vel,dim=-1).mean()),
          "vxy":float(torch.linalg.vector_norm(d.root_lin_vel_b[:,:2],dim=-1).mean()),
         };feats.append(f)
         with torch.no_grad():a,lp,u=m.act_with_preference_latent(cur,w)
         nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
         ratios.append(torch.exp(m.logp_from_pre_tanh_with_preference(cur,w,u)-lp.detach()));rews.append(torch.tensor(vec[:,J],device="cuda")*env.unwrapped.step_dt);cur=ot(nxt).cuda()
        gs={}
        for t in range(T):
         loss=-(ratios[t]*rews[t].detach()).mean()
         gs[t]=flat(torch.autograd.grad(loss,aps,retain_graph=True,allow_unused=True),aps).detach()
        rows.append({"g0":gs[0],"g1":gs[1],"g2":gs[2],"g3":gs[3],"f2":feats[2],"f3":feats[3]})
       ref=torch.stack([(r["g0"]+r["g1"])/2 for r in rows]).mean(0)
       out={}
       for tt in (2,3):
        gs=[r[f"g{tt}"] for r in rows]; G=torch.stack(gs)
        feats={k:np.array([r[f"f{tt}"][k] for r in rows],float) for k in FEATURES}
        base={"mean_cos_to_ref":cos(G.mean(0),ref),"mean_norm":float(G.mean(0).norm())}
        source={}
        for k,x in feats.items():
          med=float(np.median(x));lo=[gs[i] for i,v in enumerate(x) if v<=med];hi=[gs[i] for i,v in enumerate(x) if v>med]
          source[k]={
            "median":med,
            "low_cos":cos(torch.stack(lo).mean(0),ref),
            "high_cos":cos(torch.stack(hi).mean(0),ref),
            "delta_alignment":float(cos(torch.stack(lo).mean(0),ref)-cos(torch.stack(hi).mean(0),ref)),
            "corr_feature_to_perreset_cos":float(np.corrcoef(zscore(x),np.array([cos(g,ref) for g in gs]))[0,1]),
          }
        # multivariate linear residualization of gradient coefficients in PCA-like low-rank projection basis
        X=np.column_stack([zscore(feats[k]) for k in FEATURES]);X=np.column_stack([np.ones(len(X)),X])
        # project gradients to 64 deterministic chunks to make regression tractable
        chunks=64; P=G.reshape(G.shape[0],chunks,-1).mean(-1).cpu().numpy()
        B=np.linalg.lstsq(X,P,rcond=None)[0]
        basecoef=B[0]
        single_restore={}
        for j,k in enumerate(FEATURES, start=1):
          Pres=P-np.outer(X[:,j],B[j])
          gm=torch.tensor(Pres.mean(0),device=G.device,dtype=G.dtype)
          rr=torch.tensor(basecoef,device=G.device,dtype=G.dtype)
          single_restore[k]={"residual_chunk_cos_to_intercept":cos(gm,rr)}
        out[str(tt)]={"base":base,"sources":source,"residualized_chunk_proxy":single_restore,
                      "feature_corr":{a:{b:float(np.corrcoef(feats[a],feats[b])[0,1]) for b in FEATURES} for a in FEATURES}}
       rep["snapshots"][str(snap)]=out
      # tensors removed before json
      (OUT/"audit.json").write_text(json.dumps(rep,indent=2)+"\n");print(json.dumps(rep,indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c31_stratify_single():
    """Run former post_v2_t5_c31_stratify_single.py stage."""
    from pathlib import Path
    import sys,json,argparse,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c25_actor_updating25-2026-09-23";OUT=ROOT/"runs/post_v2_t5_c31_dynamics_source-2026-09-23";OUT.mkdir(parents=True,exist_ok=True)
    NENV=8;NB=12;J=1;T=4;WREF=np.array([.1,.7,.1,.1],np.float32)
    FEATURES=("vz_abs","tilt","wx_abs","wy_abs","joint_pos_norm","joint_vel_norm","vxy")
    def ot(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def flat(gs,ps):return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,ps)])
    def cos(a,b):return float(torch.dot(a,b)/(a.norm()*b.norm()+1e-12))
    def tilt(q):
     _,x,y,_=[q[:,i] for i in range(4)]
     return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def main():
     ap=argparse.ArgumentParser();ap.add_argument("--snap",type=int,required=True);a=ap.parse_args();snap=a.snap
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
      m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"A_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"]);m.eval();aps=[p for n,p in m.named_parameters() if n.startswith("actor_")];w=torch.tensor(WREF,device="cuda").repeat(NENV,1)
      rows=[]
      for bi in range(NB):
       cur,_=env.reset(seed=5810000+snap*10000+bi*211);cur=ot(cur).cuda();rat=[];rw=[];ff=[]
       for t in range(T):
        d=robot.data;ff.append({
          "vz_abs":float(d.root_lin_vel_b[:,2].abs().mean()),"tilt":float(tilt(d.root_quat_w).mean()),
          "wx_abs":float(d.root_ang_vel_b[:,0].abs().mean()),"wy_abs":float(d.root_ang_vel_b[:,1].abs().mean()),
          "joint_pos_norm":float(torch.linalg.vector_norm(d.joint_pos,dim=-1).mean()),"joint_vel_norm":float(torch.linalg.vector_norm(d.joint_vel,dim=-1).mean()),
          "vxy":float(torch.linalg.vector_norm(d.root_lin_vel_b[:,:2],dim=-1).mean())})
        with torch.no_grad():aa,lp,u=m.act_with_preference_latent(cur,w)
        nxt,_,te,tr,_=env.step(aa);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
        rat.append(torch.exp(m.logp_from_pre_tanh_with_preference(cur,w,u)-lp.detach()));rw.append(torch.tensor(vec[:,J],device="cuda")*env.unwrapped.step_dt);cur=ot(nxt).cuda()
       gg=[]
       for t in range(T):
        gg.append(flat(torch.autograd.grad(-(rat[t]*rw[t].detach()).mean(),aps,retain_graph=True,allow_unused=True),aps).detach().cpu())
       rows.append({"g":gg,"f":ff})
      ref=torch.stack([(r["g"][0]+r["g"][1])/2 for r in rows]).mean(0)
      out={"schema":"c31_stratify_v1","snap":snap,"windows":{}}
      for t in (2,3):
       gs=[r["g"][t] for r in rows];allg=torch.stack(gs).mean(0);src={}
       for k in FEATURES:
        vals=np.array([r["f"][t][k] for r in rows]);med=float(np.median(vals));lo=[gs[i] for i,v in enumerate(vals) if v<=med];hi=[gs[i] for i,v in enumerate(vals) if v>med]
        gl=torch.stack(lo).mean(0);gh=torch.stack(hi).mean(0)
        src[k]={"median":med,"low_cos_to_ref":cos(gl,ref),"high_cos_to_ref":cos(gh,ref),"low_minus_high":cos(gl,ref)-cos(gh,ref),
                "low_norm":float(gl.norm()),"high_norm":float(gh.norm())}
       out["windows"][str(t)]={"all_cos_to_ref":cos(allg,ref),"all_norm":float(allg.norm()),"sources":src}
      path=OUT/f"stratify_u{snap}.json";path.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2),flush=True)
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c31_u10():
    """Run former post_v2_t5_c31_u10.py stage."""
    from pathlib import Path
    import sys,json,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c25_actor_updating25-2026-09-23"
    OUT=ROOT/"runs/post_v2_t5_c31_dynamics_source_u10-2026-09-23";OUT.mkdir(parents=True,exist_ok=True)
    SNAPS=(10,); NENV=8; NB=16; J=1; T=4
    WREF=np.array([.1,.7,.1,.1],np.float32)
    FEATURES=("vz_abs","tilt","wx_abs","wy_abs","joint_pos_norm","joint_vel_norm","vxy")
    def ot(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def flat(gs,ps):return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,ps)])
    def cos(a,b):return float(torch.dot(a,b)/(a.norm()*b.norm()+1e-12))
    def tilt(q):
     _,x,y,_=[q[:,i] for i in range(4)]
     return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def zscore(x):
     x=np.asarray(x,float);return (x-x.mean())/(x.std()+1e-12)
    def main():
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda()
      od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
      rep={"schema":"c31_dynamics_source_v1","snapshots":{}}
      for snap in SNAPS:
       m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"A_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
       aps=[p for n,p in m.named_parameters() if n.startswith("actor_")];w=torch.tensor(WREF,device="cuda").repeat(NENV,1)
       rows=[]
       for bi in range(NB):
        cur,_=env.reset(seed=5610000+snap*10000+bi*211);cur=ot(cur).cuda()
        ratios=[];rews=[];feats=[]
        for t in range(T):
         d=robot.data
         f={
          "vz_abs":float(d.root_lin_vel_b[:,2].abs().mean()),
          "tilt":float(tilt(d.root_quat_w).mean()),
          "wx_abs":float(d.root_ang_vel_b[:,0].abs().mean()),
          "wy_abs":float(d.root_ang_vel_b[:,1].abs().mean()),
          "joint_pos_norm":float(torch.linalg.vector_norm(d.joint_pos,dim=-1).mean()),
          "joint_vel_norm":float(torch.linalg.vector_norm(d.joint_vel,dim=-1).mean()),
          "vxy":float(torch.linalg.vector_norm(d.root_lin_vel_b[:,:2],dim=-1).mean()),
         };feats.append(f)
         with torch.no_grad():a,lp,u=m.act_with_preference_latent(cur,w)
         nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
         ratios.append(torch.exp(m.logp_from_pre_tanh_with_preference(cur,w,u)-lp.detach()));rews.append(torch.tensor(vec[:,J],device="cuda")*env.unwrapped.step_dt);cur=ot(nxt).cuda()
        gs={}
        for t in range(T):
         loss=-(ratios[t]*rews[t].detach()).mean()
         gs[t]=flat(torch.autograd.grad(loss,aps,retain_graph=True,allow_unused=True),aps).detach()
        rows.append({"g0":gs[0],"g1":gs[1],"g2":gs[2],"g3":gs[3],"f2":feats[2],"f3":feats[3]})
       ref=torch.stack([(r["g0"]+r["g1"])/2 for r in rows]).mean(0)
       out={}
       for tt in (2,3):
        gs=[r[f"g{tt}"] for r in rows]; G=torch.stack(gs)
        feats={k:np.array([r[f"f{tt}"][k] for r in rows],float) for k in FEATURES}
        base={"mean_cos_to_ref":cos(G.mean(0),ref),"mean_norm":float(G.mean(0).norm())}
        source={}
        for k,x in feats.items():
          med=float(np.median(x));lo=[gs[i] for i,v in enumerate(x) if v<=med];hi=[gs[i] for i,v in enumerate(x) if v>med]
          source[k]={
            "median":med,
            "low_cos":cos(torch.stack(lo).mean(0),ref),
            "high_cos":cos(torch.stack(hi).mean(0),ref),
            "delta_alignment":float(cos(torch.stack(lo).mean(0),ref)-cos(torch.stack(hi).mean(0),ref)),
            "corr_feature_to_perreset_cos":float(np.corrcoef(zscore(x),np.array([cos(g,ref) for g in gs]))[0,1]),
          }
        # multivariate linear residualization of gradient coefficients in PCA-like low-rank projection basis
        X=np.column_stack([zscore(feats[k]) for k in FEATURES]);X=np.column_stack([np.ones(len(X)),X])
        # project gradients to 64 deterministic chunks to make regression tractable
        chunks=64; P=G.reshape(G.shape[0],chunks,-1).mean(-1).cpu().numpy()
        B=np.linalg.lstsq(X,P,rcond=None)[0]
        basecoef=B[0]
        single_restore={}
        for j,k in enumerate(FEATURES, start=1):
          Pres=P-np.outer(X[:,j],B[j])
          gm=torch.tensor(Pres.mean(0),device=G.device,dtype=G.dtype)
          rr=torch.tensor(basecoef,device=G.device,dtype=G.dtype)
          single_restore[k]={"residual_chunk_cos_to_intercept":cos(gm,rr)}
        out[str(tt)]={"base":base,"sources":source,"residualized_chunk_proxy":single_restore,
                      "feature_corr":{a:{b:float(np.corrcoef(feats[a],feats[b])[0,1]) for b in FEATURES} for a in FEATURES}}
       rep["snapshots"][str(snap)]=out
      # tensors removed before json
      (OUT/"audit.json").write_text(json.dumps(rep,indent=2)+"\n");print(json.dumps(rep,indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c32_state_intervention():
    """Run former post_v2_t5_c32_state_intervention.py stage."""
    from pathlib import Path
    import sys,json,copy,numpy as np,torch,argparse
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c25_actor_updating25-2026-09-23";OUT=ROOT/"runs/post_v2_t5_c32_state_intervention-2026-09-23";OUT.mkdir(parents=True,exist_ok=True)
    NENV=8;NB=8;J=1;WREF=np.array([.1,.7,.1,.1],np.float32)
    def ot(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def flat(gs,ps):return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,ps)])
    def cos(a,b):return float(torch.dot(a,b)/(a.norm()*b.norm()+1e-12))
    def qnorm(q):return q/(torch.linalg.vector_norm(q,dim=-1,keepdim=True)+1e-12)
    def tilt_deg(q):
     _,x,y,_=[q[:,i] for i in range(4)]
     return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    def get_state(env):
     r=env.unwrapped.scene["robot"].data
     return {
      "root_state_w":r.root_state_w.clone(),
      "joint_pos":r.joint_pos.clone(),
      "joint_vel":r.joint_vel.clone(),
     }
    def set_state(env,s):
     robot=env.unwrapped.scene["robot"]
     robot.write_root_state_to_sim(s["root_state_w"])
     robot.write_joint_state_to_sim(s["joint_pos"],s["joint_vel"])
     env.unwrapped.sim.forward()
    def intervene(s,mode,scale=1.0):
     z={k:v.clone() for k,v in s.items()}
     rs=z["root_state_w"]
     # root_state_w: pos3 quat4 lin3 ang3
     if mode in ("vz","both"):
      rs[:,9] = rs[:,9]*(1-scale)  # drive vertical vel toward zero
     if mode in ("att","both"):
      q=rs[:,3:7]
      # linearly shrink x,y quaternion components, preserve yaw-ish w,z then renormalize
      q2=q.clone(); q2[:,1]*=(1-scale); q2[:,2]*=(1-scale); rs[:,3:7]=qnorm(q2)
     return z
    def metric(env):
     d=env.unwrapped.scene["robot"].data
     return {
      "ang":float(torch.linalg.vector_norm(d.root_ang_vel_b[:,:2],dim=-1).mean()),
      "tilt":float(tilt_deg(d.root_quat_w).mean()),
      "vz":float(d.root_lin_vel_b[:,2].abs().mean())
     }
    def one_branch(env,m,w,mgr,state,mode,scale,href):
     from talon_rl.rewards.objectives import normalized_objective_vector
     set_state(env,intervene(state,mode,scale))
     cur=ot(env.unwrapped.observation_manager.compute()["policy"]).cuda()
     aps=[p for n,p in m.named_parameters() if n.startswith("actor_")]
     grads=[]; phys=[]; objs=[]
     for h in range(2):
      pre=metric(env)
      with torch.no_grad():a,lp,u=m.act_with_preference_latent(cur,w)
      ratio=torch.exp(m.logp_from_pre_tanh_with_preference(cur,w,u)-lp.detach())
      nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
      vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
      rr=torch.tensor(vec[:,J],device="cuda")*env.unwrapped.step_dt
      g=flat(torch.autograd.grad(-(ratio*rr.detach()).mean(),aps,retain_graph=False,allow_unused=True),aps).detach()
      grads.append(g);post=metric(env);phys.append(post["ang"]-pre["ang"]);objs.append(float(np.mean(vec[:,J])))
      cur=ot(nxt).cuda()
     gh1=grads[0]; gh2=(grads[0]+grads[1])/2
     return {"g1_cos":cos(gh1,href),"g2_cos":cos(gh2,href),"g1_norm":float(gh1.norm()),"g2_norm":float(gh2.norm()),
             "delta_ang_h1":phys[0],"delta_ang_h2_mean":float(np.mean(phys)),"obj_h1":objs[0],"obj_h2_mean":float(np.mean(objs)),
             "post_tilt":metric(env)["tilt"],"post_vz":metric(env)["vz"]}
    def main():
     ap=argparse.ArgumentParser();ap.add_argument("--snap",type=int,required=True);a=ap.parse_args();snap=a.snap
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
      m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"A_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"]);m.eval();w=torch.tensor(WREF,device="cuda").repeat(NENV,1)
      # build H2 local reference from un-intervened t0/t1 over matched reset suites
      refs=[];states=[]
      aps=[p for n,p in m.named_parameters() if n.startswith("actor_")]
      from talon_rl.rewards.objectives import normalized_objective_vector
      for bi in range(NB):
       cur,_=env.reset(seed=6310000+snap*10000+bi*211);cur=ot(cur).cuda();gs=[]
       for t in range(2):
        with torch.no_grad():aa,lp,u=m.act_with_preference_latent(cur,w)
        ratio=torch.exp(m.logp_from_pre_tanh_with_preference(cur,w,u)-lp.detach())
        nxt,_,te,tr,_=env.step(aa);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
        rr=torch.tensor(vec[:,J],device="cuda")*env.unwrapped.step_dt
        gs.append(flat(torch.autograd.grad(-(ratio*rr.detach()).mean(),aps,allow_unused=True),aps).detach())
        cur=ot(nxt).cuda()
       refs.append((gs[0]+gs[1])/2);states.append(get_state(env))
      href=torch.stack(refs).mean(0)
      rows=[]
      for bi,state in enumerate(states):
       base_pre=metric(env)
       # individual 50% correction; joint uses 50% each. matched_mag uses 35.355% each ~ same L2 state-space scale as one 50% axis under equal normalization
       conds={"base":("base",0.0),"vz":("vz",0.5),"att":("att",0.5),"both":("both",0.5),"both_matched":("both",0.353553)}
       r={}
       for name,(mode,scale) in conds.items():
        r[name]=one_branch(env,m,w,mgr,state,mode,scale,href)
       # effect relative to base; beneficial physical = more negative delta_ang
       for metric_name in ("g1_cos","g2_cos","delta_ang_h1","delta_ang_h2_mean","obj_h1","obj_h2_mean"):
        b=r["base"][metric_name];vz=r["vz"][metric_name];att=r["att"][metric_name];both=r["both"][metric_name]
        r.setdefault("interaction",{})[metric_name]=both-vz-att+b
       rows.append(r)
      out={"schema":"c32_state_intervention_v1","snap":snap,"rows":rows,"summary":{}}
      for c in ("base","vz","att","both","both_matched"):
       out["summary"][c]={k:float(np.mean([r[c][k] for r in rows])) for k in rows[0][c]}
      out["summary"]["interaction"]={k:float(np.mean([r["interaction"][k] for r in rows])) for k in rows[0]["interaction"]}
      (OUT/f"audit_u{snap}.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out["summary"],indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c33_fd_worker():
    """Run former post_v2_t5_c33_fd_worker.py stage."""
    from pathlib import Path
    import sys,json,argparse,torch,numpy as np
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c25_actor_updating25-2026-09-23"
    OUT=ROOT/"runs/post_v2_t5_c33_tangent_preservation-2026-09-23";OUT.mkdir(parents=True,exist_ok=True)
    WREF=np.array([.1,.7,.1,.1],np.float32)
    def ot(x):
     if isinstance(x,dict): x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def main():
     ap=argparse.ArgumentParser();ap.add_argument("--snap",type=int,required=True);ap.add_argument("--suite",type=int,required=True);ap.add_argument("--dof",type=int,required=True);ap.add_argument("--sign",type=int,choices=[-1,1],required=True);ap.add_argument("--eps",type=float,required=True);a=ap.parse_args()
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=8;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim
      m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"A_snap_{a.snap}.pt",map_location="cuda",weights_only=False)["model"]);m.eval();w=torch.tensor(WREF,device="cuda").repeat(8,1)
      seed=7310000+a.snap*10000+a.suite*211
      cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
      for _ in range(2):
       with torch.no_grad():act=m.act_inference_with_preference(cur,w)
       nxt,_,_,_,_=env.step(act);cur=ot(nxt).cuda()
      with torch.no_grad():act=m.act_inference_with_preference(cur,w)
      act[:,a.dof]+=a.sign*a.eps
      env.step(act)
      d=env.unwrapped.scene["robot"].data
      y=float(torch.linalg.vector_norm(d.root_ang_vel_b[:,:2],dim=-1).mean())
      rec={"snap":a.snap,"suite":a.suite,"dof":a.dof,"sign":a.sign,"eps":a.eps,"y":y}
      fn=OUT/f"fd_u{a.snap}_s{a.suite}_d{a.dof}_{'p' if a.sign>0 else 'm'}_e{a.eps:.3f}.json"
      fn.write_text(json.dumps(rec)+"\n")
      print(json.dumps(rec))
     finally:
      if env is not None: env.close()
      app.close()
    if True: main()

def run_post_v2_t5_c33_one():
    """Run former post_v2_t5_c33_one.py stage."""
    from pathlib import Path
    import sys,json,argparse,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c25_actor_updating25-2026-09-23";OUT=ROOT/"runs/post_v2_t5_c33_tangent_preservation-2026-09-23";OUT.mkdir(parents=True,exist_ok=True)
    NENV=8;NB=1;J=1;WREF=np.array([.1,.7,.1,.1],np.float32);EPS=0.03
    def ot(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def flat(gs,ps):return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,ps)])
    def cos(a,b):return float(torch.dot(a,b)/(a.norm()*b.norm()+1e-12))
    def get_state(env):
     d=env.unwrapped.scene["robot"].data
     return {"root":d.root_state_w.clone(),"q":d.joint_pos.clone(),"qd":d.joint_vel.clone()}
    def set_state(env,s):
     r=env.unwrapped.scene["robot"];r.write_root_state_to_sim(s["root"]);r.write_joint_state_to_sim(s["q"],s["qd"]);env.unwrapped.sim.forward()
    def ang(env):
     d=env.unwrapped.scene["robot"].data
     return torch.linalg.vector_norm(d.root_ang_vel_b[:,:2],dim=-1)
    def main():
     ap=argparse.ArgumentParser();ap.add_argument("--snap",type=int,required=True);a=ap.parse_args();snap=a.snap
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
      m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"A_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"]);m.eval();w=torch.tensor(WREF,device="cuda").repeat(NENV,1)
      aps=[p for n,p in m.named_parameters() if n.startswith("actor_")]
      rows=[]
      for bi in range(NB):
       cur,_=env.reset(seed=6610000+snap*10000+bi*211);cur=ot(cur).cuda()
       # advance to t2
       for _ in range(2):
        with torch.no_grad():aa=m.act_inference_with_preference(cur,w)
        nxt,_,_,_,_=env.step(aa);cur=ot(nxt).cuda()
       s=get_state(env)
       set_state(env,s);obs=ot(env.unwrapped.observation_manager.compute()["policy"]).cuda()
       # nominal action and policy gradient wrt params from immediate Angular objective
       with torch.no_grad():a0,lp,u=m.act_with_preference_latent(obs,w)
       # action Jacobian wrt theta via directional JVP using grad of each action dim mean
       Jpi=[]
       for j in range(ad):
        scalar=a0[:,j].mean()
        gj=flat(torch.autograd.grad(scalar,aps,retain_graph=True,allow_unused=True),aps).detach()
        Jpi.append(gj)
       # immediate angular policy gradient
       ratio=torch.exp(m.logp_from_pre_tanh_with_preference(obs,w,u)-lp.detach())
       set_state(env,s)
       nxt,_,_,_,_=env.step(a0)
       raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
       rr=torch.tensor(vec[:,J],device="cuda")*env.unwrapped.step_dt
       gtheta=flat(torch.autograd.grad(-(ratio*rr.detach()).mean(),aps,allow_unused=True),aps).detach()
       dpi=torch.stack([torch.dot(gj,gtheta) for gj in Jpi])
       # physical finite-difference Jacobian d next |wxy| / da
       jac=[]
       for j in range(ad):
        ap_=a0.detach().clone();am_=a0.detach().clone();ap_[:,j]+=EPS;am_[:,j]-=EPS
        set_state(env,s);env.step(ap_);yp=float(ang(env).mean())
        set_state(env,s);env.step(am_);ym=float(ang(env).mean())
        jac.append((yp-ym)/(2*EPS))
       jac=torch.tensor(jac,device="cuda");dphys=-jac
       # authority / conditioning proxy from per-env finite diff matrix
       A=np.zeros((NENV,ad),dtype=np.float32)
       for j in range(ad):
        ap_=a0.detach().clone();am_=a0.detach().clone();ap_[:,j]+=EPS;am_[:,j]-=EPS
        set_state(env,s);env.step(ap_);yp=ang(env).detach().cpu().numpy()
        set_state(env,s);env.step(am_);ym=ang(env).detach().cpu().numpy()
        A[:,j]=(yp-ym)/(2*EPS)
       sv=np.linalg.svd(A,compute_uv=False);cond=float(sv[0]/max(sv[-1],1e-8))
       rows.append({
        "cos_pi_phys":cos(-dpi,dphys),
        "dpi_norm":float(dpi.norm()),"dphys_norm":float(dphys.norm()),
        "jacobian_singular_values":sv.tolist(),"jacobian_condition":cond,
        "phys_top_joints":np.argsort(-np.abs(jac.detach().cpu().numpy()))[:4].tolist(),
        "pi_top_joints":np.argsort(-np.abs(dpi.detach().cpu().numpy()))[:4].tolist(),
        "support_overlap_top4":len(set(np.argsort(-np.abs(jac.detach().cpu().numpy()))[:4]).intersection(set(np.argsort(-np.abs(dpi.detach().cpu().numpy()))[:4])))/4.0,
        "jacobian":jac.detach().cpu().tolist(),"dpi":dpi.detach().cpu().tolist()
       })
      out={"schema":"c33_tangent_preservation_v1","snap":snap,"rows":rows,"summary":{
        "cos_pi_phys_mean":float(np.mean([r["cos_pi_phys"] for r in rows])),
        "cos_pi_phys_std":float(np.std([r["cos_pi_phys"] for r in rows])),
        "condition_median":float(np.median([r["jacobian_condition"] for r in rows])),
        "support_overlap_top4_mean":float(np.mean([r["support_overlap_top4"] for r in rows])),
        "dphys_norm_mean":float(np.mean([r["dphys_norm"] for r in rows])),
        "dpi_norm_mean":float(np.mean([r["dpi_norm"] for r in rows]))
      }}
      (OUT/f"audit_u{snap}_one.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out["summary"],indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c33_policy_jvp_worker():
    """Run former post_v2_t5_c33_policy_jvp_worker.py stage."""
    from pathlib import Path
    import sys,json,argparse,torch,numpy as np
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c25_actor_updating25-2026-09-23";OUT=ROOT/"runs/post_v2_t5_c33_tangent_preservation-2026-09-23";OUT.mkdir(parents=True,exist_ok=True)
    WREF=np.array([.1,.7,.1,.1],np.float32);J=1;ETA=1e-4
    
    def ot(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def flat(gs,ps):return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,ps)])
    def apply_delta(ps,delta,scale):
     o=0
     with torch.no_grad():
      for p in ps:
       n=p.numel();p.add_(scale*delta[o:o+n].view_as(p));o+=n
    
    def main():
     ap=argparse.ArgumentParser();ap.add_argument("--snap",type=int,required=True);ap.add_argument("--suite",type=int,required=True);a=ap.parse_args()
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=8;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim
      m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"A_snap_{a.snap}.pt",map_location="cuda",weights_only=False)["model"]);m.eval();w=torch.tensor(WREF,device="cuda").repeat(8,1)
      aps=[p for n,p in m.named_parameters() if n.startswith("actor_")]
      seed=7310000+a.snap*10000+a.suite*211
      cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
      for _ in range(2):
       with torch.no_grad():act=m.act_inference_with_preference(cur,w)
       nxt,_,_,_,_=env.step(act);cur=ot(nxt).cuda()
      obs=cur.detach().clone()
      act,lp,u=m.act_with_preference_latent(obs,w)
      ratio=torch.exp(m.logp_from_pre_tanh_with_preference(obs,w,u)-lp.detach())
      nxt,_,_,_,_=env.step(act.detach());raw=env.unwrapped.reward_manager._step_reward.detach().cpu().numpy();names=list(env.unwrapped.reward_manager.active_terms)
      vec=normalized_objective_vector(terms(raw,names),shape=(8,));rr=torch.tensor(vec[:,J],device="cuda")*env.unwrapped.step_dt
      gtheta=flat(torch.autograd.grad((ratio*rr.detach()).mean(),aps,allow_unused=True),aps).detach()
      gn=float(gtheta.norm());unit=gtheta/(gtheta.norm()+1e-12)
      apply_delta(aps,unit,+ETA)
      with torch.no_grad():aplus=m.act_inference_with_preference(obs,w).mean(0)
      apply_delta(aps,unit,-2*ETA)
      with torch.no_grad():aminus=m.act_inference_with_preference(obs,w).mean(0)
      apply_delta(aps,unit,+ETA)
      dpi=(aplus-aminus)/(2*ETA)
      rec={"snap":a.snap,"suite":a.suite,"eta":ETA,"gtheta_norm":gn,"dpi":dpi.cpu().tolist(),"dpi_norm":float(dpi.norm())}
      (OUT/f"projection_u{a.snap}_s{a.suite}.json").write_text(json.dumps(rec,indent=2)+"\n");print(json.dumps(rec))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c33_policy_projection_worker():
    """Run former post_v2_t5_c33_policy_projection_worker.py stage."""
    from pathlib import Path
    import sys,json,argparse,torch,numpy as np
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c25_actor_updating25-2026-09-23";OUT=ROOT/"runs/post_v2_t5_c33_tangent_preservation-2026-09-23";OUT.mkdir(parents=True,exist_ok=True)
    WREF=np.array([.1,.7,.1,.1],np.float32);J=1
    def ot(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def flat(gs,ps):return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,ps)])
    def main():
     ap=argparse.ArgumentParser();ap.add_argument("--snap",type=int,required=True);ap.add_argument("--suite",type=int,required=True);a=ap.parse_args()
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=8;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim
      m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"A_snap_{a.snap}.pt",map_location="cuda",weights_only=False)["model"]);m.eval();w=torch.tensor(WREF,device="cuda").repeat(8,1);aps=[p for n,p in m.named_parameters() if n.startswith("actor_")]
      seed=7310000+a.snap*10000+a.suite*211
      cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
      for _ in range(2):
       with torch.no_grad():act=m.act_inference_with_preference(cur,w)
       nxt,_,_,_,_=env.step(act);cur=ot(nxt).cuda()
      a0,lp,u=m.act_with_preference_latent(cur,w)
      Jpi=[]
      for j in range(ad):
       Jpi.append(flat(torch.autograd.grad(a0[:,j].mean(),aps,retain_graph=True,allow_unused=True),aps))
      ratio=torch.exp(m.logp_from_pre_tanh_with_preference(cur,w,u)-lp.detach())
      nxt,_,_,_,_=env.step(a0.detach());raw=env.unwrapped.reward_manager._step_reward.detach().cpu().numpy();names=list(env.unwrapped.reward_manager.active_terms)
      vec=normalized_objective_vector(terms(raw,names),shape=(8,));rr=torch.tensor(vec[:,J],device="cuda")*env.unwrapped.step_dt
      gtheta=flat(torch.autograd.grad((ratio*rr.detach()).mean(),aps,allow_unused=True),aps).detach()
      dpi=torch.stack([torch.dot(g.detach(),gtheta) for g in Jpi])
      rec={"snap":a.snap,"suite":a.suite,"dpi":dpi.cpu().tolist(),"dpi_norm":float(dpi.norm()),"action_mean":a0.detach().mean(0).cpu().tolist()}
      (OUT/f"projection_u{a.snap}_s{a.suite}.json").write_text(json.dumps(rec,indent=2)+"\n");print(json.dumps(rec))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c33_reset_fd():
    """Run former post_v2_t5_c33_reset_fd.py stage."""
    from pathlib import Path
    import sys,json,argparse,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c25_actor_updating25-2026-09-23";OUT=ROOT/"runs/post_v2_t5_c33_tangent_preservation-2026-09-23";OUT.mkdir(parents=True,exist_ok=True)
    NENV=8;J=1;EPS=0.03;WREF=np.array([.1,.7,.1,.1],np.float32)
    def ot(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def flat(gs,ps):return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,ps)])
    def cos(a,b):return float(torch.dot(a,b)/(a.norm()*b.norm()+1e-12))
    def advance(env,m,w,seed):
     cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
     for _ in range(2):
      with torch.no_grad():a=m.act_inference_with_preference(cur,w)
      nxt,_,_,_,_=env.step(a);cur=ot(nxt).cuda()
     return cur
    def ang(env):
     d=env.unwrapped.scene["robot"].data
     return torch.linalg.vector_norm(d.root_ang_vel_b[:,:2],dim=-1)
    def main():
     ap=argparse.ArgumentParser();ap.add_argument("--snap",type=int,required=True);a=ap.parse_args();snap=a.snap
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
      m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"A_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"]);m.eval();w=torch.tensor(WREF,device="cuda").repeat(NENV,1);aps=[p for n,p in m.named_parameters() if n.startswith("actor_")]
      seed=7110000+snap
      # t2 policy quantities at nominal matched state
      obs=advance(env,m,w,seed)
      with torch.no_grad():a0,lp,u=m.act_with_preference_latent(obs,w)
      Jpi=[]
      for j in range(ad):
       Jpi.append(flat(torch.autograd.grad(a0[:,j].mean(),aps,retain_graph=True,allow_unused=True),aps).detach())
      ratio=torch.exp(m.logp_from_pre_tanh_with_preference(obs,w,u)-lp.detach())
      nxt,_,_,_,_=env.step(a0);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
      rr=torch.tensor(vec[:,J],device="cuda")*env.unwrapped.step_dt
      gtheta=flat(torch.autograd.grad(-(ratio*rr.detach()).mean(),aps,allow_unused=True),aps).detach()
      dpi=torch.stack([torch.dot(gj,gtheta) for gj in Jpi])
      jac=[]
      for j in range(ad):
       obs=advance(env,m,w,seed)
       with torch.no_grad():base=m.act_inference_with_preference(obs,w)
       ap_=base.clone();ap_[:,j]+=EPS
       env.step(ap_);yp=float(ang(env).mean())
       obs=advance(env,m,w,seed)
       with torch.no_grad():base=m.act_inference_with_preference(obs,w)
       am_=base.clone();am_[:,j]-=EPS
       env.step(am_);ym=float(ang(env).mean())
       jac.append((yp-ym)/(2*EPS))
      jac=torch.tensor(jac,device="cuda");dphys=-jac;dpi_descent=-dpi
      # scalar-output Jacobian rank is 1 if nonzero; conditioning use authority spread instead
      absj=jac.abs().cpu().numpy();authority_ratio=float(absj.max()/(np.median(absj)+1e-8))
      pj=np.argsort(-np.abs(jac.cpu().numpy()))[:4].tolist();pp=np.argsort(-np.abs(dpi.cpu().numpy()))[:4].tolist()
      out={"schema":"c33_reset_fd_v1","snap":snap,"cos_pi_phys":cos(dpi_descent,dphys),"dphys_norm":float(dphys.norm()),"dpi_norm":float(dpi.norm()),
           "authority_ratio_max_to_median":authority_ratio,"phys_top4":pj,"pi_top4":pp,"support_overlap_top4":len(set(pj)&set(pp))/4.0,
           "jacobian":jac.cpu().tolist(),"dpi":dpi.cpu().tolist()}
      (OUT/f"audit_u{snap}.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c33_tangent_preservation():
    """Run former post_v2_t5_c33_tangent_preservation.py stage."""
    from pathlib import Path
    import sys,json,argparse,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c25_actor_updating25-2026-09-23";OUT=ROOT/"runs/post_v2_t5_c33_tangent_preservation-2026-09-23";OUT.mkdir(parents=True,exist_ok=True)
    NENV=4;NB=6;J=1;WREF=np.array([.1,.7,.1,.1],np.float32);EPS=0.03
    def ot(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def flat(gs,ps):return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,ps)])
    def cos(a,b):return float(torch.dot(a,b)/(a.norm()*b.norm()+1e-12))
    def get_state(env):
     d=env.unwrapped.scene["robot"].data
     return {"root":d.root_state_w.clone(),"q":d.joint_pos.clone(),"qd":d.joint_vel.clone()}
    def set_state(env,s):
     r=env.unwrapped.scene["robot"];r.write_root_state_to_sim(s["root"]);r.write_joint_state_to_sim(s["q"],s["qd"]);env.unwrapped.sim.forward()
    def ang(env):
     d=env.unwrapped.scene["robot"].data
     return torch.linalg.vector_norm(d.root_ang_vel_b[:,:2],dim=-1)
    def main():
     ap=argparse.ArgumentParser();ap.add_argument("--snap",type=int,required=True);a=ap.parse_args();snap=a.snap
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
      m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"A_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"]);m.eval();w=torch.tensor(WREF,device="cuda").repeat(NENV,1)
      aps=[p for n,p in m.named_parameters() if n.startswith("actor_")]
      rows=[]
      for bi in range(NB):
       cur,_=env.reset(seed=6610000+snap*10000+bi*211);cur=ot(cur).cuda()
       # advance to t2
       for _ in range(2):
        with torch.no_grad():aa=m.act_inference_with_preference(cur,w)
        nxt,_,_,_,_=env.step(aa);cur=ot(nxt).cuda()
       s=get_state(env)
       set_state(env,s);obs=ot(env.unwrapped.observation_manager.compute()["policy"]).cuda()
       # nominal action and policy gradient wrt params from immediate Angular objective
       with torch.no_grad():a0,lp,u=m.act_with_preference_latent(obs,w)
       # action Jacobian wrt theta via directional JVP using grad of each action dim mean
       Jpi=[]
       for j in range(ad):
        scalar=a0[:,j].mean()
        gj=flat(torch.autograd.grad(scalar,aps,retain_graph=True,allow_unused=True),aps).detach()
        Jpi.append(gj)
       # immediate angular policy gradient
       ratio=torch.exp(m.logp_from_pre_tanh_with_preference(obs,w,u)-lp.detach())
       set_state(env,s)
       nxt,_,_,_,_=env.step(a0)
       raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
       rr=torch.tensor(vec[:,J],device="cuda")*env.unwrapped.step_dt
       gtheta=flat(torch.autograd.grad(-(ratio*rr.detach()).mean(),aps,allow_unused=True),aps).detach()
       dpi=torch.stack([torch.dot(gj,gtheta) for gj in Jpi])
       # physical finite-difference Jacobian d next |wxy| / da
       jac=[]
       for j in range(ad):
        ap_=a0.detach().clone();am_=a0.detach().clone();ap_[:,j]+=EPS;am_[:,j]-=EPS
        set_state(env,s);env.step(ap_);yp=float(ang(env).mean())
        set_state(env,s);env.step(am_);ym=float(ang(env).mean())
        jac.append((yp-ym)/(2*EPS))
       jac=torch.tensor(jac,device="cuda");dphys=-jac
       # authority / conditioning proxy from per-env finite diff matrix
       A=np.zeros((NENV,ad),dtype=np.float32)
       for j in range(ad):
        ap_=a0.detach().clone();am_=a0.detach().clone();ap_[:,j]+=EPS;am_[:,j]-=EPS
        set_state(env,s);env.step(ap_);yp=ang(env).detach().cpu().numpy()
        set_state(env,s);env.step(am_);ym=ang(env).detach().cpu().numpy()
        A[:,j]=(yp-ym)/(2*EPS)
       sv=np.linalg.svd(A,compute_uv=False);cond=float(sv[0]/max(sv[-1],1e-8))
       rows.append({
        "cos_pi_phys":cos(-dpi,dphys),
        "dpi_norm":float(dpi.norm()),"dphys_norm":float(dphys.norm()),
        "jacobian_singular_values":sv.tolist(),"jacobian_condition":cond,
        "phys_top_joints":np.argsort(-np.abs(jac.detach().cpu().numpy()))[:4].tolist(),
        "pi_top_joints":np.argsort(-np.abs(dpi.detach().cpu().numpy()))[:4].tolist(),
        "support_overlap_top4":len(set(np.argsort(-np.abs(jac.detach().cpu().numpy()))[:4]).intersection(set(np.argsort(-np.abs(dpi.detach().cpu().numpy()))[:4])))/4.0,
        "jacobian":jac.detach().cpu().tolist(),"dpi":dpi.detach().cpu().tolist()
       })
      out={"schema":"c33_tangent_preservation_v1","snap":snap,"rows":rows,"summary":{
        "cos_pi_phys_mean":float(np.mean([r["cos_pi_phys"] for r in rows])),
        "cos_pi_phys_std":float(np.std([r["cos_pi_phys"] for r in rows])),
        "condition_median":float(np.median([r["jacobian_condition"] for r in rows])),
        "support_overlap_top4_mean":float(np.mean([r["support_overlap_top4"] for r in rows])),
        "dphys_norm_mean":float(np.mean([r["dphys_norm"] for r in rows])),
        "dpi_norm_mean":float(np.mean([r["dpi_norm"] for r in rows]))
      }}
      (OUT/f"audit_u{snap}.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out["summary"],indent=2))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c33_vectorized():
    """Run former post_v2_t5_c33_vectorized.py stage."""
    from pathlib import Path
    import sys,json,argparse,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c25_actor_updating25-2026-09-23";OUT=ROOT/"runs/post_v2_t5_c33_tangent_preservation-2026-09-23";OUT.mkdir(parents=True,exist_ok=True)
    NENV=8;J=1;EPS=0.03;WREF=np.array([.1,.7,.1,.1],np.float32)
    def ot(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def flat(gs,ps):return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,ps)])
    def cos(a,b):return float(torch.dot(a,b)/(a.norm()*b.norm()+1e-12))
    def clone_env0(env):
     d=env.unwrapped.scene["robot"].data;r=env.unwrapped.scene["robot"]
     root=d.root_state_w[0:1].repeat(NENV,1);q=d.joint_pos[0:1].repeat(NENV,1);qd=d.joint_vel[0:1].repeat(NENV,1)
     r.write_root_state_to_sim(root);r.write_joint_state_to_sim(q,qd);env.unwrapped.sim.forward()
    def main():
     ap=argparse.ArgumentParser();ap.add_argument("--snap",type=int,required=True);ap.add_argument("--dof",type=int,default=-1);a=ap.parse_args();snap=a.snap
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=6910000+snap);o=ot(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
      m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"A_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"]);m.eval();w=torch.tensor(WREF,device="cuda").repeat(NENV,1);aps=[p for n,p in m.named_parameters() if n.startswith("actor_")]
      cur=o
      for _ in range(2):
       with torch.no_grad():act=m.act_inference_with_preference(cur,w)
       nxt,_,_,_,_=env.step(act);cur=ot(nxt).cuda()
      clone_env0(env);obs=ot(env.unwrapped.observation_manager.compute()["policy"]).cuda()
      with torch.no_grad():a0,lp,u=m.act_with_preference_latent(obs,w)
      # immediate angular gradient wrt theta from one nominal step on cloned states
      ratio=torch.exp(m.logp_from_pre_tanh_with_preference(obs,w,u)-lp.detach())
      # Jpi rows at cloned state
      Jpi=[]
      for j in range(ad):
       gj=flat(torch.autograd.grad(a0[:,j].mean(),aps,retain_graph=True,allow_unused=True),aps).detach();Jpi.append(gj)
      # policy gradient uses same-step angular reward proxy from nominal action in first half only; all cloned so equivalent
      # physical jacobian one dof per process if requested; no restore loop
      dofs=range(ad) if a.dof<0 else [a.dof]
      result={}
      for j in dofs:
       clone_env0(env);obs=ot(env.unwrapped.observation_manager.compute()["policy"]).cuda()
       with torch.no_grad():base=m.act_inference_with_preference(obs,w)
       test=base.clone();test[:NENV//2,j]+=EPS;test[NENV//2:,j]-=EPS
       env.step(test);ww=torch.linalg.vector_norm(env.unwrapped.scene["robot"].data.root_ang_vel_b[:,:2],dim=-1)
       deriv=float((ww[:NENV//2].mean()-ww[NENV//2:].mean())/(2*EPS))
       result[str(j)]=deriv
      path=OUT/f"phys_u{snap}_d{a.dof}.json";path.write_text(json.dumps({"snap":snap,"dof":a.dof,"jac":result},indent=2)+"\n");print(json.dumps(result))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c34_action_credit_worker():
    """Run former post_v2_t5_c34_action_credit_worker.py stage."""
    from pathlib import Path
    import sys,json,argparse,torch,numpy as np
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c25_actor_updating25-2026-09-23"
    OUT=ROOT/"runs/post_v2_t5_c34_action_credit-2026-09-23";OUT.mkdir(parents=True,exist_ok=True)
    WREF=np.array([.1,.7,.1,.1],np.float32); J=1
    def ot(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names): return {n:raw[:,i] for i,n in enumerate(names)}
    def main():
     ap=argparse.ArgumentParser();ap.add_argument("--snap",type=int,required=True);ap.add_argument("--suite",type=int,required=True);ap.add_argument("--dof",type=int,required=True);ap.add_argument("--sign",type=int,choices=[-1,1],required=True);ap.add_argument("--eps",type=float,required=True);a=ap.parse_args()
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=8;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim
      m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"A_snap_{a.snap}.pt",map_location="cuda",weights_only=False)["model"]);m.eval();w=torch.tensor(WREF,device="cuda").repeat(8,1)
      seed=7310000+a.snap*10000+a.suite*211
      cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
      for _ in range(2):
       with torch.no_grad():act=m.act_inference_with_preference(cur,w)
       nxt,_,_,_,_=env.step(act);cur=ot(nxt).cuda()
      with torch.no_grad():act=m.act_inference_with_preference(cur,w)
      act[:,a.dof]+=a.sign*a.eps
      env.step(act)
      raw=env.unwrapped.reward_manager._step_reward.detach().cpu().numpy();names=list(env.unwrapped.reward_manager.active_terms)
      vec=normalized_objective_vector(terms(raw,names),shape=(8,))
      y_obj=float(np.mean(vec[:,J]))
      d=env.unwrapped.scene["robot"].data
      y_phys=float(torch.linalg.vector_norm(d.root_ang_vel_b[:,:2],dim=-1).mean())
      rec={"snap":a.snap,"suite":a.suite,"dof":a.dof,"sign":a.sign,"eps":a.eps,"angular_obj":y_obj,"wxy":y_phys}
      fn=OUT/f"u{a.snap}_s{a.suite}_d{a.dof}_{'p' if a.sign>0 else 'm'}_e{a.eps:.3f}.json"
      fn.write_text(json.dumps(rec)+"\n");print(json.dumps(rec))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c35_p_column_worker():
    """Run former post_v2_t5_c35_p_column_worker.py stage."""
    from pathlib import Path
    import sys,json,argparse,torch,numpy as np
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c25_actor_updating25-2026-09-23";OUT=ROOT/"runs/post_v2_t5_c35_policy_tangent-2026-09-23";OUT.mkdir(parents=True,exist_ok=True)
    WREF=np.array([.1,.7,.1,.1],np.float32);ETA=1e-4
    
    def ot(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def flat(gs,ps):return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,ps)])
    def apply_delta(ps,delta,scale):
     o=0
     with torch.no_grad():
      for p in ps:
       n=p.numel();p.add_(scale*delta[o:o+n].view_as(p));o+=n
    
    def main():
     ap=argparse.ArgumentParser();ap.add_argument("--snap",type=int,required=True);ap.add_argument("--suite",type=int,required=True);ap.add_argument("--col",type=int,required=True);a=ap.parse_args()
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=8;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim
      m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"A_snap_{a.snap}.pt",map_location="cuda",weights_only=False)["model"]);m.eval();w=torch.tensor(WREF,device="cuda").repeat(8,1)
      aps=[p for n,p in m.named_parameters() if n.startswith("actor_")]
      seed=7310000+a.snap*10000+a.suite*211
      cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
      for _ in range(2):
       with torch.no_grad():act=m.act_inference_with_preference(cur,w)
       nxt,_,_,_,_=env.step(act);cur=ot(nxt).cuda()
      obs=cur.detach().clone(); act=m.act_inference_with_preference(obs,w)
      scalar=act[:,a.col].mean()
      gi=flat(torch.autograd.grad(scalar,aps,allow_unused=True),aps).detach(); gin=float(gi.norm())
      if gin<1e-12: direction=gi
      else: direction=gi/gin
      apply_delta(aps,direction,+ETA)
      with torch.no_grad():aplus=m.act_inference_with_preference(obs,w).mean(0)
      apply_delta(aps,direction,-2*ETA)
      with torch.no_grad():aminus=m.act_inference_with_preference(obs,w).mean(0)
      apply_delta(aps,direction,+ETA)
      # derivative wrt unit gi direction = J gi/||gi||; multiply back by ||gi|| => J gi = P e_i
      col=((aplus-aminus)/(2*ETA))*gin
      rec={"schema":"c35_p_column_v1","snap":a.snap,"suite":a.suite,"col":a.col,"eta":ETA,"gi_norm":gin,"p_col":col.cpu().tolist()}
      (OUT/f"P_u{a.snap}_s{a.suite}_c{a.col}.json").write_text(json.dumps(rec,indent=2)+"\n");print(json.dumps(rec))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c36_aggregate():
    """Run former post_v2_t5_c36_aggregate.py stage."""
    from pathlib import Path
    import json, numpy as np
    ROOT=Path(__file__).resolve().parents[4]
    R=ROOT/"runs/post_v2_t5_c36_score_attribution-2026-09-23"
    C34=ROOT/"runs/post_v2_t5_c34_action_credit-2026-09-23"
    C33=ROOT/"runs/post_v2_t5_c33_tangent_preservation-2026-09-23"
    def cos(a,b):
        a=np.asarray(a,float); b=np.asarray(b,float)
        return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
    def refs(snap,suite):
        ga=[]; dp=[]
        for d in range(12):
            p=json.load(open(C34/f"u{snap}_s{suite}_d{d}_p_e0.030.json"))
            m=json.load(open(C34/f"u{snap}_s{suite}_d{d}_m_e0.030.json"))
            ga.append((p["angular_obj"]-m["angular_obj"])/.06)
            dp.append(-(p["wxy"]-m["wxy"])/.06)
        return np.array(ga),np.array(dp)
    def stats(D,ga,dp):
        D=np.asarray(D,float)
        if len(D)==0:return None
        mean=D.mean(0)
        cga=np.array([cos(x,ga) for x in D]); cdp=np.array([cos(x,dp) for x in D])
        return {"n":len(D),"norm":float(np.linalg.norm(mean)),"cos_ga":cos(mean,ga),"cos_phys":cos(mean,dp),
                "sample_cos_ga_mean":float(cga.mean()),"sample_cos_ga_negfrac":float((cga<0).mean()),
                "sample_cos_phys_mean":float(cdp.mean()),"sample_cos_phys_negfrac":float((cdp<0).mean())}
    out={"schema":"c36_aggregate_v1","contexts":{}}
    for snap in (10,25):
        for suite in (0,1):
            data=json.load(open(R/f"audit_u{snap}_s{suite}.json"))["records"]
            ga,dp=refs(snap,suite)
            adv=np.array([x["adv"] for x in data]); res=np.array([x["residual_norm"] for x in data])
            sc=np.array([x["score_norm"] for x in data]); gn=np.array([x["gtheta_norm"] for x in data])
            D=np.array([x["d"] for x in data])
            medr=np.median(res); meds=np.median(sc); medgn=np.median(gn)
            groups={"all":np.ones(len(data),bool),"adv_pos":adv>0,"adv_neg":adv<0,
                    "near_mean":res<=medr,"tail_action":res>medr,
                    "low_score":sc<=meds,"high_score":sc>meds,
                    "low_gtheta":gn<=medgn,"high_gtheta":gn>medgn}
            ctx={"adv_positive_fraction":float((adv>0).mean()),"groups":{}}
            for k,m in groups.items(): ctx["groups"][k]=stats(D[m],ga,dp)
            agg=D.mean(0); dpi=np.array(json.load(open(C33/f"projection_u{snap}_s{suite}.json"))["dpi"])
            cga=np.array([cos(x,ga) for x in D])
            ctx["aggregate"]={"cos_to_dpi":cos(agg,dpi),"cos_to_ga":cos(agg,ga),"cos_to_phys":cos(agg,dp),
                              "dpi_to_ga":cos(dpi,ga),"dpi_to_phys":cos(dpi,dp)}
            ctx["correlations"]={"residual_vs_sample_cos_ga":float(np.corrcoef(res,cga)[0,1]),
                                 "score_vs_sample_cos_ga":float(np.corrcoef(sc,cga)[0,1]),
                                 "abs_adv_vs_sample_cos_ga":float(np.corrcoef(np.abs(adv),cga)[0,1])}
            keys=sorted(set().union(*[x["layer_norms"].keys() for x in data]))
            means={k:float(np.mean([x["layer_norms"].get(k,0) for x in data])) for k in keys}
            total=sum(means.values())+1e-12
            ctx["layer_mean_norms"]=means
            ctx["layer_share_proxy"]={k:v/total for k,v in means.items()}
            out["contexts"][f"u{snap}_s{suite}"]=ctx
    (R/"aggregate.json").write_text(json.dumps(out,indent=2)+"\n")
    for k,v in out["contexts"].items():
        print("\n",k)
        print("aggregate",v["aggregate"])
        for g in ("adv_pos","adv_neg","near_mean","tail_action","low_score","high_score","low_gtheta","high_gtheta"):
            print(g,v["groups"][g])
        print("corr",v["correlations"])
        print("layers",v["layer_mean_norms"],v["layer_share_proxy"])

def run_post_v2_t5_c36_controls_aggregate():
    """Run former post_v2_t5_c36_controls_aggregate.py stage."""
    from pathlib import Path
    import json,numpy as np
    ROOT=Path(__file__).resolve().parents[4]
    R=ROOT/"runs/post_v2_t5_c36_score_controls-2026-09-23"
    C34=ROOT/"runs/post_v2_t5_c34_action_credit-2026-09-23"
    def cos(a,b):
     a=np.asarray(a,float);b=np.asarray(b,float)
     return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
    def refs(snap,suite):
     ga=[];dp=[]
     for d in range(12):
      p=json.load(open(C34/f"u{snap}_s{suite}_d{d}_p_e0.030.json"))
      m=json.load(open(C34/f"u{snap}_s{suite}_d{d}_m_e0.030.json"))
      ga.append((p["angular_obj"]-m["angular_obj"])/.06);dp.append(-(p["wxy"]-m["wxy"])/.06)
     return np.array(ga),np.array(dp)
    out={"schema":"c36_controls_aggregate_v1","contexts":{}}
    for snap in (10,25):
     for suite in (0,1):
      ds=[json.load(open(R/f"u{snap}_s{suite}_draw{d}.json")) for d in range(4)]
      ga,dp=refs(snap,suite);ctx={}
      ctx["param_cos"]={k:{"mean":float(np.mean([x["cos_param"][k] for x in ds])),"std":float(np.std([x["cos_param"][k] for x in ds])),
                            "values":[x["cos_param"][k] for x in ds]} for k in ("gae_det","r_det","rc_det")}
      acts={}
      for k in ("det","gae","r","rc"):
       arr=np.array([x["action_proj"][k] for x in ds])
       acts[k]={"cos_ga_mean":float(np.mean([cos(v,ga) for v in arr])),"cos_ga_std":float(np.std([cos(v,ga) for v in arr])),
                "cos_phys_mean":float(np.mean([cos(v,dp) for v in arr])),"cos_phys_std":float(np.std([cos(v,dp) for v in arr])),
                "mean_direction_cos_ga":cos(arr.mean(0),ga),"mean_direction_cos_phys":cos(arr.mean(0),dp)}
      ctx["action_projection"]=acts
      ctx["layers"]={k:{q:float(np.mean([x["layer"][k].get(q,0) for x in ds])) for q in ("trunk","mean_head","logstd")} for k in ("det","gae","r","rc")}
      ctx["adv_pos_fraction"]=float(np.mean([x["adv_pos_fraction"] for x in ds]))
      out["contexts"][f"u{snap}_s{suite}"]=ctx
    (R/"aggregate.json").write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(out,indent=2))

def run_post_v2_t5_c36_matched_aggregate():
    """Run former post_v2_t5_c36_matched_aggregate.py stage."""
    from pathlib import Path
    import json,numpy as np
    ROOT=Path(__file__).resolve().parents[4]
    R=ROOT/"runs/post_v2_t5_c36_score_attribution_matched-2026-09-23"
    C34=ROOT/"runs/post_v2_t5_c34_action_credit-2026-09-23"
    def cos(a,b):
     a=np.asarray(a,float);b=np.asarray(b,float)
     return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
    def refs(snap,suite):
     ga=[];dp=[]
     for d in range(12):
      p=json.load(open(C34/f"u{snap}_s{suite}_d{d}_p_e0.030.json"))
      m=json.load(open(C34/f"u{snap}_s{suite}_d{d}_m_e0.030.json"))
      ga.append((p["angular_obj"]-m["angular_obj"])/.06)
      dp.append(-(p["wxy"]-m["wxy"])/.06)
     return np.array(ga),np.array(dp)
    def stat(D,ga,dp):
     D=np.asarray(D,float)
     if len(D)==0:return None
     md=D.mean(0);cg=np.array([cos(x,ga) for x in D]);cp=np.array([cos(x,dp) for x in D])
     return {"n":len(D),"mean_norm":float(np.linalg.norm(md)),"mean_cos_ga":cos(md,ga),"mean_cos_phys":cos(md,dp),
             "sample_cos_ga_mean":float(cg.mean()),"sample_cos_ga_negfrac":float((cg<0).mean()),
             "sample_cos_phys_mean":float(cp.mean()),"sample_cos_phys_negfrac":float((cp<0).mean())}
    out={"schema":"c36_matched_aggregate_v1","contexts":{}}
    for snap in (10,25):
     for suite in (0,1):
      draws=[json.load(open(R/f"u{snap}_s{suite}_draw{d}.json")) for d in range(4)]
      rec=[r for x in draws for r in x["records"]];ga,dp=refs(snap,suite)
      adv=np.array([r["adv"] for r in rec]);res=np.array([r["residual_norm"] for r in rec])
      score=np.array([r["score_norm"] for r in rec]);gn=np.array([r["gtheta_norm"] for r in rec])
      D=np.array([r["d"] for r in rec]);mr=np.median(res);ms=np.median(score);mg=np.median(gn)
      groups={"all":np.ones(len(rec),bool),"adv_pos":adv>0,"adv_neg":adv<0,
              "near_mean":res<=mr,"tail_action":res>mr,"low_score":score<=ms,"high_score":score>ms,
              "low_gtheta":gn<=mg,"high_gtheta":gn>mg}
      ctx={"groups":{k:stat(D[m],ga,dp) for k,m in groups.items()},
           "adv_positive_fraction":float((adv>0).mean()),
           "dglobal":stat(np.array([x["dglobal"] for x in draws]),ga,dp),
           "mean_sample_d_per_draw":stat(np.array([x["mean_sample_d"] for x in draws]),ga,dp)}
      cga=np.array([cos(x,ga) for x in D])
      ctx["correlations"]={"residual_vs_cos_ga":float(np.corrcoef(res,cga)[0,1]),
                           "score_vs_cos_ga":float(np.corrcoef(score,cga)[0,1]),
                           "abs_adv_vs_cos_ga":float(np.corrcoef(np.abs(adv),cga)[0,1]),
                           "gtheta_norm_vs_cos_ga":float(np.corrcoef(gn,cga)[0,1])}
      keys=sorted(set().union(*[r["layer_norms"].keys() for r in rec]))
      means={k:float(np.mean([r["layer_norms"].get(k,0) for r in rec])) for k in keys}
      total=sum(means.values())+1e-12
      ctx["layer_mean_norms"]=means;ctx["layer_share_proxy"]={k:v/total for k,v in means.items()}
      out["contexts"][f"u{snap}_s{suite}"]=ctx
    (R/"aggregate.json").write_text(json.dumps(out,indent=2)+"\n")
    for k,v in out["contexts"].items():
     print("\n",k)
     print("dglobal",v["dglobal"])
     for g in ("adv_pos","adv_neg","near_mean","tail_action","low_score","high_score","low_gtheta","high_gtheta"):
      print(g,v["groups"][g])
     print("corr",v["correlations"])
     print("layers",v["layer_mean_norms"],v["layer_share_proxy"])

def run_post_v2_t5_c36_matched_draw():
    """Run former post_v2_t5_c36_matched_draw.py stage."""
    from pathlib import Path
    import sys,json,argparse,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c25_actor_updating25-2026-09-23"
    OUT=ROOT/"runs/post_v2_t5_c36_score_attribution_matched-2026-09-23";OUT.mkdir(parents=True,exist_ok=True)
    WREF=np.array([.1,.7,.1,.1],np.float32); J=1; NENV=8; H=8; ETA=1e-4
    def ot(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def flat(gs,ps):return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,ps)])
    def apply_delta(ps,delta,scale):
     o=0
     with torch.no_grad():
      for p in ps:
       n=p.numel();p.add_(scale*delta[o:o+n].view_as(p));o+=n
    def main():
     ap=argparse.ArgumentParser();ap.add_argument("--snap",type=int,required=True);ap.add_argument("--suite",type=int,required=True);ap.add_argument("--draw",type=int,required=True);a=ap.parse_args()
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic,vector_gae
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim
      m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"A_snap_{a.snap}.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
      named=[(n,p) for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std"];ps=[p for _,p in named]
      w=torch.tensor(WREF,device="cuda").repeat(NENV,1)
      seed=7310000+a.snap*10000+a.suite*211
      cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
      for _ in range(2):
       with torch.no_grad():act=m.act_inference_with_preference(cur,w)
       nxt,_,_,_,_=env.step(act);cur=ot(nxt).cuda()
      obs0=cur.detach().clone(); torch.manual_seed(8010000+a.snap*10000+a.suite*100+a.draw)
      obs=[];us=[];lps=[];acts=[];mus=[];rews=[];vals=[];dns=[]
      for _ in range(H):
       with torch.no_grad():
        act,lp,u=m.act_with_preference_latent(cur,w);v=m.value_with_preference(cur,w);mu=m.act_inference_with_preference(cur,w)
       nxt,_,te,tr,_=env.step(act)
       raw=env.unwrapped.reward_manager._step_reward.detach().cpu().numpy();names=list(env.unwrapped.reward_manager.active_terms)
       vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
       obs.append(cur);us.append(u);lps.append(lp);acts.append(act);mus.append(mu);rews.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt);vals.append(v);dns.append((te|tr).cuda());cur=ot(nxt).cuda()
      with torch.no_grad():nv=m.value_with_preference(cur,w)
      rt=torch.stack(rews);vt=torch.stack(vals);dt=torch.stack(dns).bool();adv,_=vector_gae(rt,vt,nv,dt,lam=.95)
      aa=adv[0,:,J].detach(); O=obs[0]; U=us[0]; A=acts[0]; MU=mus[0]
      records=[]; gis=[]
      for i in range(NENV):
       logp=m.logp_from_pre_tanh_with_preference(O[i:i+1],w[i:i+1],U[i:i+1])
       score=flat(torch.autograd.grad(logp.mean(),ps,retain_graph=True,allow_unused=True),ps)
       gi=aa[i]*score;gis.append(gi.detach());gn=float(gi.norm())
       if gn<1e-12:di=torch.zeros(ad,device="cuda")
       else:
        direction=gi.detach()/(gi.detach().norm()+1e-12)
        apply_delta(ps,direction,+ETA)
        with torch.no_grad():aplus=m.act_inference_with_preference(O[i:i+1],w[i:i+1]).squeeze(0)
        apply_delta(ps,direction,-2*ETA)
        with torch.no_grad():aminus=m.act_inference_with_preference(O[i:i+1],w[i:i+1]).squeeze(0)
        apply_delta(ps,direction,+ETA);di=((aplus-aminus)/(2*ETA))*gn
       off=0;ln={}
       for name,p in named:
        nn=p.numel();seg=gi[off:off+nn];off+=nn
        key="logstd" if name=="log_std" else ("mean_head" if name.startswith("actor_mean") else "trunk")
        ln[key]=ln.get(key,0.0)+float(seg.norm()**2)
       ln={k:v**0.5 for k,v in ln.items()}
       records.append({"env":i,"adv":float(aa[i]),"score_norm":float(score.norm()),"gtheta_norm":gn,"d":di.detach().cpu().tolist(),
                       "residual_norm":float(torch.linalg.vector_norm(A[i]-MU[i])),"layer_norms":ln})
      gbar=torch.stack(gis).mean(0);gbn=float(gbar.norm())
      if gbn<1e-12:dglobal=torch.zeros(ad,device="cuda")
      else:
       direction=gbar/(gbar.norm()+1e-12);apply_delta(ps,direction,+ETA)
       with torch.no_grad():aplus=m.act_inference_with_preference(O,w).mean(0)
       apply_delta(ps,direction,-2*ETA)
       with torch.no_grad():aminus=m.act_inference_with_preference(O,w).mean(0)
       apply_delta(ps,direction,+ETA);dglobal=((aplus-aminus)/(2*ETA))*gbn
      out={"schema":"c36_matched_draw_v1","snap":a.snap,"suite":a.suite,"draw":a.draw,"seed":seed,
           "records":records,"gbar_norm":gbn,"dglobal":dglobal.detach().cpu().tolist(),
           "mean_sample_d":np.mean(np.array([r["d"] for r in records]),axis=0).tolist()}
      (OUT/f"u{a.snap}_s{a.suite}_draw{a.draw}.json").write_text(json.dumps(out,indent=2)+"\n")
      print("WROTE",a.snap,a.suite,a.draw)
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c36_score_attribution():
    """Run former post_v2_t5_c36_score_attribution.py stage."""
    from pathlib import Path
    import sys,json,argparse,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c25_actor_updating25-2026-09-23"; OUT=ROOT/"runs/post_v2_t5_c36_score_attribution-2026-09-23"; OUT.mkdir(parents=True,exist_ok=True)
    WREF=np.array([.1,.7,.1,.1],np.float32); J=1; NENV=8; H=8
    
    def ot(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def flat(gs,ps):return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,ps)])
    def cos(a,b):
     a=np.asarray(a,float);b=np.asarray(b,float)
     return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
    def apply_delta(ps,delta,scale):
     o=0
     with torch.no_grad():
      for p in ps:
       n=p.numel();p.add_(scale*delta[o:o+n].view_as(p));o+=n
    
    def main():
     ap=argparse.ArgumentParser();ap.add_argument("--snap",type=int,required=True);ap.add_argument("--suite",type=int,required=True);a=ap.parse_args()
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic,vector_gae
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim
      m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"A_snap_{a.snap}.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
      aps=[p for n,p in m.named_parameters() if n.startswith("actor_")]
      named=[(n,p) for n,p in m.named_parameters() if n.startswith("actor_")]
      w=torch.tensor(WREF,device="cuda").repeat(NENV,1)
      seed=7610000+a.snap*10000+a.suite*211
      cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
      obs=[];us=[];lps=[];acts=[];mus=[];rews=[];vals=[];dns=[]
      for _ in range(H):
       with torch.no_grad():
        act,lp,u=m.act_with_preference_latent(cur,w);v=m.value_with_preference(cur,w)
        mu=m.act_inference_with_preference(cur,w)
       nxt,_,te,tr,_=env.step(act)
       raw=env.unwrapped.reward_manager._step_reward.detach().cpu().numpy();names=list(env.unwrapped.reward_manager.active_terms)
       vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
       obs.append(cur);us.append(u);lps.append(lp);acts.append(act);mus.append(mu);rews.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt);vals.append(v);dns.append((te|tr).cuda());cur=ot(nxt).cuda()
      with torch.no_grad():nv=m.value_with_preference(cur,w)
      rt=torch.stack(rews);vt=torch.stack(vals);dt=torch.stack(dns).bool();adv,_=vector_gae(rt,vt,nv,dt,lam=.95)
      O=torch.cat(obs);U=torch.cat(us);LP=torch.cat(lps);A=torch.cat(acts);MU=torch.cat(mus);WW=w.repeat(H,1)
      aa=adv.reshape(-1,4)[:,J].detach()
      n=O.shape[0]
      # matched deterministic J_i g via finite-diff parameter perturbation around each sample is too costly; use local mean-action Jacobian rows and per-sample score vector in chunks
      records=[]
      layer_sums={}
      for i in range(n):
       oi=O[i:i+1];ui=U[i:i+1];wi=WW[i:i+1];lpi=LP[i:i+1]
       logp=m.logp_from_pre_tanh_with_preference(oi,wi,ui)
       score=flat(torch.autograd.grad(logp.mean(),aps,retain_graph=True,allow_unused=True),aps)
       gi=aa[i]*score
       # J_i gi by directional finite diff in parameter space
       gn=float(gi.norm())
       if gn<1e-12:
        di=torch.zeros(ad,device="cuda")
       else:
        direction=gi/(gi.norm()+1e-12); eta=1e-4
        apply_delta(aps,direction,+eta)
        with torch.no_grad():aplus=m.act_inference_with_preference(oi,wi).squeeze(0)
        apply_delta(aps,direction,-2*eta)
        with torch.no_grad():aminus=m.act_inference_with_preference(oi,wi).squeeze(0)
        apply_delta(aps,direction,+eta)
        di=((aplus-aminus)/(2*eta))*gn
       # layer contributions
       off=0
       lnorm={}
       for name,p in named:
        nn=p.numel();seg=gi[off:off+nn];off+=nn
        key="logstd" if "log_std" in name or "std" in name else ("mean_head" if "actor_mean" in name or "mu" in name or "actor_out" in name else "trunk")
        lnorm[key]=lnorm.get(key,0.0)+float(seg.norm()**2)
       for k in lnorm: lnorm[k]=lnorm[k]**0.5
       records.append({
        "adv":float(aa[i]),"score_norm":float(score.norm()),"gtheta_norm":gn,"d":di.detach().cpu().tolist(),
        "residual_norm":float(torch.linalg.vector_norm(A[i]-MU[i])),"layer_norms":lnorm
       })
      out={"schema":"c36_score_attribution_v1","snap":a.snap,"suite":a.suite,"records":records}
      (OUT/f"audit_u{a.snap}_s{a.suite}.json").write_text(json.dumps(out,indent=2)+"\n")
      print("WROTE",len(records))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c36_score_controls():
    """Run former post_v2_t5_c36_score_controls.py stage."""
    from pathlib import Path
    import sys,json,argparse,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c25_actor_updating25-2026-09-23"
    C34=ROOT/"runs/post_v2_t5_c34_action_credit-2026-09-23"
    OUT=ROOT/"runs/post_v2_t5_c36_score_controls-2026-09-23";OUT.mkdir(parents=True,exist_ok=True)
    WREF=np.array([.1,.7,.1,.1],np.float32);J=1;NENV=8;H=8;ETA=1e-4
    def ot(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def flat(gs,ps):return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,ps)])
    def cos(a,b):return float(torch.dot(a,b)/(a.norm()*b.norm()+1e-12))
    def apply_delta(ps,d,scale):
     o=0
     with torch.no_grad():
      for p in ps:
       n=p.numel();p.add_(scale*d[o:o+n].view_as(p));o+=n
    def ga_ref(snap,suite):
     g=[]
     for d in range(12):
      p=json.load(open(C34/f"u{snap}_s{suite}_d{d}_p_e0.030.json"));m=json.load(open(C34/f"u{snap}_s{suite}_d{d}_m_e0.030.json"))
      g.append((p["angular_obj"]-m["angular_obj"])/.06)
     return torch.tensor(g,device="cuda",dtype=torch.float32)
    def project_mean(m,ps,O,w,g):
     gn=float(g.norm())
     if gn<1e-12:return torch.zeros(m.actor_mean.out_features,device="cuda")
     direction=g/(g.norm()+1e-12)
     apply_delta(ps,direction,+ETA)
     with torch.no_grad():ap=m.act_inference_with_preference(O,w).mean(0)
     apply_delta(ps,direction,-2*ETA)
     with torch.no_grad():am=m.act_inference_with_preference(O,w).mean(0)
     apply_delta(ps,direction,+ETA)
     return ((ap-am)/(2*ETA))*gn
    def layer_stats(g,named):
     off=0;out={}
     for name,p in named:
      n=p.numel();seg=g[off:off+n];off+=n
      key="logstd" if name=="log_std" else ("mean_head" if name.startswith("actor_mean") else "trunk")
      out[key]=out.get(key,0.0)+float(seg.norm()**2)
     return {k:v**0.5 for k,v in out.items()}
    def main():
     ap=argparse.ArgumentParser();ap.add_argument("--snap",type=int,required=True);ap.add_argument("--suite",type=int,required=True);ap.add_argument("--draw",type=int,required=True);a=ap.parse_args()
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic,vector_gae
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim
      m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"A_snap_{a.snap}.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
      named=[(n,p) for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std"];ps=[p for _,p in named]
      w=torch.tensor(WREF,device="cuda").repeat(NENV,1)
      seed=7310000+a.snap*10000+a.suite*211
      cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
      for _ in range(2):
       with torch.no_grad():act=m.act_inference_with_preference(cur,w)
       nxt,_,_,_,_=env.step(act);cur=ot(nxt).cuda()
      O=cur.detach().clone();ga=ga_ref(a.snap,a.suite)
      mu=m.act_inference_with_preference(O,w)
      gdet=flat(torch.autograd.grad((mu*ga).sum(-1).mean(),ps,retain_graph=True,allow_unused=True),ps).detach()
      torch.manual_seed(8010000+a.snap*10000+a.suite*100+a.draw)
      obs=[];us=[];rews=[];vals=[];dns=[]
      for _ in range(H):
       with torch.no_grad():act,lp,u=m.act_with_preference_latent(cur,w);v=m.value_with_preference(cur,w)
       nxt,_,te,tr,_=env.step(act)
       raw=env.unwrapped.reward_manager._step_reward.detach().cpu().numpy();names=list(env.unwrapped.reward_manager.active_terms)
       vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
       obs.append(cur);us.append(u);rews.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt);vals.append(v);dns.append((te|tr).cuda());cur=ot(nxt).cuda()
      with torch.no_grad():nv=m.value_with_preference(cur,w)
      rt=torch.stack(rews);vt=torch.stack(vals);dt=torch.stack(dns).bool();adv,_=vector_gae(rt,vt,nv,dt,lam=.95)
      A=adv[0,:,J].detach();R0=rt[0,:,J].detach();Rc=R0-R0.mean();U=us[0];O0=obs[0]
      scores=[]
      for i in range(NENV):
       lp=m.logp_from_pre_tanh_with_preference(O0[i:i+1],w[i:i+1],U[i:i+1])
       scores.append(flat(torch.autograd.grad(lp.mean(),ps,retain_graph=True,allow_unused=True),ps).detach())
      S=torch.stack(scores)
      g_gae=(S*A[:,None]).mean(0);g_r=(S*R0[:,None]).mean(0);g_rc=(S*Rc[:,None]).mean(0)
      d_det=project_mean(m,ps,O,w,gdet);d_gae=project_mean(m,ps,O,w,g_gae);d_r=project_mean(m,ps,O,w,g_r);d_rc=project_mean(m,ps,O,w,g_rc)
      rec={"schema":"c36_score_controls_v1","snap":a.snap,"suite":a.suite,"draw":a.draw,
           "cos_param":{"gae_det":cos(g_gae,gdet),"r_det":cos(g_r,gdet),"rc_det":cos(g_rc,gdet)},
           "norm_param":{"det":float(gdet.norm()),"gae":float(g_gae.norm()),"r":float(g_r.norm()),"rc":float(g_rc.norm())},
           "action_proj":{"det":d_det.cpu().tolist(),"gae":d_gae.cpu().tolist(),"r":d_r.cpu().tolist(),"rc":d_rc.cpu().tolist()},
           "layer":{"det":layer_stats(gdet,named),"gae":layer_stats(g_gae,named),"r":layer_stats(g_r,named),"rc":layer_stats(g_rc,named)},
           "adv_pos_fraction":float((A>0).float().mean()),"reward0_mean":float(R0.mean()),"reward0_std":float(R0.std())}
      (OUT/f"u{a.snap}_s{a.suite}_draw{a.draw}.json").write_text(json.dumps(rec,indent=2)+"\n");print("WROTE",a.snap,a.suite,a.draw)
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c37_aggregate():
    """Run former post_v2_t5_c37_aggregate.py stage."""
    from pathlib import Path
    import json,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4]
    R=ROOT/"runs/post_v2_t5_c37_stochasticity-2026-09-23"
    C34=ROOT/"runs/post_v2_t5_c34_action_credit-2026-09-23"
    def cos(a,b):
     a=np.asarray(a,float);b=np.asarray(b,float)
     return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
    def refs(snap,suite):
     ga=[];dp=[]
     for d in range(12):
      p=json.load(open(C34/f"u{snap}_s{suite}_d{d}_p_e0.030.json"))
      m=json.load(open(C34/f"u{snap}_s{suite}_d{d}_m_e0.030.json"))
      ga.append((p["angular_obj"]-m["angular_obj"])/.06);dp.append(-(p["wxy"]-m["wxy"])/.06)
     return np.array(ga),np.array(dp)
    def vartrace(X):
     X=np.asarray(X,float);mu=X.mean(0)
     return float(np.mean(np.sum((X-mu)**2,axis=1)))
    out={"schema":"c37_stochasticity_aggregate_v1","contexts":{}}
    for snap in (10,25):
     for suite in (0,1):
      ga,dp=refs(snap,suite);ctx={}
      for sc in (1.0,.5,.25):
       singles=[];single_g=[];pairs=[];pair_g=[];res=[]
       for d in range(4):
        recs=[];gs=[]
        for sg,suf in ((-1,'m'),(1,'p')):
         stem=f"u{snap}_s{suite}_sc{sc:.2f}_d{d}_{suf}"
         rec=json.load(open(R/f"{stem}.json"));t=torch.load(R/f"{stem}.pt",map_location="cpu",weights_only=False)
         recs.append(rec);gs.append(t["gscore"].numpy())
         singles.append(np.array(rec["dscore"]));single_g.append(gs[-1]);res.append(rec["residual_mean"])
        pairs.append((np.array(recs[0]["dscore"])+np.array(recs[1]["dscore"]))/2)
        pair_g.append((gs[0]+gs[1])/2)
       singles=np.array(singles);single_g=np.array(single_g);pairs=np.array(pairs);pair_g=np.array(pair_g)
       gdet=torch.load(R/f"u{snap}_s{suite}_sc{sc:.2f}_d0_p.pt",map_location="cpu",weights_only=False)["gdet"].numpy()
       one={"residual_mean":float(np.mean(res)),
            "single":{"param_cos_det_mean":float(np.mean([cos(g,gdet) for g in single_g])),
                      "param_cos_det_std":float(np.std([cos(g,gdet) for g in single_g])),
                      "action_cos_ga_mean":float(np.mean([cos(x,ga) for x in singles])),
                      "action_cos_ga_std":float(np.std([cos(x,ga) for x in singles])),
                      "action_cos_phys_mean":float(np.mean([cos(x,dp) for x in singles])),
                      "mean_direction_cos_ga":cos(singles.mean(0),ga),
                      "mean_direction_cos_phys":cos(singles.mean(0),dp),
                      "param_vartrace":vartrace(single_g),"action_vartrace":vartrace(singles)},
            "antithetic":{"param_cos_det_mean":float(np.mean([cos(g,gdet) for g in pair_g])),
                          "param_cos_det_std":float(np.std([cos(g,gdet) for g in pair_g])),
                          "action_cos_ga_mean":float(np.mean([cos(x,ga) for x in pairs])),
                          "action_cos_ga_std":float(np.std([cos(x,ga) for x in pairs])),
                          "action_cos_phys_mean":float(np.mean([cos(x,dp) for x in pairs])),
                          "mean_direction_cos_ga":cos(pairs.mean(0),ga),
                          "mean_direction_cos_phys":cos(pairs.mean(0),dp),
                          "param_vartrace":vartrace(pair_g),"action_vartrace":vartrace(pairs)}}
       one["variance_reduction"]={"param_pair_over_single":one["antithetic"]["param_vartrace"]/(one["single"]["param_vartrace"]+1e-12),
                                  "action_pair_over_single":one["antithetic"]["action_vartrace"]/(one["single"]["action_vartrace"]+1e-12)}
       ctx[f"{sc:.2f}"]=one
      out["contexts"][f"u{snap}_s{suite}"]=ctx
    (R/"aggregate.json").write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(out,indent=2))

def run_post_v2_t5_c37_stochasticity_audit():
    """Run former post_v2_t5_c37_stochasticity_audit.py stage."""
    from pathlib import Path
    import sys,json,argparse,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c25_actor_updating25-2026-09-23"
    C34=ROOT/"runs/post_v2_t5_c34_action_credit-2026-09-23"
    OUT=ROOT/"runs/post_v2_t5_c37_stochasticity-2026-09-23";OUT.mkdir(parents=True,exist_ok=True)
    WREF=np.array([.1,.7,.1,.1],np.float32);J=1;NENV=8;ETA=1e-4
    def ot(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def flat(gs,ps):return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,ps)])
    def cos(a,b):return float(torch.dot(a,b)/(a.norm()*b.norm()+1e-12))
    def apply_delta(ps,d,scale):
     o=0
     with torch.no_grad():
      for p in ps:
       n=p.numel();p.add_(scale*d[o:o+n].view_as(p));o+=n
    def ga_ref(snap,suite):
     g=[]
     for d in range(12):
      p=json.load(open(C34/f"u{snap}_s{suite}_d{d}_p_e0.030.json"));m=json.load(open(C34/f"u{snap}_s{suite}_d{d}_m_e0.030.json"))
      g.append((p["angular_obj"]-m["angular_obj"])/.06)
     return torch.tensor(g,device="cuda",dtype=torch.float32)
    def project_mean(m,ps,O,w,g):
     gn=float(g.norm())
     if gn<1e-12:return torch.zeros(m.actor_mean.out_features,device="cuda")
     d=g/(g.norm()+1e-12);apply_delta(ps,d,+ETA)
     with torch.no_grad():ap=m.act_inference_with_preference(O,w).mean(0)
     apply_delta(ps,d,-2*ETA)
     with torch.no_grad():am=m.act_inference_with_preference(O,w).mean(0)
     apply_delta(ps,d,+ETA)
     return ((ap-am)/(2*ETA))*gn

def run_post_v2_t5_c37_worker():
    """Run former post_v2_t5_c37_worker.py stage."""
    from pathlib import Path
    import sys,json,argparse,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c25_actor_updating25-2026-09-23"
    C34=ROOT/"runs/post_v2_t5_c34_action_credit-2026-09-23"
    OUT=ROOT/"runs/post_v2_t5_c37_stochasticity-2026-09-23";OUT.mkdir(parents=True,exist_ok=True)
    WREF=np.array([.1,.7,.1,.1],np.float32);J=1;NENV=8;ETA=1e-4
    def ot(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def flat(gs,ps):return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,ps)])
    def cos(a,b):return float(torch.dot(a,b)/(a.norm()*b.norm()+1e-12))
    def ga_ref(snap,suite):
     g=[];dp=[]
     for d in range(12):
      p=json.load(open(C34/f"u{snap}_s{suite}_d{d}_p_e0.030.json"));m=json.load(open(C34/f"u{snap}_s{suite}_d{d}_m_e0.030.json"))
      g.append((p["angular_obj"]-m["angular_obj"])/.06);dp.append(-(p["wxy"]-m["wxy"])/.06)
     return torch.tensor(g,device="cuda",dtype=torch.float32),torch.tensor(dp,device="cuda",dtype=torch.float32)
    def apply_delta(ps,d,scale):
     o=0
     with torch.no_grad():
      for p in ps:
       n=p.numel();p.add_(scale*d[o:o+n].view_as(p));o+=n
    def project_mean(m,ps,O,w,g):
     gn=float(g.norm())
     if gn<1e-12:return torch.zeros(m.actor_mean.out_features,device="cuda")
     d=g/(g.norm()+1e-12);apply_delta(ps,d,+ETA)
     with torch.no_grad():ap=m.act_inference_with_preference(O,w).mean(0)
     apply_delta(ps,d,-2*ETA)
     with torch.no_grad():am=m.act_inference_with_preference(O,w).mean(0)
     apply_delta(ps,d,+ETA)
     return ((ap-am)/(2*ETA))*gn
    def main():
     ap=argparse.ArgumentParser();ap.add_argument("--snap",type=int,required=True);ap.add_argument("--suite",type=int,required=True);ap.add_argument("--scale",type=float,required=True);ap.add_argument("--draw",type=int,required=True);ap.add_argument("--sign",type=int,choices=[-1,1],required=True);a=ap.parse_args()
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from torch.distributions import Normal
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic
      from talon_rl.rewards.objectives import normalized_objective_vector
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim
      m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"A_snap_{a.snap}.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
      named=[(n,p) for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std"];ps=[p for _,p in named]
      w=torch.tensor(WREF,device="cuda").repeat(NENV,1)
      seed=7310000+a.snap*10000+a.suite*211
      cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
      for _ in range(2):
       with torch.no_grad():act=m.act_inference_with_preference(cur,w)
       nxt,_,_,_,_=env.step(act);cur=ot(nxt).cuda()
      O=cur.detach().clone();ga,dp=ga_ref(a.snap,a.suite)
      mu=m.act_inference_with_preference(O,w)
      gdet=flat(torch.autograd.grad((mu*ga).sum(-1).mean(),ps,retain_graph=True,allow_unused=True),ps).detach()
      # counterfactual std scaling in pre-tanh Gaussian
      actor_obs=m._with_w(O,w);dist0=m._pre_tanh_dist(actor_obs);std=dist0.scale*a.scale
      gen=torch.Generator(device="cuda");gen.manual_seed(9010000+a.snap*10000+a.suite*1000+a.draw)
      z=torch.randn(dist0.loc.shape,device="cuda",generator=gen)
      u=(dist0.loc + a.sign*std*z).detach()
      act=torch.tanh(u)*m.ACTION_CLIP
      dist=Normal(dist0.loc,std)
      logp=(dist.log_prob(u)-m._log_det_jacobian(u)).sum(-1)
      env.step(act.detach())
      raw=env.unwrapped.reward_manager._step_reward.detach().cpu().numpy();names=list(env.unwrapped.reward_manager.active_terms)
      vec=normalized_objective_vector(terms(raw,names),shape=(NENV,));r=torch.tensor(vec[:,J],device="cuda")*env.unwrapped.step_dt
      gscore=flat(torch.autograd.grad((logp*r.detach()).mean(),ps,allow_unused=True),ps).detach()
      ddet=project_mean(m,ps,O,w,gdet);dscore=project_mean(m,ps,O,w,gscore)
      residual=torch.linalg.vector_norm(act.detach()-mu.detach(),dim=-1)
      rec={"schema":"c37_worker_v1","snap":a.snap,"suite":a.suite,"scale":a.scale,"draw":a.draw,"sign":a.sign,
           "param_cos_det":cos(gscore,gdet),"gscore_norm":float(gscore.norm()),"gdet_norm":float(gdet.norm()),
           "dscore":dscore.cpu().tolist(),"ddet":ddet.cpu().tolist(),
           "dscore_cos_ga":cos(dscore,ga),"dscore_cos_phys":cos(dscore,dp),
           "ddet_cos_ga":cos(ddet,ga),"ddet_cos_phys":cos(ddet,dp),
           "reward_mean":float(r.mean()),"reward_std":float(r.std()),
           "residual_mean":float(residual.mean()),"residual_std":float(residual.std())}
      stem=f"u{a.snap}_s{a.suite}_sc{a.scale:.2f}_d{a.draw}_{'p' if a.sign>0 else 'm'}"
      (OUT/f"{stem}.json").write_text(json.dumps(rec,indent=2)+"\n")
      torch.save({"gscore":gscore.cpu(),"gdet":gdet.cpu()},OUT/f"{stem}.pt")
      print(json.dumps(rec))
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c38_expected_score_path():
    """Run former post_v2_t5_c38_expected_score_path.py stage."""
    from pathlib import Path
    import sys,json,argparse,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c25_actor_updating25-2026-09-23"
    C34=ROOT/"runs/post_v2_t5_c34_action_credit-2026-09-23"
    OUT=ROOT/"runs/post_v2_t5_c38_expected_score_path-2026-09-23";OUT.mkdir(parents=True,exist_ok=True)
    WREF=np.array([.1,.7,.1,.1],np.float32);J=1;NENV=8;HFD=0.03
    BUDGETS=(64,256,1024,4096)
    def ot(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def flat(gs,ps):return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,ps)])
    def cos(a,b):return float(torch.dot(a,b)/(a.norm()*b.norm()+1e-12))
    def ga_ref(snap,suite):
     g=[]
     for d in range(12):
      p=json.load(open(C34/f"u{snap}_s{suite}_d{d}_p_e0.030.json"));m=json.load(open(C34/f"u{snap}_s{suite}_d{d}_m_e0.030.json"))
      g.append((p["angular_obj"]-m["angular_obj"])/.06)
     return torch.tensor(g,device="cuda",dtype=torch.float32)
    def reset_t2(env,m,w,seed):
     cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
     for _ in range(2):
      with torch.no_grad():a=m.act_inference_with_preference(cur,w)
      nxt,_,_,_,_=env.step(a);cur=ot(nxt).cuda()
     return cur
    def step_reward(env,action):
     from talon_rl.rewards.objectives import normalized_objective_vector
     env.step(action)
     raw=env.unwrapped.reward_manager._step_reward.detach().cpu().numpy();names=list(env.unwrapped.reward_manager.active_terms)
     vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
     return torch.tensor(vec[:,J],device="cuda")*env.unwrapped.step_dt
    def main():
     ap=argparse.ArgumentParser();ap.add_argument("--snap",type=int,required=True);ap.add_argument("--suite",type=int,required=True);ap.add_argument("--draws",type=int,default=512);a=ap.parse_args()
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from torch.distributions import Normal
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim
      m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"A_snap_{a.snap}.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
      named=[(n,p) for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std"];ps=[p for _,p in named]
      actor_named=[(n,p) for n,p in named if n.startswith("actor_")];actor_ps=[p for _,p in actor_named]
      w=torch.tensor(WREF,device="cuda").repeat(NENV,1);seed=7310000+a.snap*10000+a.suite*211
      O0=reset_t2(env,m,w,seed).detach()
      ga=ga_ref(a.snap,a.suite)
      mu=m.act_inference_with_preference(O0,w)
      gdet=flat(torch.autograd.grad((mu*ga).sum(-1).mean(),ps,retain_graph=True,allow_unused=True),ps).detach()
      score_sum=torch.zeros_like(gdet);path_sum=torch.zeros_like(gdet)
      score_draws=[];path_draws=[];results={}
      gen=torch.Generator(device="cuda");gen.manual_seed(9910000+a.snap*10000+a.suite*100)
      for draw in range(a.draws):
       # all random quantities are fixed across center / +/- branches for this draw
       z=torch.randn((NENV,ad),device="cuda",generator=gen)
       dl=(torch.randint(0,2,(NENV,ad),device="cuda",generator=gen,dtype=torch.int64)*2-1).float()
       ds=(torch.randint(0,2,(ad,),device="cuda",generator=gen,dtype=torch.int64)*2-1).float()
       # center score-function branch
       _=reset_t2(env,m,w,seed)
       dist0=m._pre_tanh_dist(m._with_w(O0,w));loc=dist0.loc;std=dist0.scale
       u=(loc.detach()+std.detach()*z).detach();act=torch.tanh(u)*m.ACTION_CLIP
       r=step_reward(env,act.detach())
       logp=(dist0.log_prob(u)-m._log_det_jacobian(u)).sum(-1)
       gscore=flat(torch.autograd.grad((logp*r.detach()).mean(),ps,retain_graph=True,allow_unused=True),ps).detach()
       # pathwise finite difference wrt pre-tanh mean, common random numbers
       _=reset_t2(env,m,w,seed)
       ap=torch.tanh(loc.detach()+std.detach()*z+HFD*dl)*m.ACTION_CLIP
       rp=step_reward(env,ap)
       _=reset_t2(env,m,w,seed)
       am=torch.tanh(loc.detach()+std.detach()*z-HFD*dl)*m.ACTION_CLIP
       rm=step_reward(env,am)
       gu=((rp-rm)/(2*HFD))[:,None]*dl
       gloc=m._pre_tanh_dist(m._with_w(O0,w)).loc
       gactor=flat(torch.autograd.grad((gloc*gu.detach()).sum(-1).mean(),actor_ps,allow_unused=True),actor_ps).detach()
       # pathwise finite difference wrt shared log_std
       _=reset_t2(env,m,w,seed)
       stdp=std.detach()*torch.exp(HFD*ds)
       rsp=step_reward(env,torch.tanh(loc.detach()+stdp*z)*m.ACTION_CLIP)
       _=reset_t2(env,m,w,seed)
       stdm=std.detach()*torch.exp(-HFD*ds)
       rsm=step_reward(env,torch.tanh(loc.detach()+stdm*z)*m.ACTION_CLIP)
       glog=((rsp-rsm).mean()/(2*HFD))*ds
       # merge actor and log_std parts in named-parameter order
       parts=[];off=0
       for n,p in named:
        if n=="log_std":parts.append(glog.reshape(-1))
        else:
         nn=p.numel();parts.append(gactor[off:off+nn]);off+=nn
       gpath=torch.cat(parts)
       score_sum+=gscore;path_sum+=gpath;score_draws.append(gscore.cpu());path_draws.append(gpath.cpu())
       n_samples=(draw+1)*NENV
       if n_samples in BUDGETS:
        gs=score_sum/(draw+1);gp=path_sum/(draw+1)
        results[str(n_samples)]={"score_cos_det":cos(gs,gdet),"path_cos_det":cos(gp,gdet),"score_cos_path":cos(gs,gp),
                                "score_norm":float(gs.norm()),"path_norm":float(gp.norm()),"det_norm":float(gdet.norm())}
        print("BUDGET",n_samples,results[str(n_samples)],flush=True)
        def proj(g):
         gn=float(g.norm())
         if gn<1e-12:return torch.zeros(ad,device="cuda")
         direction=g/(g.norm()+1e-12);eta=1e-4;o=0
         with torch.no_grad():
          for _,p in named:
           nn=p.numel();p.add_(eta*direction[o:o+nn].view_as(p));o+=nn
          apm=m.act_inference_with_preference(O0,w).mean(0)
          o=0
          for _,p in named:
           nn=p.numel();p.add_(-2*eta*direction[o:o+nn].view_as(p));o+=nn
          amm=m.act_inference_with_preference(O0,w).mean(0)
          o=0
          for _,p in named:
           nn=p.numel();p.add_(eta*direction[o:o+nn].view_as(p));o+=nn
         return ((apm-amm)/(2*eta))*gn
        dscr=proj(gs);dpth=proj(gp);ddet=proj(gdet)
        results[str(n_samples)].update({"score_action_cos_ga":cos(dscr,ga),"path_action_cos_ga":cos(dpth,ga),
                                        "det_action_cos_ga":cos(ddet,ga),"score_action_cos_det_action":cos(dscr,ddet),
                                        "path_action_cos_det_action":cos(dpth,ddet),"score_action_cos_path_action":cos(dscr,dpth)})
      # Monte Carlo variance trace over per-draw gradient estimators
      SD=torch.stack(score_draws).numpy();PD=torch.stack(path_draws).numpy()
      def vt(X):
       mu=X.mean(0);return float(np.mean(np.sum((X-mu)**2,axis=1)))
      out={"schema":"c38_expected_score_path_v1","snap":a.snap,"suite":a.suite,"draws":a.draws,
           "results":results,"score_draw_vartrace":vt(SD),"path_draw_vartrace":vt(PD)}
      (OUT/f"audit_u{a.snap}_s{a.suite}.json").write_text(json.dumps(out,indent=2)+"\n")
      print("WROTE",a.snap,a.suite,flush=True)
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_post_v2_t5_c39_variance_curvature():
    """Run former post_v2_t5_c39_variance_curvature.py stage."""
    from pathlib import Path
    import sys,json,argparse,numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    RUN=ROOT/"runs/post_v2_t5_c25_actor_updating25-2026-09-23"
    C34=ROOT/"runs/post_v2_t5_c34_action_credit-2026-09-23"
    OUT=ROOT/"runs/post_v2_t5_c39_variance_curvature-2026-09-23";OUT.mkdir(parents=True,exist_ok=True)
    WREF=np.array([.1,.7,.1,.1],np.float32);J=1;NENV=8;ETA=1e-4;HFD=0.03
    SCALES=(1.0,.5,.25,.125,.0625)
    def ot(x):
     if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
     return x if torch.is_tensor(x) else torch.as_tensor(x)
    def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
    def flat(gs,ps):return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,ps)])
    def cos(a,b):return float(torch.dot(a,b)/(a.norm()*b.norm()+1e-12))
    def ga_ref(snap,suite):
     g=[];dp=[]
     for d in range(12):
      p=json.load(open(C34/f"u{snap}_s{suite}_d{d}_p_e0.030.json"));m=json.load(open(C34/f"u{snap}_s{suite}_d{d}_m_e0.030.json"))
      g.append((p["angular_obj"]-m["angular_obj"])/.06);dp.append(-(p["wxy"]-m["wxy"])/.06)
     return torch.tensor(g,device="cuda",dtype=torch.float32),torch.tensor(dp,device="cuda",dtype=torch.float32)
    def reset_t2(env,m,w,seed):
     cur,_=env.reset(seed=seed);cur=ot(cur).cuda()
     for _ in range(2):
      with torch.no_grad():a=m.act_inference_with_preference(cur,w)
      nxt,_,_,_,_=env.step(a);cur=ot(nxt).cuda()
     return cur
    def step_reward(env,action):
     from talon_rl.rewards.objectives import normalized_objective_vector
     env.step(action)
     raw=env.unwrapped.reward_manager._step_reward.detach().cpu().numpy();names=list(env.unwrapped.reward_manager.active_terms)
     vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
     return torch.tensor(vec[:,J],device="cuda")*env.unwrapped.step_dt
    def apply_delta(ps,d,scale):
     o=0
     with torch.no_grad():
      for p in ps:
       n=p.numel();p.add_(scale*d[o:o+n].view_as(p));o+=n
    def project_mean(m,ps,O,w,g):
     gn=float(g.norm())
     if gn<1e-12:return torch.zeros(m.actor_mean.out_features,device="cuda")
     d=g/(g.norm()+1e-12);apply_delta(ps,d,+ETA)
     with torch.no_grad():ap=m.act_inference_with_preference(O,w).mean(0)
     apply_delta(ps,d,-2*ETA)
     with torch.no_grad():am=m.act_inference_with_preference(O,w).mean(0)
     apply_delta(ps,d,+ETA)
     return ((ap-am)/(2*ETA))*gn
    def main():
     ap=argparse.ArgumentParser();ap.add_argument("--snap",type=int,required=True);ap.add_argument("--suite",type=int,required=True);ap.add_argument("--draws",type=int,default=256);a=ap.parse_args()
     from isaaclab.app import AppLauncher
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.models.foundations.four_objective import T4SharedActorCritic
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=ot(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim
      m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"A_snap_{a.snap}.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
      named=[(n,p) for n,p in m.named_parameters() if n.startswith("actor_") or n=="log_std"];ps=[p for _,p in named]
      actor_ps=[p for n,p in named if n.startswith("actor_")]
      w=torch.tensor(WREF,device="cuda").repeat(NENV,1);seed=7310000+a.snap*10000+a.suite*211
      O=reset_t2(env,m,w,seed).detach();ga,dp=ga_ref(a.snap,a.suite)
      mu=m.act_inference_with_preference(O,w)
      gdet=flat(torch.autograd.grad((mu*ga).sum(-1).mean(),ps,retain_graph=True,allow_unused=True),ps).detach()
      ddet=project_mean(m,ps,O,w,gdet)
      dist0=m._pre_tanh_dist(m._with_w(O,w));loc=dist0.loc.detach();std0=dist0.scale.detach()
      gen=torch.Generator(device="cuda");gen.manual_seed(10010000+a.snap*10000+a.suite*211)
      Z=[torch.randn((NENV,ad),device="cuda",generator=gen) for _ in range(a.draws)]
      out={"schema":"c39_variance_curvature_v1","snap":a.snap,"suite":a.suite,"draws":a.draws,"scales":{}}
      for sc in SCALES:
       gsum=torch.zeros_like(gdet); samples=[]
       for draw,z in enumerate(Z):
        # SPSA pathwise derivative wrt pre-tanh location, with common random numbers
        dl=(torch.randint(0,2,(NENV,ad),device="cuda",generator=gen,dtype=torch.int64)*2-1).float()
        std=std0*sc
        _=reset_t2(env,m,w,seed)
        rp=step_reward(env,torch.tanh(loc+std*z+HFD*dl)*m.ACTION_CLIP)
        _=reset_t2(env,m,w,seed)
        rm=step_reward(env,torch.tanh(loc+std*z-HFD*dl)*m.ACTION_CLIP)
        gu=((rp-rm)/(2*HFD))[:,None]*dl
        gloc=m._pre_tanh_dist(m._with_w(O,w)).loc
        gactor=flat(torch.autograd.grad((gloc*gu.detach()).sum(-1).mean(),actor_ps,allow_unused=True),actor_ps).detach()
        # pathwise derivative wrt original log_std parameters at this counterfactual scale
        ds=(torch.randint(0,2,(ad,),device="cuda",generator=gen,dtype=torch.int64)*2-1).float()
        _=reset_t2(env,m,w,seed)
        rsp=step_reward(env,torch.tanh(loc+(std*torch.exp(HFD*ds))*z)*m.ACTION_CLIP)
        _=reset_t2(env,m,w,seed)
        rsm=step_reward(env,torch.tanh(loc+(std*torch.exp(-HFD*ds))*z)*m.ACTION_CLIP)
        glog=((rsp-rsm).mean()/(2*HFD))*ds
        parts=[];off=0
        for n,p in named:
         if n=="log_std":parts.append(glog.reshape(-1))
         else:
          nn=p.numel();parts.append(gactor[off:off+nn]);off+=nn
        gp=torch.cat(parts);gsum+=gp;samples.append(gp.cpu())
       gpath=gsum/a.draws;dpath=project_mean(m,ps,O,w,gpath)
       X=torch.stack(samples).numpy();xm=X.mean(0)
       vt=float(np.mean(np.sum((X-xm)**2,axis=1)))
       out["scales"][str(sc)]={"path_cos_det":cos(gpath,gdet),"path_action_cos_ga":cos(dpath,ga),
                               "path_action_cos_phys":cos(dpath,dp),"path_action_cos_det_action":cos(dpath,ddet),
                               "path_norm":float(gpath.norm()),"det_norm":float(gdet.norm()),
                               "norm_ratio":float(gpath.norm()/(gdet.norm()+1e-12)),"draw_vartrace":vt,
                               "dpath":dpath.cpu().tolist()}
       print("SCALE",sc,out["scales"][str(sc)],flush=True)
      # deterministic mean-action diagonal curvature using C34 +/- probes + one center reward
      _=reset_t2(env,m,w,seed)
      r0=float(step_reward(env,mu.detach()).mean())
      curv=[]
      for d in range(ad):
       p=json.load(open(C34/f"u{a.snap}_s{a.suite}_d{d}_p_e0.030.json"))["angular_obj"]*env.unwrapped.step_dt
       mm=json.load(open(C34/f"u{a.snap}_s{a.suite}_d{d}_m_e0.030.json"))["angular_obj"]*env.unwrapped.step_dt
       curv.append((p-2*r0+mm)/(0.03**2))
      out["curvature"]={"center_reward":r0,"diag":curv,"abs_rank":np.argsort(-np.abs(np.array(curv))).tolist()}
      # action-space shift between lowest-variance path direction and deterministic valid credit
      low=np.array(out["scales"][str(SCALES[-1])]["dpath"]);ga_np=ga.cpu().numpy()
      out["curvature"]["low_sigma_action_shift"]= (low/(np.linalg.norm(low)+1e-12)-ga_np/(np.linalg.norm(ga_np)+1e-12)).tolist()
      out["curvature"]["abs_shift_rank"]=np.argsort(-np.abs(np.array(out["curvature"]["low_sigma_action_shift"]))).tolist()
      (OUT/f"audit_u{a.snap}_s{a.suite}.json").write_text(json.dumps(out,indent=2)+"\n")
      print("WROTE",a.snap,a.suite,flush=True)
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

STAGES = {
    "post_v2_t5_c30_timestep_window": run_post_v2_t5_c30_timestep_window,
    "post_v2_t5_c30_window_core": run_post_v2_t5_c30_window_core,
    "post_v2_t5_c31_combo_single": run_post_v2_t5_c31_combo_single,
    "post_v2_t5_c31_dynamics_source": run_post_v2_t5_c31_dynamics_source,
    "post_v2_t5_c31_stratify_single": run_post_v2_t5_c31_stratify_single,
    "post_v2_t5_c31_u10": run_post_v2_t5_c31_u10,
    "post_v2_t5_c32_state_intervention": run_post_v2_t5_c32_state_intervention,
    "post_v2_t5_c33_fd_worker": run_post_v2_t5_c33_fd_worker,
    "post_v2_t5_c33_one": run_post_v2_t5_c33_one,
    "post_v2_t5_c33_policy_jvp_worker": run_post_v2_t5_c33_policy_jvp_worker,
    "post_v2_t5_c33_policy_projection_worker": run_post_v2_t5_c33_policy_projection_worker,
    "post_v2_t5_c33_reset_fd": run_post_v2_t5_c33_reset_fd,
    "post_v2_t5_c33_tangent_preservation": run_post_v2_t5_c33_tangent_preservation,
    "post_v2_t5_c33_vectorized": run_post_v2_t5_c33_vectorized,
    "post_v2_t5_c34_action_credit_worker": run_post_v2_t5_c34_action_credit_worker,
    "post_v2_t5_c35_p_column_worker": run_post_v2_t5_c35_p_column_worker,
    "post_v2_t5_c36_aggregate": run_post_v2_t5_c36_aggregate,
    "post_v2_t5_c36_controls_aggregate": run_post_v2_t5_c36_controls_aggregate,
    "post_v2_t5_c36_matched_aggregate": run_post_v2_t5_c36_matched_aggregate,
    "post_v2_t5_c36_matched_draw": run_post_v2_t5_c36_matched_draw,
    "post_v2_t5_c36_score_attribution": run_post_v2_t5_c36_score_attribution,
    "post_v2_t5_c36_score_controls": run_post_v2_t5_c36_score_controls,
    "post_v2_t5_c37_aggregate": run_post_v2_t5_c37_aggregate,
    "post_v2_t5_c37_stochasticity_audit": run_post_v2_t5_c37_stochasticity_audit,
    "post_v2_t5_c37_worker": run_post_v2_t5_c37_worker,
    "post_v2_t5_c38_expected_score_path": run_post_v2_t5_c38_expected_score_path,
    "post_v2_t5_c39_variance_curvature": run_post_v2_t5_c39_variance_curvature,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
