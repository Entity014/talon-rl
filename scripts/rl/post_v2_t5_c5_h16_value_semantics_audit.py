#!/usr/bin/env python3
from __future__ import annotations
import json,sys,traceback
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
RUN=ROOT/"runs/post_v2_t5_c5_h16-2026-09-23"
ORDER=("T","A","O","S")
PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
IDX={"T":0,"A":1,"O":2,"S":3};SNAPS=(0,10,25,50,100);PERT=(10,25,50,100);H=(1,2,4,8,16,32);G=.99
def obs_tensor(x):
    if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
    return x if torch.is_tensor(x) else torch.as_tensor(x)
def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
def ev(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float);return float(1-np.var(y-p)/(np.var(y)+1e-12))
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
    rows=[];done_any=np.zeros(len(cur),bool)
    with torch.no_grad():
      for _ in range(32):
        a=m.act_inference_with_preference(cur,w);nxt,_,term,trunc,_=env.step(a);data=robot.data
        rows.append({"ang_vel_xy":float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),"tilt_deg":float(tilt(data.root_quat_w).mean()),"action_rate":float(torch.linalg.vector_norm(a-prev,dim=-1).mean())})
        done_any|=(term|trunc).cpu().numpy();prev=a;cur=obs_tensor(nxt).cuda()
    metrics={str(h):{k:float(np.mean([r[k] for r in rows[:h]])) for k in rows[0]} for h in H}
    return metrics,float(1-done_any.mean())
def main():
 from isaaclab.app import AppLauncher
 app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
 try:
  import gymnasium as gym,isaaclab_tasks
  from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
  from talon_rl.t4_actor_critic import T4SharedActorCritic,vector_gae
  from talon_rl.t3b_objectives import normalized_objective_vector,raw_objective_vector
  cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=32;cfg.seed=0;env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
  obs,_=env.reset(seed=0);obs=obs_tensor(obs).cuda();od=obs.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]
  critic={};pert=[]
  for bi,lab in enumerate(ORDER):
    w=torch.tensor(PREFS[lab],device="cuda").repeat(32,1);critic[lab]={}
    for snap in SNAPS:
      cp=RUN/f"{lab}_snap_{snap}.pt";m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(cp,map_location="cuda",weights_only=False)["model"]);m.eval()
      cur,_=env.reset(seed=500000+bi*1000+snap);cur=obs_tensor(cur).cuda();vals=[];rews=[];dones=[]
      with torch.no_grad():
       for _ in range(32):
        vals.append(m.value_with_preference(cur,w).cpu().numpy());a=m.act_inference_with_preference(cur,w);nxt,_,term,trunc,_=env.step(a)
        raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);rews.append(normalized_objective_vector(terms(raw,names),shape=(32,))*env.unwrapped.step_dt);dones.append((term|trunc).cpu().numpy());cur=obs_tensor(nxt).cuda()
      R=np.asarray(rews);D=np.asarray(dones,bool);V=np.asarray(vals);MC=np.zeros_like(R);run=np.zeros_like(R[0])
      for t in range(31,-1,-1):run=R[t]+G*run*(~D[t])[:,None];MC[t]=run
      critic[lab][str(snap)]={"ev":[ev(MC[:,:,j].reshape(-1),V[:,:,j].reshape(-1)) for j in range(4)],"bias":[float(np.mean(V[:,:,j]-MC[:,:,j])) for j in range(4)]}
    if lab in ("A","O"):
      j=IDX[lab]
      for snap in PERT:
        cp=RUN/f"{lab}_snap_{snap}.pt";m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(cp,map_location="cuda",weights_only=False)["model"])
        cur,_=env.reset(seed=510000+bi*1000+snap);cur=obs_tensor(cur).cuda();ob=[];pre=[];old=[];rw=[];val=[];dn=[]
        for _ in range(16):
          with torch.no_grad():a,lp,u=m.act_with_preference_latent(cur,w);v=m.value_with_preference(cur,w)
          nxt,_,term,trunc,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);vec=normalized_objective_vector(terms(raw,names),shape=(32,))
          ob.append(cur);pre.append(u);old.append(lp);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt);val.append(v);dn.append((term|trunc).cuda());cur=obs_tensor(nxt).cuda()
        with torch.no_grad():nv=m.value_with_preference(cur,w)
        rt=torch.stack(rw);vt=torch.stack(val);dt=torch.stack(dn).bool();adv,_=vector_gae(rt,vt,nv,dt)
        fo=torch.cat(ob);fu=torch.cat(pre);fold=torch.cat(old);fw=w.repeat(16,1);ratio=torch.exp(m.logp_from_pre_tanh_with_preference(fo,fw,fu)-fold.detach())
        aps=[p for n,p in m.named_parameters() if n.startswith("actor_")]
        loss=-(ratio*adv.reshape(-1,4)[:,j].detach()).mean();g=flat(torch.autograd.grad(loss,aps,allow_unused=True),aps).detach()
        basevec=torch.cat([p.detach().reshape(-1) for p in aps]);stepnorm=1e-4*(basevec.norm()+1e-12);delta=-g/(g.norm()+1e-12)*stepnorm
        pm=T4SharedActorCritic(od,ad).cuda();pm.load_state_dict(torch.load(cp,map_location="cuda",weights_only=False)["model"]);paps=[p for n,p in pm.named_parameters() if n.startswith("actor_")];setflat(paps,cloneps(paps),delta);pm.eval();m.eval()
        metric="ang_vel_xy" if lab=="A" else "tilt_deg";suites=[]
        for ss in range(2):
          seed=520000+bi*1000+snap*10+ss;b,_=rollout_metric(env,m,w,seed);p,_=rollout_metric(env,pm,w,seed);suites.append({"suite":ss,"baseline":b,"perturbed":p,"metric":metric})
        pert.append({"branch":lab,"snapshot":snap,"grad_norm":float(g.norm()),"step_norm":float(stepnorm),"suites":suites})
  out={"schema":"t5_c5_h16_value_semantics_audit_v1","critic":critic,"perturbations":pert}
  (RUN/"value_semantics_audit.json").write_text(json.dumps(out,indent=2)+"\n")
  print(json.dumps({"critic":critic,"perturbations":len(pert)},indent=2))
 finally:
  if env is not None:env.close()
  app.close()
if __name__=="__main__":main()
