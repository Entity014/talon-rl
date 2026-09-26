#!/usr/bin/env python3
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
  from talon_rl.t4_actor_critic import T4SharedActorCritic,vector_gae
  from talon_rl.t3b_objectives import normalized_objective_vector
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
if __name__=="__main__":main()
