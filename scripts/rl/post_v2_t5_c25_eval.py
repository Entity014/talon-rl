#!/usr/bin/env python3
from pathlib import Path
import sys,json,numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
RUN=ROOT/"runs/post_v2_t5_c25_actor_updating-2026-09-23"
ORDER=("T","A","O","S");IDX={"T":0,"A":1,"O":2,"S":3};G=.99;NENV=8
PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
SNAPS=(0,1,5,10)
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
def flat(gs,ps):
 return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,ps)])
def setflat(ps,base,delta):
 o=0
 with torch.no_grad():
  for p,b in zip(ps,base):
   n=p.numel();p.copy_(b+delta[o:o+n].view_as(p));o+=n
def tilt(q):
 _,x,y,_=[q[:,i] for i in range(4)]
 return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
def value_eval(env,m,w,mgr,seed):
 from talon_rl.t3b_objectives import normalized_objective_vector
 cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda();R=[];D=[];V=[]
 with torch.no_grad():
  for _ in range(64):
   V.append(m.value_with_preference(cur,w).cpu().numpy());a=m.act_inference_with_preference(cur,w)
   nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
   R.append(normalized_objective_vector(terms(raw,names),shape=(NENV,))*env.unwrapped.step_dt)
   D.append((te|tr).cpu().numpy());cur=obs_tensor(nxt).cuda()
 R=np.asarray(R);D=np.asarray(D,bool);V=np.asarray(V);H=ret(R,D,32);M=ret(R,D,None)
 return {"h32_ev":[ev(H[:,:,j],V[:,:,j]) for j in range(4)],"mc64_ev":[ev(M[:,:,j],V[:,:,j]) for j in range(4)],
  "h32_bias":[float(np.mean(V[:,:,j]-H[:,:,j])) for j in range(4)],"survival":float(1-D.any(0).mean())}
def phys_roll(env,m,w,seed):
 cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda();robot=env.unwrapped.scene["robot"];rows=[];done=np.zeros(NENV,bool)
 with torch.no_grad():
  for _ in range(32):
   a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a);data=robot.data
   rows.append({"ang_vel_xy":float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),
                "tilt_deg":float(tilt(data.root_quat_w).mean())})
   done|=(te|tr).cpu().numpy();cur=obs_tensor(nxt).cuda()
 return {k:float(np.mean([r[k] for r in rows])) for k in rows[0]},float(1-done.mean())
def grad_perturb(env,m,w,mgr,j,seed):
 from talon_rl.t4_actor_critic import vector_gae
 from talon_rl.t3b_objectives import normalized_objective_vector
 cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda();ob=[];pre=[];old=[];rw=[];val=[];dn=[]
 for _ in range(16):
  with torch.no_grad():a,lp,u=m.act_with_preference_latent(cur,w);v=m.value_with_preference(cur,w)
  nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
  ob.append(cur);pre.append(u);old.append(lp);rw.append(torch.tensor(normalized_objective_vector(terms(raw,names),shape=(NENV,)),device="cuda")*env.unwrapped.step_dt)
  val.append(v);dn.append((te|tr).cuda());cur=obs_tensor(nxt).cuda()
 with torch.no_grad():nv=m.value_with_preference(cur,w)
 adv,_=vector_gae(torch.stack(rw),torch.stack(val),nv,torch.stack(dn).bool(),lam=.95)
 fo=torch.cat(ob);fu=torch.cat(pre);fold=torch.cat(old);fw=w.repeat(16,1)
 ratio=torch.exp(m.logp_from_pre_tanh_with_preference(fo,fw,fu)-fold.detach())
 aps=[p for n,p in m.named_parameters() if n.startswith("actor_")]
 loss=-(ratio*adv.reshape(-1,4)[:,j].detach()).mean();g=flat(torch.autograd.grad(loss,aps,allow_unused=True),aps).detach()
 return g
def main():
 from isaaclab.app import AppLauncher
 app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
 try:
  import gymnasium as gym,isaaclab_tasks
  from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
  from talon_rl.t4_actor_critic import T4SharedActorCritic
  cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
  cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
  env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=obs_tensor(o).cuda()
  od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
  out={"schema":"c25_eval_v1","value":{},"perturbations":[]}
  for snap in SNAPS:
   vals=[]
   for bi,lab in enumerate(ORDER):
    m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"{lab}_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
    w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1)
    for ss in range(3):
     q=value_eval(env,m,w,mgr,2510000+snap*10000+bi*1000+ss*113);q.update({"policy":lab,"suite":ss});vals.append(q)
   out["value"][str(snap)]=vals
  for snap in (1,5,10):
   for lab in ("A","O"):
    bi=ORDER.index(lab);j=IDX[lab]
    m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"{lab}_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
    w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1);g=grad_perturb(env,m,w,mgr,j,2610000+snap*10000+bi*1000)
    aps=[p for n,p in m.named_parameters() if n.startswith("actor_")];base=[p.detach().clone() for p in aps];basevec=torch.cat([p.reshape(-1) for p in base])
    delta=-g/(g.norm()+1e-12)*(1e-4*(basevec.norm()+1e-12))
    pm=T4SharedActorCritic(od,ad).cuda();pm.load_state_dict(torch.load(RUN/f"{lab}_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"])
    paps=[p for n,p in pm.named_parameters() if n.startswith("actor_")];setflat(paps,[p.detach().clone() for p in paps],delta);pm.eval()
    suites=[];metric="ang_vel_xy" if lab=="A" else "tilt_deg"
    for ss in range(3):
     seed=2710000+snap*10000+bi*1000+ss*127;b,sb=phys_roll(env,m,w,seed);p,sp=phys_roll(env,pm,w,seed)
     suites.append({"suite":ss,"baseline":b,"perturbed":p,"delta_metric":p[metric]-b[metric],"survival_base":sb,"survival_perturbed":sp})
    out["perturbations"].append({"snapshot":snap,"branch":lab,"metric":metric,"grad_norm":float(g.norm()),"suites":suites})
  agg={}
  for snap in SNAPS:
   vals=out["value"][str(snap)];h=sum([x["h32_ev"] for x in vals],[]);mc=sum([x["mc64_ev"] for x in vals],[]);b=sum([x["h32_bias"] for x in vals],[])
   by=[[] for _ in range(4)]
   for x in vals:
    for j,z in enumerate(x["h32_ev"]):by[j].append(z)
   agg[str(snap)]={"h32_ev_mean":float(np.mean(h)),"h32_negative_fraction":float(np.mean(np.array(h)<0)),
    "h32_ev_by_head":[float(np.mean(x)) for x in by],"mc64_ev_mean":float(np.mean(mc)),"mc64_negative_fraction":float(np.mean(np.array(mc)<0)),
    "h32_mean_abs_bias":float(np.mean(np.abs(b))),"min_survival":float(min(x["survival"] for x in vals))}
  out["aggregate"]=agg;(RUN/"eval.json").write_text(json.dumps(out,indent=2)+"\n")
  print(json.dumps({"value":agg,"perturbations":out["perturbations"]},indent=2))
 finally:
  if env is not None:env.close()
  app.close()
if __name__=="__main__":main()
