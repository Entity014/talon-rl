#!/usr/bin/env python3
from pathlib import Path
import sys,json,numpy as np,torch
ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
RUN=ROOT/"runs/post_v2_t5_c25_actor_updating25-2026-09-23"
ORDER=("T","A","O","S");IDX={"T":0,"A":1,"O":2,"S":3};G=.99;H=16;NENV=8
PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
def obs_tensor(x):
 if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
 return x if torch.is_tensor(x) else torch.as_tensor(x)
def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
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
def rollout(env,m,w,seed):
 robot=env.unwrapped.scene["robot"];cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda()
 rows=[];done=np.zeros(NENV,bool);prev=torch.zeros((NENV,env.unwrapped.action_manager.total_action_dim),device="cuda")
 with torch.no_grad():
  for _ in range(32):
   a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a);data=robot.data
   rows.append({"ang_vel_xy":float(torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).mean()),
                "tilt_deg":float(tilt(data.root_quat_w).mean()),
                "action_rate":float(torch.linalg.vector_norm(a-prev,dim=-1).mean())})
   done|=(te|tr).cpu().numpy();prev=a;cur=obs_tensor(nxt).cuda()
 return rows,float(1-done.mean())
def main():
 from isaaclab.app import AppLauncher
 app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
 try:
  import gymnasium as gym,isaaclab_tasks
  from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
  from talon_rl.t4_actor_critic import T4SharedActorCritic,vector_gae
  from talon_rl.t3b_objectives import normalized_objective_vector
  cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
  env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
  o,_=env.reset(seed=0);o=obs_tensor(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
  out={"schema":"c25_semantic_perturb_v1","rows":[]}
  for snap in (25,):
   for lab in ("A","O"):
    m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"{lab}_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
    w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1);j=IDX[lab]
    cur,_=env.reset(seed=2510000+snap*10000+j*1000);cur=obs_tensor(cur).cuda()
    ob=[];pre=[];old=[];rw=[];val=[];dn=[]
    for _ in range(H):
     with torch.no_grad():a,lp,u=m.act_with_preference_latent(cur,w);v=m.value_with_preference(cur,w)
     nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
     vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
     ob.append(cur);pre.append(u);old.append(lp);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt);val.append(v);dn.append((te|tr).cuda());cur=obs_tensor(nxt).cuda()
    with torch.no_grad():nv=m.value_with_preference(cur,w)
    rt=torch.stack(rw);vt=torch.stack(val);dt=torch.stack(dn).bool();adv,_=vector_gae(rt,vt,nv,dt,lam=.95)
    fo=torch.cat(ob);fu=torch.cat(pre);fold=torch.cat(old);fw=w.repeat(H,1)
    ratio=torch.exp(m.logp_from_pre_tanh_with_preference(fo,fw,fu)-fold.detach())
    aps=[p for n,p in m.named_parameters() if n.startswith("actor_")]
    loss=-(ratio*adv.reshape(-1,4)[:,j].detach()).mean()
    g=flat(torch.autograd.grad(loss,aps,allow_unused=True),aps).detach()
    basevec=torch.cat([p.detach().reshape(-1) for p in aps]);delta=-g/(g.norm()+1e-12)*(1e-4*(basevec.norm()+1e-12))
    pm=T4SharedActorCritic(od,ad).cuda();pm.load_state_dict(torch.load(RUN/f"{lab}_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"]);pm.eval()
    paps=[p for n,p in pm.named_parameters() if n.startswith("actor_")];setflat(paps,cloneps(paps),delta)
    metric="ang_vel_xy" if lab=="A" else "tilt_deg";suites=[]
    for ss in range(3):
     seed=2610000+snap*10000+j*1000+ss
     b,bs=rollout(env,m,w,seed);p,ps=rollout(env,pm,w,seed)
     q={}
     for h in (1,2,4,8,16,32):
      q[str(h)]={"baseline":float(np.mean([x[metric] for x in b[:h]])),
                 "perturbed":float(np.mean([x[metric] for x in p[:h]]))}
     suites.append({"suite":ss,"metric":metric,"horizons":q,"baseline_survival":bs,"perturbed_survival":ps})
    out["rows"].append({"snapshot":snap,"branch":lab,"grad_norm":float(g.norm()),"suites":suites})
  (RUN/"semantic_u25.json").write_text(json.dumps(out,indent=2)+"\n")
  for row in out["rows"]:
   print("\n",row["branch"],"u",row["snapshot"])
   for h in ("1","2","4","8","16","32"):
    ds=[s["horizons"][h]["perturbed"]-s["horizons"][h]["baseline"] for s in row["suites"]]
    print(h,"delta",round(float(np.mean(ds)),5),"improve",round(float(np.mean(np.array(ds)<0)),2))
 finally:
  if env is not None:env.close()
  app.close()
if __name__=="__main__":main()
