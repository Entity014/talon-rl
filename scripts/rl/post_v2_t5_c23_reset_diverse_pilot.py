#!/usr/bin/env python3
from pathlib import Path
import sys,json,numpy as np,torch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
OUT=ROOT/"runs/post_v2_t5_c23_reset_diverse-2026-09-23"
OUT.mkdir(parents=True,exist_ok=True)
ORDER=("T","A","O","S");G=.99;H=32;NENV=8;NROUND=4;NSUP=12
PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
def obs_tensor(x):
 if isinstance(x,dict): x=x.get("policy",next(iter(x.values())))
 return x if torch.is_tensor(x) else torch.as_tensor(x)
def terms(raw,names): return {n:raw[:,i] for i,n in enumerate(names)}
def mc(rt,dt):
 out=torch.zeros_like(rt);run=torch.zeros_like(rt[-1])
 for t in range(len(rt)-1,-1,-1):
  run=rt[t]+G*run*(~dt[t]).unsqueeze(-1);out[t]=run
 return out
def ev(y,p):
 y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
 return float(1-np.var(y-p)/(np.var(y)+1e-12))
def ridge(F,Y,l2=1.0):
 A=np.c_[F,np.ones(len(F))];I=np.eye(A.shape[1]);I[-1,-1]=0
 sol=np.linalg.solve(A.T@A+l2*I,A.T@Y)
 return sol[:-1].T,sol[-1]
def collect_segment(env,m,w,mgr,cur=None,seed=None):
 if seed is not None:
  cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda()
 ob=[];rw=[];dn=[];cmds=[]
 with torch.no_grad():
  for _ in range(H):
   cmds.append(env.unwrapped.command_manager.get_command("base_velocity").detach().cpu().numpy())
   a=m.act_inference_with_preference(cur,w)
   nxt,_,te,tr,_=env.step(a)
   raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
   from talon_rl.t3b_objectives import normalized_objective_vector
   vec=normalized_objective_vector(terms(raw,names),shape=(NENV,))
   ob.append(cur);rw.append(torch.tensor(vec,device="cuda")*env.unwrapped.step_dt)
   dn.append((te|tr).cuda());cur=obs_tensor(nxt).cuda()
 fo=torch.cat(ob);fw=w.repeat(H,1);Y=mc(torch.stack(rw),torch.stack(dn).bool()).reshape(-1,4)
 with torch.no_grad():F=m.critic_body(m._with_w(fo,fw))
 C=np.concatenate(cmds,0)
 S=np.r_[C.mean(0),C.std(0),F.mean(0).cpu().numpy(),F.std(0).cpu().numpy(),Y.mean(0).cpu().numpy(),Y.std(0).cpu().numpy()]
 return F.cpu().numpy(),Y.cpu().numpy(),S.astype(np.float32),cur
def eval_head(env,m,w,mgr,W,b,seed,horizon=64):
 cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda();R=[];D=[];F=[]
 from talon_rl.t3b_objectives import normalized_objective_vector
 with torch.no_grad():
  for _ in range(horizon):
   f=m.critic_body(m._with_w(cur,w));F.append(f.cpu().numpy())
   a=m.act_inference_with_preference(cur,w)
   nxt,_,te,tr,_=env.step(a)
   raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
   R.append(normalized_objective_vector(terms(raw,names),shape=(NENV,))*env.unwrapped.step_dt)
   D.append((te|tr).cpu().numpy());cur=obs_tensor(nxt).cuda()
 R=np.asarray(R);D=np.asarray(D,bool);F=np.asarray(F)
 P=np.einsum("ted,hd->teh",F,W)+b
 def ret(seg=None):
  out=np.zeros_like(R)
  if seg is None:
   run=np.zeros_like(R[0])
   for t in range(len(R)-1,-1,-1):
    run=R[t]+G*run*(~D[t])[:,None];out[t]=run
  else:
   for st in range(0,len(R),seg):
    en=min(st+seg,len(R));run=np.zeros_like(R[0])
    for t in range(en-1,st-1,-1):
     run=R[t]+G*run*(~D[t])[:,None];out[t]=run
  return out
 H32=ret(32);MC64=ret(None)
 return {"h32_ev":[ev(H32[:,:,j],P[:,:,j]) for j in range(4)],
         "mc64_ev":[ev(MC64[:,:,j],P[:,:,j]) for j in range(4)],
         "h32_bias":[float(np.mean(P[:,:,j]-H32[:,:,j])) for j in range(4)],
         "survival":float(1-D.any(0).mean())}
def main():
 from isaaclab.app import AppLauncher
 app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
 try:
  import gymnasium as gym,isaaclab_tasks
  from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
  from talon_rl.t4_actor_critic import T4SharedActorCritic,initialize_from_rsl_m01
  cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
  cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
  env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
  o,_=env.reset(seed=0);o=obs_tensor(o).cuda();ad=env.unwrapped.action_manager.total_action_dim
  mgr=env.unwrapped.reward_manager;report={"schema":"c23_reset_diverse_v1","rounds":{}}
  for bi,lab in enumerate(ORDER):
   m=T4SharedActorCritic(o.shape[-1],ad).cuda()
   initialize_from_rsl_m01(m,ROOT/"runs/m0_1_seed0_2026-09-22/model_299.pt",device="cpu",critic_head_init="zero")
   m.eval();w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1)
   prev={"consecutive":None,"reset_diverse":None};rows=[]
   stream,_=env.reset(seed=1510000+bi*10000);stream=obs_tensor(stream).cuda()
   for rnd in range(NROUND):
    arms={}
    for mode in ("consecutive","reset_diverse"):
     Fs=[];Ys=[];Ss=[]
     if mode=="consecutive":
      cur=stream
      for k in range(NSUP):
       F,Y,S,cur=collect_segment(env,m,w,mgr,cur=cur)
       Fs.append(F);Ys.append(Y);Ss.append(S)
      stream=cur
     else:
      for k in range(NSUP):
       seed=1610000+bi*100000+rnd*1000+k*37
       F,Y,S,_=collect_segment(env,m,w,mgr,seed=seed)
       Fs.append(F);Ys.append(Y);Ss.append(S)
     F=np.concatenate(Fs);Y=np.concatenate(Ys);W,b=ridge(F,Y,1.0)
     sol=np.r_[W.reshape(-1),b]
     drift=0.0 if prev[mode] is None else float(np.linalg.norm(sol-prev[mode]))
     prev[mode]=sol
     suites=[]
     for s in range(3):
      seed=1710000+bi*100000+rnd*10000+s*503
      suites.append(eval_head(env,m,w,mgr,W,b,seed))
     arms[mode]={"solution_drift":drift,"support_summary_std_mean":float(np.std(np.stack(Ss),axis=0).mean()),"suites":suites}
    rows.append({"round":rnd,"arms":arms})
   report["rounds"][lab]=rows
  (OUT/"audit.json").write_text(json.dumps(report,indent=2)+"\n")
  agg={}
  for mode in ("consecutive","reset_diverse"):
   h=[];mc=[];bias=[];sv=[];dr=[];orient=[]
   for lab in ORDER:
    for row in report["rounds"][lab]:
     dr.append(row["arms"][mode]["solution_drift"])
     for q in row["arms"][mode]["suites"]:
      h+=q["h32_ev"];mc+=q["mc64_ev"];bias+=q["h32_bias"];sv.append(q["survival"]);orient.append(q["h32_ev"][2])
   agg[mode]={"h32_ev_mean":float(np.mean(h)),"h32_negative_fraction":float(np.mean(np.array(h)<0)),
              "orientation_h32_ev_mean":float(np.mean(orient)),"mc64_ev_mean":float(np.mean(mc)),
              "mc64_negative_fraction":float(np.mean(np.array(mc)<0)),"h32_mean_abs_bias":float(np.mean(np.abs(bias))),
              "solution_drift_mean_excluding_first":float(np.mean([x for x in dr if x>0])),"min_survival":float(np.min(sv))}
  report["aggregate"]=agg;(OUT/"audit.json").write_text(json.dumps(report,indent=2)+"\n")
  print(json.dumps(agg,indent=2))
 finally:
  if env is not None:env.close()
  app.close()
if __name__=="__main__":main()
