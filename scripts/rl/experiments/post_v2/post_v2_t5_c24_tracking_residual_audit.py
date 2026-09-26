#!/usr/bin/env python3
from pathlib import Path
import sys,json,numpy as np,torch
ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
OUT=ROOT/"runs/post_v2_t5_c24_tracking_residual-2026-09-23"
OUT.mkdir(parents=True,exist_ok=True)
ORDER=("T","A","O","S");G=.99;NENV=8;H=64
PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
LAMBDAS=(0.0,1e-4,1e-3,1e-2,1e-1,1.0,10.0,100.0)
def obs_tensor(x):
 if isinstance(x,dict): x=x.get("policy",next(iter(x.values())))
 return x if torch.is_tensor(x) else torch.as_tensor(x)
def terms(raw,names): return {n:raw[:,i] for i,n in enumerate(names)}
def ret(R,D,seg=None):
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
def ev(y,p):
 y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1)
 return float(1-np.var(y-p)/(np.var(y)+1e-12))
def ridge(F,Y,l2):
 A=np.c_[F,np.ones(len(F))]
 if l2==0:
  sol=np.linalg.lstsq(A,Y,rcond=None)[0]
 else:
  I=np.eye(A.shape[1]);I[-1,-1]=0
  sol=np.linalg.solve(A.T@A+l2*I,A.T@Y)
 return sol[:-1].T,sol[-1]
def collect(env,m,w,mgr,seed):
 from talon_rl.t3b_objectives import normalized_objective_vector
 cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda()
 Fs=[];Rs=[];Ds=[];Cs=[]
 with torch.no_grad():
  for _ in range(H):
   Fs.append(m.critic_body(m._with_w(cur,w)).cpu().numpy())
   Cs.append(env.unwrapped.command_manager.get_command("base_velocity").detach().cpu().numpy())
   a=m.act_inference_with_preference(cur,w)
   nxt,_,te,tr,_=env.step(a)
   raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
   Rs.append(normalized_objective_vector(terms(raw,names),shape=(NENV,))*env.unwrapped.step_dt)
   Ds.append((te|tr).cpu().numpy());cur=obs_tensor(nxt).cuda()
 R=np.asarray(Rs);D=np.asarray(Ds,bool);F=np.asarray(Fs);C=np.asarray(Cs)
 return {"F":F,"C":C,"H32":ret(R,D,32),"MC64":ret(R,D,None),"D":D}
def flatten(ds,key):
 return np.concatenate([x[key].reshape(-1,x[key].shape[-1]) for x in ds],axis=0)
def corr(a,b):
 a=np.asarray(a).reshape(-1);b=np.asarray(b).reshape(-1)
 return float(np.corrcoef(a,b)[0,1])
def command_bins(C):
 speed=np.linalg.norm(C[...,:2],axis=-1).reshape(-1)
 q=np.quantile(speed,[0,.25,.5,.75,1.0])
 return speed,q
def eval_by_bins(Y,P,C):
 speed,q=command_bins(C);rows=[]
 for i in range(4):
  lo,hi=q[i],q[i+1]
  mask=(speed>=lo)&(speed<=hi if i==3 else speed<hi)
  rows.append({"bin":i,"lo":float(lo),"hi":float(hi),"n":int(mask.sum()),
               "ev":ev(Y.reshape(-1)[mask],P.reshape(-1)[mask]),
               "mse":float(np.mean((Y.reshape(-1)[mask]-P.reshape(-1)[mask])**2)),
               "bias":float(np.mean(P.reshape(-1)[mask]-Y.reshape(-1)[mask]))})
 return rows
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
  mgr=env.unwrapped.reward_manager;report={"schema":"c24_tracking_residual_v1","specialists":{}}
  for bi,lab in enumerate(ORDER):
   m=T4SharedActorCritic(o.shape[-1],ad).cuda()
   initialize_from_rsl_m01(m,ROOT/"runs/m0_1_seed0_2026-09-22/model_299.pt",device="cpu",critic_head_init="zero")
   m.eval();w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1)
   train=[collect(env,m,w,mgr,2110000+bi*100000+i*97) for i in range(16)]
   test=[collect(env,m,w,mgr,2310000+bi*100000+i*101) for i in range(8)]
   Ftr=flatten(train,"F");Fte=flatten(test,"F")
   Cte=flatten(test,"C");Htr=flatten(train,"H32");Hte=flatten(test,"H32")
   Mtr=flatten(train,"MC64");Mte=flatten(test,"MC64")
   q={"target_stats":{},"fits":{}}
   for j,name in enumerate(("Tracking","Angular","Orientation","Smoothness")):
    q["target_stats"][name]={
      "h32_mean":float(Hte[:,j].mean()),"h32_std":float(Hte[:,j].std()),"h32_var":float(Hte[:,j].var()),
      "mc64_mean":float(Mte[:,j].mean()),"mc64_std":float(Mte[:,j].std()),"mc64_var":float(Mte[:,j].var()),
      "h32_mc64_corr":corr(Hte[:,j],Mte[:,j])}
    sweeps={}
    for lam in LAMBDAS:
     W,b=ridge(Ftr,Htr[:,j:j+1],lam);P=Fte@W.T+b
     sweeps[str(lam)]={"h32_ev":ev(Hte[:,j],P[:,0]),"h32_mse":float(np.mean((Hte[:,j]-P[:,0])**2)),
       "h32_bias":float(np.mean(P[:,0]-Hte[:,j])),"weight_norm":float(np.linalg.norm(W))}
    Wm,bm=ridge(Ftr,Mtr[:,j:j+1],1.0);Pm=Fte@Wm.T+bm
    q["fits"][name]={"h32_ridge_sweep":sweeps,"mc64_ridge1_ev":ev(Mte[:,j],Pm[:,0]),
      "mc64_ridge1_mse":float(np.mean((Mte[:,j]-Pm[:,0])**2))}
    if name=="Tracking":
     W,b=ridge(Ftr,Htr[:,j:j+1],1.0);P=Fte@W.T+b
     q["tracking_command_bins"]=eval_by_bins(Hte[:,j],P[:,0],Cte)
   report["specialists"][lab]=q
  # aggregate across specialist policies
  agg={"target_stats":{},"fit_summary":{}}
  for name in ("Tracking","Angular","Orientation","Smoothness"):
   ts=[report["specialists"][lab]["target_stats"][name] for lab in ORDER]
   agg["target_stats"][name]={k:float(np.mean([x[k] for x in ts])) for k in ts[0]}
   fs={}
   for lam in LAMBDAS:
    vals=[report["specialists"][lab]["fits"][name]["h32_ridge_sweep"][str(lam)]["h32_ev"] for lab in ORDER]
    fs[str(lam)]={"h32_ev_mean":float(np.mean(vals)),"h32_ev_min":float(np.min(vals))}
   mc=[report["specialists"][lab]["fits"][name]["mc64_ridge1_ev"] for lab in ORDER]
   agg["fit_summary"][name]={"h32_ridge_sweep":fs,"mc64_ridge1_ev_mean":float(np.mean(mc))}
  report["aggregate"]=agg
  (OUT/"audit.json").write_text(json.dumps(report,indent=2)+"\n")
  print(json.dumps(agg,indent=2))
 finally:
  if env is not None:env.close()
  app.close()
if __name__=="__main__":main()
