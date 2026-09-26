#!/usr/bin/env python3
from pathlib import Path
import sys,json,numpy as np,torch
ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
OUT=ROOT/"runs/post_v2_t5_c24_tracking_residual-2026-09-23"
ORDER=("T","A","O","S");G=.99;NENV=8;H=64
PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
def obs_tensor(x):
 if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
 return x if torch.is_tensor(x) else torch.as_tensor(x)
def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
def returns(R,D,seg=None):
 out=np.zeros_like(R)
 if seg is None:
  run=np.zeros_like(R[0])
  for t in range(len(R)-1,-1,-1):run=R[t]+G*run*(~D[t])[:,None];out[t]=run
 else:
  for st in range(0,len(R),seg):
   en=min(st+seg,len(R));run=np.zeros_like(R[0])
   for t in range(en-1,st-1,-1):run=R[t]+G*run*(~D[t])[:,None];out[t]=run
 return out
def ev(y,p):
 return float(1-np.var(np.asarray(y).reshape(-1)-np.asarray(p).reshape(-1))/(np.var(np.asarray(y).reshape(-1))+1e-12))
def ridge(F,Y):
 A=np.c_[F,np.ones(len(F))];I=np.eye(A.shape[1]);I[-1,-1]=0
 sol=np.linalg.solve(A.T@A+I,A.T@Y);return sol[:-1],sol[-1]
def collect(env,m,w,mgr,seed):
 from talon_rl.t3b_objectives import normalized_objective_vector
 cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda();F=[];R=[];D=[];C=[]
 with torch.no_grad():
  for _ in range(H):
   F.append(m.critic_body(m._with_w(cur,w)).cpu().numpy())
   C.append(env.unwrapped.command_manager.get_command("base_velocity").cpu().numpy())
   a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a)
   raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
   R.append(normalized_objective_vector(terms(raw,names),shape=(NENV,))*env.unwrapped.step_dt)
   D.append((te|tr).cpu().numpy());cur=obs_tensor(nxt).cuda()
 R=np.asarray(R);D=np.asarray(D,bool)
 return np.asarray(F),returns(R,D,32)[:,:,0],returns(R,D,None)[:,:,0],np.asarray(C)
def main():
 from isaaclab.app import AppLauncher
 app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
 try:
  import gymnasium as gym,isaaclab_tasks
  from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
  from talon_rl.t4_actor_critic import T4SharedActorCritic,initialize_from_rsl_m01
  cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
  cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
  env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=obs_tensor(o).cuda()
  ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager;out={}
  for bi,lab in enumerate(ORDER):
   m=T4SharedActorCritic(o.shape[-1],ad).cuda()
   initialize_from_rsl_m01(m,ROOT/"runs/m0_1_seed0_2026-09-22/model_299.pt",device="cpu",critic_head_init="zero");m.eval()
   w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1)
   tr=[collect(env,m,w,mgr,2710000+bi*100000+i*97) for i in range(12)]
   F=np.concatenate([x[0].reshape(-1,x[0].shape[-1]) for x in tr])
   Y=np.concatenate([x[1].reshape(-1) for x in tr]);W,b=ridge(F,Y)
   rows=[]
   for i in range(6):
    seed=2910000+bi*100000+i*103;f,h32,mc64,c=collect(env,m,w,mgr,seed)
    p=f.reshape(-1,f.shape[-1])@W+b;p=p.reshape(h32.shape)
    cmd=np.linalg.norm(c[...,:2],axis=-1)
    rows.append({"seed":seed,"h32_ev":ev(h32,p),"h32_mc64_corr":float(np.corrcoef(h32.reshape(-1),mc64.reshape(-1))[0,1]),
      "command_speed_mean":float(cmd.mean()),"command_speed_std":float(cmd.std()),"h32_target_std":float(h32.std())})
   out[lab]=rows
  agg=[x for lab in ORDER for x in out[lab]]
  report={"schema":"c24_reset_breakdown_v1","specialists":out,
    "aggregate":{"h32_ev_mean":float(np.mean([x["h32_ev"] for x in agg])),"h32_ev_min":float(np.min([x["h32_ev"] for x in agg])),
      "negative_fraction":float(np.mean(np.array([x["h32_ev"] for x in agg])<0)),
      "ev_command_corr":float(np.corrcoef([x["h32_ev"] for x in agg],[x["command_speed_mean"] for x in agg])[0,1]),
      "ev_targetstd_corr":float(np.corrcoef([x["h32_ev"] for x in agg],[x["h32_target_std"] for x in agg])[0,1])}}
  (OUT/"reset_breakdown.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report["aggregate"],indent=2))
 finally:
  if env is not None:env.close()
  app.close()
if __name__=="__main__":main()
