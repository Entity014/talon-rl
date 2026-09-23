#!/usr/bin/env python3
from pathlib import Path
import sys,json,numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
RUN=ROOT/"runs/post_v2_t5_c25_actor_updating25-2026-09-23"
ORDER=("T","A","O","S");SNAPS=(0,1,5,10,25);G=.99;NENV=8
PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
def obs_tensor(x):
 if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
 return x if torch.is_tensor(x) else torch.as_tensor(x)
def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
def ev(y,p):
 y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1);return float(1-np.var(y-p)/(np.var(y)+1e-12))
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
  from talon_rl.t4_actor_critic import T4SharedActorCritic
  from talon_rl.t3b_objectives import normalized_objective_vector
  cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
  env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
  o,_=env.reset(seed=0);o=obs_tensor(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
  out={"schema":"c25_replay_v1","snapshots":list(SNAPS),"specialists":{}}
  for bi,lab in enumerate(ORDER):
   rows=[];w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1)
   for snap in SNAPS:
    m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"{lab}_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
    suite_rows=[]
    for ss in range(3):
     cur,_=env.reset(seed=2410000+bi*10000+ss*503);cur=obs_tensor(cur).cuda();R=[];D=[];V=[]
     with torch.no_grad():
      for _ in range(64):
       V.append(m.value_with_preference(cur,w).cpu().numpy());a=m.act_inference_with_preference(cur,w)
       nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
       R.append(normalized_objective_vector(terms(raw,names),shape=(NENV,))*env.unwrapped.step_dt);D.append((te|tr).cpu().numpy());cur=obs_tensor(nxt).cuda()
     R=np.asarray(R);D=np.asarray(D,bool);V=np.asarray(V);H=ret(R,D,32);M=ret(R,D,None)
     suite_rows.append({"h32_ev":[ev(H[:,:,j],V[:,:,j]) for j in range(4)],"mc64_ev":[ev(M[:,:,j],V[:,:,j]) for j in range(4)],"h32_bias":[float(np.mean(V[:,:,j]-H[:,:,j])) for j in range(4)],"survival":float(1-D.any(0).mean())})
    rows.append({"snapshot":snap,"suites":suite_rows})
   out["specialists"][lab]=rows
  agg={}
  for snap in SNAPS:
   h=[];m=[];b=[];sv=[];by=[[] for _ in range(4)]
   for lab in ORDER:
    row=next(r for r in out["specialists"][lab] if r["snapshot"]==snap)
    for q in row["suites"]:
     h+=q["h32_ev"];m+=q["mc64_ev"];b+=q["h32_bias"];sv.append(q["survival"])
     for j,x in enumerate(q["h32_ev"]):by[j].append(x)
   agg[str(snap)]={"h32_ev_mean":float(np.mean(h)),"h32_negative_fraction":float(np.mean(np.array(h)<0)),"h32_ev_by_head":[float(np.mean(x)) for x in by],"mc64_ev_mean":float(np.mean(m)),"mc64_negative_fraction":float(np.mean(np.array(m)<0)),"h32_mean_abs_bias":float(np.mean(np.abs(b))),"min_survival":float(np.min(sv))}
  out["aggregate"]=agg;(RUN/"replay_audit25.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(agg,indent=2))
 finally:
  if env is not None:env.close()
  app.close()
if __name__=="__main__":main()
