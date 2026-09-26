#!/usr/bin/env python3
from pathlib import Path
import sys,json,numpy as np,torch
ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
RUNS={"C5_u1":ROOT/"runs/post_v2_t5_c5_h16-2026-09-23","C7_u10":ROOT/"runs/post_v2_t5_c7_critic_updates10-2026-09-23"}
ORDER=("T","A","O","S");SNAPS=(10,25,50);G=.99
PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
def obs_tensor(x):
 if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
 return x if torch.is_tensor(x) else torch.as_tensor(x)
def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
def mc(R,D):
 out=np.zeros_like(R);run=np.zeros_like(R[0])
 for t in range(len(R)-1,-1,-1):run=R[t]+G*run*(~D[t])[:,None];out[t]=run
 return out
def main():
 from isaaclab.app import AppLauncher
 app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
 try:
  import gymnasium as gym,isaaclab_tasks
  from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
  from talon_rl.t4_actor_critic import T4SharedActorCritic,vector_gae
  from talon_rl.t3b_objectives import normalized_objective_vector
  cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=32;cfg.seed=0;env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
  o,_=env.reset(seed=0);o=obs_tensor(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
  out={}
  for tag,run in RUNS.items():
   out[tag]={}
   for bi,lab in enumerate(ORDER):
    w=torch.tensor(PREFS[lab],device="cuda").repeat(32,1);out[tag][lab]={}
    for snap in SNAPS:
     m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(run/f"{lab}_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
     cur,_=env.reset(seed=800000+bi*1000+snap);cur=obs_tensor(cur).cuda();R=[];D=[];V=[]
     with torch.no_grad():
      for _ in range(64):
       V.append(m.value_with_preference(cur,w).cpu().numpy());a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a)
       raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms);R.append(normalized_objective_vector(terms(raw,names),shape=(32,))*env.unwrapped.step_dt);D.append((te|tr).cpu().numpy());cur=obs_tensor(nxt).cuda()
      nv=m.value_with_preference(cur,w)
     R=np.asarray(R);D=np.asarray(D,bool);V=np.asarray(V);MC=mc(R,D)
     rt=torch.tensor(R,dtype=torch.float32,device="cuda");vt=torch.tensor(V,dtype=torch.float32,device="cuda");dt=torch.tensor(D,device="cuda")
     _,ret=vector_gae(rt,vt,nv,dt);RET=ret.cpu().numpy()
     out[tag][lab][str(snap)]={
      "gae_mc_mae":[float(np.mean(np.abs(RET[:,:,j]-MC[:,:,j]))) for j in range(4)],
      "gae_mc_bias":[float(np.mean(RET[:,:,j]-MC[:,:,j])) for j in range(4)],
      "value_mc_mae":[float(np.mean(np.abs(V[:,:,j]-MC[:,:,j]))) for j in range(4)]
     }
  p=ROOT/"runs/post_v2_t5_c7_critic_updates10-2026-09-23/target_gap_audit.json";p.write_text(json.dumps(out,indent=2)+"\n")
  for s in SNAPS:
   for tag in RUNS:
    a=[];b=[]
    for lab in ORDER:a+=out[tag][lab][str(s)]["gae_mc_mae"];b+=out[tag][lab][str(s)]["value_mc_mae"]
    print(s,tag,"GAE-MC",round(float(np.mean(a)),4),"V-MC",round(float(np.mean(b)),4))
 finally:
  if env is not None:env.close()
  app.close()
if __name__=="__main__":main()
