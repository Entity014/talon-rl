#!/usr/bin/env python3
from pathlib import Path
import sys,json,numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
RUNS={"control":ROOT/"runs/post_v2_t5_c18_control-2026-09-23","ridge3":ROOT/"runs/post_v2_t5_c18_ridge3-2026-09-23"}
ORDER=("T","A","O","S");G=.99
PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
def obs_tensor(x):
 if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
 return x if torch.is_tensor(x) else torch.as_tensor(x)
def terms(raw,names):return {n:raw[:,i] for i,n in enumerate(names)}
def ev(y,p):
 y=np.asarray(y,float).reshape(-1);p=np.asarray(p,float).reshape(-1);return float(1-np.var(y-p)/(np.var(y)+1e-12))
def returns(R,D,segment=None):
 out=np.zeros_like(R)
 if segment is None:
  run=np.zeros_like(R[0])
  for t in range(len(R)-1,-1,-1):run=R[t]+G*run*(~D[t])[:,None];out[t]=run
 else:
  for st in range(0,len(R),segment):
   en=min(st+segment,len(R));run=np.zeros_like(R[0])
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
  cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=32;cfg.seed=0;env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
  o,_=env.reset(seed=0);o=obs_tensor(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
  out={}
  for tag,run in RUNS.items():
   out[tag]={}
   for bi,lab in enumerate(ORDER):
    m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(run/f"{lab}_snap_25.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
    w=torch.tensor(PREFS[lab],device="cuda").repeat(32,1);cur,_=env.reset(seed=1150000+bi*1000);cur=obs_tensor(cur).cuda();R=[];D=[];V=[]
    with torch.no_grad():
     for _ in range(64):
      V.append(m.value_with_preference(cur,w).cpu().numpy());a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a);raw=mgr._step_reward.detach().cpu().numpy();names=list(mgr.active_terms)
      R.append(normalized_objective_vector(terms(raw,names),shape=(32,))*env.unwrapped.step_dt);D.append((te|tr).cpu().numpy());cur=obs_tensor(nxt).cuda()
    R=np.asarray(R);D=np.asarray(D,bool);V=np.asarray(V);H32=returns(R,D,32);MC64=returns(R,D,None)
    out[tag][lab]={
      "h32_ev":[ev(H32[:,:,j],V[:,:,j]) for j in range(4)],
      "h32_bias":[float(np.mean(V[:,:,j]-H32[:,:,j])) for j in range(4)],
      "mc64_ev":[ev(MC64[:,:,j],V[:,:,j]) for j in range(4)],
      "mc64_bias":[float(np.mean(V[:,:,j]-MC64[:,:,j])) for j in range(4)]
    }
  p=RUNS["ridge3"]/"horizon_value_compare.json";p.write_text(json.dumps(out,indent=2)+"\n")
  for tag in out:
   h=[];m=[];hb=[];mb=[]
   for lab in ORDER:h+=out[tag][lab]["h32_ev"];m+=out[tag][lab]["mc64_ev"];hb+=out[tag][lab]["h32_bias"];mb+=out[tag][lab]["mc64_bias"]
   print(tag,"H32 EV",np.mean(h),"MC64 EV",np.mean(m),"|bias|",np.mean(np.abs(hb)),np.mean(np.abs(mb)))
 finally:
  if env is not None:env.close()
  app.close()
if __name__=="__main__":main()
