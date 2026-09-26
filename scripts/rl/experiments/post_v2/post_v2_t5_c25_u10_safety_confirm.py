#!/usr/bin/env python3
from pathlib import Path
import sys,json,numpy as np,torch
ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
RUN=ROOT/"runs/post_v2_t5_c25_actor_updating-2026-09-23";ORDER=("T","A","O","S");NENV=8
PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"A":np.array([.1,.7,.1,.1],np.float32),"O":np.array([.1,.1,.7,.1],np.float32),"S":np.array([.1,.1,.1,.7],np.float32)}
def obs_tensor(x):
 if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
 return x if torch.is_tensor(x) else torch.as_tensor(x)
def main():
 from isaaclab.app import AppLauncher
 app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
 try:
  import gymnasium as gym,isaaclab_tasks
  from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
  from talon_rl.t4_actor_critic import T4SharedActorCritic
  cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
  env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=obs_tensor(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim
  out={}
  for bi,lab in enumerate(ORDER):
   m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"{lab}_snap_10.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
   w=torch.tensor(PREFS[lab],device="cuda").repeat(NENV,1);vals=[]
   for ss in range(8):
    cur,_=env.reset(seed=2810000+bi*10000+ss*503);cur=obs_tensor(cur).cuda();done=np.zeros(NENV,bool)
    with torch.no_grad():
     for _ in range(64):
      a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a);done|=(te|tr).cpu().numpy();cur=obs_tensor(nxt).cuda()
    vals.append(float(1-done.mean()))
   out[lab]={"survival_by_suite":vals,"mean":float(np.mean(vals)),"min":float(np.min(vals)),"failure_suite_fraction":float(np.mean(np.array(vals)<1))}
  (RUN/"u10_safety_confirm.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
 finally:
  if env is not None:env.close()
  app.close()
if __name__=="__main__":main()
