#!/usr/bin/env python3
from pathlib import Path
import sys,json,numpy as np,torch
ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
RUN=ROOT/"runs/post_v2_t5_c25_actor_updating25-2026-09-23";OUT=ROOT/"runs/post_v2_t5_c27_smoothness_safety-2026-09-23";OUT.mkdir(parents=True,exist_ok=True)
SNAPS=(0,5,10,25);NENV=8;SEED=2840503
W=np.array([.1,.1,.1,.7],np.float32)
def obs_tensor(x):
 if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
 return x if torch.is_tensor(x) else torch.as_tensor(x)
def tilt(q):
 _,x,y,_=[q[:,i] for i in range(4)]
 return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
def main():
 from isaaclab.app import AppLauncher
 app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
 try:
  import gymnasium as gym,isaaclab_tasks
  from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
  from talon_rl.t4_actor_critic import T4SharedActorCritic
  cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
  env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=obs_tensor(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim
  robot=env.unwrapped.scene["robot"];out={"schema":"c27_smoothness_safety_v1","seed":SEED,"snapshots":{}}
  for snap in SNAPS:
   m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/f"S_snap_{snap}.pt",map_location="cuda",weights_only=False)["model"]);m.eval()
   w=torch.tensor(W,device="cuda").repeat(NENV,1);cur,_=env.reset(seed=SEED);cur=obs_tensor(cur).cuda();prev=torch.zeros((NENV,ad),device="cuda")
   first_done=[None]*NENV;rows=[]
   with torch.no_grad():
    for t in range(64):
     a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a);done=(te|tr).cpu().numpy();data=robot.data
     cmd=env.unwrapped.command_manager.get_command("base_velocity")
     ar=torch.linalg.vector_norm(a-prev,dim=-1);av=torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1);ti=tilt(data.root_quat_w)
     vz=data.root_lin_vel_b[:,2].abs();track=(data.root_lin_vel_b[:,0]-cmd[:,0]).abs()+(data.root_ang_vel_b[:,2]-cmd[:,2]).abs()
     rows.append({"t":t,"action_rate":ar.cpu().tolist(),"action_norm":torch.linalg.vector_norm(a,dim=-1).cpu().tolist(),
      "ang_vel_xy":av.cpu().tolist(),"tilt_deg":ti.cpu().tolist(),"abs_lin_vel_z":vz.cpu().tolist(),"tracking_abs_error":track.cpu().tolist(),"done":done.astype(int).tolist()})
     for i,d in enumerate(done):
      if d and first_done[i] is None:first_done[i]=t
     prev=a;cur=obs_tensor(nxt).cuda()
   failed=[i for i,x in enumerate(first_done) if x is not None]
   metrics={}
   for key in ("action_rate","action_norm","ang_vel_xy","tilt_deg","abs_lin_vel_z","tracking_abs_error"):
    A=np.array([r[key] for r in rows],float)
    metrics[key]={"all_mean":float(A.mean()),"failed_env_mean":float(A[:,failed].mean()) if failed else None,
                  "survivor_mean":float(A[:,[i for i in range(NENV) if i not in failed]].mean()) if len(failed)<NENV else None}
   out["snapshots"][str(snap)]={"first_done_step":first_done,"failed_envs":failed,"survival":float(1-len(failed)/NENV),"metrics":metrics,"rows":rows}
  (OUT/"audit.json").write_text(json.dumps(out,indent=2)+"\n")
  for s,x in out["snapshots"].items():
   print("\n",s,"survival",x["survival"],"failed",x["failed_envs"],"done",x["first_done_step"])
   for k,v in x["metrics"].items():print(k,round(v["all_mean"],4),v["failed_env_mean"],v["survivor_mean"])
 finally:
  if env is not None:env.close()
  app.close()
if __name__=="__main__":main()
