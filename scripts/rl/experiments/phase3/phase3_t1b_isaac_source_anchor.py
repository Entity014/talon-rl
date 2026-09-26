#!/usr/bin/env python3
from pathlib import Path
import json,math,sys
import numpy as np,torch

ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
ART=ROOT/"artifacts/phase1_canonical_actor_state.pt"
OUT=ROOT/"runs/phase3_t1b_source_anchor";OUT.mkdir(parents=True,exist_ok=True)
PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"C":np.array([.25]*4,np.float32)}
SEEDS=(840001,840002,840003,840004)
NENV=8;STEPS=64

def obs_tensor(x):
    if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
    return x if torch.is_tensor(x) else torch.as_tensor(x)

def main():
 from isaaclab.app import AppLauncher
 saved=sys.argv[:];sys.argv=[sys.argv[0]]
 app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
 env=None
 try:
  import gymnasium as gym,isaaclab_tasks
  from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
  from talon_rl.phase1_deployment import Phase1EagerStateRuntime
  cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
  cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
  cfg.observations.policy.enable_corruption=False
  env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
  u=env.unwrapped;robot=u.scene["robot"];rt=Phase1EagerStateRuntime(str(ART),device="cuda")
  reports=[];snapshots=[]
  for si,seed in enumerate(SEEDS):
   suite={}
   for lab,pref in PREFS.items():
    cur,_=env.reset(seed=seed);cur=obs_tensor(cur)
    cmd=u.command_manager.get_command("base_velocity").detach().cpu().numpy()
    if lab=="T":
      local_pos=(robot.data.root_link_pos_w-u.scene.env_origins).detach().cpu().numpy()
      snapshots.append({
        "suite":si,"seed":seed,
        "root_link_pos_local":local_pos.tolist(),
        "root_link_quat_w":robot.data.root_link_quat_w.detach().cpu().tolist(),
        "root_link_lin_vel_w":robot.data.root_link_lin_vel_w.detach().cpu().tolist(),
        "root_link_ang_vel_w":robot.data.root_link_ang_vel_w.detach().cpu().tolist(),
        "root_com_lin_vel_w":robot.data.root_com_lin_vel_w.detach().cpu().tolist(),
        "joint_pos":robot.data.joint_pos.detach().cpu().tolist(),
        "joint_vel":robot.data.joint_vel.detach().cpu().tolist(),
        "joint_names":list(robot.data.joint_names),
        "command":cmd.tolist(),
      })
    rt.last_action=np.zeros((NENV,12),np.float32);rt.estop_latched=False
    rows=[];prev=np.zeros((NENV,12),np.float32)
    pref_batch=np.repeat(pref[None,:],NENV,axis=0)
    for t in range(STEPS):
      a=rt.act(cur.detach().cpu().numpy().astype(np.float32),pref_batch)
      a_t=torch.from_numpy(a).to(u.device)
      nxt,_,te,tr,_=env.step(a_t)
      data=robot.data;v=data.root_lin_vel_b.detach().cpu().numpy();w=data.root_ang_vel_b.detach().cpu().numpy()
      cm=u.command_manager.get_command("base_velocity").detach().cpu().numpy()
      errxy=np.sum((cm[:,:2]-v[:,:2])**2,axis=1);erryaw=(cm[:,2]-w[:,2])**2
      T=(1.5*np.exp(-errxy/.25)+.75*np.exp(-erryaw/.25))/1.7194554805755615*.02
      phys=np.abs(v[:,0]-cm[:,0])+np.abs(w[:,2]-cm[:,2])
      rows.append({"T_obj":T.tolist(),"tracking":phys.tolist(),"action_rate":np.linalg.norm(a-prev,axis=1).tolist()})
      prev=a.copy();cur=obs_tensor(nxt)
    suite[lab]={
      "T_obj_mean":float(np.mean([x for r in rows for x in r["T_obj"]])),
      "tracking_mean":float(np.mean([x for r in rows for x in r["tracking"]])),
      "action_rate_mean":float(np.mean([x for r in rows for x in r["action_rate"]])),
    }
   dObj=suite["T"]["T_obj_mean"]-suite["C"]["T_obj_mean"]
   dPhys=suite["T"]["tracking_mean"]-suite["C"]["tracking_mean"]
   reports.append({"suite":si,"seed":seed,"T":suite["T"],"C":suite["C"],
                   "delta_T_obj":dObj,"delta_tracking":dPhys,
                   "objective_correct":bool(dObj>0),"physical_correct":bool(dPhys<0)})
   print("SUITE",si,"dObj",dObj,"dPhys",dPhys,flush=True)
  obj_frac=float(np.mean([r["objective_correct"] for r in reports]))
  phys_frac=float(np.mean([r["physical_correct"] for r in reports]))
  rep={"schema":"phase3_t1b_isaac_source_anchor_v1","clean_observation":True,
       "suites":reports,"objective_correct_fraction":obj_frac,"physical_correct_fraction":phys_frac,
       "anchor_pass":bool(obj_frac>=.75 and phys_frac>=.75),"snapshots":snapshots}
  (OUT/"isaac_source_anchor.json").write_text(json.dumps(rep,indent=2)+"\n")
  print("FINAL",json.dumps({k:v for k,v in rep.items() if k!="snapshots"},indent=2),flush=True)
 finally:
  if env is not None:env.close()
  app.close()

if __name__=="__main__":main()
