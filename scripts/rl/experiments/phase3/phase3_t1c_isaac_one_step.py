#!/usr/bin/env python3
from pathlib import Path
import json,sys
import numpy as np,torch

ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
ART=ROOT/"artifacts/phase1_canonical_actor_state.pt"
OUT=ROOT/"runs/phase3_t1c_one_step";OUT.mkdir(parents=True,exist_ok=True)
SEEDS=(840001,840002,840003,840004);NENV=8
PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"C":np.array([.25]*4,np.float32)}

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
  suites=[]
  for si,seed in enumerate(SEEDS):
   # Capture source state and actions from the same reset.
   cur,_=env.reset(seed=seed);cur=obs_tensor(cur)
   cmd=u.command_manager.get_command("base_velocity").detach().cpu().numpy()
   actions={}
   for lab,pref in PREFS.items():
    rt.last_action=np.zeros((NENV,12),np.float32);rt.estop_latched=False
    actions[lab]=rt.act(cur.detach().cpu().numpy().astype(np.float32),np.repeat(pref[None,:],NENV,axis=0))
   init={
    "root_link_pos_local":(robot.data.root_link_pos_w-u.scene.env_origins).detach().cpu().tolist(),
    "root_link_quat_w":robot.data.root_link_quat_w.detach().cpu().tolist(),
    "root_link_lin_vel_w":robot.data.root_link_lin_vel_w.detach().cpu().tolist(),
    "root_link_ang_vel_w":robot.data.root_link_ang_vel_w.detach().cpu().tolist(),
    "joint_pos":robot.data.joint_pos.detach().cpu().tolist(),
    "joint_vel":robot.data.joint_vel.detach().cpu().tolist(),
    "joint_names":list(robot.data.joint_names),"command":cmd.tolist(),
   }
   nexts={}
   for lab in ("T","C"):
    cur,_=env.reset(seed=seed);cur=obs_tensor(cur)
    a=torch.from_numpy(actions[lab]).to(u.device)
    env.step(a)
    v=robot.data.root_lin_vel_b.detach().cpu().numpy()
    w=robot.data.root_ang_vel_b.detach().cpu().numpy()
    g=robot.data.projected_gravity_b.detach().cpu().numpy()
    cm=u.command_manager.get_command("base_velocity").detach().cpu().numpy()
    q=robot.data.joint_pos.detach().cpu().numpy();qd=robot.data.joint_vel.detach().cpu().numpy()
    tr=np.abs(v[:,0]-cm[:,0])+np.abs(w[:,2]-cm[:,2])
    nexts[lab]={"v":v.tolist(),"w":w.tolist(),"g":g.tolist(),"q":q.tolist(),"qd":qd.tolist(),
                "height":robot.data.root_link_pos_w[:,2].detach().cpu().tolist(),"tracking":tr.tolist()}
   suites.append({"suite":si,"seed":seed,"initial":init,
                  "actions":{k:v.tolist() for k,v in actions.items()},"next":nexts})
   print("SUITE",si,flush=True)
  rep={"schema":"phase3_t1c_isaac_one_step_v1","clean_observation":True,"suites":suites}
  (OUT/"isaac_one_step.json").write_text(json.dumps(rep,indent=2)+"\n")
 finally:
  if env is not None:env.close()
  app.close()

if __name__=="__main__":main()
