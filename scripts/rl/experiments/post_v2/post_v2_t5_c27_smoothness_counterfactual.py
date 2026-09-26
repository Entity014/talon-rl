#!/usr/bin/env python3
from pathlib import Path
import sys,json,numpy as np,torch
ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
RUN=ROOT/"runs/post_v2_t5_c25_actor_updating25-2026-09-23";OUT=ROOT/"runs/post_v2_t5_c27_smoothness_safety-2026-09-23"
exec(open(ROOT/"scripts/rl/post_v2_t5_c25_eval.py").read().split("def main():")[0])
SEEDS=[2840503]
def rollout(env,m,w,seed):
 cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda();robot=env.unwrapped.scene["robot"];done=np.zeros(NENV,bool);prev=torch.zeros((NENV,env.unwrapped.action_manager.total_action_dim),device="cuda")
 ar=[];ti=[];track=[]
 with torch.no_grad():
  for _ in range(64):
   a=m.act_inference_with_preference(cur,w);nxt,_,te,tr,_=env.step(a);data=robot.data;cmd=env.unwrapped.command_manager.get_command("base_velocity")
   done|=(te|tr).cpu().numpy();ar.append(float(torch.linalg.vector_norm(a-prev,dim=-1).mean()));ti.append(float(tilt(data.root_quat_w).mean()))
   track.append(float(((data.root_lin_vel_b[:,0]-cmd[:,0]).abs()+(data.root_ang_vel_b[:,2]-cmd[:,2]).abs()).mean()))
   prev=a;cur=obs_tensor(nxt).cuda()
 return {"survival":float(1-done.mean()),"action_rate":float(np.mean(ar)),"tilt_deg":float(np.mean(ti)),"tracking_error":float(np.mean(track))}
def main():
 from isaaclab.app import AppLauncher
 app=AppLauncher({"headless":True,"enable_cameras":False}).app;env=None
 try:
  import gymnasium as gym,isaaclab_tasks
  from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
  from talon_rl.t4_actor_critic import T4SharedActorCritic
  cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0;cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
  env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);o,_=env.reset(seed=0);o=obs_tensor(o).cuda();od=o.shape[-1];ad=env.unwrapped.action_manager.total_action_dim;mgr=env.unwrapped.reward_manager
  m=T4SharedActorCritic(od,ad).cuda();m.load_state_dict(torch.load(RUN/"S_snap_25.pt",map_location="cuda",weights_only=False)["model"]);m.eval();w=torch.tensor(PREFS["S"],device="cuda").repeat(NENV,1)
  gs=[grad_perturb(env,m,w,mgr,3,3510000+i*211) for i in range(8)]
  g=torch.stack(gs).mean(0);aps=[p for n,p in m.named_parameters() if n.startswith("actor_")];base=[p.detach().clone() for p in aps];basevec=torch.cat([p.reshape(-1) for p in base])
  unit=-g/(g.norm()+1e-12)*(1e-4*(basevec.norm()+1e-12))
  models={"baseline":m}
  for name,scale in (("toward_smooth",1.0),("against_smooth", -1.0),("against_smooth_2x",-2.0)):
   pm=T4SharedActorCritic(od,ad).cuda();pm.load_state_dict(torch.load(RUN/"S_snap_25.pt",map_location="cuda",weights_only=False)["model"]);paps=[p for n,p in pm.named_parameters() if n.startswith("actor_")]
   setflat(paps,[p.detach().clone() for p in paps],unit*scale);pm.eval();models[name]=pm
  out={"schema":"c27_smoothness_counterfactual_v1","grad_norm":float(g.norm()),"rows":[]}
  for seed in SEEDS:
   row={"seed":seed}
   for name,pm in models.items():row[name]=rollout(env,pm,w,seed)
   out["rows"].append(row)
  (OUT/"counterfactual.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
 finally:
  if env is not None:env.close()
  app.close()
if __name__=="__main__":main()
