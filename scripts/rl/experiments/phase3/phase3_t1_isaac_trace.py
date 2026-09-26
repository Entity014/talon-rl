#!/usr/bin/env python3
from pathlib import Path
import json,math,sys
import numpy as np, torch

ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
ART=ROOT/"artifacts/phase1_canonical_actor_state.pt"
OUT=ROOT/"runs/phase3_t1_matched_trace";OUT.mkdir(parents=True,exist_ok=True)
COMMANDS={
 "forward":[.5,0.,0.],
 "turn_left":[.3,0.,.3],
 "turn_right":[.3,0.,-.3],
 "lateral":[0.,.25,0.],
}
PREFS={"T":[.7,.1,.1,.1],"C":[.25,.25,.25,.25]}
CANON=("FL_hip_joint","FR_hip_joint","RL_hip_joint","RR_hip_joint",
       "FL_thigh_joint","FR_thigh_joint","RL_thigh_joint","RR_thigh_joint",
       "FL_calf_joint","FR_calf_joint","RL_calf_joint","RR_calf_joint")
Q0=torch.tensor([.1,-.1,.1,-.1,.8,.8,1.,1.,-1.5,-1.5,-1.5,-1.5])

def tilt(q):
    _,x,y,_=[q[:,i] for i in range(4)]
    return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))

def main():
 from isaaclab.app import AppLauncher
 saved=sys.argv[:];sys.argv=[sys.argv[0]]
 app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
 env=None
 try:
  import gymnasium as gym,isaaclab_tasks
  from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
  from talon_rl.phase1_deployment import Phase1EagerStateRuntime
  cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=1;cfg.seed=0
  cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
  cfg.observations.policy.enable_corruption=False
  cfg.commands.base_velocity.heading_command=False
  cfg.commands.base_velocity.rel_heading_envs=0.0
  cfg.commands.base_velocity.rel_standing_envs=0.0
  cfg.commands.base_velocity.resampling_time_range=(1000.0,1000.0)
  env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
  u=env.unwrapped;robot=u.scene["robot"];sensor=u.scene["contact_forces"]
  rt=Phase1EagerStateRuntime(str(ART),device="cuda")
  joint_ids=[robot.data.joint_names.index(n) for n in CANON]
  foot_ids=[i for i,n in enumerate(sensor.body_names) if "foot" in n.lower()]
  term=u.command_manager.get_term("base_velocity")
  traces={}
  for cname,cmdlist in COMMANDS.items():
   cmd=torch.tensor(cmdlist,device=u.device,dtype=torch.float32).view(1,3)
   for plab,pref in PREFS.items():
    env.reset(seed=260926)
    root_pose=torch.tensor([[0.,0.,.43,1.,0.,0.,0.]],device=u.device)
    root_vel=torch.zeros((1,6),device=u.device)
    robot.write_root_pose_to_sim(root_pose);robot.write_root_velocity_to_sim(root_vel)
    qp=torch.zeros((1,robot.num_joints),device=u.device);qv=torch.zeros_like(qp)
    qp[:,joint_ids]=Q0.to(u.device)
    robot.write_joint_state_to_sim(qp,qv)
    u.scene.write_data_to_sim();u.sim.forward()
    term.vel_command_b[:]=cmd
    term.is_standing_env[:]=False;term.is_heading_env[:]=False
    u.obs_buf=u.observation_manager.compute(update_history=True)
    prev=np.zeros(12,np.float32);rt.last_action=np.zeros((1,12),np.float32);rt.estop_latched=False
    pref_np=np.asarray(pref,np.float32)[None,:]
    rows=[];cumT=0.0
    for t in range(64):
      term.vel_command_b[:]=cmd
      obs=u.obs_buf["policy"] if isinstance(u.obs_buf,dict) else u.obs_buf
      a=rt.act(obs.detach().cpu().numpy().astype(np.float32),pref_np)[0]
      a_t=torch.from_numpy(a).to(u.device).view(1,-1)
      nxt,_,te,tr,_=env.step(a_t)
      data=robot.data;v=data.root_lin_vel_b[0].detach().cpu().numpy();w=data.root_ang_vel_b[0].detach().cpu().numpy()
      g=data.projected_gravity_b[0].detach().cpu().numpy()
      errxy=float(np.sum((np.asarray(cmdlist[:2])-v[:2])**2));erryaw=float((cmdlist[2]-w[2])**2)
      Treward=1.5*math.exp(-errxy/.25)+.75*math.exp(-erryaw/.25);normT=Treward/1.7194554805755615*.02;cumT+=normT
      f=sensor.data.net_forces_w[0,foot_ids].norm(dim=-1)
      rows.append({"t":t,"vx":float(v[0]),"vy":float(v[1]),"vz":float(v[2]),
        "wx":float(w[0]),"wy":float(w[1]),"wz":float(w[2]),
        "g":g.tolist(),"height":float(data.root_link_pos_w[0,2]),"tilt_deg":float(tilt(data.root_link_quat_w)[0]),
        "tracking_error":float(abs(v[0]-cmdlist[0])+abs(w[2]-cmdlist[2])),
        "T_obj":float(normT),"cum_T":float(cumT),"action":a.tolist(),
        "action_rate":float(np.linalg.norm(a-prev)),"sat_frac":float(np.mean(np.abs(a)>=.98)),
        "contacts":int((f>1.0).sum().item()),"q":data.joint_pos[0,joint_ids].detach().cpu().tolist(),
        "qd":data.joint_vel[0,joint_ids].detach().cpu().tolist(),
        "done":bool((te|tr)[0].item())})
      prev=a.copy();u.obs_buf=nxt
    traces[f"{cname}:{plab}"]=rows
    print("DONE",cname,plab,flush=True)
  rep={"schema":"phase3_t1_isaac_matched_trace_v1","engine":"isaac","clean_obs":True,
       "matched_root_pose":[0,0,.43,1,0,0,0],"commands":COMMANDS,"traces":traces}
  (OUT/"isaac_trace.json").write_text(json.dumps(rep,indent=2)+"\n")
 finally:
  if env is not None:env.close()
  app.close()

if __name__=="__main__":main()
