#!/usr/bin/env python3
from pathlib import Path
import json,sys
import torch
ROOT=Path(__file__).resolve().parents[4]
from isaaclab.app import AppLauncher
app=AppLauncher({"headless":True,"enable_cameras":False}).app
env=None
try:
    import gymnasium as gym,isaaclab_tasks
    from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
    cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=1
    cfg.events.add_base_mass=None
    cfg.sim.gravity=(0.0,0.0,0.0)
    cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
    env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
    u=env.unwrapped;robot=u.scene["robot"];dev=u.device
    env.reset(seed=1)
    names=list(robot.data.joint_names);jid=names.index("FL_thigh_joint")
    q0=robot.data.default_joint_pos.clone()
    root_pose=torch.tensor([[0.,0.,1.0,1.,0.,0.,0.]],device=dev)
    root_vel=torch.zeros((1,6),device=dev)
    robot.write_root_pose_to_sim(root_pose);robot.write_root_velocity_to_sim(root_vel)
    robot.write_joint_state_to_sim(q0,torch.zeros_like(q0))
    u.scene.write_data_to_sim();u.sim.forward()
    rows=[];dt=float(cfg.sim.dt);env_ids=torch.tensor([0],device=dev,dtype=torch.int64)
    total_steps=22  # 110 ms
    for k in range(total_steps):
        tau=torch.zeros((1,robot.num_joints),device=dev)
        if k<2: tau[0,jid]=5.0  # 10 ms pulse
        robot.root_physx_view.set_dof_actuation_forces(tau,env_ids)
        u.sim.step(render=False);u.scene.update(dt)
        rows.append({
          "t":(k+1)*dt,
          "q":float(robot.data.joint_pos[0,jid]),
          "qdot":float(robot.data.joint_vel[0,jid]),
          "base_ang_vel_b":robot.data.root_ang_vel_b[0].detach().cpu().tolist(),
          "base_lin_vel_b":robot.data.root_lin_vel_b[0].detach().cpu().tolist(),
        })
    rep={"schema":"phase3_plant_isaac_joint_pulse_v1","joint":"FL_thigh_joint","torque_nm":5.0,
         "pulse_s":0.01,"gravity_off":True,"dt":dt,"rows":rows}
    out=ROOT/"runs/phase3_plant_equivalence_audit";out.mkdir(parents=True,exist_ok=True)
    (out/"isaac_joint_pulse.json").write_text(json.dumps(rep,indent=2)+"\n")
    print(json.dumps(rep,indent=2),flush=True)
finally:
    if env is not None:env.close()
    app.close()
