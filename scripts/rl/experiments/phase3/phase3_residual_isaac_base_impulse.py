#!/usr/bin/env python3
from pathlib import Path
import json,torch
ROOT=Path(__file__).resolve().parents[4]
from isaaclab.app import AppLauncher
app=AppLauncher({"headless":True,"enable_cameras":False}).app
env=None
try:
    import gymnasium as gym,isaaclab_tasks
    from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
    cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=1;cfg.events.add_base_mass=None;cfg.sim.gravity=(0.,0.,0.)
    cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
    env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
    u=env.unwrapped;robot=u.scene["robot"];dev=u.device;env.reset(seed=1)
    robot.write_root_pose_to_sim(torch.tensor([[0.,0.,1.,1.,0.,0.,0.]],device=dev))
    robot.write_root_velocity_to_sim(torch.zeros((1,6),device=dev))
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(),torch.zeros_like(robot.data.default_joint_pos))
    u.scene.write_data_to_sim();u.sim.forward()
    dt=float(cfg.sim.dt);env_ids=torch.tensor([0],device=dev,dtype=torch.long)
    rows=[]
    for k in range(12):
        f=torch.zeros((1,1,3),device=dev);t=torch.zeros_like(f)
        if k<2:f[0,0,0]=20.0
        robot.set_external_force_and_torque(f,t,body_ids=[0],env_ids=[0],is_global=True)
        u.scene.write_data_to_sim()
        robot.root_physx_view.set_dof_actuation_forces(torch.zeros((1,robot.num_joints),device=dev),env_ids)
        u.sim.step(render=False);u.scene.update(dt)
        rows.append({"t":(k+1)*dt,"vcom":robot.data.root_com_lin_vel_w[0].detach().cpu().tolist()})
    rep={"schema":"phase3_residual_isaac_base_impulse_v1","force_n":20.0,"pulse_s":.01,"dt":dt,"rows":rows}
    out=ROOT/"runs/phase3_residual_plant_audit";out.mkdir(parents=True,exist_ok=True)
    (out/"isaac_base_impulse.json").write_text(json.dumps(rep,indent=2)+"\n")
    print("dv10",rows[1]["vcom"][0],flush=True)
finally:
    if env is not None:env.close()
    app.close()
