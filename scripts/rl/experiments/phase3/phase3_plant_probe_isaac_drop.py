#!/usr/bin/env python3
from pathlib import Path
import json
import torch
ROOT=Path(__file__).resolve().parents[4]
from isaaclab.app import AppLauncher
app=AppLauncher({"headless":True,"enable_cameras":False}).app
env=None
try:
    import gymnasium as gym,isaaclab_tasks
    from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
    cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=1;cfg.events.add_base_mass=None
    cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
    env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
    u=env.unwrapped;robot=u.scene["robot"];sensor=u.scene["contact_forces"];dev=u.device
    env.reset(seed=1)
    names=list(robot.data.joint_names)
    q0=robot.data.default_joint_pos.clone()
    root_pose=torch.tensor([[0.,0.,.60,1.,0.,0.,0.]],device=dev)
    robot.write_root_pose_to_sim(root_pose);robot.write_root_velocity_to_sim(torch.zeros((1,6),device=dev))
    robot.write_joint_state_to_sim(q0,torch.zeros_like(q0));u.scene.write_data_to_sim();u.sim.forward()
    foot_ids=[i for i,n in enumerate(sensor.body_names) if "foot" in n.lower()]
    env_ids=torch.tensor([0],device=dev,dtype=torch.int64);dt=float(cfg.sim.dt)
    rows=[];first=None
    for k in range(int(round(.50/dt))):
        tau=torch.zeros((1,robot.num_joints),device=dev)
        robot.root_physx_view.set_dof_actuation_forces(tau,env_ids)
        u.sim.step(render=False);u.scene.update(dt)
        f=sensor.data.net_forces_w[0,foot_ids].norm(dim=-1)
        contact=bool((f>1.0).any().item())
        if contact and first is None:first=k+1
        rows.append({"t":(k+1)*dt,"z":float(robot.data.root_com_pos_w[0,2]),
                     "vz":float(robot.data.root_com_vel_w[0,2]),"foot_contact":contact,
                     "foot_force_sum":float(f.sum())})
    rep={"schema":"phase3_plant_isaac_drop_v1","root_z0":.60,"dt":dt,
         "first_contact_t":None if first is None else first*dt,"rows":rows}
    out=ROOT/"runs/phase3_plant_equivalence_audit";out.mkdir(parents=True,exist_ok=True)
    (out/"isaac_drop.json").write_text(json.dumps(rep,indent=2)+"\n")
    print("first_contact_t",rep["first_contact_t"],"min_z",min(x["z"] for x in rows),"final_z",rows[-1]["z"],flush=True)
finally:
    if env is not None:env.close()
    app.close()
