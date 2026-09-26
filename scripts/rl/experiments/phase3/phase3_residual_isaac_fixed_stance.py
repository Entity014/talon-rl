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
    cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=1;cfg.events.add_base_mass=None
    cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
    env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);u=env.unwrapped;robot=u.scene["robot"];dev=u.device
    env.reset(seed=1)
    root=torch.tensor([[0.,0.,.42,1.,0.,0.,0.]],device=dev)
    robot.write_root_pose_to_sim(root);robot.write_root_velocity_to_sim(torch.zeros((1,6),device=dev))
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(),torch.zeros_like(robot.data.default_joint_pos))
    u.scene.write_data_to_sim();u.sim.forward()
    rows=[]
    for k in range(64):
        env.step(torch.zeros((1,12),device=dev))
        rows.append({"t":(k+1)*u.step_dt,"z":float(robot.data.root_link_pos_w[0,2]),
                     "tilt":float(torch.rad2deg(torch.acos((-robot.data.projected_gravity_b[0,2]).clamp(-1,1))))})
    rep={"schema":"phase3_residual_isaac_fixed_stance_v1","rows":rows}
    out=ROOT/"runs/phase3_residual_plant_audit";out.mkdir(parents=True,exist_ok=True)
    (out/"isaac_fixed_stance.json").write_text(json.dumps(rep,indent=2)+"\n")
    print("minz",min(x["z"] for x in rows),"final",rows[-1]["z"],flush=True)
finally:
    if env is not None:env.close()
    app.close()
