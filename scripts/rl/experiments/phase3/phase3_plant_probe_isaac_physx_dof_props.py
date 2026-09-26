#!/usr/bin/env python3
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[4]
from isaaclab.app import AppLauncher
app=AppLauncher({"headless":True,"enable_cameras":False}).app
env=None
try:
    import gymnasium as gym,isaaclab_tasks
    from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
    cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=1;cfg.events.add_base_mass=None
    cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
    env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);env.reset(seed=1)
    robot=env.unwrapped.scene["robot"];v=robot.root_physx_view
    def get(name):
        fn=getattr(v,name,None)
        if fn is None:return None
        x=fn()
        try:return x[0].cpu().tolist()
        except:return str(x)
    rep={
      "schema":"phase3_plant_isaac_physx_dof_props_v1",
      "joint_names":list(robot.data.joint_names),
      "physx_stiffness":get("get_dof_stiffnesses"),
      "physx_damping":get("get_dof_dampings"),
      "physx_friction":get("get_dof_friction_coefficients"),
      "physx_armature":get("get_dof_armatures"),
      "data_default_stiffness":robot.data.default_joint_stiffness[0].cpu().tolist(),
      "data_default_damping":robot.data.default_joint_damping[0].cpu().tolist(),
      "data_default_friction":robot.data.default_joint_friction_coeff[0].cpu().tolist(),
      "data_default_armature":robot.data.default_joint_armature[0].cpu().tolist(),
    }
    out=ROOT/"runs/phase3_plant_equivalence_audit";out.mkdir(parents=True,exist_ok=True)
    (out/"isaac_physx_dof_props.json").write_text(json.dumps(rep,indent=2)+"\n")
    print(json.dumps(rep,indent=2),flush=True)
finally:
    if env is not None:env.close()
    app.close()
