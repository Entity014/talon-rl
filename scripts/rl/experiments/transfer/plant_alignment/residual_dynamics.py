"""Consolidated experiment stage module."""
from __future__ import annotations
import argparse

def run_phase3_residual_isaac_base_impulse():
    """Run former phase3_residual_isaac_base_impulse.py stage."""
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

def run_phase3_residual_isaac_fixed_stance():
    """Run former phase3_residual_isaac_fixed_stance.py stage."""
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

def run_phase3_residual_mujoco_base_impulse():
    """Run former phase3_residual_mujoco_base_impulse.py stage."""
    from pathlib import Path
    import json
    import mujoco
    import numpy as np
    ROOT=Path(__file__).resolve().parents[4]
    SCENE=ROOT/"talon_rl/assets/data/Robots/unitree_a1/mujoco/scene_t4_passive_corrected.xml"
    OUT=ROOT/"runs/phase3_residual_plant_audit";OUT.mkdir(parents=True,exist_ok=True)
    
    def main():
        m=mujoco.MjModel.from_xml_path(str(SCENE));d=mujoco.MjData(m);m.opt.gravity[:]=0
        qa=int(m.jnt_qposadr[0]);trunk=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"trunk")
        mujoco.mj_resetData(m,d);d.qpos[qa:qa+3]=[0,0,1.0];d.qpos[qa+3:qa+7]=[1,0,0,0];d.qvel[:]=0;d.ctrl[:]=0
        names=("FL_hip_joint","FR_hip_joint","RL_hip_joint","RR_hip_joint","FL_thigh_joint","FR_thigh_joint","RL_thigh_joint","RR_thigh_joint","FL_calf_joint","FR_calf_joint","RL_calf_joint","RR_calf_joint")
        q0=np.array([.1,-.1,.1,-.1,.8,.8,1.,1.,-1.5,-1.5,-1.5,-1.5])
        for i,n in enumerate(names):
            j=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n);d.qpos[m.jnt_qposadr[j]]=q0[i]
        mujoco.mj_forward(m,d)
        dt=float(m.opt.timestep);steps=int(round(.06/dt));pulse=int(round(.01/dt));rows=[]
        for k in range(steps):
            d.xfrc_applied[:]=0
            if k<pulse:d.xfrc_applied[trunk,0]=20.0
            d.ctrl[:]=0;mujoco.mj_step(m,d)
            vw=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,trunk,vw,0)
            rows.append({"t":(k+1)*dt,"vcom":vw[3:].tolist()})
        rep={"schema":"phase3_residual_mujoco_base_impulse_v1","force_n":20.0,"pulse_s":.01,"dt":dt,"rows":rows}
        (OUT/"mujoco_base_impulse.json").write_text(json.dumps(rep,indent=2)+"\n")
        print("dv10",rows[pulse-1]["vcom"][0],flush=True)
    if True:main()

def run_phase3_residual_mujoco_fixed_stance():
    """Run former phase3_residual_mujoco_fixed_stance.py stage."""
    from pathlib import Path
    import json,sys,math
    import mujoco
    import numpy as np
    ROOT=Path(__file__).resolve().parents[4]
    sys.path.insert(0,str(ROOT))
    from talon_rl.deployment.phase1 import CANONICAL_JOINT_ORDER,CANONICAL_DEFAULT_Q,source_dcmotor_torque,canonical_base_kinematics
    SCENE=ROOT/"talon_rl/assets/data/Robots/unitree_a1/mujoco/scene_t4_passive_corrected.xml"
    OUT=ROOT/"runs/phase3_residual_plant_audit";OUT.mkdir(parents=True,exist_ok=True)
    def main():
        m=mujoco.MjModel.from_xml_path(str(SCENE));d=mujoco.MjData(m)
        jmap={}
        for j in range(m.njnt):
            n=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j)
            if n:jmap[n]=(int(m.jnt_qposadr[j]),int(m.jnt_dofadr[j]))
        act_joints=[]
        for a in range(m.nu):
            jid=int(m.actuator_trnid[a,0]);act_joints.append(mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,jid))
        qa=int(m.jnt_qposadr[0]);va=int(m.jnt_dofadr[0])
        mujoco.mj_resetData(m,d);d.qpos[qa:qa+3]=[0,0,.42];d.qpos[qa+3:qa+7]=[1,0,0,0];d.qvel[:]=0
        for i,n in enumerate(CANONICAL_JOINT_ORDER):d.qpos[jmap[n][0]]=CANONICAL_DEFAULT_Q[i]
        d.ctrl[:]=0;mujoco.mj_forward(m,d)
        hold=int(round(.02/m.opt.timestep));rows=[]
        target=CANONICAL_DEFAULT_Q.astype(float)
        for k in range(64):
            for _ in range(hold):
                q=np.array([d.qpos[jmap[n][0]] for n in CANONICAL_JOINT_ORDER])
                qd=np.array([d.qvel[jmap[n][1]] for n in CANONICAL_JOINT_ORDER])
                tau=source_dcmotor_torque(target,q,qd)
                d.ctrl[:]=np.array([tau[CANONICAL_JOINT_ORDER.index(n)] for n in act_joints])
                mujoco.mj_step(m,d)
            quat=d.qpos[qa+3:qa+7].copy();qv=d.qvel[va:va+6].copy();_,_,g=canonical_base_kinematics(quat,qv)
            tilt=math.degrees(math.acos(float(np.clip(-g[2],-1,1))))
            rows.append({"t":(k+1)*.02,"z":float(d.qpos[qa+2]),"tilt":tilt})
        rep={"schema":"phase3_residual_mujoco_fixed_stance_v1","rows":rows}
        (OUT/"mujoco_fixed_stance.json").write_text(json.dumps(rep,indent=2)+"\n")
        print("minz",min(x["z"] for x in rows),"final",rows[-1]["z"],"maxtilt",max(x["tilt"] for x in rows),flush=True)
    if True:main()

STAGES = {
    "phase3_residual_isaac_base_impulse": run_phase3_residual_isaac_base_impulse,
    "phase3_residual_isaac_fixed_stance": run_phase3_residual_isaac_fixed_stance,
    "phase3_residual_mujoco_base_impulse": run_phase3_residual_mujoco_base_impulse,
    "phase3_residual_mujoco_fixed_stance": run_phase3_residual_mujoco_fixed_stance,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
