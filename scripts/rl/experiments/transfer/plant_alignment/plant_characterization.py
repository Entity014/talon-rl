"""Consolidated experiment stage module."""
from __future__ import annotations
import argparse

def run_phase3_plant_audit_isaac_nominal_snapshot():
    """Run former phase3_plant_audit_isaac_nominal_snapshot.py stage."""
    """Isaac plant snapshot with base-mass randomisation switched off."""
    import sys
    from pathlib import Path
    
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    
    from rl.experiments.common.utilities.isaac_plant_snapshot import IsaacPlantSnapshot
    
    
    class IsaacNominalPlantSnapshot(IsaacPlantSnapshot):
        """Isaac plant snapshot of the nominal model, not one sampled instance."""
    
        report = "isaac_nominal_snapshot.json"
        schema = "phase3_plant_isaac_nominal_snapshot_v1"
    
        def configure(self, cfg):
            cfg.events.add_base_mass = None
    
    
    if True:
        IsaacNominalPlantSnapshot.main()

def run_phase3_plant_audit_isaac_snapshot():
    """Run former phase3_plant_audit_isaac_snapshot.py stage."""
    """Isaac plant snapshot of the scene exactly as configured."""
    import sys
    from pathlib import Path
    
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    
    from rl.experiments.common.utilities.isaac_plant_snapshot import IsaacPlantSnapshot
    
    
    class IsaacConfiguredPlantSnapshot(IsaacPlantSnapshot):
        """Isaac plant snapshot of the scene exactly as configured."""
    
        report = "isaac_snapshot.json"
        schema = "phase3_plant_isaac_snapshot_v1"
    
    
    if True:
        IsaacConfiguredPlantSnapshot.main()

def run_phase3_plant_audit_mujoco_snapshot():
    """Run former phase3_plant_audit_mujoco_snapshot.py stage."""
    """Every plant parameter the MuJoCo A1 scene declares, as one record.
    
    Read-only. This is the MuJoCo half of the plant-equivalence comparison, so it
    reports what the model says rather than what a rollout does.
    """
    import json
    import sys
    from pathlib import Path
    
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    
    from rl.core.diagnostics.offline_audit import REPO, OfflineAudit
    
    SCENE = REPO / "talon_rl/assets/data/Robots/unitree_a1/mujoco/scene.xml"
    
    
    class MujocoPlantSnapshot(OfflineAudit):
        """Every plant parameter the MuJoCo A1 scene declares."""
    
        run = "phase3_plant_equivalence_audit"
        report = "mujoco_snapshot.json"
        schema = "phase3_plant_mujoco_snapshot_v1"
    
        def analyze(self):
            import mujoco
    
            m = mujoco.MjModel.from_xml_path(str(SCENE))
            name = lambda kind, i: mujoco.mj_id2name(m, kind, i)  # noqa: E731
    
            bodies = [(name(mujoco.mjtObj.mjOBJ_BODY, i), i) for i in range(1, m.nbody)]
            # the free joint carries no damping/armature/limits, so it is skipped
            joints = [(name(mujoco.mjtObj.mjOBJ_JOINT, j), j) for j in range(m.njnt)
                      if int(m.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_FREE)]
            geoms = [{"name": name(mujoco.mjtObj.mjOBJ_GEOM, g),
                      "body": name(mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])),
                      "friction": m.geom_friction[g].tolist(),
                      "solref": m.geom_solref[g].tolist(),
                      "solimp": m.geom_solimp[g].tolist(),
                      "condim": int(m.geom_condim[g])}
                     for g in range(m.ngeom)]
    
            return {
                "schema": self.schema,
                "scene": str(SCENE.relative_to(REPO)),
                "sim_dt": float(m.opt.timestep),
                "integrator": int(m.opt.integrator),
                "solver": int(m.opt.solver),
                "iterations": int(m.opt.iterations),
                "ls_iterations": int(m.opt.ls_iterations),
                "body_names": [n for n, _ in bodies],
                "mass": [float(m.body_mass[i]) for _, i in bodies],
                "body_ipos": [m.body_ipos[i].tolist() for _, i in bodies],
                "body_iquat": [m.body_iquat[i].tolist() for _, i in bodies],
                "body_inertia_diagonal": [m.body_inertia[i].tolist() for _, i in bodies],
                "joint_names": [n for n, _ in joints],
                "joint_damping": [float(m.dof_damping[int(m.jnt_dofadr[j])]) for _, j in joints],
                "joint_armature": [float(m.dof_armature[int(m.jnt_dofadr[j])]) for _, j in joints],
                "joint_frictionloss": [float(m.dof_frictionloss[int(m.jnt_dofadr[j])]) for _, j in joints],
                "joint_range": [m.jnt_range[j].tolist() for _, j in joints],
                "geoms": geoms,
            }
    
        def summarize(self, report):
            print(json.dumps(report, indent=2), flush=True)
    
    
    if True:
        MujocoPlantSnapshot.main()

def run_phase3_plant_probe_isaac_drop():
    """Run former phase3_plant_probe_isaac_drop.py stage."""
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

def run_phase3_plant_probe_isaac_joint_pulse():
    """Run former phase3_plant_probe_isaac_joint_pulse.py stage."""
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

def run_phase3_plant_probe_isaac_physx_dof_props():
    """Run former phase3_plant_probe_isaac_physx_dof_props.py stage."""
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

def run_phase3_plant_probe_mujoco_drop():
    """Run former phase3_plant_probe_mujoco_drop.py stage."""
    from pathlib import Path
    import json
    import mujoco
    import numpy as np
    ROOT=Path(__file__).resolve().parents[4]
    SCENE=ROOT/"talon_rl/assets/data/Robots/unitree_a1/mujoco/scene_t3_dcmotor.xml"
    OUT=ROOT/"runs/phase3_plant_equivalence_audit";OUT.mkdir(parents=True,exist_ok=True)
    CANON=("FL_hip_joint","FR_hip_joint","RL_hip_joint","RR_hip_joint",
           "FL_thigh_joint","FR_thigh_joint","RL_thigh_joint","RR_thigh_joint",
           "FL_calf_joint","FR_calf_joint","RL_calf_joint","RR_calf_joint")
    Q0=np.array([.1,-.1,.1,-.1,.8,.8,1.,1.,-1.5,-1.5,-1.5,-1.5],float)
    def main():
        m=mujoco.MjModel.from_xml_path(str(SCENE));d=mujoco.MjData(m)
        jmap={}
        for j in range(m.njnt):
            n=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j)
            if n:jmap[n]=(int(m.jnt_qposadr[j]),int(m.jnt_dofadr[j]))
        qa=int(m.jnt_qposadr[0]);va=int(m.jnt_dofadr[0])
        trunk=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"trunk")
        mujoco.mj_resetData(m,d);d.qpos[qa:qa+3]=[0,0,.60];d.qpos[qa+3:qa+7]=[1,0,0,0];d.qvel[:]=0;d.ctrl[:]=0
        for i,n in enumerate(CANON):d.qpos[jmap[n][0]]=Q0[i]
        mujoco.mj_forward(m,d)
        rows=[];first=None;dt=float(m.opt.timestep);steps=int(round(.50/dt))
        for k in range(steps):
            d.ctrl[:]=0;mujoco.mj_step(m,d)
            contact=False
            for c in range(d.ncon):
                con=d.contact[c];b1=int(m.geom_bodyid[con.geom1]);b2=int(m.geom_bodyid[con.geom2])
                if b1==0 or b2==0:
                    contact=True;break
            if contact and first is None:first=k+1
            # COM vertical velocity: body COM cvel world linear part
            vw=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,trunk,vw,0)
            rows.append({"t":(k+1)*dt,"z":float(d.xipos[trunk,2]),"vz":float(vw[5]),"contact":contact})
        rep={"schema":"phase3_plant_mujoco_drop_v1","root_z0":.60,"dt":dt,
             "first_contact_t":None if first is None else first*dt,"rows":rows}
        (OUT/"mujoco_drop.json").write_text(json.dumps(rep,indent=2)+"\n")
        print("first",rep["first_contact_t"],"min_z",min(x["z"] for x in rows),"final_z",rows[-1]["z"],flush=True)
    if True:main()

def run_phase3_plant_probe_mujoco_joint_pulse():
    """Run former phase3_plant_probe_mujoco_joint_pulse.py stage."""
    from pathlib import Path
    import json,sys
    import mujoco
    import numpy as np
    
    ROOT=Path(__file__).resolve().parents[4]
    SCENE=ROOT/"talon_rl/assets/data/Robots/unitree_a1/mujoco/scene_t3_dcmotor.xml"
    OUT=ROOT/"runs/phase3_plant_equivalence_audit";OUT.mkdir(parents=True,exist_ok=True)
    CANON=("FL_hip_joint","FR_hip_joint","RL_hip_joint","RR_hip_joint",
           "FL_thigh_joint","FR_thigh_joint","RL_thigh_joint","RR_thigh_joint",
           "FL_calf_joint","FR_calf_joint","RL_calf_joint","RR_calf_joint")
    Q0=np.array([.1,-.1,.1,-.1,.8,.8,1.,1.,-1.5,-1.5,-1.5,-1.5],float)
    
    def main():
        m=mujoco.MjModel.from_xml_path(str(SCENE));d=mujoco.MjData(m)
        m.opt.gravity[:]=0
        jmap={}
        for j in range(m.njnt):
            n=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j)
            if n:jmap[n]=(int(m.jnt_qposadr[j]),int(m.jnt_dofadr[j]))
        act_joints=[]
        for a in range(m.nu):
            jid=int(m.actuator_trnid[a,0]);act_joints.append(mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,jid))
        mujoco.mj_resetData(m,d)
        qa=int(m.jnt_qposadr[0]);va=int(m.jnt_dofadr[0])
        d.qpos[qa:qa+3]=[0,0,1.0];d.qpos[qa+3:qa+7]=[1,0,0,0];d.qvel[:]=0
        for i,n in enumerate(CANON):d.qpos[jmap[n][0]]=Q0[i]
        d.ctrl[:]=0;mujoco.mj_forward(m,d)
        target_joint="FL_thigh_joint";aidx=act_joints.index(target_joint);qadr,vadr=jmap[target_joint]
        rows=[];dt=float(m.opt.timestep);steps=int(round(.110/dt));pulse_steps=int(round(.010/dt))
        for k in range(steps):
            d.ctrl[:]=0
            if k<pulse_steps:d.ctrl[aidx]=5.0
            mujoco.mj_step(m,d)
            # free-joint rotational qvel is body frame
            rows.append({"t":(k+1)*dt,"q":float(d.qpos[qadr]),"qdot":float(d.qvel[vadr]),
                         "base_ang_vel_b":d.qvel[va+3:va+6].tolist(),
                         "base_lin_vel_world":d.qvel[va:va+3].tolist()})
        rep={"schema":"phase3_plant_mujoco_joint_pulse_v1","joint":target_joint,"torque_nm":5.0,
             "pulse_s":0.01,"gravity_off":True,"dt":dt,"rows":rows}
        (OUT/"mujoco_joint_pulse.json").write_text(json.dumps(rep,indent=2)+"\n")
        print(json.dumps(rep,indent=2),flush=True)
    
    if True:main()

STAGES = {
    "phase3_plant_audit_isaac_nominal_snapshot": run_phase3_plant_audit_isaac_nominal_snapshot,
    "phase3_plant_audit_isaac_snapshot": run_phase3_plant_audit_isaac_snapshot,
    "phase3_plant_audit_mujoco_snapshot": run_phase3_plant_audit_mujoco_snapshot,
    "phase3_plant_probe_isaac_drop": run_phase3_plant_probe_isaac_drop,
    "phase3_plant_probe_isaac_joint_pulse": run_phase3_plant_probe_isaac_joint_pulse,
    "phase3_plant_probe_isaac_physx_dof_props": run_phase3_plant_probe_isaac_physx_dof_props,
    "phase3_plant_probe_mujoco_drop": run_phase3_plant_probe_mujoco_drop,
    "phase3_plant_probe_mujoco_joint_pulse": run_phase3_plant_probe_mujoco_joint_pulse,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
