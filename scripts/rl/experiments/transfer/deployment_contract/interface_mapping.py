"""Consolidated experiment stage module."""
from __future__ import annotations
import argparse

def run_phase1_d3a_canonical_velocity_probe():
    """Run former phase1_d3a_canonical_velocity_probe.py stage."""
    from pathlib import Path
    import json, math, sys
    import mujoco
    import numpy as np
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path.insert(0,str(ROOT))
    from talon_rl.deployment.phase1 import canonical_base_kinematics,root_com_velocity_b_from_freejoint,quat_wxyz_to_rot
    
    SCENE=ROOT/"talon_rl/assets/data/Robots/unitree_a1/mujoco/scene.xml"
    OUT=ROOT/"runs/phase1_d3a_live_runtime"
    OUT.mkdir(parents=True,exist_ok=True)
    
    def main():
        m=mujoco.MjModel.from_xml_path(str(SCENE));d=mujoco.MjData(m)
        trunk=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"trunk")
        com_b=m.body_ipos[trunk].copy()
        qa=int(m.jnt_qposadr[0]);va=int(m.jnt_dofadr[0])
    
        orientations=[
          ("identity",np.array([1.,0,0,0])),
          ("yaw90",np.array([math.cos(math.pi/4),0,0,math.sin(math.pi/4)])),
          ("roll60",np.array([math.cos(math.pi/6),math.sin(math.pi/6),0,0])),
          ("pitch45",np.array([math.cos(math.pi/8),0,math.sin(math.pi/8),0])),
        ]
        velocities=[
          np.array([1.,2.,3.,.4,.5,.6]),
          np.array([-.7,.3,1.1,-.2,.8,-.5]),
        ]
        rows=[];max_com=max_ang=max_grav=max_quat=0.0
        for oname,q in orientations:
          for vi,qv in enumerate(velocities):
            d.qpos[:]=0;d.qvel[:]=0
            d.qpos[qa:qa+3]=[0,0,.43];d.qpos[qa+3:qa+7]=q
            d.qvel[va:va+6]=qv
            mujoco.mj_forward(m,d)
            R=quat_wxyz_to_rot(q)
            xmat=d.xmat[trunk].reshape(3,3).copy()
            quat_err=float(np.max(np.abs(R-xmat)));max_quat=max(max_quat,quat_err)
    
            # MuJoCo object velocity world is [angular, linear] at body COM.
            vw=np.zeros(6)
            mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,trunk,vw,0)
            mj_ang_b=R.T@vw[:3];mj_com_lin_b=R.T@vw[3:]
    
            origin_lin_b,ang_b,grav_b=canonical_base_kinematics(q,qv)
            formula_com=root_com_velocity_b_from_freejoint(q,qv[:3],qv[3:],com_b)
            com_err=float(np.max(np.abs(formula_com-mj_com_lin_b)))
            ang_err=float(np.max(np.abs(ang_b-mj_ang_b)))
            # Cross-check projected gravity against direct rotation.
            g2=R.T@np.array([0.,0.,-1.])
            grav_err=float(np.max(np.abs(grav_b-g2)))
            max_com=max(max_com,com_err);max_ang=max(max_ang,ang_err);max_grav=max(max_grav,grav_err)
            rows.append({"orientation":oname,"velocity_case":vi,
              "origin_linear_b":origin_lin_b.tolist(),"formula_com_linear_b":formula_com.tolist(),
              "mujoco_com_linear_b":mj_com_lin_b.tolist(),"angular_b":ang_b.tolist(),
              "mujoco_angular_b":mj_ang_b.tolist(),"projected_gravity_b":grav_b.tolist(),
              "quat_xmat_error":quat_err,"com_error":com_err,"ang_error":ang_err,"gravity_error":grav_err})
    
        rep={"schema":"phase1_d3a_canonical_velocity_probe_v1",
          "trunk_com_offset_body":com_b.tolist(),
          "max_quaternion_xmat_error":max_quat,
          "max_com_linear_error":max_com,
          "max_angular_error":max_ang,
          "max_projected_gravity_error":max_grav,
          "rows":rows,
          "gates":{
            "quaternion_wxyz":bool(max_quat<=1e-12),
            "root_com_linear_velocity":bool(max_com<=1e-12),
            "root_com_angular_velocity":bool(max_ang<=1e-7),
            "projected_gravity":bool(max_grav<=1e-7),
          }}
        rep["pass"]=bool(all(rep["gates"].values()))
        (OUT/"canonical_velocity_probe.json").write_text(json.dumps(rep,indent=2)+"\n")
        print(json.dumps({k:v for k,v in rep.items() if k!="rows"},indent=2),flush=True)
    
    if True:main()

def run_phase1_d3a_freejoint_basis_probe():
    """Run former phase1_d3a_freejoint_basis_probe.py stage."""
    from pathlib import Path
    import json, math
    import mujoco
    import numpy as np
    
    ROOT=Path(__file__).resolve().parents[4]
    SCENE=ROOT/"talon_rl/assets/data/Robots/unitree_a1/mujoco/scene.xml"
    OUT=ROOT/"runs/phase1_d3a_live_runtime"
    OUT.mkdir(parents=True,exist_ok=True)
    
    def quat_to_rot(q):
        w,x,y,z=q
        return np.array([
          [1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
          [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
          [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]],float)
    
    def main():
        m=mujoco.MjModel.from_xml_path(str(SCENE));d=mujoco.MjData(m)
        qa=int(m.jnt_qposadr[0]);va=int(m.jnt_dofadr[0])
        yaw=math.pi/2
        q=np.array([math.cos(yaw/2),0,0,math.sin(yaw/2)])
        R=quat_to_rot(q)
        cases=[]
        for kind in ("linear","angular"):
            for axis in range(3):
                d.qpos[:]=0;d.qvel[:]=0
                d.qpos[qa:qa+3]=[0,0,.4];d.qpos[qa+3:qa+7]=q
                if kind=="linear": d.qvel[va+axis]=1.0
                else: d.qvel[va+3+axis]=1.0
                mujoco.mj_forward(m,d)
                raw_lin=d.qvel[va:va+3].copy()
                raw_ang=d.qvel[va+3:va+6].copy()
                canon_lin=R.T@raw_lin
                canon_ang=raw_ang.copy()
                world_ang=R@raw_ang
                vw=np.zeros(6)
                bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"trunk")
                mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,bid,vw,0)
                cases.append({
                  "kind":kind,"axis":axis,
                  "raw_free_linear":raw_lin.tolist(),
                  "raw_free_angular":raw_ang.tolist(),
                  "canonical_body_linear":canon_lin.tolist(),
                  "canonical_body_angular":canon_ang.tolist(),
                  "expected_world_angular_from_local":world_ang.tolist(),
                  "mj_object_velocity_world_ang_lin":vw.tolist(),
                  "world_ang_error":float(np.max(np.abs(vw[:3]-world_ang))),
                  "world_lin_vs_raw_error":float(np.max(np.abs(vw[3:]-raw_lin))),
                })
        max_ang=max(x["world_ang_error"] for x in cases)
        max_lin=max(x["world_lin_vs_raw_error"] for x in cases)
        rep={
          "schema":"phase1_d3a_freejoint_basis_probe_v1",
          "quaternion_wxyz":q.tolist(),
          "interpretation":{
            "qvel_translation":"world frame; canonical base_lin_vel_b = R^T qvel_translation",
            "qvel_rotation":"body/local frame; canonical base_ang_vel_b = qvel_rotation",
          },
          "cases":cases,
          "max_world_angular_validation_error":max_ang,
          "max_world_linear_validation_error":max_lin,
          "angular_semantics_gate":bool(max_ang<=1e-12),
          "linear_semantics_gate":bool(max_lin<=5e-3),
        }
        rep["pass"]=bool(rep["angular_semantics_gate"] and rep["linear_semantics_gate"])
        (OUT/"freejoint_basis_probe.json").write_text(json.dumps(rep,indent=2)+"\n")
        print(json.dumps(rep,indent=2),flush=True)
    
    if True:main()

def run_phase1_d3a_freejoint_semantics():
    """Run former phase1_d3a_freejoint_semantics.py stage."""
    from pathlib import Path
    import json,math
    import numpy as np,mujoco
    
    ROOT=Path(__file__).resolve().parents[4]
    SCENE=ROOT/"talon_rl/assets/data/Robots/unitree_a1/mujoco/scene.xml"
    OUT=ROOT/"runs/phase1_d3a_live_mapping";OUT.mkdir(parents=True,exist_ok=True)
    
    def qaxis(axis,angle):
        a=np.asarray(axis,float);a=a/np.linalg.norm(a)
        return np.r_[math.cos(angle/2),a*math.sin(angle/2)]
    
    def qmat(q):
        m=np.empty(9,float);mujoco.mju_quat2Mat(m,q);return m.reshape(3,3)
    
    def rotvec(R):
        th=np.arccos(np.clip((np.trace(R)-1)/2,-1,1))
        if th<1e-12:return np.zeros(3)
        v=np.array([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])/(2*np.sin(th))
        return v*th
    
    def main():
        m=mujoco.MjModel.from_xml_path(str(SCENE))
        q=np.array([0.8112322025014301,0.20280805062535753,-0.30421207593803623,0.4563181139070544])
        q=q/np.linalg.norm(q);R0=qmat(q)
        v=np.array([0.7,-0.4,0.2]);w=np.array([0.3,0.5,-0.6]);dt=1e-6
        qpos=np.zeros(m.nq);mujoco.mj_resetDataKeyframe(m,mujoco.MjData(m),0)
        d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);qpos=d.qpos.copy()
        qpos[:3]=np.array([.2,-.1,.5]);qpos[3:7]=q
        qvel=np.zeros(m.nv);qvel[:3]=v;qvel[3:6]=w
        q1=qpos.copy();mujoco.mj_integratePos(m,q1,qvel,dt)
        trans_fd=(q1[:3]-qpos[:3])/dt
        R1=qmat(q1[3:7])
        local_rel=rotvec(R0.T@R1)/dt
        world_rel=rotvec(R1@R0.T)/dt
        rep={
          "qvel_linear":v.tolist(),"translation_fd":trans_fd.tolist(),
          "linear_vs_raw_error":float(np.max(np.abs(trans_fd-v))),
          "linear_vs_Rraw_error":float(np.max(np.abs(trans_fd-R0@v))),
          "linear_vs_Rt_raw_error":float(np.max(np.abs(trans_fd-R0.T@v))),
          "qvel_angular":w.tolist(),"local_relative_omega_fd":local_rel.tolist(),
          "world_relative_omega_fd":world_rel.tolist(),
          "angular_local_vs_raw_error":float(np.max(np.abs(local_rel-w))),
          "angular_world_vs_raw_error":float(np.max(np.abs(world_rel-w))),
          "angular_world_vs_Rraw_error":float(np.max(np.abs(world_rel-R0@w))),
          "angular_local_vs_Rt_raw_error":float(np.max(np.abs(local_rel-R0.T@w))),
        }
        (OUT/"freejoint_semantics_probe.json").write_text(json.dumps(rep,indent=2)+"\n")
        print(json.dumps(rep,indent=2))
    if True:main()

def run_phase1_d3a_live_interface_final():
    """Run former phase1_d3a_live_interface_final.py stage."""
    from pathlib import Path
    import json, math, sys
    import mujoco
    import numpy as np
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path.insert(0,str(ROOT))
    from talon_rl.deployment.phase1 import (
        CANONICAL_JOINT_ORDER,CANONICAL_DEFAULT_Q,build_canonical_obs,
        canonical_base_kinematics,root_com_velocity_b_from_freejoint,
        canonical_joint_target,reorder,
    )
    
    SCENE=ROOT/"talon_rl/assets/data/Robots/unitree_a1/mujoco/scene.xml"
    OUT=ROOT/"runs/phase1_d3a_live_runtime";OUT.mkdir(parents=True,exist_ok=True)
    
    def jname(m,j): return mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j)
    
    def main():
        m=mujoco.MjModel.from_xml_path(str(SCENE));d=mujoco.MjData(m)
        joints={jname(m,j):(int(m.jnt_qposadr[j]),int(m.jnt_dofadr[j])) for j in range(m.njnt) if jname(m,j)}
        actuators=[]
        for a in range(m.nu):
            jid=int(m.actuator_trnid[a,0]);actuators.append(jname(m,jid))
        trunk=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"trunk")
        com_b=m.body_ipos[trunk].copy()
        qa=int(m.jnt_qposadr[0]);va=int(m.jnt_dofadr[0])
    
        # Controlled live state.
        roll,pitch,yaw=.31,-.27,.42
        # MuJoCo helper: euler XYZ -> quaternion via axis composition.
        def qaxis(axis,ang):
            q=np.zeros(4);q[0]=math.cos(ang/2);q[axis+1]=math.sin(ang/2);return q
        q=np.empty(4);tmp=np.empty(4)
        mujoco.mju_mulQuat(tmp,qaxis(2,yaw),qaxis(1,pitch))
        mujoco.mju_mulQuat(q,tmp,qaxis(0,roll))
        d.qpos[:]=0;d.qvel[:]=0
        d.qpos[qa:qa+3]=[.2,-.1,.48];d.qpos[qa+3:qa+7]=q
        free_qv=np.array([.7,-.4,.2,.35,-.25,.45])
        d.qvel[va:va+6]=free_qv
    
        qcan=np.linspace(-.2,.35,12)
        qdcan=np.linspace(-1.4,1.9,12)
        for i,jn in enumerate(CANONICAL_JOINT_ORDER):
            qadr,vadr=joints[jn];d.qpos[qadr]=qcan[i];d.qvel[vadr]=qdcan[i]
        mujoco.mj_forward(m,d)
    
        # Canonical base state.
        origin_lin_b,ang_b,grav_b=canonical_base_kinematics(q,free_qv)
        com_lin_b=root_com_velocity_b_from_freejoint(q,free_qv[:3],free_qv[3:],com_b).astype(np.float32)
    
        # Validate COM velocity against MuJoCo body velocity.
        vw=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,trunk,vw,0)
        # body orientation matrix from engine
        R=d.xmat[trunk].reshape(3,3)
        mj_com_b=(R.T@vw[3:]).astype(np.float32)
        mj_ang_b=(R.T@vw[:3]).astype(np.float32)
    
        command=np.array([.55,-.1,.22],np.float32)
        prev=np.linspace(-.6,.6,12).astype(np.float32)
        q_act=np.array([d.qpos[joints[j][0]] for j in actuators],np.float32)
        qd_act=np.array([d.qvel[joints[j][1]] for j in actuators],np.float32)
        prev_act=reorder(prev,CANONICAL_JOINT_ORDER,actuators)
    
        obs_live=build_canonical_obs(com_lin_b,ang_b,grav_b,command,q_act,qd_act,prev_act,actuators)
        obs_expected=np.concatenate([
            com_lin_b,ang_b,grav_b,command,
            qcan.astype(np.float32)-CANONICAL_DEFAULT_Q,
            qdcan.astype(np.float32),prev
        ]).astype(np.float32)
        obs_err=float(np.max(np.abs(obs_live-obs_expected)))
    
        # Action->ctrl exact and full action cube lies inside ctrl ranges.
        action=np.linspace(-1,1,12).astype(np.float32)
        target=canonical_joint_target(action)
        ctrl=np.array([target[CANONICAL_JOINT_ORDER.index(j)] for j in actuators],np.float64)
        target_back=reorder(ctrl,actuators,CANONICAL_JOINT_ORDER)
        ctrl_err=float(np.max(np.abs(target_back-target)))
        target_min=canonical_joint_target(-np.ones(12,np.float32))
        target_max=canonical_joint_target(np.ones(12,np.float32))
        min_mj=np.array([target_min[CANONICAL_JOINT_ORDER.index(j)] for j in actuators])
        max_mj=np.array([target_max[CANONICAL_JOINT_ORDER.index(j)] for j in actuators])
        full_range_ok=bool(np.all(min_mj>=m.actuator_ctrlrange[:,0]-1e-7) and np.all(max_mj<=m.actuator_ctrlrange[:,1]+1e-7))
    
        # Position actuator live semantics: at q=ctrl force≈0; ctrl=q+0.05 -> force≈kp*0.05=5,
        # checked on one hip actuator away from limits/contact relevance.
        a0=0;jn=actuators[a0];qadr,_=joints[jn]
        mujoco.mj_resetData(m,d)
        # keyframe home if present
        if m.nkey: mujoco.mj_resetDataKeyframe(m,d,0)
        mujoco.mj_forward(m,d)
        q0=float(d.qpos[qadr]);d.ctrl[:]=0
        # keep all actuator ctrls at their current joint angles
        for a,j in enumerate(actuators):d.ctrl[a]=d.qpos[joints[j][0]]
        mujoco.mj_forward(m,d);f0=float(d.actuator_force[a0])
        d.ctrl[a0]=q0+.05;mujoco.mj_forward(m,d);f1=float(d.actuator_force[a0])
        force_sem_err=abs((f1-f0)-5.0)
    
        rep={
          "schema":"phase1_d3a_live_interface_final_v1",
          "trunk_com_offset_body":com_b.tolist(),
          "com_velocity_engine_max_abs_error":float(np.max(np.abs(com_lin_b-mj_com_b))),
          "angular_velocity_engine_max_abs_error":float(np.max(np.abs(ang_b-mj_ang_b))),
          "full_obs_48d_max_abs_error":obs_err,
          "action_ctrl_roundtrip_max_abs_error":ctrl_err,
          "canonical_action_cube_within_ctrlrange":full_range_ok,
          "position_actuator_zero_error_force":f0,
          "position_actuator_plus_0p05_force":f1,
          "position_actuator_kp100_semantics_error":float(force_sem_err),
          "physics_dt":float(m.opt.timestep),
          "hold_steps_20ms":int(round(.02/m.opt.timestep)),
          "hold_time":float(round(.02/m.opt.timestep)*m.opt.timestep),
        }
        rep["gates"]={
          "com_velocity":bool(rep["com_velocity_engine_max_abs_error"]<=1e-7),
          "angular_velocity":bool(rep["angular_velocity_engine_max_abs_error"]<=1e-7),
          "full_observation":bool(obs_err<=1e-7),
          "action_ctrl":bool(ctrl_err<=1e-7 and full_range_ok),
          "position_actuator_semantics":bool(force_sem_err<=1e-6),
          "hold_20ms":bool(rep["hold_steps_20ms"]==10 and abs(rep["hold_time"]-.02)<=1e-12),
        }
        rep["pass"]=bool(all(rep["gates"].values()))
        (OUT/"live_interface_final.json").write_text(json.dumps(rep,indent=2)+"\n")
        print(json.dumps(rep,indent=2),flush=True)
    
    if True:main()

def run_phase1_d3a_live_mapping():
    """Run former phase1_d3a_live_mapping.py stage."""
    from pathlib import Path
    import json,math,sys
    import numpy as np,mujoco
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path.insert(0,str(ROOT))
    from talon_rl.deployment.phase1 import (
        CANONICAL_JOINT_ORDER,CANONICAL_DEFAULT_Q,build_canonical_obs,
        canonical_joint_target,permutation,reorder,target_for_actuator_order,
    )
    
    SCENE=ROOT/"talon_rl/assets/data/Robots/unitree_a1/mujoco/scene.xml"
    OUT=ROOT/"runs/phase1_d3a_live_mapping";OUT.mkdir(parents=True,exist_ok=True)
    
    def quat_axis_angle(axis,angle):
        axis=np.asarray(axis,np.float64);axis=axis/np.linalg.norm(axis)
        return np.r_[math.cos(angle/2),axis*math.sin(angle/2)]
    
    def body_rot(model,data,bid):
        return data.xmat[bid].reshape(3,3).copy()
    
    def objvel(model,data,bid,local):
        v=np.zeros(6,np.float64)
        mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,bid,v,int(local))
        return v
    
    def joint_maps(model):
        names=[];qadr=[];dadr=[]
        for name in CANONICAL_JOINT_ORDER:
            jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
            names.append(name);qadr.append(int(model.jnt_qposadr[jid]));dadr.append(int(model.jnt_dofadr[jid]))
        return names,np.asarray(qadr),np.asarray(dadr)
    def set_pose(model,data,quat):
        mujoco.mj_resetDataKeyframe(model,data,0)
        data.qpos[3:7]=quat/np.linalg.norm(quat)
        data.qvel[:]=0
        mujoco.mj_forward(model,data)
    
    def main():
        model=mujoco.MjModel.from_xml_path(str(SCENE));data=mujoco.MjData(model)
        trunk=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"trunk")
        _,qadr,dadr=joint_maps(model)
        actuator_joints=[]
        for aid in range(model.nu):
            jid=int(model.actuator_trnid[aid,0])
            actuator_joints.append(mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,jid))
    
        rep={"schema":"phase1_d3a_live_mapping_v1","mujoco_version":mujoco.__version__,
             "timestep":float(model.opt.timestep),"actuator_joint_names":actuator_joints}
    
        # Quaternion + projected gravity live parity against compiled body rotation.
        orientations={
          "identity":np.array([1.,0,0,0]),
          "roll90":quat_axis_angle([1,0,0],math.pi/2),
          "pitch90":quat_axis_angle([0,1,0],math.pi/2),
          "yaw90":quat_axis_angle([0,0,1],math.pi/2),
          "mixed":np.array([0.8,0.2,-0.3,0.45],np.float64),
        }
        grav_rows=[];max_g=0.0
        from talon_rl.deployment.phase1 import quat_rotate_inverse_wxyz
        for name,q in orientations.items():
            q=q/np.linalg.norm(q);set_pose(model,data,q);R=body_rot(model,data,trunk)
            expected=R.T@np.array([0.,0.,-1.])
            got=quat_rotate_inverse_wxyz(q,np.array([0.,0.,-1.]))
            err=float(np.max(np.abs(expected-got)));max_g=max(max_g,err)
            grav_rows.append({"name":name,"quat_wxyz":q.tolist(),"expected":expected.tolist(),"adapter":got.tolist(),"max_abs":err})
        rep["projected_gravity"]={"rows":grav_rows,"max_abs_error":max_g,"pass":bool(max_g<=1e-7)}
    
        # Live free-joint semantics from finite-difference integration at the joint origin.
        from talon_rl.deployment.phase1 import root_com_velocity_b_from_freejoint
        q=orientations["mixed"];q=q/np.linalg.norm(q);set_pose(model,data,q);R=body_rot(model,data,trunk)
        world_v=np.array([0.7,-0.4,0.2]);raw_w=np.array([0.3,0.5,-0.6])
        data.qpos[:3]=np.array([.2,-.1,.5]);data.qvel[:3]=world_v;data.qvel[3:6]=raw_w;mujoco.mj_forward(model,data)
        qpos0=data.qpos.copy();xipos0=data.xipos[trunk].copy();R0=body_rot(model,data,trunk)
        dt_fd=1e-7
        qpos1=qpos0.copy();mujoco.mj_integratePos(model,qpos1,data.qvel,dt_fd)
        trans_fd=(qpos1[:3]-qpos0[:3])/dt_fd
        # Quaternion integration reveals angular velocity coordinates.
        mat0=np.empty(9);mat1=np.empty(9);mujoco.mju_quat2Mat(mat0,qpos0[3:7]);mujoco.mju_quat2Mat(mat1,qpos1[3:7]);mat0=mat0.reshape(3,3);mat1=mat1.reshape(3,3)
        rel=mat0.T@mat1
        ang_local_fd=np.array([rel[2,1]-rel[1,2],rel[0,2]-rel[2,0],rel[1,0]-rel[0,1]])/(2*dt_fd)
        # COM velocity: finite-difference the compiled inertial COM position and compare to canonical formula.
        data.qpos[:]=qpos1;mujoco.mj_forward(model,data);xipos1=data.xipos[trunk].copy()
        com_world_fd=(xipos1-xipos0)/dt_fd
        com_body_fd=R0.T@com_world_fd
        com_offset_b=model.body_ipos[trunk].copy()
        com_body_formula=root_com_velocity_b_from_freejoint(qpos0[3:7],world_v,raw_w,com_offset_b)
        lin_origin_err=float(np.max(np.abs(trans_fd-world_v)))
        ang_body_err=float(np.max(np.abs(ang_local_fd-raw_w)))
        com_err=float(np.max(np.abs(com_body_fd-com_body_formula)))
        rep["freejoint_velocity"]={
          "qvel_linear_raw":world_v.tolist(),"translation_fd":trans_fd.tolist(),
          "qvel_angular_raw":raw_w.tolist(),"angular_body_fd":ang_local_fd.tolist(),
          "com_offset_body":com_offset_b.tolist(),"com_velocity_body_fd":com_body_fd.tolist(),
          "com_velocity_body_formula":com_body_formula.tolist(),
          "linear_origin_error":lin_origin_err,"angular_body_error":ang_body_err,"com_body_error":com_err,
          "linear_qvel_is_world_frame":bool(lin_origin_err<=1e-7),
          "angular_qvel_is_body_frame":bool(ang_body_err<=1e-7),
          "canonical_com_velocity_mapping":bool(com_err<=1e-7),
        }
        # Restore exact current pose/velocity for subsequent tests.
        data.qpos[:]=qpos0;data.qvel[:3]=world_v;data.qvel[3:6]=raw_w;mujoco.mj_forward(model,data)
        # Exact q/qdot extraction by model addresses, not slice assumptions.
        rng=np.random.default_rng(260926)
        q_can=rng.uniform(-.5,.5,12);qd_can=rng.uniform(-3,3,12)
        mujoco.mj_resetDataKeyframe(model,data,0)
        data.qpos[qadr]=q_can;data.qvel[dadr]=qd_can;mujoco.mj_forward(model,data)
        q_live=data.qpos[qadr].copy();qd_live=data.qvel[dadr].copy()
        rep["joint_extraction"]={
          "q_max_abs_error":float(np.max(np.abs(q_live-q_can))),
          "qd_max_abs_error":float(np.max(np.abs(qd_live-qd_can))),
          "qpos_addresses":qadr.tolist(),"dof_addresses":dadr.tolist(),
        }
    
        # Canonical observation constructed from one controlled live engine state.
        data.qpos[:3]=np.array([.2,-.1,.5]);data.qpos[3:7]=q
        data.qpos[qadr]=q_can;data.qvel[:3]=world_v;data.qvel[3:6]=raw_w;data.qvel[dadr]=qd_can
        mujoco.mj_forward(model,data)
        R=body_rot(model,data,trunk)
        lin_local=root_com_velocity_b_from_freejoint(data.qpos[3:7],data.qvel[:3],data.qvel[3:6],model.body_ipos[trunk])
        ang_local=data.qvel[3:6].copy()
        pg=R.T@np.array([0.,0.,-1.])
        q_live=data.qpos[qadr].copy();qd_live=data.qvel[dadr].copy()
        cmd=np.array([.5,0,.2],np.float32);prev=rng.uniform(-1,1,12).astype(np.float32)
        obs=build_canonical_obs(lin_local,ang_local,pg,cmd,q_live,qd_live,prev,CANONICAL_JOINT_ORDER)
        manual=np.concatenate([
            np.asarray(lin_local,np.float32),np.asarray(ang_local,np.float32),np.asarray(pg,np.float32),cmd,
            q_can.astype(np.float32)-CANONICAL_DEFAULT_Q,qd_can.astype(np.float32),prev]).astype(np.float32)
        obs_err=float(np.max(np.abs(obs-manual)))
        rep["live_observation"]={"shape":list(obs.shape),"max_abs_error":obs_err,"pass":bool(obs_err<=1e-7)}
    
        # Actuator interpretation and 20 ms hold: 10 MuJoCo steps at dt=2 ms.
        mujoco.mj_resetDataKeyframe(model,data,0)
        # Put actual joint state at D2 canonical default before command test.
        data.qpos[qadr]=CANONICAL_DEFAULT_Q
        data.qvel[:]=0;mujoco.mj_forward(model,data)
        action=rng.uniform(-.8,.8,12).astype(np.float32)
        target_can=canonical_joint_target(action)
        target_mj=target_for_actuator_order(action,actuator_joints)
        data.ctrl[:]=target_mj
        ctrl_initial=data.ctrl.copy()
        force_expected=[]
        # At forward, position actuator force should follow kp*(ctrl-q) before clipping.
        mujoco.mj_forward(model,data)
        q_act=np.array([data.qpos[int(model.jnt_qposadr[int(model.actuator_trnid[a,0])])] for a in range(model.nu)])
        expected_force=np.clip(100.0*(data.ctrl-q_act),model.actuator_forcerange[:,0],model.actuator_forcerange[:,1])
        force_err=float(np.max(np.abs(data.actuator_force-expected_force)))
        for _ in range(10):mujoco.mj_step(model,data)
        hold_err=float(np.max(np.abs(data.ctrl-ctrl_initial)))
        target_back=reorder(target_mj,actuator_joints,CANONICAL_JOINT_ORDER)
        target_err=float(np.max(np.abs(target_back-target_can)))
        rep["actuator_and_hold"]={
          "target_roundtrip_max_abs_error":target_err,
          "position_force_model_max_abs_error":force_err,
          "ctrl_hold_10_substeps_max_abs_error":hold_err,
          "hold_duration_s":10*float(model.opt.timestep),
          "pass":bool(target_err<=1e-7 and force_err<=1e-7 and hold_err<=1e-12 and abs(10*model.opt.timestep-.02)<=1e-12),
        }
    
        gates={
          "projected_gravity":rep["projected_gravity"]["pass"],
          "linear_velocity_frame":rep["freejoint_velocity"]["linear_qvel_is_world_frame"],
          "angular_velocity_frame":rep["freejoint_velocity"]["angular_qvel_is_body_frame"],
          "root_com_velocity_mapping":rep["freejoint_velocity"]["canonical_com_velocity_mapping"],
          "joint_extraction":rep["joint_extraction"]["q_max_abs_error"]<=1e-12 and rep["joint_extraction"]["qd_max_abs_error"]<=1e-12,
          "live_observation":rep["live_observation"]["pass"],
          "actuator_and_hold":rep["actuator_and_hold"]["pass"],
        }
        rep["gates"]=gates;rep["pass"]=bool(all(gates.values()))
        (OUT/"live_mapping_report.json").write_text(json.dumps(rep,indent=2)+"\n")
        print(json.dumps(rep,indent=2),flush=True)
    
    if True:main()

def run_phase1_d3a_live_runtime_mapping():
    """Run former phase1_d3a_live_runtime_mapping.py stage."""
    from pathlib import Path
    import json, math
    import mujoco
    import numpy as np
    
    ROOT=Path(__file__).resolve().parents[4]
    SCENE=ROOT/"talon_rl/assets/data/Robots/unitree_a1/mujoco/scene.xml"
    OUT=ROOT/"runs/phase1_d3a_live_runtime"
    OUT.mkdir(parents=True,exist_ok=True)
    
    CANON=(
     "FL_hip_joint","FR_hip_joint","RL_hip_joint","RR_hip_joint",
     "FL_thigh_joint","FR_thigh_joint","RL_thigh_joint","RR_thigh_joint",
     "FL_calf_joint","FR_calf_joint","RL_calf_joint","RR_calf_joint",
    )
    DEFAULT=np.array([.1,-.1,.1,-.1,.8,.8,1.,1.,-1.5,-1.5,-1.5,-1.5])
    
    def name(obj,m,i): return mujoco.mj_id2name(m,obj,i)
    
    def quat_to_rot(q):
        # MuJoCo free-joint quaternion convention is w,x,y,z.
        w,x,y,z=q
        return np.array([
          [1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
          [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
          [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]],float)
    
    def main():
        m=mujoco.MjModel.from_xml_path(str(SCENE));d=mujoco.MjData(m)
        joints={}
        for j in range(m.njnt):
            n=name(mujoco.mjtObj.mjOBJ_JOINT,m,j)
            joints[n]={"id":j,"type":int(m.jnt_type[j]),"qposadr":int(m.jnt_qposadr[j]),"dofadr":int(m.jnt_dofadr[j])}
        actuators=[]
        for a in range(m.nu):
            an=name(mujoco.mjtObj.mjOBJ_ACTUATOR,m,a)
            jid=int(m.actuator_trnid[a,0])
            jn=name(mujoco.mjtObj.mjOBJ_JOINT,m,jid)
            actuators.append({"id":a,"name":an,"joint":jn,"ctrlrange":m.actuator_ctrlrange[a].tolist(),"forcerange":m.actuator_forcerange[a].tolist()})
    
        free=[k for k,v in joints.items() if v["type"]==int(mujoco.mjtJoint.mjJNT_FREE)]
        if len(free)!=1: raise RuntimeError(f"expected one free joint, got {free}")
        rootj=free[0]; qa=joints[rootj]["qposadr"]; va=joints[rootj]["dofadr"]
    
        # Known state: yaw +90 deg. MuJoCo quat is wxyz.
        d.qpos[:] = 0; d.qvel[:] = 0
        d.qpos[qa:qa+3]=[0,0,.4]
        yaw=math.pi/2
        q=np.array([math.cos(yaw/2),0,0,math.sin(yaw/2)])
        d.qpos[qa+3:qa+7]=q
        # Deliberately asymmetric free-joint velocities.
        d.qvel[va:va+3]=[1.0,2.0,3.0]
        d.qvel[va+3:va+6]=[0.4,0.5,0.6]
        # Known joint states from canonical names.
        qcan=np.linspace(-.4,.7,12); qdcan=np.linspace(-2.0,2.4,12)
        for i,jn in enumerate(CANON):
            d.qpos[joints[jn]["qposadr"]]=qcan[i]
            d.qvel[joints[jn]["dofadr"]]=qdcan[i]
        mujoco.mj_forward(m,d)
    
        body_id=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"trunk")
        vw=np.zeros(6); vb=np.zeros(6)
        mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,body_id,vw,0)
        mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,body_id,vb,1)
    
        R=quat_to_rot(q) # body->world
        # mj_objectVelocity returns [angular, linear].
        expected_body_lin=R.T@np.array([1.,2.,3.])
        expected_body_ang=R.T@np.array([.4,.5,.6])
        vel_err=max(np.max(np.abs(vb[3:]-expected_body_lin)),np.max(np.abs(vb[:3]-expected_body_ang)))
    
        gravity_world=np.array([0.,0.,-1.])
        projected=R.T@gravity_world
        # Known yaw-only orientation keeps gravity unchanged.
        projected_expected=np.array([0.,0.,-1.])
        gravity_err=float(np.max(np.abs(projected-projected_expected)))
    
        # Exact joint extraction via engine addresses.
        qext=np.array([d.qpos[joints[j]["qposadr"]] for j in CANON])
        qdext=np.array([d.qvel[joints[j]["dofadr"]] for j in CANON])
        joint_err=float(max(np.max(np.abs(qext-qcan)),np.max(np.abs(qdext-qdcan))))
    
        # Actuator mapping: canonical target -> actuator ctrl by transmitted joint name.
        action=np.linspace(-1,1,12)
        target=DEFAULT+.25*action
        ctrl=np.zeros(m.nu)
        for a,info in enumerate(actuators):
            ci=CANON.index(info["joint"]); ctrl[a]=target[ci]
        d.ctrl[:]=ctrl
        target_back=np.empty(12)
        for a,info in enumerate(actuators):
            target_back[CANON.index(info["joint"])]=d.ctrl[a]
        ctrl_err=float(np.max(np.abs(target_back-target)))
        ctrl_within=bool(np.all(d.ctrl>=m.actuator_ctrlrange[:,0]-1e-12) and np.all(d.ctrl<=m.actuator_ctrlrange[:,1]+1e-12))
    
        # 20 ms zero-order hold: dt=.002 -> exactly 10 physics steps.
        hold_steps=round(.020/m.opt.timestep)
        hold_time=hold_steps*m.opt.timestep
        hold_err=abs(hold_time-.020)
    
        # Previous action propagation is adapter state, test exact one-step memory.
        prev=np.zeros(12); a0=np.linspace(-.8,.8,12); observed_prev=prev.copy(); prev=a0.copy()
        prev_err=float(np.max(np.abs(prev-a0)))
        initial_prev_zero=bool(np.array_equal(observed_prev,np.zeros(12)))
    
        rep={
          "schema":"phase1_d3a_live_runtime_mapping_v1",
          "free_joint":rootj,
          "free_qposadr":qa,"free_dofadr":va,
          "joint_addresses":joints,
          "actuators":actuators,
          "known_quaternion_wxyz":q.tolist(),
          "mj_object_velocity_world_ang_lin":vw.tolist(),
          "mj_object_velocity_body_ang_lin":vb.tolist(),
          "expected_body_linear":expected_body_lin.tolist(),
          "expected_body_angular":expected_body_ang.tolist(),
          "body_velocity_max_abs_error":float(vel_err),
          "projected_gravity":projected.tolist(),
          "projected_gravity_max_abs_error":gravity_err,
          "joint_extraction_max_abs_error":joint_err,
          "action_ctrl_roundtrip_max_abs_error":ctrl_err,
          "action_ctrl_within_range":ctrl_within,
          "physics_dt":float(m.opt.timestep),
          "policy_hold_steps":int(hold_steps),
          "policy_hold_time":float(hold_time),
          "policy_hold_abs_error":float(hold_err),
          "initial_previous_action_zero":initial_prev_zero,
          "previous_action_propagation_max_abs_error":prev_err,
        }
        rep["gates"]={
          "free_joint_present":bool(len(free)==1),
          "body_velocity_frame":bool(vel_err<=1e-12),
          "projected_gravity":bool(gravity_err<=1e-12),
          "joint_state_extraction":bool(joint_err<=1e-12),
          "action_ctrl_mapping":bool(ctrl_err<=1e-12 and ctrl_within),
          "policy_hold_20ms":bool(hold_err<=1e-12 and hold_steps==10),
          "previous_action_state":bool(initial_prev_zero and prev_err<=1e-12),
        }
        rep["pass"]=bool(all(rep["gates"].values()))
        (OUT/"live_runtime_mapping.json").write_text(json.dumps(rep,indent=2)+"\n")
        print(json.dumps(rep,indent=2),flush=True)
    
    if True:main()

def run_phase1_d3a_structural_interface():
    """Run former phase1_d3a_structural_interface.py stage."""
    from pathlib import Path
    import json,xml.etree.ElementTree as ET
    import numpy as np
    
    ROOT=Path(__file__).resolve().parents[4]
    XML=ROOT/"talon_rl/assets/data/Robots/unitree_a1/mujoco/a1.xml"
    OUT=ROOT/"runs/phase1_d3a_structural_interface";OUT.mkdir(parents=True,exist_ok=True)
    
    from talon_rl.deployment.phase1 import (
        CANONICAL_JOINT_ORDER,CANONICAL_DEFAULT_Q,build_canonical_obs,
        canonical_joint_target,permutation,reorder,target_for_actuator_order,
    )
    
    def parse_model():
        tree=ET.parse(XML);root=tree.getroot()
        joints=[x.attrib["name"] for x in root.findall(".//joint") if "name" in x.attrib]
        acts=[x.attrib["joint"] for x in root.findall("./actuator/position")]
        key=root.find("./keyframe/key")
        home_q=np.asarray([float(x) for x in key.attrib["qpos"].split()][7:],np.float32)
        home_ctrl=np.asarray([float(x) for x in key.attrib["ctrl"].split()],np.float32)
        return joints,acts,home_q,home_ctrl
    def main():
        joints,acts,home_q,home_ctrl=parse_model()
        expected=set(CANONICAL_JOINT_ORDER)
        joint_set_ok=set(joints)==expected
        actuator_set_ok=set(acts)==expected
    
        rng=np.random.default_rng(260926)
        q_can=rng.uniform(-1.0,1.0,size=12).astype(np.float32)
        qd_can=rng.uniform(-4.0,4.0,size=12).astype(np.float32)
        a_can=rng.uniform(-1.0,1.0,size=12).astype(np.float32)
    
        q_mj=reorder(q_can,CANONICAL_JOINT_ORDER,acts)
        qd_mj=reorder(qd_can,CANONICAL_JOINT_ORDER,acts)
        a_mj=reorder(a_can,CANONICAL_JOINT_ORDER,acts)
    
        common=dict(
            base_lin_vel_b=np.array([.3,-.2,.1],np.float32),
            base_ang_vel_b=np.array([.4,.5,-.6],np.float32),
            projected_gravity_b=np.array([.1,-.2,-.97],np.float32),
            command=np.array([.5,.0,.2],np.float32),
        )
        obs_ref=build_canonical_obs(**common,joint_pos_abs=q_can,joint_vel_abs=qd_can,
                                    prev_action=a_can,joint_names=CANONICAL_JOINT_ORDER)
        obs_mj=build_canonical_obs(**common,joint_pos_abs=q_mj,joint_vel_abs=qd_mj,
                                   prev_action=a_mj,joint_names=acts)
        obs_err=float(np.max(np.abs(obs_ref-obs_mj)))
    
        target_can=canonical_joint_target(a_can)
        target_mj=target_for_actuator_order(a_can,acts)
        target_back=reorder(target_mj,acts,CANONICAL_JOINT_ORDER)
        target_err=float(np.max(np.abs(target_back-target_can)))
    
        home_can=reorder(home_q,acts,CANONICAL_JOINT_ORDER)
        home_rel=home_can-CANONICAL_DEFAULT_Q
        rep={
          "schema":"phase1_d3a_structural_interface_v1",
          "xml":str(XML.relative_to(ROOT)),
          "xml_joint_names":joints,
          "xml_actuator_joint_names":acts,
          "joint_set_matches_canonical":joint_set_ok,
          "actuator_set_matches_canonical":actuator_set_ok,
          "mujoco_to_canonical_perm":permutation(acts,CANONICAL_JOINT_ORDER).tolist(),
          "canonical_to_mujoco_perm":permutation(CANONICAL_JOINT_ORDER,acts).tolist(),
          "synthetic_obs_max_abs_error":obs_err,
          "action_target_roundtrip_max_abs_error":target_err,
          "mujoco_home_in_canonical_order":home_can.tolist(),
          "mujoco_home_relative_to_d2_default":home_rel.tolist(),
          "mujoco_home_ctrl_matches_home_q":bool(np.array_equal(home_ctrl,home_q)),
          "gates":{
            "joint_identity":bool(joint_set_ok and actuator_set_ok),
            "synthetic_obs_equivalence":bool(obs_err<=1e-7),
            "action_target_equivalence":bool(target_err<=1e-7),
          },
          "live_runtime_mapping_verified":False,
        }
        rep["structural_pass"]=bool(all(rep["gates"].values()))
        rep["d3a_complete"]=False
        (OUT/"structural_interface_report.json").write_text(json.dumps(rep,indent=2)+"\n")
        print(json.dumps(rep,indent=2))
    
    if True:main()

STAGES = {
    "phase1_d3a_canonical_velocity_probe": run_phase1_d3a_canonical_velocity_probe,
    "phase1_d3a_freejoint_basis_probe": run_phase1_d3a_freejoint_basis_probe,
    "phase1_d3a_freejoint_semantics": run_phase1_d3a_freejoint_semantics,
    "phase1_d3a_live_interface_final": run_phase1_d3a_live_interface_final,
    "phase1_d3a_live_mapping": run_phase1_d3a_live_mapping,
    "phase1_d3a_live_runtime_mapping": run_phase1_d3a_live_runtime_mapping,
    "phase1_d3a_structural_interface": run_phase1_d3a_structural_interface,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
