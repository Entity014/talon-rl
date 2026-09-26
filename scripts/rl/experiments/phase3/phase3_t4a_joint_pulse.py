#!/usr/bin/env python3
from pathlib import Path
import json,sys
import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
SCENE=ROOT/"talon_rl/assets/data/Robots/unitree_a1/mujoco/scene_t4_passive_corrected.xml"
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
    (OUT/"t4_joint_pulse.json").write_text(json.dumps(rep,indent=2)+"\n")
    print(json.dumps(rep,indent=2),flush=True)

if __name__=="__main__":main()
