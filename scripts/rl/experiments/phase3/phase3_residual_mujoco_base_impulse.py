#!/usr/bin/env python3
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
if __name__=="__main__":main()
