#!/usr/bin/env python3
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
if __name__=="__main__":main()
