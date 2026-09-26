#!/usr/bin/env python3
from pathlib import Path
import json,sys,math
import mujoco
import numpy as np
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from talon_rl.phase1_mujoco_adapter import CANONICAL_JOINT_ORDER,CANONICAL_DEFAULT_Q,source_dcmotor_torque,canonical_base_kinematics
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
if __name__=="__main__":main()
