#!/usr/bin/env python3
from pathlib import Path
import json,sys
import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from talon_rl.phase1_mujoco_adapter import (
    CANONICAL_JOINT_ORDER,canonical_base_kinematics,root_com_velocity_b_from_freejoint,
    canonical_joint_target,quat_wxyz_to_rot,source_dcmotor_torque,
)
SRC=ROOT/"runs/phase3_t1c_one_step/isaac_one_step.json"
SCENE=ROOT/"talon_rl/assets/data/Robots/unitree_a1/mujoco/scene_t3_dcmotor.xml"
OUT=ROOT/"runs/phase3_t3a_one_step";OUT.mkdir(parents=True,exist_ok=True)

def main():
 src=json.load(open(SRC));m=mujoco.MjModel.from_xml_path(str(SCENE));d=mujoco.MjData(m)
 jmap={}
 for j in range(m.njnt):
  n=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j)
  if n:jmap[n]=(int(m.jnt_qposadr[j]),int(m.jnt_dofadr[j]))
 act_joints=[]
 for a in range(m.nu):
  jid=int(m.actuator_trnid[a,0]);act_joints.append(mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,jid))
 trunk=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"trunk");com_b=m.body_ipos[trunk].copy()
 qa=int(m.jnt_qposadr[0]);va=int(m.jnt_dofadr[0]);hold=int(round(.02/m.opt.timestep))

 def load(init,e):
  mujoco.mj_resetData(m,d)
  pos=np.asarray(init["root_link_pos_local"][e],float);quat=np.asarray(init["root_link_quat_w"][e],float)
  lin_w=np.asarray(init["root_link_lin_vel_w"][e],float);ang_w=np.asarray(init["root_link_ang_vel_w"][e],float)
  R=quat_wxyz_to_rot(quat);d.qpos[qa:qa+3]=pos;d.qpos[qa+3:qa+7]=quat
  d.qvel[va:va+3]=lin_w;d.qvel[va+3:va+6]=R.T@ang_w
  names=init["joint_names"];qp=np.asarray(init["joint_pos"][e]);qv=np.asarray(init["joint_vel"][e])
  for jn in CANONICAL_JOINT_ORDER:
   ii=names.index(jn);qadr,vadr=jmap[jn];d.qpos[qadr]=qp[ii];d.qvel[vadr]=qv[ii]
  d.ctrl[:]=0.0
  mujoco.mj_forward(m,d)

 def next_state():
  quat=d.qpos[qa+3:qa+7].copy();qv=d.qvel[va:va+6].copy()
  _,ang,g=canonical_base_kinematics(quat,qv)
  lin=root_com_velocity_b_from_freejoint(quat,qv[:3],qv[3:],com_b).astype(np.float32)
  q=np.array([d.qpos[jmap[j][0]] for j in CANONICAL_JOINT_ORDER])
  qd=np.array([d.qvel[jmap[j][1]] for j in CANONICAL_JOINT_ORDER])
  return lin,ang,g,q,qd,float(d.qpos[qa+2])

 suite_reports=[];raw=[]
 for suite in src["suites"]:
  si=suite["suite"];init=suite["initial"];names=init["joint_names"]
  labdata={}
  for lab in ("T","C"):
   entries=[]
   for e,a in enumerate(suite["actions"][lab]):
    load(init,e)
    target=canonical_joint_target(np.asarray(a,np.float32))
    for _ in range(hold):
      q_now=np.array([d.qpos[jmap[j][0]] for j in CANONICAL_JOINT_ORDER],float)
      qd_now=np.array([d.qvel[jmap[j][1]] for j in CANONICAL_JOINT_ORDER],float)
      tau=source_dcmotor_torque(target,q_now,qd_now)
      d.ctrl[:]=np.array([tau[CANONICAL_JOINT_ORDER.index(j)] for j in act_joints])
      mujoco.mj_step(m,d)
    v,w,g,q,qd,h=next_state()
    cmd=np.asarray(init["command"][e],float)
    tracking=float(abs(v[0]-cmd[0])+abs(w[2]-cmd[2]))
    entries.append({"v":v.tolist(),"w":w.tolist(),"g":g.tolist(),"q":q.tolist(),"qd":qd.tolist(),"height":h,"tracking":tracking})
   labdata[lab]=entries
  # Isaac next state arrays and target arrays in canonical order.
  stats={}
  for lab in ("T","C"):
   isa=suite["next"][lab];tar=labdata[lab]
   iq=np.asarray(isa["q"],float);iqd=np.asarray(isa["qd"],float)
   perm=[names.index(j) for j in CANONICAL_JOINT_ORDER]
   iq=iq[:,perm];iqd=iqd[:,perm]
   iv=np.asarray(isa["v"],float);iw=np.asarray(isa["w"],float);ig=np.asarray(isa["g"],float);ih=np.asarray(isa["height"],float)
   tv=np.asarray([x["v"] for x in tar]);tw=np.asarray([x["w"] for x in tar]);tg=np.asarray([x["g"] for x in tar])
   tq=np.asarray([x["q"] for x in tar]);tqd=np.asarray([x["qd"] for x in tar]);th=np.asarray([x["height"] for x in tar])
   stats[lab]={
    "v_rmse":float(np.sqrt(np.mean((tv-iv)**2))),
    "w_rmse":float(np.sqrt(np.mean((tw-iw)**2))),
    "g_rmse":float(np.sqrt(np.mean((tg-ig)**2))),
    "q_rmse":float(np.sqrt(np.mean((tq-iq)**2))),
    "qd_rmse":float(np.sqrt(np.mean((tqd-iqd)**2))),
    "height_rmse":float(np.sqrt(np.mean((th-ih)**2))),
    "isaac_tracking_mean":float(np.mean(isa["tracking"])),
    "mujoco_tracking_mean":float(np.mean([x["tracking"] for x in tar])),
   }
  isa_delta=float(np.mean(suite["next"]["T"]["tracking"])-np.mean(suite["next"]["C"]["tracking"]))
  mj_delta=float(np.mean([x["tracking"] for x in labdata["T"]])-np.mean([x["tracking"] for x in labdata["C"]]))
  suite_reports.append({"suite":si,"T":stats["T"],"C":stats["C"],
                        "isaac_T_minus_C_tracking_after_20ms":isa_delta,
                        "mujoco_T_minus_C_tracking_after_20ms":mj_delta,
                        "one_step_semantic_sign_flip":bool(np.sign(isa_delta)!=np.sign(mj_delta))})
  raw.append({"suite":si,"target_next":labdata})
  print("SUITE",si,"isa_delta",isa_delta,"mj_delta",mj_delta,flush=True)

 agg={
  "sign_flip_suites":int(sum(x["one_step_semantic_sign_flip"] for x in suite_reports)),
  "isaac_T_better_suites":int(sum(x["isaac_T_minus_C_tracking_after_20ms"]<0 for x in suite_reports)),
  "mujoco_T_better_suites":int(sum(x["mujoco_T_minus_C_tracking_after_20ms"]<0 for x in suite_reports)),
  "mean_abs_isaac_delta":float(np.mean([abs(x["isaac_T_minus_C_tracking_after_20ms"]) for x in suite_reports])),
  "mean_abs_mujoco_delta":float(np.mean([abs(x["mujoco_T_minus_C_tracking_after_20ms"]) for x in suite_reports])),
  "mean_T_v_rmse":float(np.mean([x["T"]["v_rmse"] for x in suite_reports])),
  "mean_T_w_rmse":float(np.mean([x["T"]["w_rmse"] for x in suite_reports])),
  "mean_T_qd_rmse":float(np.mean([x["T"]["qd_rmse"] for x in suite_reports])),
 }
 rep={"schema":"phase3_t3a_one_step_compare_v1","suites":suite_reports,"aggregate":agg,"raw_target":raw}
 (OUT/"one_step_compare.json").write_text(json.dumps(rep,indent=2)+"\n")
 print("FINAL",json.dumps({"suites":suite_reports,"aggregate":agg},indent=2),flush=True)

if __name__=="__main__":main()
