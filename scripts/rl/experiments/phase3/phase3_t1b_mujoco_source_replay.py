#!/usr/bin/env python3
from pathlib import Path
import json,math,sys
import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from talon_rl.phase1_deployment import Phase1EagerStateRuntime
from talon_rl.phase1_mujoco_adapter import (
    CANONICAL_JOINT_ORDER,CANONICAL_DEFAULT_Q,build_canonical_obs,
    canonical_base_kinematics,root_com_velocity_b_from_freejoint,
    canonical_joint_target,quat_wxyz_to_rot,
)

SRC=ROOT/"runs/phase3_t1b_source_anchor/isaac_source_anchor.json"
SCENE=ROOT/"talon_rl/assets/data/Robots/unitree_a1/mujoco/scene.xml"
ART=ROOT/"artifacts/phase1_canonical_actor_state.pt"
OUT=ROOT/"runs/phase3_t1b_source_anchor";OUT.mkdir(parents=True,exist_ok=True)
PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"C":np.array([.25]*4,np.float32)}

def main():
 src=json.load(open(SRC));m=mujoco.MjModel.from_xml_path(str(SCENE));d=mujoco.MjData(m)
 rt=Phase1EagerStateRuntime(str(ART),device="cuda")
 jmap={}
 for j in range(m.njnt):
  n=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j)
  if n:jmap[n]=(int(m.jnt_qposadr[j]),int(m.jnt_dofadr[j]))
 act_joints=[]
 for a in range(m.nu):
  jid=int(m.actuator_trnid[a,0]);act_joints.append(mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,jid))
 trunk=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"trunk")
 com_b=m.body_ipos[trunk].copy();qa=int(m.jnt_qposadr[0]);va=int(m.jnt_dofadr[0]);hold=int(round(.02/m.opt.timestep))

 def load_snapshot(snap,e):
  mujoco.mj_resetData(m,d)
  pos=np.asarray(snap["root_link_pos_local"][e],float);quat=np.asarray(snap["root_link_quat_w"][e],float)
  lin_w=np.asarray(snap["root_link_lin_vel_w"][e],float);ang_w=np.asarray(snap["root_link_ang_vel_w"][e],float)
  R=quat_wxyz_to_rot(quat);ang_b=R.T@ang_w
  d.qpos[qa:qa+3]=pos;d.qpos[qa+3:qa+7]=quat
  d.qvel[va:va+3]=lin_w;d.qvel[va+3:va+6]=ang_b
  names=snap["joint_names"];qp=np.asarray(snap["joint_pos"][e],float);qv=np.asarray(snap["joint_vel"][e],float)
  for jn in CANONICAL_JOINT_ORDER:
    ii=names.index(jn);qadr,vadr=jmap[jn];d.qpos[qadr]=qp[ii];d.qvel[vadr]=qv[ii]
  for a,jn in enumerate(act_joints):d.ctrl[a]=d.qpos[jmap[jn][0]]
  mujoco.mj_forward(m,d)

 def state(cmd,prev):
  quat=d.qpos[qa+3:qa+7].copy();qv=d.qvel[va:va+6].copy()
  _,ang_b,g=canonical_base_kinematics(quat,qv)
  lin_b=root_com_velocity_b_from_freejoint(quat,qv[:3],qv[3:],com_b).astype(np.float32)
  qp=np.array([d.qpos[jmap[j][0]] for j in CANONICAL_JOINT_ORDER],np.float32)
  jv=np.array([d.qvel[jmap[j][1]] for j in CANONICAL_JOINT_ORDER],np.float32)
  obs=build_canonical_obs(lin_b,ang_b,g,cmd,qp,jv,prev,CANONICAL_JOINT_ORDER)
  return obs,lin_b,ang_b,g

 suite_reports=[];all_traces={}
 for snap in src["snapshots"]:
  si=snap["suite"];suite_by_pref={"T":[],"C":[]}
  for e in range(len(snap["command"])):
   cmd=np.asarray(snap["command"][e],np.float32)
   for lab,pref in PREFS.items():
    load_snapshot(snap,e);prev=np.zeros(12,np.float32);rt.last_action=np.zeros((1,12),np.float32);rt.estop_latched=False
    vals=[];trace=[];cum=0.0
    for t in range(64):
      obs,v,w,g=state(cmd,prev);a=rt.act(obs[None,:],pref[None,:])[0]
      errxy=float(np.sum((cmd[:2]-v[:2])**2));erryaw=float((cmd[2]-w[2])**2)
      T=(1.5*math.exp(-errxy/.25)+.75*math.exp(-erryaw/.25))/1.7194554805755615*.02
      phys=float(abs(v[0]-cmd[0])+abs(w[2]-cmd[2]));cum+=T
      vals.append((T,phys,float(np.linalg.norm(a-prev)),float(d.qpos[qa+2]),float(np.mean(np.abs(a)>=.98))))
      trace.append({"t":t,"T_obj":T,"cum_T":cum,"tracking":phys,"vx":float(v[0]),"vy":float(v[1]),"wz":float(w[2]),
                    "height":float(d.qpos[qa+2]),"action":a.tolist(),"action_rate":float(np.linalg.norm(a-prev)),
                    "sat_frac":float(np.mean(np.abs(a)>=.98))})
      target=canonical_joint_target(a);d.ctrl[:]=np.array([target[CANONICAL_JOINT_ORDER.index(j)] for j in act_joints])
      for _ in range(hold):mujoco.mj_step(m,d)
      prev=a.copy()
    ar=np.asarray(vals,float)
    suite_by_pref[lab].append({"T_obj_mean":float(ar[:,0].mean()),"tracking_mean":float(ar[:,1].mean()),
                               "action_rate_mean":float(ar[:,2].mean()),"height_mean":float(ar[:,3].mean()),
                               "sat_mean":float(ar[:,4].mean())})
    all_traces[f"s{si}:e{e}:{lab}"]=trace
  agg={}
  for lab in ("T","C"):
    agg[lab]={k:float(np.mean([x[k] for x in suite_by_pref[lab]])) for k in suite_by_pref[lab][0]}
  dObj=agg["T"]["T_obj_mean"]-agg["C"]["T_obj_mean"];dPhys=agg["T"]["tracking_mean"]-agg["C"]["tracking_mean"]
  suite_reports.append({"suite":si,"T":agg["T"],"C":agg["C"],"delta_T_obj":dObj,"delta_tracking":dPhys,
                        "objective_correct":bool(dObj>0),"physical_correct":bool(dPhys<0)})
  print("SUITE",si,"dObj",dObj,"dPhys",dPhys,flush=True)

 obj_frac=float(np.mean([x["objective_correct"] for x in suite_reports]))
 phys_frac=float(np.mean([x["physical_correct"] for x in suite_reports]))
 rep={"schema":"phase3_t1b_mujoco_source_replay_v1","suites":suite_reports,
      "objective_correct_fraction":obj_frac,"physical_correct_fraction":phys_frac,
      "semantic_pass":bool(obj_frac>=.75 and phys_frac>=.75),"traces":all_traces}
 (OUT/"mujoco_source_replay.json").write_text(json.dumps(rep,indent=2)+"\n")
 print("FINAL",json.dumps({k:v for k,v in rep.items() if k!="traces"},indent=2),flush=True)

if __name__=="__main__":main()
