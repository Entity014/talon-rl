#!/usr/bin/env python3
from pathlib import Path
import json,math,sys
import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from talon_rl.phase1_deployment import Phase1EagerStateRuntime
from talon_rl.phase1_mujoco_adapter import (
    CANONICAL_JOINT_ORDER,CANONICAL_DEFAULT_Q,
    build_canonical_obs,canonical_base_kinematics,
    root_com_velocity_b_from_freejoint,canonical_joint_target,
)

SCENE=ROOT/"talon_rl/assets/data/Robots/unitree_a1/mujoco/scene.xml"
ART=ROOT/"artifacts/phase1_canonical_actor_state.pt"
OUT=ROOT/"runs/phase3_t1_matched_trace";OUT.mkdir(parents=True,exist_ok=True)
COMMANDS={
 "forward":[.5,0.,0.],
 "turn_left":[.3,0.,.3],
 "turn_right":[.3,0.,-.3],
 "lateral":[0.,.25,0.],
}
PREFS={"T":[.7,.1,.1,.1],"C":[.25,.25,.25,.25]}

def tilt_deg(g): return math.degrees(math.acos(float(np.clip(-g[2],-1.,1.))))

def main():
 m=mujoco.MjModel.from_xml_path(str(SCENE));d=mujoco.MjData(m)
 rt=Phase1EagerStateRuntime(str(ART),device="cuda")
 jmap={}
 for j in range(m.njnt):
  n=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j)
  if n:jmap[n]=(int(m.jnt_qposadr[j]),int(m.jnt_dofadr[j]))
 act_joints=[]
 for a in range(m.nu):
  jid=int(m.actuator_trnid[a,0]);act_joints.append(mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,jid))
 trunk=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"trunk")
 com_b=m.body_ipos[trunk].copy();qa=int(m.jnt_qposadr[0]);va=int(m.jnt_dofadr[0])
 hold=int(round(.02/m.opt.timestep))

 def reset():
  mujoco.mj_resetData(m,d)
  d.qpos[qa:qa+3]=[0,0,.43];d.qpos[qa+3:qa+7]=[1,0,0,0];d.qvel[:]=0
  for i,jn in enumerate(CANONICAL_JOINT_ORDER):d.qpos[jmap[jn][0]]=CANONICAL_DEFAULT_Q[i]
  for a,jn in enumerate(act_joints):d.ctrl[a]=d.qpos[jmap[jn][0]]
  mujoco.mj_forward(m,d)

 def state(cmd,prev):
  quat=d.qpos[qa+3:qa+7].copy();qv=d.qvel[va:va+6].copy()
  _,ang_b,grav_b=canonical_base_kinematics(quat,qv)
  lin_b=root_com_velocity_b_from_freejoint(quat,qv[:3],qv[3:],com_b).astype(np.float32)
  q=np.array([d.qpos[jmap[j][0]] for j in CANONICAL_JOINT_ORDER],np.float32)
  qd=np.array([d.qvel[jmap[j][1]] for j in CANONICAL_JOINT_ORDER],np.float32)
  obs=build_canonical_obs(lin_b,ang_b,grav_b,np.asarray(cmd,np.float32),q,qd,prev,CANONICAL_JOINT_ORDER)
  return obs,lin_b,ang_b,grav_b,q,qd

 def contacts():
  c=0
  for k in range(d.ncon):
   con=d.contact[k]
   b1=int(m.geom_bodyid[con.geom1]);b2=int(m.geom_bodyid[con.geom2])
   if b1!=0 and b2==0 or b2!=0 and b1==0:c+=1
  return c

 traces={}
 for cname,cmd in COMMANDS.items():
  for plab,pref in PREFS.items():
   reset();prev=np.zeros(12,np.float32);rt.last_action=np.zeros((1,12),np.float32);rt.estop_latched=False
   pref_np=np.asarray(pref,np.float32)[None,:];rows=[];cumT=0.0
   for t in range(64):
    obs,v,w,g,q,qd=state(cmd,prev)
    a=rt.act(obs[None,:],pref_np)[0]
    errxy=float(np.sum((np.asarray(cmd[:2])-v[:2])**2));erryaw=float((cmd[2]-w[2])**2)
    Treward=1.5*math.exp(-errxy/.25)+.75*math.exp(-erryaw/.25);normT=Treward/1.7194554805755615*.02;cumT+=normT
    rows.append({"t":t,"vx":float(v[0]),"vy":float(v[1]),"vz":float(v[2]),
      "wx":float(w[0]),"wy":float(w[1]),"wz":float(w[2]),"g":g.tolist(),
      "height":float(d.qpos[qa+2]),"tilt_deg":float(tilt_deg(g)),
      "tracking_error":float(abs(v[0]-cmd[0])+abs(w[2]-cmd[2])),
      "T_obj":float(normT),"cum_T":float(cumT),"action":a.tolist(),
      "action_rate":float(np.linalg.norm(a-prev)),"sat_frac":float(np.mean(np.abs(a)>=.98)),
      "contacts":int(contacts()),"q":q.tolist(),"qd":qd.tolist(),"done":False})
    target=canonical_joint_target(a);d.ctrl[:]=np.array([target[CANONICAL_JOINT_ORDER.index(j)] for j in act_joints])
    for _ in range(hold):mujoco.mj_step(m,d)
    prev=a.copy()
   traces[f"{cname}:{plab}"]=rows
   print("DONE",cname,plab,flush=True)
 rep={"schema":"phase3_t1_mujoco_matched_trace_v1","engine":"mujoco",
      "matched_root_pose":[0,0,.43,1,0,0,0],"commands":COMMANDS,"traces":traces}
 (OUT/"mujoco_trace.json").write_text(json.dumps(rep,indent=2)+"\n")

if __name__=="__main__":main()
