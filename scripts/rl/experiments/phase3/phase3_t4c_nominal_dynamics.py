#!/usr/bin/env python3
from pathlib import Path
import json, math, sys
import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))

from talon_rl.phase1_deployment import Phase1EagerStateRuntime
from talon_rl.phase1_mujoco_adapter import (
    CANONICAL_JOINT_ORDER,CANONICAL_DEFAULT_Q,
    build_canonical_obs,canonical_base_kinematics,
    root_com_velocity_b_from_freejoint,canonical_joint_target,source_dcmotor_torque,
)

SCENE=ROOT/"talon_rl/assets/data/Robots/unitree_a1/mujoco/scene_t4_passive_corrected.xml"
ART=ROOT/"artifacts/phase1_canonical_actor_state.pt"
OUT=ROOT/"runs/phase3_t4c_nominal_dynamics";OUT.mkdir(parents=True,exist_ok=True)

COMMANDS={
 "stand":[0.,0.,0.],
 "forward":[.5,0.,0.],
 "turn_left":[.3,0.,.3],
 "turn_right":[.3,0.,-.3],
 "lateral":[0.,.25,0.],
}
PREFS={
 "T":[.7,.1,.1,.1],
 "A":[.1,.7,.1,.1],
 "O":[.1,.1,.7,.1],
 "S":[.1,.1,.1,.7],
 "C":[.25,.25,.25,.25],
}
POLICY_STEPS=64

def tilt_deg(gravity_b):
    return math.degrees(math.acos(float(np.clip(-gravity_b[2],-1.,1.))))

def main():
    m=mujoco.MjModel.from_xml_path(str(SCENE))
    d=mujoco.MjData(m)
    rt=Phase1EagerStateRuntime(str(ART),device="cuda")

    # Runtime maps.
    jmap={}
    for j in range(m.njnt):
        n=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j)
        if n:jmap[n]=(int(m.jnt_qposadr[j]),int(m.jnt_dofadr[j]))
    act_joints=[]
    for a in range(m.nu):
        jid=int(m.actuator_trnid[a,0])
        act_joints.append(mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,jid))
    trunk=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"trunk")
    com_b=m.body_ipos[trunk].copy()
    qa=int(m.jnt_qposadr[0]);va=int(m.jnt_dofadr[0])
    hold_steps=int(round(.020/m.opt.timestep))

    def reset(arm):
        if m.nkey:mujoco.mj_resetDataKeyframe(m,d,0)
        else:mujoco.mj_resetData(m,d)
        d.qvel[:]=0
        if arm=="canonical":
            for i,jn in enumerate(CANONICAL_JOINT_ORDER):
                d.qpos[jmap[jn][0]]=CANONICAL_DEFAULT_Q[i]
        # Explicit torque actuator starts at zero torque.
        d.ctrl[:]=0.0
        mujoco.mj_forward(m,d)

    def observation(command,prev):
        quat=d.qpos[qa+3:qa+7].copy()
        qv=d.qvel[va:va+6].copy()
        _,ang_b,grav_b=canonical_base_kinematics(quat,qv)
        lin_b=root_com_velocity_b_from_freejoint(quat,qv[:3],qv[3:],com_b).astype(np.float32)
        q=np.array([d.qpos[jmap[j][0]] for j in CANONICAL_JOINT_ORDER],np.float32)
        qd=np.array([d.qvel[jmap[j][1]] for j in CANONICAL_JOINT_ORDER],np.float32)
        obs=build_canonical_obs(lin_b,ang_b,grav_b,np.asarray(command,np.float32),q,qd,prev,CANONICAL_JOINT_ORDER)
        return obs,lin_b,ang_b,grav_b

    def trunk_floor_contact():
        for k in range(d.ncon):
            c=d.contact[k]
            b1=int(m.geom_bodyid[c.geom1]);b2=int(m.geom_bodyid[c.geom2])
            if (b1==trunk and b2==0) or (b2==trunk and b1==0):
                return True
        return False

    rows=[]
    for arm in ("canonical","native"):
      for cname,cmd in COMMANDS.items():
       for pname,pref in PREFS.items():
        reset(arm);prev=np.zeros(12,np.float32)
        rt.last_action=np.zeros((1,12),np.float32);rt.estop_latched=False
        alive_steps=0;immediate_ok=True;sat_count=0;sat_total=0
        ztrace=[];tilttrace=[];action_norm=[];ctrl_violation=False;finite_ok=True;contact_step=None
        for step in range(POLICY_STEPS):
            obs,lin_b,ang_b,grav_b=observation(cmd,prev)
            if not np.isfinite(obs).all():finite_ok=False;break
            action=rt.act(obs[None,:],np.asarray(pref,np.float32)[None,:],age_ms=0.0)[0]
            if not np.isfinite(action).all():finite_ok=False;break
            sat_count+=int(np.sum(np.abs(action)>=.98));sat_total+=action.size
            action_norm.append(float(np.linalg.norm(action)))
            target=canonical_joint_target(action)
            for _ in range(hold_steps):
                q_now=np.array([d.qpos[jmap[j][0]] for j in CANONICAL_JOINT_ORDER],float)
                qd_now=np.array([d.qvel[jmap[j][1]] for j in CANONICAL_JOINT_ORDER],float)
                tau=source_dcmotor_torque(target,q_now,qd_now)
                if np.any(~np.isfinite(tau)) or np.any(tau<-33.5-1e-7) or np.any(tau>33.5+1e-7):
                    ctrl_violation=True;break
                d.ctrl[:]=np.array([tau[CANONICAL_JOINT_ORDER.index(j)] for j in act_joints],float)
                mujoco.mj_step(m,d)
            if ctrl_violation:break
            prev=action.astype(np.float32,copy=True)
            quat=d.qpos[qa+3:qa+7].copy();qv=d.qvel[va:va+6].copy()
            _,_,g=canonical_base_kinematics(quat,qv)
            z=float(d.qpos[qa+2]);tilt=tilt_deg(g)
            ztrace.append(z);tilttrace.append(tilt)
            finite_state=bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all())
            contact=trunk_floor_contact()
            collapsed=(not finite_state) or z<.18 or tilt>75.0 or contact
            if collapsed:
                if contact:contact_step=step+1
                if step+1<=10:immediate_ok=False
                break
            alive_steps=step+1
        survived=alive_steps>=POLICY_STEPS
        rows.append({
          "arm":arm,"command":cname,"preference":pname,
          "alive_steps":alive_steps,"survived64":survived,
          "immediate10_ok":immediate_ok,
          "finite_ok":finite_ok,"ctrl_violation":ctrl_violation,
          "trunk_contact_step":contact_step,
          "min_height":float(min(ztrace)) if ztrace else None,
          "median_height":float(np.median(ztrace)) if ztrace else None,
          "max_tilt_deg":float(max(tilttrace)) if tilttrace else None,
          "mean_action_norm":float(np.mean(action_norm)) if action_norm else None,
          "saturation_fraction":float(sat_count/max(sat_total,1)),
        })
        print("ROLLOUT",arm,cname,pname,rows[-1]["alive_steps"],rows[-1]["survived64"],flush=True)

    def summary(arm):
        q=[x for x in rows if x["arm"]==arm]
        return {
          "n":len(q),
          "survival64_fraction":float(np.mean([x["survived64"] for x in q])),
          "immediate10_ok_fraction":float(np.mean([x["immediate10_ok"] for x in q])),
          "finite_fraction":float(np.mean([x["finite_ok"] for x in q])),
          "ctrl_contract_fraction":float(np.mean([not x["ctrl_violation"] for x in q])),
          "aggregate_saturation_fraction":float(np.mean([x["saturation_fraction"] for x in q])),
          "median_of_median_height":float(np.median([x["median_height"] for x in q if x["median_height"] is not None])),
          "max_observed_tilt_deg":float(max(x["max_tilt_deg"] for x in q if x["max_tilt_deg"] is not None)),
        }
    can=summary("canonical");nat=summary("native")
    combined={
      "survival64_fraction":float(np.mean([x["survived64"] for x in rows])),
      "immediate10_ok_fraction":float(np.mean([x["immediate10_ok"] for x in rows])),
      "aggregate_saturation_fraction":float(np.mean([x["saturation_fraction"] for x in rows])),
    }
    gates={
      "finite":bool(all(x["finite_ok"] for x in rows)),
      "ctrl_contract":bool(all(not x["ctrl_violation"] for x in rows)),
      "canonical_immediate":bool(can["immediate10_ok_fraction"]>=.75),
      "canonical_survival64":bool(can["survival64_fraction"]>=.75),
      "canonical_saturation":bool(can["aggregate_saturation_fraction"]<.95),
    }
    rep={"schema":"phase3_t4c_nominal_dynamics_v1","rows":rows,
         "canonical_arm":can,"native_arm":nat,"combined":combined,"gates":gates}
    rep["canonical_pass"]=bool(all(gates.values()))
    rep["native_pass"]=bool(nat["immediate10_ok_fraction"]>=.75 and nat["survival64_fraction"]>=.75 and nat["aggregate_saturation_fraction"]<.95)
    rep["d3b_pass"]=rep["canonical_pass"]
    (OUT/"nominal_dynamics_report.json").write_text(json.dumps(rep,indent=2)+"\n")
    print("FINAL",json.dumps({"canonical":can,"native":nat,"gates":gates,"d3b_pass":rep["d3b_pass"]},indent=2),flush=True)

if __name__=="__main__":main()
