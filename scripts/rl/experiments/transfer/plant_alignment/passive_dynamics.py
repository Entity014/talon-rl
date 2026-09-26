"""Consolidated experiment stage module."""
from __future__ import annotations
import argparse

def run_phase3_t4_generate_passive_corrected_model():
    """Run former phase3_t4_generate_passive_corrected_model.py stage."""
    from pathlib import Path
    import hashlib,json
    
    ROOT=Path(__file__).resolve().parents[4]
    D=ROOT/"talon_rl/assets/data/Robots/unitree_a1/mujoco"
    SRC=D/"a1_t3_dcmotor.xml";SCENE_SRC=D/"scene_t3_dcmotor.xml"
    A1_OUT=D/"a1_t4_passive_corrected.xml";SCENE_OUT=D/"scene_t4_passive_corrected.xml"
    OUT=ROOT/"runs/phase3_t4_passive_correction";OUT.mkdir(parents=True,exist_ok=True)
    
    def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
    
    def main():
        s=SRC.read_text()
        repl=[
          ('<joint axis="0 1 0" damping="2" armature="0.01" frictionloss="0.2"/>',
           '<joint axis="0 1 0" damping="0" armature="0" frictionloss="0"/>'),
          ('<joint axis="1 0 0" damping="1" range="-0.802851 0.802851"/>',
           '<joint axis="1 0 0" damping="0" range="-0.802851 0.802851"/>'),
        ]
        counts=[]
        for old,new in repl:
            n=s.count(old);counts.append(n)
            if n!=1: raise RuntimeError(f"expected one replacement, got {n}: {old}")
            s=s.replace(old,new)
        A1_OUT.write_text(s)
        sc=SCENE_SRC.read_text()
        token='<include file="a1_t3_dcmotor.xml"/>'
        if sc.count(token)!=1: raise RuntimeError("scene include mismatch")
        SCENE_OUT.write_text(sc.replace(token,'<include file="a1_t4_passive_corrected.xml"/>'))
        rep={"schema":"phase3_t4_model_generation_v1",
             "t3_a1_sha256":sha(SRC),"t4_a1_sha256":sha(A1_OUT),
             "t3_scene_sha256":sha(SCENE_SRC),"t4_scene_sha256":sha(SCENE_OUT),
             "intervention":{"joint_damping":0.0,"frictionloss":0.0,"armature":0.0}}
        (OUT/"model_generation.json").write_text(json.dumps(rep,indent=2)+"\n")
        print(json.dumps(rep,indent=2),flush=True)
    
    if True:main()

def run_phase3_t4a_joint_pulse():
    """Run former phase3_t4a_joint_pulse.py stage."""
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
    
    if True:main()

def run_phase3_t4b_one_step_compare():
    """Run former phase3_t4b_one_step_compare.py stage."""
    from pathlib import Path
    import json,sys
    import mujoco
    import numpy as np
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path.insert(0,str(ROOT))
    from talon_rl.deployment.phase1 import (
        CANONICAL_JOINT_ORDER,canonical_base_kinematics,root_com_velocity_b_from_freejoint,
        canonical_joint_target,quat_wxyz_to_rot,source_dcmotor_torque,
    )
    SRC=ROOT/"runs/phase3_t1c_one_step/isaac_one_step.json"
    SCENE=ROOT/"talon_rl/assets/data/Robots/unitree_a1/mujoco/scene_t4_passive_corrected.xml"
    OUT=ROOT/"runs/phase3_t4b_one_step";OUT.mkdir(parents=True,exist_ok=True)
    
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
     rep={"schema":"phase3_t4b_one_step_compare_v1","suites":suite_reports,"aggregate":agg,"raw_target":raw}
     (OUT/"one_step_compare.json").write_text(json.dumps(rep,indent=2)+"\n")
     print("FINAL",json.dumps({"suites":suite_reports,"aggregate":agg},indent=2),flush=True)
    
    if True:main()

def run_phase3_t4c_nominal_dynamics():
    """Run former phase3_t4c_nominal_dynamics.py stage."""
    from pathlib import Path
    import json, math, sys
    import mujoco
    import numpy as np
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path.insert(0,str(ROOT))
    
    from talon_rl.deployment.phase1 import Phase1EagerStateRuntime
    from talon_rl.deployment.phase1 import (
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
    
    if True:main()

STAGES = {
    "phase3_t4_generate_passive_corrected_model": run_phase3_t4_generate_passive_corrected_model,
    "phase3_t4a_joint_pulse": run_phase3_t4a_joint_pulse,
    "phase3_t4b_one_step_compare": run_phase3_t4b_one_step_compare,
    "phase3_t4c_nominal_dynamics": run_phase3_t4c_nominal_dynamics,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
