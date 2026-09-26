"""Consolidated experiment stages. Use the first CLI argument to select a former script stage."""
from __future__ import annotations
import argparse

def run_phase4_r2_isaac_source_trajectories():
    """Run former phase4_r2_isaac_source_trajectories.py stage."""
    from pathlib import Path
    import json,sys,math
    import numpy as np,torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    ART=ROOT/"artifacts/phase1_canonical_actor_state.pt"
    OUT=ROOT/"runs/phase4_controller_robustness";OUT.mkdir(parents=True,exist_ok=True)
    SEEDS=(840001,840002,840003,840004);NENV=8;STEPS=64
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"C":np.array([.25]*4,np.float32)}
    
    def obs_tensor(x):
        if isinstance(x,dict): x=x.get("policy",next(iter(x.values())))
        return x if torch.is_tensor(x) else torch.as_tensor(x)
    
    def main():
     from isaaclab.app import AppLauncher
     saved=sys.argv[:];sys.argv=[sys.argv[0]]
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
     env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.deployment.phase1 import Phase1EagerStateRuntime
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.observations.policy.enable_corruption=False
      cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
      u=env.unwrapped;robot=u.scene["robot"];rt=Phase1EagerStateRuntime(str(ART),device="cuda")
      term=u.command_manager.get_term("base_velocity")
      suites=[]
      for si,seed in enumerate(SEEDS):
        obs,_=env.reset(seed=seed);obs=obs_tensor(obs)
        cmd=u.command_manager.get_command("base_velocity").detach().clone()
        snap={
          "root_link_pos_local":(robot.data.root_link_pos_w-u.scene.env_origins).detach().cpu().tolist(),
          "root_link_quat_w":robot.data.root_link_quat_w.detach().cpu().tolist(),
          "root_link_lin_vel_w":robot.data.root_link_lin_vel_w.detach().cpu().tolist(),
          "root_link_ang_vel_w":robot.data.root_link_ang_vel_w.detach().cpu().tolist(),
          "root_com_lin_vel_w":robot.data.root_com_lin_vel_w.detach().cpu().tolist(),
          "joint_pos":robot.data.joint_pos.detach().cpu().tolist(),
          "joint_vel":robot.data.joint_vel.detach().cpu().tolist(),
          "joint_names":list(robot.data.joint_names),"command":cmd.detach().cpu().tolist(),
        }
        traces={}
        for plab,pref in PREFS.items():
          # restore exact captured physical state and command
          rp=torch.tensor(snap["root_link_pos_local"],device=u.device)+u.scene.env_origins
          rq=torch.tensor(snap["root_link_quat_w"],device=u.device)
          rv=torch.cat([torch.tensor(snap["root_link_lin_vel_w"],device=u.device),
                        torch.tensor(snap["root_link_ang_vel_w"],device=u.device)],dim=1)
          robot.write_root_pose_to_sim(torch.cat([rp,rq],dim=1));robot.write_root_velocity_to_sim(rv)
          robot.write_joint_state_to_sim(torch.tensor(snap["joint_pos"],device=u.device),
                                         torch.tensor(snap["joint_vel"],device=u.device))
          term.vel_command_b[:]=cmd;term.is_standing_env[:]=False
          if hasattr(term,"is_heading_env"): term.is_heading_env[:]=False
          u.action_manager.reset()
          u.scene.write_data_to_sim();u.sim.forward()
          u.obs_buf=u.observation_manager.compute(update_history=True)
          rt.last_action=np.zeros((NENV,12),np.float32);rt.estop_latched=False
          prev=np.zeros((NENV,12),np.float32);pref_batch=np.repeat(pref[None,:],NENV,axis=0);rows=[]
          for t in range(STEPS):
            term.vel_command_b[:]=cmd
            x=obs_tensor(u.obs_buf).detach().cpu().numpy().astype(np.float32)
            a=rt.act(x,pref_batch)
            nxt,_,te,tr,_=env.step(torch.from_numpy(a).to(u.device))
            d=robot.data;v=d.root_lin_vel_b.detach().cpu().numpy();w=d.root_ang_vel_b.detach().cpu().numpy()
            g=d.projected_gravity_b.detach().cpu().numpy();q=d.joint_pos.detach().cpu().numpy();qd=d.joint_vel.detach().cpu().numpy()
            h=d.root_link_pos_w[:,2].detach().cpu().numpy();cm=u.command_manager.get_command("base_velocity").detach().cpu().numpy()
            err=np.abs(v[:,0]-cm[:,0])+np.abs(w[:,2]-cm[:,2])
            errxy=np.sum((cm[:,:2]-v[:,:2])**2,axis=1);erryaw=(cm[:,2]-w[:,2])**2
            Tobj=(1.5*np.exp(-errxy/.25)+.75*np.exp(-erryaw/.25))/1.7194554805755615*.02
            rows.append({"v":v.tolist(),"w":w.tolist(),"g":g.tolist(),"q":q.tolist(),"qd":qd.tolist(),
                         "height":h.tolist(),"action":a.tolist(),"tracking":err.tolist(),"T_obj":Tobj.tolist(),
                         "done":(te|tr).detach().cpu().tolist()})
            prev=a.copy();u.obs_buf=nxt
          traces[plab]=rows
        # source semantic sanity
        def mean_metric(plab,key):
          return float(np.mean([z for r in traces[plab] for z in r[key]]))
        dphys=mean_metric("T","tracking")-mean_metric("C","tracking")
        dobj=mean_metric("T","T_obj")-mean_metric("C","T_obj")
        suites.append({"suite":si,"seed":seed,"snapshot":snap,"traces":traces,
                       "delta_tracking_T_C":dphys,"delta_Tobj_T_C":dobj,
                       "physical_correct":bool(dphys<0),"objective_correct":bool(dobj>0)})
        print("SUITE",si,"dtrack",dphys,"dT",dobj,flush=True)
      rep={"schema":"phase4_r2_isaac_source_trajectories_v1","suites":suites,
           "physical_correct_fraction":float(np.mean([s["physical_correct"] for s in suites])),
           "objective_correct_fraction":float(np.mean([s["objective_correct"] for s in suites]))}
      (OUT/"r2_isaac_source_trajectories.json").write_text(json.dumps(rep,indent=2)+"\n")
      print("FINAL",rep["physical_correct_fraction"],rep["objective_correct_fraction"],flush=True)
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_phase4_r2_mujoco_target_trajectories():
    """Run former phase4_r2_mujoco_target_trajectories.py stage."""
    from pathlib import Path
    import json,math,sys
    import numpy as np,mujoco
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path.insert(0,str(ROOT))
    from talon_rl.deployment.phase1 import Phase1EagerStateRuntime
    from talon_rl.deployment.phase1 import (
     CANONICAL_JOINT_ORDER,build_canonical_obs,canonical_base_kinematics,
     root_com_velocity_b_from_freejoint,canonical_joint_target,source_dcmotor_torque,quat_wxyz_to_rot,
    )
    
    SRC=ROOT/"runs/phase4_controller_robustness/r2_isaac_source_trajectories.json"
    ART=ROOT/"artifacts/phase1_canonical_actor_state.pt"
    SCENE=ROOT/"talon_rl/assets/data/Robots/unitree_a1/mujoco/scene_t4_passive_corrected.xml"
    OUT=ROOT/"runs/phase4_controller_robustness"
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
    
     def load(snap,e):
      mujoco.mj_resetData(m,d)
      pos=np.asarray(snap["root_link_pos_local"][e]);quat=np.asarray(snap["root_link_quat_w"][e])
      lin=np.asarray(snap["root_link_lin_vel_w"][e]);ang=np.asarray(snap["root_link_ang_vel_w"][e]);R=quat_wxyz_to_rot(quat)
      d.qpos[qa:qa+3]=pos;d.qpos[qa+3:qa+7]=quat;d.qvel[va:va+3]=lin;d.qvel[va+3:va+6]=R.T@ang
      names=snap["joint_names"];qp=np.asarray(snap["joint_pos"][e]);qv=np.asarray(snap["joint_vel"][e])
      for n in CANONICAL_JOINT_ORDER:
       ii=names.index(n);qadr,vadr=jmap[n];d.qpos[qadr]=qp[ii];d.qvel[vadr]=qv[ii]
      d.ctrl[:]=0;mujoco.mj_forward(m,d)
    
     def state(cmd,prev):
      quat=d.qpos[qa+3:qa+7].copy();qv=d.qvel[va:va+6].copy()
      _,w,g=canonical_base_kinematics(quat,qv);v=root_com_velocity_b_from_freejoint(quat,qv[:3],qv[3:],com_b).astype(np.float32)
      q=np.array([d.qpos[jmap[n][0]] for n in CANONICAL_JOINT_ORDER],np.float32)
      qd=np.array([d.qvel[jmap[n][1]] for n in CANONICAL_JOINT_ORDER],np.float32)
      obs=build_canonical_obs(v,w,g,cmd,q,qd,prev,CANONICAL_JOINT_ORDER)
      return obs,v,w,g,q,qd
    
     suites=[]
     for ss in src["suites"]:
      si=ss["suite"];snap=ss["snapshot"];traces={}
      for plab,pref in PREFS.items():
       envtr=[]
       for e in range(8):
        load(snap,e);cmd=np.asarray(snap["command"][e],np.float32);prev=np.zeros(12,np.float32)
        rt.last_action=np.zeros((1,12),np.float32);rt.estop_latched=False;rows=[]
        for t in range(64):
         obs,_,_,_,_,_=state(cmd,prev);a=rt.act(obs[None,:],pref[None,:])[0]
         target=canonical_joint_target(a)
         for _ in range(hold):
          qn=np.array([d.qpos[jmap[n][0]] for n in CANONICAL_JOINT_ORDER]);qdn=np.array([d.qvel[jmap[n][1]] for n in CANONICAL_JOINT_ORDER])
          tau=source_dcmotor_torque(target,qn,qdn)
          d.ctrl[:]=np.array([tau[CANONICAL_JOINT_ORDER.index(n)] for n in act_joints]);mujoco.mj_step(m,d)
         prev=a.copy()
         _,v,w,g,q,qd=state(cmd,prev)
         err=float(abs(v[0]-cmd[0])+abs(w[2]-cmd[2]))
         errxy=float(np.sum((cmd[:2]-v[:2])**2));erryaw=float((cmd[2]-w[2])**2)
         Tobj=float((1.5*math.exp(-errxy/.25)+.75*math.exp(-erryaw/.25))/1.7194554805755615*.02)
         rows.append({"v":v.tolist(),"w":w.tolist(),"g":g.tolist(),"q":q.tolist(),"qd":qd.tolist(),
                      "height":float(d.qpos[qa+2]),"action":a.tolist(),"tracking":err,"T_obj":Tobj})
        envtr.append(rows)
       traces[plab]=envtr
      suites.append({"suite":si,"traces":traces})
      print("SUITE",si,"done",flush=True)
     rep={"schema":"phase4_r2_mujoco_target_trajectories_v1","suites":suites}
     (OUT/"r2_mujoco_target_trajectories.json").write_text(json.dumps(rep,indent=2)+"\n")
    if True:main()

def run_phase4_r2_compare_divergence():
    """Run former phase4_r2_compare_divergence.py stage."""
    from pathlib import Path
    import json
    import numpy as np
    
    ROOT=Path(__file__).resolve().parents[4]
    D=ROOT/"runs/phase4_controller_robustness"
    S=json.load(open(D/"r2_isaac_source_trajectories.json"))
    T=json.load(open(D/"r2_mujoco_target_trajectories.json"))
    H=(1,2,4,8,16,32,64);PREFS=("T","C")
    
    def norm_state(r,ei=None,perm=None):
        if ei is None:
            v=np.asarray(r["v"]);w=np.asarray(r["w"]);g=np.asarray(r["g"]);q=np.asarray(r["q"]);qd=np.asarray(r["qd"]);h=float(r["height"])
        else:
            v=np.asarray(r["v"][ei]);w=np.asarray(r["w"][ei]);g=np.asarray(r["g"][ei])
            q=np.asarray(r["q"][ei])[perm];qd=np.asarray(r["qd"][ei])[perm];h=float(r["height"][ei])
        return np.concatenate([v/.01,w/.01,g/.005,q/.005,qd/.05,np.array([h/.01])])
    
    rows=[];semantic=[]
    for ss,tt in zip(S["suites"],T["suites"]):
        suite=ss["suite"];names=ss["snapshot"]["joint_names"]
        canon=("FL_hip_joint","FR_hip_joint","RL_hip_joint","RR_hip_joint","FL_thigh_joint","FR_thigh_joint",
               "RL_thigh_joint","RR_thigh_joint","FL_calf_joint","FR_calf_joint","RL_calf_joint","RR_calf_joint")
        perm=[names.index(n) for n in canon]
        for p in PREFS:
          for e in range(8):
            src=ss["traces"][p];tar=tt["traces"][p][e]
            sd=[];ad=[];hd=[];td=[];od=[]
            for k in range(64):
                xs=norm_state(src[k],e,perm);xt=norm_state(tar[k])
                sd.append(float(np.linalg.norm(xt-xs)))
                ad.append(float(np.linalg.norm(np.asarray(tar[k]["action"])-np.asarray(src[k]["action"][e]))))
                hd.append(float(tar[k]["height"]-src[k]["height"][e]))
                td.append(float(tar[k]["tracking"]-src[k]["tracking"][e]))
                od.append(float(tar[k]["T_obj"]-src[k]["T_obj"][e]))
            for h in H:
                rows.append({"suite":suite,"env":e,"pref":p,"h":h,
                             "state_div":float(np.mean(sd[:h])),
                             "action_div":float(np.mean(ad[:h])),
                             "height_abs_div":float(np.mean(np.abs(hd[:h]))),
                             "tracking_abs_div":float(np.mean(np.abs(td[:h]))),
                             "Tobj_abs_div":float(np.mean(np.abs(od[:h])))})
        # semantic margin within each engine over suite/envs
        for h in H:
          src_T=[];src_C=[];tar_T=[];tar_C=[];src_To=[];src_Co=[];tar_To=[];tar_Co=[]
          for e in range(8):
            src_T += [ss["traces"]["T"][k]["tracking"][e] for k in range(h)]
            src_C += [ss["traces"]["C"][k]["tracking"][e] for k in range(h)]
            src_To += [ss["traces"]["T"][k]["T_obj"][e] for k in range(h)]
            src_Co += [ss["traces"]["C"][k]["T_obj"][e] for k in range(h)]
            tar_T += [tt["traces"]["T"][e][k]["tracking"] for k in range(h)]
            tar_C += [tt["traces"]["C"][e][k]["tracking"] for k in range(h)]
            tar_To += [tt["traces"]["T"][e][k]["T_obj"] for k in range(h)]
            tar_Co += [tt["traces"]["C"][e][k]["T_obj"] for k in range(h)]
          semantic.append({"suite":suite,"h":h,
                           "source_tracking_margin":float(np.mean(src_T)-np.mean(src_C)),
                           "target_tracking_margin":float(np.mean(tar_T)-np.mean(tar_C)),
                           "source_Tobj_margin":float(np.mean(src_To)-np.mean(src_Co)),
                           "target_Tobj_margin":float(np.mean(tar_To)-np.mean(tar_Co))})
    
    summary={}
    for p in PREFS:
     summary[p]={}
     for h in H:
        rr=[r for r in rows if r["pref"]==p and r["h"]==h]
        summary[p][str(h)]={k:float(np.mean([x[k] for x in rr])) for k in
                            ("state_div","action_div","height_abs_div","tracking_abs_div","Tobj_abs_div")}
     # fit early log divergence using H-point instantaneous means approximated from cumulative H=1..8 needs raw reconstruction:
     vals=[]
     for step in range(1,9):
        inst=[]
        for ss,tt in zip(S["suites"],T["suites"]):
          names=ss["snapshot"]["joint_names"];perm=[names.index(n) for n in canon]
          for e in range(8):
            inst.append(np.linalg.norm(norm_state(tt["traces"][p][e][step-1])-norm_state(ss["traces"][p][step-1],e,perm)))
        vals.append(float(np.mean(inst)))
     x=np.arange(1,9,dtype=float);y=np.log(np.asarray(vals)+1e-9)
     summary[p]["early_log_slope_per_step"]=float(np.polyfit(x,y,1)[0])
     summary[p]["instant_state_div_H1_to_H8_ratio"]=float(vals[-1]/(vals[0]+1e-12))
     summary[p]["instant_state_div_steps_1_8"]=vals
    
    sem_summary={}
    for h in H:
     rr=[x for x in semantic if x["h"]==h]
     sem_summary[str(h)]={
       "source_tracking_correct_fraction":float(np.mean([x["source_tracking_margin"]<0 for x in rr])),
       "target_tracking_correct_fraction":float(np.mean([x["target_tracking_margin"]<0 for x in rr])),
       "source_Tobj_correct_fraction":float(np.mean([x["source_Tobj_margin"]>0 for x in rr])),
       "target_Tobj_correct_fraction":float(np.mean([x["target_Tobj_margin"]>0 for x in rr])),
       "mean_source_tracking_margin":float(np.mean([x["source_tracking_margin"] for x in rr])),
       "mean_target_tracking_margin":float(np.mean([x["target_tracking_margin"] for x in rr])),
     }
    rep={"schema":"phase4_r2_divergence_v1","summary":summary,"semantic_ladder":sem_summary,"rows":rows,"semantic_rows":semantic}
    (D/"r2_divergence_report.json").write_text(json.dumps(rep,indent=2)+"\n")
    print(json.dumps({"summary":summary,"semantic_ladder":sem_summary},indent=2))

STAGES = {
    "phase4_r2_isaac_source_trajectories": run_phase4_r2_isaac_source_trajectories,
    "phase4_r2_mujoco_target_trajectories": run_phase4_r2_mujoco_target_trajectories,
    "phase4_r2_compare_divergence": run_phase4_r2_compare_divergence,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
