"""Consolidated experiment stage module."""
from __future__ import annotations
import argparse

def run_phase3_t1_compare():
    """Run former phase3_t1_compare.py stage."""
    from pathlib import Path
    import json
    import numpy as np
    
    ROOT=Path(__file__).resolve().parents[4]
    DIR=ROOT/"runs/phase3_t1_matched_trace"
    I=json.load(open(DIR/"isaac_trace.json"))
    M=json.load(open(DIR/"mujoco_trace.json"))
    CMDS=("forward","turn_left","turn_right","lateral")
    
    def arr(rows,key):
        if key in ("action","q","qd"):
            return np.asarray([x[key] for x in rows],float)
        if key=="state":
            return np.asarray([[x["vx"],x["vy"],x["vz"],x["wx"],x["wy"],x["wz"],*x["g"],*x["q"],*x["qd"]] for x in rows],float)
        return np.asarray([x[key] for x in rows],float)
    
    def engine_metrics(rep,cmd):
        T=rep["traces"][f"{cmd}:T"];C=rep["traces"][f"{cmd}:C"]
        aT=arr(T,"action");aC=arr(C,"action");da=aT-aC
        st=arr(T,"state");sc=arr(C,"state")
        trT=arr(T,"tracking_error");trC=arr(C,"tracking_error")
        tObj=arr(T,"T_obj");cObj=arr(C,"T_obj")
        out={
          "t0_action_sep":float(np.linalg.norm(da[0])),
          "mean_action_sep_h8":float(np.mean(np.linalg.norm(da[:8],axis=1))),
          "mean_action_sep_h16":float(np.mean(np.linalg.norm(da[:16],axis=1))),
          "mean_action_sep_h64":float(np.mean(np.linalg.norm(da,axis=1))),
          "mean_state_sep_h8":float(np.mean(np.linalg.norm(st[:8]-sc[:8],axis=1))),
          "mean_state_sep_h16":float(np.mean(np.linalg.norm(st[:16]-sc[:16],axis=1))),
          "mean_state_sep_h64":float(np.mean(np.linalg.norm(st-sc,axis=1))),
          "tracking_delta_h8":float(np.mean(trT[:8]-trC[:8])),
          "tracking_delta_h16":float(np.mean(trT[:16]-trC[:16])),
          "tracking_delta_h64":float(np.mean(trT-trC)),
          "cum_T_margin_h8":float(np.sum(tObj[:8]-cObj[:8])),
          "cum_T_margin_h16":float(np.sum(tObj[:16]-cObj[:16])),
          "cum_T_margin_h64":float(np.sum(tObj-cObj)),
          "T_tracking_mean":float(np.mean(trT)),
          "C_tracking_mean":float(np.mean(trC)),
          "T_height_mean":float(np.mean(arr(T,"height"))),
          "C_height_mean":float(np.mean(arr(C,"height"))),
          "T_sat_mean":float(np.mean(arr(T,"sat_frac"))),
          "C_sat_mean":float(np.mean(arr(C,"sat_frac"))),
          "T_contacts_mean":float(np.mean(arr(T,"contacts"))),
          "C_contacts_mean":float(np.mean(arr(C,"contacts"))),
          "mean_joint_sep":float(np.mean(np.linalg.norm(arr(T,"q")-arr(C,"q"),axis=1))),
        }
        return out,da
    
    rows={};cross={}
    for cmd in CMDS:
        im,ida=engine_metrics(I,cmd);mm,mda=engine_metrics(M,cmd)
        # Same initial observation should imply same T-vs-C action vector.
        t0_vec_err=float(np.max(np.abs(ida[0]-mda[0])))
        # cosine of preference response after divergence.
        cos=[]
        for x,y in zip(ida,mda):
            nx=np.linalg.norm(x);ny=np.linalg.norm(y)
            cos.append(float(np.dot(x,y)/(nx*ny+1e-12)))
        rows[cmd]={"isaac":im,"mujoco":mm}
        cross[cmd]={
          "t0_preference_action_vector_max_error":t0_vec_err,
          "mean_action_response_cosine_h8":float(np.mean(cos[:8])),
          "mean_action_response_cosine_h16":float(np.mean(cos[:16])),
          "mean_action_response_cosine_h64":float(np.mean(cos)),
          "action_sep_ratio_mj_over_isaac_h64":float(mm["mean_action_sep_h64"]/(im["mean_action_sep_h64"]+1e-12)),
          "center_tracking_shift_mj_minus_isaac":float(mm["C_tracking_mean"]-im["C_tracking_mean"]),
          "center_height_shift_mj_minus_isaac":float(mm["C_height_mean"]-im["C_height_mean"]),
          "center_saturation_shift":float(mm["C_sat_mean"]-im["C_sat_mean"]),
          "tracking_margin_sign_flip":bool(np.sign(im["tracking_delta_h64"])!=np.sign(mm["tracking_delta_h64"])),
          "T_return_margin_sign_flip":bool(np.sign(im["cum_T_margin_h64"])!=np.sign(mm["cum_T_margin_h64"])),
        }
    
    agg={
     "isaac_T_semantic_correct_commands":int(sum(rows[c]["isaac"]["tracking_delta_h64"]<0 and rows[c]["isaac"]["cum_T_margin_h64"]>0 for c in CMDS)),
     "mujoco_T_semantic_correct_commands":int(sum(rows[c]["mujoco"]["tracking_delta_h64"]<0 and rows[c]["mujoco"]["cum_T_margin_h64"]>0 for c in CMDS)),
     "t0_exact_commands":int(sum(cross[c]["t0_preference_action_vector_max_error"]<=1e-6 for c in CMDS)),
     "tracking_sign_flip_commands":int(sum(cross[c]["tracking_margin_sign_flip"] for c in CMDS)),
     "return_sign_flip_commands":int(sum(cross[c]["T_return_margin_sign_flip"] for c in CMDS)),
     "mean_action_sep_ratio":float(np.mean([cross[c]["action_sep_ratio_mj_over_isaac_h64"] for c in CMDS])),
     "mean_action_cosine_h8":float(np.mean([cross[c]["mean_action_response_cosine_h8"] for c in CMDS])),
     "mean_action_cosine_h64":float(np.mean([cross[c]["mean_action_response_cosine_h64"] for c in CMDS])),
     "mean_center_tracking_shift":float(np.mean([cross[c]["center_tracking_shift_mj_minus_isaac"] for c in CMDS])),
     "mean_center_height_shift":float(np.mean([cross[c]["center_height_shift_mj_minus_isaac"] for c in CMDS])),
     "mean_center_saturation_shift":float(np.mean([cross[c]["center_saturation_shift"] for c in CMDS])),
    }
    
    # Evidence-oriented mechanism flags, not adaptation decisions.
    mechanism={
     "initial_policy_function_changed":bool(agg["t0_exact_commands"]<4),
     "closed_loop_policy_response_diverges":bool(agg["mean_action_cosine_h64"]<0.9 or abs(np.log(max(agg["mean_action_sep_ratio"],1e-12)))>0.2),
     "dynamics_effect_sign_reversal":bool(agg["tracking_sign_flip_commands"]>=3 and agg["return_sign_flip_commands"]>=3),
     "center_baseline_shift_material":bool(abs(agg["mean_center_tracking_shift"])>0.1),
     "operating_regime_shift":bool(abs(agg["mean_center_height_shift"])>0.05 or agg["mean_center_saturation_shift"]>0.2),
    }
    rep={"schema":"phase3_t1_compare_v1","commands":rows,"cross_engine":cross,"aggregate":agg,"mechanism_flags":mechanism}
    (DIR/"comparison_report.json").write_text(json.dumps(rep,indent=2)+"\n")
    print(json.dumps({"aggregate":agg,"mechanism_flags":mechanism,"cross_engine":cross},indent=2))

def run_phase3_t1_isaac_trace():
    """Run former phase3_t1_isaac_trace.py stage."""
    from pathlib import Path
    import json,math,sys
    import numpy as np, torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    ART=ROOT/"artifacts/phase1_canonical_actor_state.pt"
    OUT=ROOT/"runs/phase3_t1_matched_trace";OUT.mkdir(parents=True,exist_ok=True)
    COMMANDS={
     "forward":[.5,0.,0.],
     "turn_left":[.3,0.,.3],
     "turn_right":[.3,0.,-.3],
     "lateral":[0.,.25,0.],
    }
    PREFS={"T":[.7,.1,.1,.1],"C":[.25,.25,.25,.25]}
    CANON=("FL_hip_joint","FR_hip_joint","RL_hip_joint","RR_hip_joint",
           "FL_thigh_joint","FR_thigh_joint","RL_thigh_joint","RR_thigh_joint",
           "FL_calf_joint","FR_calf_joint","RL_calf_joint","RR_calf_joint")
    Q0=torch.tensor([.1,-.1,.1,-.1,.8,.8,1.,1.,-1.5,-1.5,-1.5,-1.5])
    
    def tilt(q):
        _,x,y,_=[q[:,i] for i in range(4)]
        return torch.rad2deg(torch.acos((1-2*(x*x+y*y)).clamp(-1,1)))
    
    def main():
     from isaaclab.app import AppLauncher
     saved=sys.argv[:];sys.argv=[sys.argv[0]]
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
     env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.deployment.phase1 import Phase1EagerStateRuntime
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=1;cfg.seed=0
      cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      cfg.observations.policy.enable_corruption=False
      cfg.commands.base_velocity.heading_command=False
      cfg.commands.base_velocity.rel_heading_envs=0.0
      cfg.commands.base_velocity.rel_standing_envs=0.0
      cfg.commands.base_velocity.resampling_time_range=(1000.0,1000.0)
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
      u=env.unwrapped;robot=u.scene["robot"];sensor=u.scene["contact_forces"]
      rt=Phase1EagerStateRuntime(str(ART),device="cuda")
      joint_ids=[robot.data.joint_names.index(n) for n in CANON]
      foot_ids=[i for i,n in enumerate(sensor.body_names) if "foot" in n.lower()]
      term=u.command_manager.get_term("base_velocity")
      traces={}
      for cname,cmdlist in COMMANDS.items():
       cmd=torch.tensor(cmdlist,device=u.device,dtype=torch.float32).view(1,3)
       for plab,pref in PREFS.items():
        env.reset(seed=260926)
        root_pose=torch.tensor([[0.,0.,.43,1.,0.,0.,0.]],device=u.device)
        root_vel=torch.zeros((1,6),device=u.device)
        robot.write_root_pose_to_sim(root_pose);robot.write_root_velocity_to_sim(root_vel)
        qp=torch.zeros((1,robot.num_joints),device=u.device);qv=torch.zeros_like(qp)
        qp[:,joint_ids]=Q0.to(u.device)
        robot.write_joint_state_to_sim(qp,qv)
        u.scene.write_data_to_sim();u.sim.forward()
        term.vel_command_b[:]=cmd
        term.is_standing_env[:]=False;term.is_heading_env[:]=False
        u.obs_buf=u.observation_manager.compute(update_history=True)
        prev=np.zeros(12,np.float32);rt.last_action=np.zeros((1,12),np.float32);rt.estop_latched=False
        pref_np=np.asarray(pref,np.float32)[None,:]
        rows=[];cumT=0.0
        for t in range(64):
          term.vel_command_b[:]=cmd
          obs=u.obs_buf["policy"] if isinstance(u.obs_buf,dict) else u.obs_buf
          a=rt.act(obs.detach().cpu().numpy().astype(np.float32),pref_np)[0]
          a_t=torch.from_numpy(a).to(u.device).view(1,-1)
          nxt,_,te,tr,_=env.step(a_t)
          data=robot.data;v=data.root_lin_vel_b[0].detach().cpu().numpy();w=data.root_ang_vel_b[0].detach().cpu().numpy()
          g=data.projected_gravity_b[0].detach().cpu().numpy()
          errxy=float(np.sum((np.asarray(cmdlist[:2])-v[:2])**2));erryaw=float((cmdlist[2]-w[2])**2)
          Treward=1.5*math.exp(-errxy/.25)+.75*math.exp(-erryaw/.25);normT=Treward/1.7194554805755615*.02;cumT+=normT
          f=sensor.data.net_forces_w[0,foot_ids].norm(dim=-1)
          rows.append({"t":t,"vx":float(v[0]),"vy":float(v[1]),"vz":float(v[2]),
            "wx":float(w[0]),"wy":float(w[1]),"wz":float(w[2]),
            "g":g.tolist(),"height":float(data.root_link_pos_w[0,2]),"tilt_deg":float(tilt(data.root_link_quat_w)[0]),
            "tracking_error":float(abs(v[0]-cmdlist[0])+abs(w[2]-cmdlist[2])),
            "T_obj":float(normT),"cum_T":float(cumT),"action":a.tolist(),
            "action_rate":float(np.linalg.norm(a-prev)),"sat_frac":float(np.mean(np.abs(a)>=.98)),
            "contacts":int((f>1.0).sum().item()),"q":data.joint_pos[0,joint_ids].detach().cpu().tolist(),
            "qd":data.joint_vel[0,joint_ids].detach().cpu().tolist(),
            "done":bool((te|tr)[0].item())})
          prev=a.copy();u.obs_buf=nxt
        traces[f"{cname}:{plab}"]=rows
        print("DONE",cname,plab,flush=True)
      rep={"schema":"phase3_t1_isaac_matched_trace_v1","engine":"isaac","clean_obs":True,
           "matched_root_pose":[0,0,.43,1,0,0,0],"commands":COMMANDS,"traces":traces}
      (OUT/"isaac_trace.json").write_text(json.dumps(rep,indent=2)+"\n")
     finally:
      if env is not None:env.close()
      app.close()
    
    if True:main()

def run_phase3_t1_mujoco_trace():
    """Run former phase3_t1_mujoco_trace.py stage."""
    from pathlib import Path
    import json,math,sys
    import mujoco
    import numpy as np
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path.insert(0,str(ROOT))
    from talon_rl.deployment.phase1 import Phase1EagerStateRuntime
    from talon_rl.deployment.phase1 import (
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
    
    if True:main()

def run_phase3_t1b_isaac_source_anchor():
    """Run former phase3_t1b_isaac_source_anchor.py stage."""
    from pathlib import Path
    import json,math,sys
    import numpy as np,torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    ART=ROOT/"artifacts/phase1_canonical_actor_state.pt"
    OUT=ROOT/"runs/phase3_t1b_source_anchor";OUT.mkdir(parents=True,exist_ok=True)
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"C":np.array([.25]*4,np.float32)}
    SEEDS=(840001,840002,840003,840004)
    NENV=8;STEPS=64
    
    def obs_tensor(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
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
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
      cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      cfg.observations.policy.enable_corruption=False
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
      u=env.unwrapped;robot=u.scene["robot"];rt=Phase1EagerStateRuntime(str(ART),device="cuda")
      reports=[];snapshots=[]
      for si,seed in enumerate(SEEDS):
       suite={}
       for lab,pref in PREFS.items():
        cur,_=env.reset(seed=seed);cur=obs_tensor(cur)
        cmd=u.command_manager.get_command("base_velocity").detach().cpu().numpy()
        if lab=="T":
          local_pos=(robot.data.root_link_pos_w-u.scene.env_origins).detach().cpu().numpy()
          snapshots.append({
            "suite":si,"seed":seed,
            "root_link_pos_local":local_pos.tolist(),
            "root_link_quat_w":robot.data.root_link_quat_w.detach().cpu().tolist(),
            "root_link_lin_vel_w":robot.data.root_link_lin_vel_w.detach().cpu().tolist(),
            "root_link_ang_vel_w":robot.data.root_link_ang_vel_w.detach().cpu().tolist(),
            "root_com_lin_vel_w":robot.data.root_com_lin_vel_w.detach().cpu().tolist(),
            "joint_pos":robot.data.joint_pos.detach().cpu().tolist(),
            "joint_vel":robot.data.joint_vel.detach().cpu().tolist(),
            "joint_names":list(robot.data.joint_names),
            "command":cmd.tolist(),
          })
        rt.last_action=np.zeros((NENV,12),np.float32);rt.estop_latched=False
        rows=[];prev=np.zeros((NENV,12),np.float32)
        pref_batch=np.repeat(pref[None,:],NENV,axis=0)
        for t in range(STEPS):
          a=rt.act(cur.detach().cpu().numpy().astype(np.float32),pref_batch)
          a_t=torch.from_numpy(a).to(u.device)
          nxt,_,te,tr,_=env.step(a_t)
          data=robot.data;v=data.root_lin_vel_b.detach().cpu().numpy();w=data.root_ang_vel_b.detach().cpu().numpy()
          cm=u.command_manager.get_command("base_velocity").detach().cpu().numpy()
          errxy=np.sum((cm[:,:2]-v[:,:2])**2,axis=1);erryaw=(cm[:,2]-w[:,2])**2
          T=(1.5*np.exp(-errxy/.25)+.75*np.exp(-erryaw/.25))/1.7194554805755615*.02
          phys=np.abs(v[:,0]-cm[:,0])+np.abs(w[:,2]-cm[:,2])
          rows.append({"T_obj":T.tolist(),"tracking":phys.tolist(),"action_rate":np.linalg.norm(a-prev,axis=1).tolist()})
          prev=a.copy();cur=obs_tensor(nxt)
        suite[lab]={
          "T_obj_mean":float(np.mean([x for r in rows for x in r["T_obj"]])),
          "tracking_mean":float(np.mean([x for r in rows for x in r["tracking"]])),
          "action_rate_mean":float(np.mean([x for r in rows for x in r["action_rate"]])),
        }
       dObj=suite["T"]["T_obj_mean"]-suite["C"]["T_obj_mean"]
       dPhys=suite["T"]["tracking_mean"]-suite["C"]["tracking_mean"]
       reports.append({"suite":si,"seed":seed,"T":suite["T"],"C":suite["C"],
                       "delta_T_obj":dObj,"delta_tracking":dPhys,
                       "objective_correct":bool(dObj>0),"physical_correct":bool(dPhys<0)})
       print("SUITE",si,"dObj",dObj,"dPhys",dPhys,flush=True)
      obj_frac=float(np.mean([r["objective_correct"] for r in reports]))
      phys_frac=float(np.mean([r["physical_correct"] for r in reports]))
      rep={"schema":"phase3_t1b_isaac_source_anchor_v1","clean_observation":True,
           "suites":reports,"objective_correct_fraction":obj_frac,"physical_correct_fraction":phys_frac,
           "anchor_pass":bool(obj_frac>=.75 and phys_frac>=.75),"snapshots":snapshots}
      (OUT/"isaac_source_anchor.json").write_text(json.dumps(rep,indent=2)+"\n")
      print("FINAL",json.dumps({k:v for k,v in rep.items() if k!="snapshots"},indent=2),flush=True)
     finally:
      if env is not None:env.close()
      app.close()
    
    if True:main()

def run_phase3_t1b_mujoco_source_replay():
    """Run former phase3_t1b_mujoco_source_replay.py stage."""
    from pathlib import Path
    import json,math,sys
    import mujoco
    import numpy as np
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path.insert(0,str(ROOT))
    from talon_rl.deployment.phase1 import Phase1EagerStateRuntime
    from talon_rl.deployment.phase1 import (
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
    
    if True:main()

def run_phase3_t1c_isaac_one_step():
    """Run former phase3_t1c_isaac_one_step.py stage."""
    from pathlib import Path
    import json,sys
    import numpy as np,torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    ART=ROOT/"artifacts/phase1_canonical_actor_state.pt"
    OUT=ROOT/"runs/phase3_t1c_one_step";OUT.mkdir(parents=True,exist_ok=True)
    SEEDS=(840001,840002,840003,840004);NENV=8
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"C":np.array([.25]*4,np.float32)}
    
    def obs_tensor(x):
        if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
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
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=NENV;cfg.seed=0
      cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      cfg.observations.policy.enable_corruption=False
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
      u=env.unwrapped;robot=u.scene["robot"];rt=Phase1EagerStateRuntime(str(ART),device="cuda")
      suites=[]
      for si,seed in enumerate(SEEDS):
       # Capture source state and actions from the same reset.
       cur,_=env.reset(seed=seed);cur=obs_tensor(cur)
       cmd=u.command_manager.get_command("base_velocity").detach().cpu().numpy()
       actions={}
       for lab,pref in PREFS.items():
        rt.last_action=np.zeros((NENV,12),np.float32);rt.estop_latched=False
        actions[lab]=rt.act(cur.detach().cpu().numpy().astype(np.float32),np.repeat(pref[None,:],NENV,axis=0))
       init={
        "root_link_pos_local":(robot.data.root_link_pos_w-u.scene.env_origins).detach().cpu().tolist(),
        "root_link_quat_w":robot.data.root_link_quat_w.detach().cpu().tolist(),
        "root_link_lin_vel_w":robot.data.root_link_lin_vel_w.detach().cpu().tolist(),
        "root_link_ang_vel_w":robot.data.root_link_ang_vel_w.detach().cpu().tolist(),
        "joint_pos":robot.data.joint_pos.detach().cpu().tolist(),
        "joint_vel":robot.data.joint_vel.detach().cpu().tolist(),
        "joint_names":list(robot.data.joint_names),"command":cmd.tolist(),
       }
       nexts={}
       for lab in ("T","C"):
        cur,_=env.reset(seed=seed);cur=obs_tensor(cur)
        a=torch.from_numpy(actions[lab]).to(u.device)
        env.step(a)
        v=robot.data.root_lin_vel_b.detach().cpu().numpy()
        w=robot.data.root_ang_vel_b.detach().cpu().numpy()
        g=robot.data.projected_gravity_b.detach().cpu().numpy()
        cm=u.command_manager.get_command("base_velocity").detach().cpu().numpy()
        q=robot.data.joint_pos.detach().cpu().numpy();qd=robot.data.joint_vel.detach().cpu().numpy()
        tr=np.abs(v[:,0]-cm[:,0])+np.abs(w[:,2]-cm[:,2])
        nexts[lab]={"v":v.tolist(),"w":w.tolist(),"g":g.tolist(),"q":q.tolist(),"qd":qd.tolist(),
                    "height":robot.data.root_link_pos_w[:,2].detach().cpu().tolist(),"tracking":tr.tolist()}
       suites.append({"suite":si,"seed":seed,"initial":init,
                      "actions":{k:v.tolist() for k,v in actions.items()},"next":nexts})
       print("SUITE",si,flush=True)
      rep={"schema":"phase3_t1c_isaac_one_step_v1","clean_observation":True,"suites":suites}
      (OUT/"isaac_one_step.json").write_text(json.dumps(rep,indent=2)+"\n")
     finally:
      if env is not None:env.close()
      app.close()
    
    if True:main()

def run_phase3_t1c_mujoco_compare():
    """Run former phase3_t1c_mujoco_compare.py stage."""
    from pathlib import Path
    import json,sys
    import mujoco
    import numpy as np
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path.insert(0,str(ROOT))
    from talon_rl.deployment.phase1 import (
        CANONICAL_JOINT_ORDER,canonical_base_kinematics,root_com_velocity_b_from_freejoint,
        canonical_joint_target,quat_wxyz_to_rot,
    )
    SRC=ROOT/"runs/phase3_t1c_one_step/isaac_one_step.json"
    SCENE=ROOT/"talon_rl/assets/data/Robots/unitree_a1/mujoco/scene.xml"
    OUT=ROOT/"runs/phase3_t1c_one_step";OUT.mkdir(parents=True,exist_ok=True)
    
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
      for a,jn in enumerate(act_joints):d.ctrl[a]=d.qpos[jmap[jn][0]]
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
        d.ctrl[:]=np.array([target[CANONICAL_JOINT_ORDER.index(j)] for j in act_joints])
        for _ in range(hold):mujoco.mj_step(m,d)
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
     rep={"schema":"phase3_t1c_one_step_compare_v1","suites":suite_reports,"aggregate":agg,"raw_target":raw}
     (OUT/"one_step_compare.json").write_text(json.dumps(rep,indent=2)+"\n")
     print("FINAL",json.dumps({"suites":suite_reports,"aggregate":agg},indent=2),flush=True)
    
    if True:main()

def run_phase3_t1d_temporal_reversal():
    """Run former phase3_t1d_temporal_reversal.py stage."""
    from pathlib import Path
    import json
    import numpy as np
    
    ROOT=Path(__file__).resolve().parents[4]
    SRC=ROOT/"runs/phase3_t1b_source_anchor/mujoco_source_replay.json"
    OUT=ROOT/"runs/phase3_t1d_temporal_reversal";OUT.mkdir(parents=True,exist_ok=True)
    H=(1,4,8,16,32,64)
    
    def main():
     d=json.load(open(SRC));tr=d["traces"];reports=[]
     for suite in range(4):
      envs=range(8);rows=[]
      # tensors [env,time]
      TT=[];CT=[];TP=[];CP=[];TH=[];CH=[];TS=[];CS=[];AD=[]
      for e in envs:
       T=tr[f"s{suite}:e{e}:T"];C=tr[f"s{suite}:e{e}:C"]
       TT.append([x["T_obj"] for x in T]);CT.append([x["T_obj"] for x in C])
       TP.append([x["tracking"] for x in T]);CP.append([x["tracking"] for x in C])
       TH.append([x["height"] for x in T]);CH.append([x["height"] for x in C])
       TS.append([x["sat_frac"] for x in T]);CS.append([x["sat_frac"] for x in C])
       AD.append([np.linalg.norm(np.asarray(x["action"])-np.asarray(y["action"])) for x,y in zip(T,C)])
      TT=np.asarray(TT);CT=np.asarray(CT);TP=np.asarray(TP);CP=np.asarray(CP)
      TH=np.asarray(TH);CH=np.asarray(CH);TS=np.asarray(TS);CS=np.asarray(CS);AD=np.asarray(AD)
      first_phys_wrong=None;first_obj_wrong=None
      for h in H:
       physical=float(np.mean(TP[:,:h]-CP[:,:h]))
       obj=float(np.mean(np.sum(TT[:,:h]-CT[:,:h],axis=1)))
       row={"horizon":h,"tracking_delta_T_minus_C":physical,"cum_T_margin":obj,
            "action_sep_mean":float(np.mean(AD[:,:h])),
            "height_delta":float(np.mean(TH[:,:h]-CH[:,:h])),
            "T_sat":float(np.mean(TS[:,:h])),"C_sat":float(np.mean(CS[:,:h]))}
       rows.append(row)
       if first_phys_wrong is None and physical>0:first_phys_wrong=h
       if first_obj_wrong is None and obj<0:first_obj_wrong=h
      reports.append({"suite":suite,"ladder":rows,"first_physical_wrong_h":first_phys_wrong,"first_objective_wrong_h":first_obj_wrong})
     agg={
      "early_H1_physical_correct_suites":int(sum(r["ladder"][0]["tracking_delta_T_minus_C"]<0 for r in reports)),
      "H64_physical_correct_suites":int(sum(r["ladder"][-1]["tracking_delta_T_minus_C"]<0 for r in reports)),
      "early_H1_objective_correct_suites":int(sum(r["ladder"][0]["cum_T_margin"]>0 for r in reports)),
      "H64_objective_correct_suites":int(sum(r["ladder"][-1]["cum_T_margin"]>0 for r in reports)),
      "first_physical_wrong_h":[r["first_physical_wrong_h"] for r in reports],
      "first_objective_wrong_h":[r["first_objective_wrong_h"] for r in reports],
     }
     rep={"schema":"phase3_t1d_temporal_reversal_v1","suites":reports,"aggregate":agg}
     (OUT/"temporal_reversal.json").write_text(json.dumps(rep,indent=2)+"\n")
     print(json.dumps(rep,indent=2),flush=True)
    
    if True:main()

STAGES = {
    "phase3_t1_compare": run_phase3_t1_compare,
    "phase3_t1_isaac_trace": run_phase3_t1_isaac_trace,
    "phase3_t1_mujoco_trace": run_phase3_t1_mujoco_trace,
    "phase3_t1b_isaac_source_anchor": run_phase3_t1b_isaac_source_anchor,
    "phase3_t1b_mujoco_source_replay": run_phase3_t1b_mujoco_source_replay,
    "phase3_t1c_isaac_one_step": run_phase3_t1c_isaac_one_step,
    "phase3_t1c_mujoco_compare": run_phase3_t1c_mujoco_compare,
    "phase3_t1d_temporal_reversal": run_phase3_t1d_temporal_reversal,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
