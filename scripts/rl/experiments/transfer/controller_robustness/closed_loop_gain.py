"""Consolidated experiment stages. Use the first CLI argument to select a former script stage."""
from __future__ import annotations
import argparse

def run_phase4_r1_isaac_closed_loop_gain():
    """Run former phase4_r1_isaac_closed_loop_gain.py stage."""
    from pathlib import Path
    import json,sys
    import numpy as np,torch
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
    OBS_SRC=ROOT/"runs/phase4_controller_robustness/source_obs.json"
    SNAP_SRC=ROOT/"runs/phase3_t1b_source_anchor/isaac_source_anchor.json"
    ART=ROOT/"artifacts/phase1_canonical_actor_state.pt"
    OUT=ROOT/"runs/phase4_controller_robustness";OUT.mkdir(parents=True,exist_ok=True)
    
    BLOCKS={
     "base_lin_vel":(0,3,.01),"base_ang_vel":(3,6,.01),"projected_gravity":(6,9,.005),
     "joint_pos":(12,24,.005),"joint_vel":(24,36,.05),"prev_action":(36,48,.01),
    }
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"C":np.array([.25]*4,np.float32)}
    VARIANTS=[("base",None,0)]
    for b in BLOCKS:
        VARIANTS += [(b,b,1),(b,b,-1)]
    
    def norm_state(v,w,g,q,qd,h):
        # Dimensionless diagnostic scaling tied to audit perturbation scales.
        return np.concatenate([v/.01,w/.01,g/.005,q/.005,qd/.05,np.array([h/.01])])
    
    def main():
     from isaaclab.app import AppLauncher
     saved=sys.argv[:];sys.argv=[sys.argv[0]]
     app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
     env=None
     try:
      import gymnasium as gym,isaaclab_tasks
      from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
      from talon_rl.deployment.phase1 import Phase1StandaloneActor
      obsrep=json.load(open(OBS_SRC));obs_all=np.asarray(obsrep["obs"],np.float32)
      snaps=json.load(open(SNAP_SRC))["snapshots"]
      bundle=torch.load(ART,map_location="cuda",weights_only=False);state=bundle.get("actor_state",bundle)
      actor=Phase1StandaloneActor().cuda();actor.load_state_dict(state);actor.eval()
    
      nenv=8*len(PREFS)*len(VARIANTS)
      cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=nenv;cfg.observations.policy.enable_corruption=False
      cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
      env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);u=env.unwrapped;robot=u.scene["robot"]
      jnames=list(robot.data.joint_names)
      records=[]
      for suite,snap in enumerate(snaps):
        # Build deterministic actor action table.
        actions=[];meta=[];delta_obs_norm=[]
        for e in range(8):
          x=obs_all[suite*8+e]
          for plab,pref in PREFS.items():
            for vname,b,sgn in VARIANTS:
              xp=x.copy()
              if b is not None:
                lo,hi,eps=BLOCKS[b]; signs=np.where(np.arange(hi-lo)%2==0,1.0,-1.0).astype(np.float32)
                xp[lo:hi]+=sgn*eps*signs
                dn=float(np.linalg.norm(xp-x))
              else: dn=0.0
              with torch.inference_mode():
                a=actor(torch.from_numpy(xp[None]).cuda(),torch.from_numpy(pref[None]).cuda()).cpu().numpy()[0]
              actions.append(a);meta.append((e,plab,vname,b,sgn));delta_obs_norm.append(dn)
        actions=np.asarray(actions,np.float32)
    
        # Repeat exact physical source states in the same order.
        root_pos=[];root_quat=[];root_lin=[];root_ang=[];q=[];qd=[]
        names=snap["joint_names"];perm=[names.index(n) for n in jnames]
        for e,plab,vname,b,sgn in meta:
          root_pos.append(snap["root_link_pos_local"][e]);root_quat.append(snap["root_link_quat_w"][e])
          root_lin.append(snap["root_link_lin_vel_w"][e]);root_ang.append(snap["root_link_ang_vel_w"][e])
          q.append(np.asarray(snap["joint_pos"][e])[perm]);qd.append(np.asarray(snap["joint_vel"][e])[perm])
        root_pos=torch.tensor(root_pos,device=u.device,dtype=torch.float32)+u.scene.env_origins
        root_pose=torch.cat([root_pos,torch.tensor(root_quat,device=u.device,dtype=torch.float32)],dim=1)
        root_vel=torch.cat([torch.tensor(root_lin,device=u.device,dtype=torch.float32),torch.tensor(root_ang,device=u.device,dtype=torch.float32)],dim=1)
        robot.write_root_pose_to_sim(root_pose);robot.write_root_velocity_to_sim(root_vel)
        robot.write_joint_state_to_sim(torch.tensor(np.asarray(q),device=u.device,dtype=torch.float32),
                                       torch.tensor(np.asarray(qd),device=u.device,dtype=torch.float32))
        u.scene.write_data_to_sim();u.sim.forward()
        action_t=torch.from_numpy(actions).to(u.device)
        u.action_manager.process_action(action_t)
        for _ in range(int(u.cfg.decimation)):
          u.action_manager.apply_action()
          u.scene.write_data_to_sim()
          u.sim.step(render=False)
          u.scene.update(float(u.cfg.sim.dt))
        d=robot.data
        v=d.root_lin_vel_b.detach().cpu().numpy();w=d.root_ang_vel_b.detach().cpu().numpy()
        g=d.projected_gravity_b.detach().cpu().numpy();qp=d.joint_pos.detach().cpu().numpy();qdp=d.joint_vel.detach().cpu().numpy()
        h=d.root_link_pos_w[:,2].detach().cpu().numpy()
        states=np.stack([norm_state(v[i],w[i],g[i],qp[i],qdp[i],h[i]) for i in range(nenv)])
        # Compare each +/- variant to paired baseline of same state/pref.
        index={(e,p,v):i for i,(e,p,v,b,s) in enumerate(meta)}
        for i,(e,p,vname,b,sgn) in enumerate(meta):
          if vname=="base":continue
          ib=index[(e,p,"base")]
          da=float(np.linalg.norm(actions[i]-actions[ib]));ds0=float(delta_obs_norm[i])
          ds1=float(np.linalg.norm(states[i]-states[ib]))
          records.append({"engine":"isaac","suite":suite,"env":e,"pref":p,"block":b,"sign":sgn,
                          "delta_obs":ds0,"delta_action":da,"delta_next_state_norm":ds1,
                          "G_pi":da/(ds0+1e-12),"G_cl":ds1/(ds0+1e-12)})
        print("SUITE",suite,"done",flush=True)
      summary={}
      for p in PREFS:
        summary[p]={}
        for b in BLOCKS:
          rr=[r for r in records if r["pref"]==p and r["block"]==b]
          summary[p][b]={"G_pi_mean":float(np.mean([r["G_pi"] for r in rr])),
                         "G_cl_mean":float(np.mean([r["G_cl"] for r in rr])),
                         "G_cl_p90":float(np.quantile([r["G_cl"] for r in rr],.9))}
      rep={"schema":"phase4_r1_isaac_gain_v1","summary":summary,"records":records}
      (OUT/"r1_isaac_gain.json").write_text(json.dumps(rep,indent=2)+"\n")
      print(json.dumps(summary,indent=2),flush=True)
     finally:
      if env is not None:env.close()
      app.close()
    if True:main()

def run_phase4_r1_mujoco_closed_loop_gain():
    """Run former phase4_r1_mujoco_closed_loop_gain.py stage."""
    from pathlib import Path
    import json,sys
    import numpy as np,torch,mujoco
    
    ROOT=Path(__file__).resolve().parents[4]
    sys.path.insert(0,str(ROOT))
    from talon_rl.deployment.phase1 import Phase1StandaloneActor
    from talon_rl.deployment.phase1 import (
        CANONICAL_JOINT_ORDER,canonical_joint_target,source_dcmotor_torque,
        canonical_base_kinematics,root_com_velocity_b_from_freejoint,quat_wxyz_to_rot,
    )
    OBS_SRC=ROOT/"runs/phase4_controller_robustness/source_obs.json"
    SNAP_SRC=ROOT/"runs/phase3_t1b_source_anchor/isaac_source_anchor.json"
    ART=ROOT/"artifacts/phase1_canonical_actor_state.pt"
    SCENE=ROOT/"talon_rl/assets/data/Robots/unitree_a1/mujoco/scene_t4_passive_corrected.xml"
    OUT=ROOT/"runs/phase4_controller_robustness"
    
    BLOCKS={
     "base_lin_vel":(0,3,.01),"base_ang_vel":(3,6,.01),"projected_gravity":(6,9,.005),
     "joint_pos":(12,24,.005),"joint_vel":(24,36,.05),"prev_action":(36,48,.01),
    }
    PREFS={"T":np.array([.7,.1,.1,.1],np.float32),"C":np.array([.25]*4,np.float32)}
    VARIANTS=[("base",None,0)]
    for b in BLOCKS: VARIANTS += [(b,b,1),(b,b,-1)]
    
    def norm_state(v,w,g,q,qd,h):
        return np.concatenate([v/.01,w/.01,g/.005,q/.005,qd/.05,np.array([h/.01])])
    
    def main():
     obsrep=json.load(open(OBS_SRC));obs_all=np.asarray(obsrep["obs"],np.float32)
     snaps=json.load(open(SNAP_SRC))["snapshots"]
     bundle=torch.load(ART,map_location="cuda",weights_only=False);state=bundle.get("actor_state",bundle)
     actor=Phase1StandaloneActor().cuda();actor.load_state_dict(state);actor.eval()
     m=mujoco.MjModel.from_xml_path(str(SCENE));d=mujoco.MjData(m)
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
      lin_w=np.asarray(snap["root_link_lin_vel_w"][e]);ang_w=np.asarray(snap["root_link_ang_vel_w"][e])
      R=quat_wxyz_to_rot(quat);d.qpos[qa:qa+3]=pos;d.qpos[qa+3:qa+7]=quat
      d.qvel[va:va+3]=lin_w;d.qvel[va+3:va+6]=R.T@ang_w
      names=snap["joint_names"];qp=np.asarray(snap["joint_pos"][e]);qv=np.asarray(snap["joint_vel"][e])
      for n in CANONICAL_JOINT_ORDER:
       ii=names.index(n);qadr,vadr=jmap[n];d.qpos[qadr]=qp[ii];d.qvel[vadr]=qv[ii]
      d.ctrl[:]=0;mujoco.mj_forward(m,d)
    
     def transition(snap,e,a):
      load(snap,e);target=canonical_joint_target(a)
      for _ in range(hold):
       q=np.array([d.qpos[jmap[n][0]] for n in CANONICAL_JOINT_ORDER])
       qd=np.array([d.qvel[jmap[n][1]] for n in CANONICAL_JOINT_ORDER])
       tau=source_dcmotor_torque(target,q,qd)
       d.ctrl[:]=np.array([tau[CANONICAL_JOINT_ORDER.index(n)] for n in act_joints])
       mujoco.mj_step(m,d)
      quat=d.qpos[qa+3:qa+7].copy();qv=d.qvel[va:va+6].copy()
      _,w,g=canonical_base_kinematics(quat,qv)
      v=root_com_velocity_b_from_freejoint(quat,qv[:3],qv[3:],com_b)
      q=np.array([d.qpos[jmap[n][0]] for n in CANONICAL_JOINT_ORDER])
      qd=np.array([d.qvel[jmap[n][1]] for n in CANONICAL_JOINT_ORDER])
      return norm_state(v,w,g,q,qd,float(d.qpos[qa+2]))
    
     records=[]
     for suite,snap in enumerate(snaps):
      for e in range(8):
       x=obs_all[suite*8+e]
       for plab,pref in PREFS.items():
        actions={};dns={}
        for vname,b,sgn in VARIANTS:
         xp=x.copy()
         if b is not None:
          lo,hi,eps=BLOCKS[b];signs=np.where(np.arange(hi-lo)%2==0,1.0,-1.0).astype(np.float32)
          xp[lo:hi]+=sgn*eps*signs;dns[vname,sgn]=float(np.linalg.norm(xp-x))
         with torch.inference_mode():
          a=actor(torch.from_numpy(xp[None]).cuda(),torch.from_numpy(pref[None]).cuda()).cpu().numpy()[0]
         actions[vname,sgn]=a
        base_a=actions["base",0];base_s=transition(snap,e,base_a)
        for b in BLOCKS:
         for sgn in (-1,1):
          a=actions[b,sgn];ns=transition(snap,e,a)
          ds0=dns[b,sgn];da=float(np.linalg.norm(a-base_a));ds1=float(np.linalg.norm(ns-base_s))
          records.append({"engine":"mujoco_t4","suite":suite,"env":e,"pref":plab,"block":b,"sign":sgn,
                          "delta_obs":ds0,"delta_action":da,"delta_next_state_norm":ds1,
                          "G_pi":da/(ds0+1e-12),"G_cl":ds1/(ds0+1e-12)})
      print("SUITE",suite,"done",flush=True)
     summary={}
     for p in PREFS:
      summary[p]={}
      for b in BLOCKS:
       rr=[r for r in records if r["pref"]==p and r["block"]==b]
       summary[p][b]={"G_pi_mean":float(np.mean([r["G_pi"] for r in rr])),
                      "G_cl_mean":float(np.mean([r["G_cl"] for r in rr])),
                      "G_cl_p90":float(np.quantile([r["G_cl"] for r in rr],.9))}
     rep={"schema":"phase4_r1_mujoco_gain_v1","summary":summary,"records":records}
     (OUT/"r1_mujoco_gain.json").write_text(json.dumps(rep,indent=2)+"\n")
     print(json.dumps(summary,indent=2),flush=True)
    if True:main()

STAGES = {
    "phase4_r1_isaac_closed_loop_gain": run_phase4_r1_isaac_closed_loop_gain,
    "phase4_r1_mujoco_closed_loop_gain": run_phase4_r1_mujoco_closed_loop_gain,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
