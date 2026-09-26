"""Consolidated experiment stages. Use the first CLI argument to select a former script stage."""
from __future__ import annotations
import argparse

def run_phase5_e1_capture_reset_bank():
    """Run former phase5_e1_capture_reset_bank.py stage."""
    from pathlib import Path
    import json,sys,torch
    ROOT=Path(__file__).resolve().parents[4]
    OUT=ROOT/"runs/phase5_e1_validity_envelope"
    SEEDS=[840001+i for i in range(4)]
    for pi in range(6):
        SEEDS += [850001+pi*1000+i for i in range(4)]
    SEEDS=sorted(set(SEEDS))
    
    from isaaclab.app import AppLauncher
    app=AppLauncher({"headless":True,"enable_cameras":False}).app
    env=None
    try:
        import gymnasium as gym,isaaclab_tasks
        from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
        cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=8;cfg.seed=0
        cfg.events.add_base_mass=None
        cfg.observations.policy.enable_corruption=False
        cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
        env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
        u=env.unwrapped;robot=u.scene["robot"];bank={}
        for seed in SEEDS:
            env.reset(seed=seed)
            bank[str(seed)]={
              "root_link_pos_local":(robot.data.root_link_pos_w-u.scene.env_origins).detach().cpu().tolist(),
              "root_link_quat_w":robot.data.root_link_quat_w.detach().cpu().tolist(),
              "root_link_lin_vel_w":robot.data.root_link_lin_vel_w.detach().cpu().tolist(),
              "root_link_ang_vel_w":robot.data.root_link_ang_vel_w.detach().cpu().tolist(),
              "joint_pos":robot.data.joint_pos.detach().cpu().tolist(),
              "joint_vel":robot.data.joint_vel.detach().cpu().tolist(),
              "joint_names":list(robot.data.joint_names),
              "command":u.command_manager.get_command("base_velocity").detach().cpu().tolist(),
            }
            print("CAPTURE",seed,flush=True)
        OUT.mkdir(parents=True,exist_ok=True)
        rep={"schema":"phase5_e1_exact_8env_reset_bank_v1","num_envs":8,
             "mass_randomization":"disabled_exact_nominal","seeds":SEEDS,"snapshots":bank}
        (OUT/"exact_8env_reset_bank.json").write_text(json.dumps(rep,indent=2)+"\n")
    finally:
        if env is not None:env.close()
        app.close()

def run_phase5_e1_nominal_setter_probe():
    """Run former phase5_e1_nominal_setter_probe.py stage."""
    from pathlib import Path
    import json,sys
    import numpy as np,torch
    ROOT=Path(__file__).resolve().parents[4]
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"scripts/rl")]
    CKPT=ROOT/"runs/authority_isolated_simplex_edge_retain-2026-09-25/model_30.pt"
    from isaaclab.app import AppLauncher
    app=AppLauncher({"headless":True,"enable_cameras":False}).app
    env=None
    try:
     import gymnasium as gym,isaaclab_tasks
     from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
     from talon_rl.models.authority.isolated import AuthorityIsolatedWideCritic
     cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=8;cfg.events.add_base_mass=None;cfg.observations.policy.enable_corruption=False
     cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
     env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);u=env.unwrapped;robot=u.scene["robot"]
     def ot(x):
      if isinstance(x,dict):x=x.get("policy",next(iter(x.values())))
      return x if torch.is_tensor(x) else torch.as_tensor(x)
     o,_=env.reset(seed=0);m=AuthorityIsolatedWideCritic(ot(o).shape[-1],12).cuda()
     m.load_state_dict(torch.load(CKPT,map_location="cuda",weights_only=False)["model"]);m.eval()
     w=torch.tensor([.7,.1,.1,.1],device="cuda").repeat(8,1)
     def inject(kind):
      ids=torch.arange(8,dtype=torch.int32,device="cpu")
      bodies=list(robot.data.body_names);trunk=bodies.index("trunk");joints=list(robot.data.joint_names)
      hips=[i for i,n in enumerate(joints) if "_hip_joint" in n]
      flex=[i for i,n in enumerate(joints) if "_thigh_joint" in n or "_calf_joint" in n]
      if kind in ("mass","all"):
       masses=robot.root_physx_view.get_masses();inertias=robot.root_physx_view.get_inertias()
       masses[:,trunk]=robot.data.default_mass[:,trunk].cpu()
       inertias[:,trunk]=robot.data.default_inertia[:,trunk].cpu()
       robot.root_physx_view.set_masses(masses,ids);robot.root_physx_view.set_inertias(inertias,ids)
      if kind in ("passive","all"):
       damp=robot.root_physx_view.get_dof_dampings();arm=robot.root_physx_view.get_dof_armatures()
       damp[:]=0;arm[:]=0
       robot.root_physx_view.set_dof_dampings(damp,indices=ids);robot.root_physx_view.set_dof_armatures(arm,indices=ids)
      if kind in ("material","all"):
       mats=robot.root_physx_view.get_material_properties();mats[:,:,0]=.8;mats[:,:,1]=.6;mats[:,:,2]=0
       robot.root_physx_view.set_material_properties(mats,ids)
    
     def run(kind):
      cur,_=env.reset(seed=840001);cur=ot(cur).cuda()
      if kind=="matched":
       rp=(robot.data.root_link_pos_w-u.scene.env_origins).detach();rq=robot.data.root_link_quat_w.detach()
       rv=torch.cat([robot.data.root_link_lin_vel_w,robot.data.root_link_ang_vel_w],dim=1).detach()
       q=robot.data.joint_pos.detach();qd=robot.data.joint_vel.detach()
       cmd=u.command_manager.get_command("base_velocity").detach().clone()
       robot.write_root_pose_to_sim(torch.cat([rp+u.scene.env_origins,rq],dim=1))
       robot.write_root_velocity_to_sim(rv);robot.write_joint_state_to_sim(q,qd)
       term=u.command_manager.get_term("base_velocity");term.vel_command_b[:]=cmd
       term.is_standing_env[:]=False
       if hasattr(term,"is_heading_env"):term.is_heading_env[:]=False
       u.action_manager.reset();u.scene.write_data_to_sim();u.sim.forward()
       cur=ot(u.observation_manager.compute(update_history=True)).cuda()
      elif kind!="baseline":
       inject(kind);u.scene.write_data_to_sim();u.sim.forward()
       cur=ot(u.observation_manager.compute(update_history=True)).cuda()
      with torch.no_grad():a=m.act_inference_with_preference(cur,w)
      nxt,_,te,tr,_=env.step(a)
      d=robot.data
      return {"obs":cur.detach().cpu().numpy(),"action":a.cpu().numpy(),
        "v":d.root_lin_vel_b.detach().cpu().numpy(),"om":d.root_ang_vel_b.detach().cpu().numpy(),
        "q":d.joint_pos.detach().cpu().numpy(),"qd":d.joint_vel.detach().cpu().numpy(),
        "done":(te|tr).cpu().numpy()}
     base=run("baseline");out={}
     for kind in ("matched","mass","passive","material","all"):
      x=run(kind)
      out[kind]={}
      for k in ("obs","action","v","om","q","qd"):
       z=np.asarray(x[k])-np.asarray(base[k])
       out[k][k] if False else None
       out[kind][k+"_maxabs"]=float(np.max(np.abs(z)))
       out[kind][k+"_rmse"]=float(np.sqrt(np.mean(z*z)))
      out[kind]["done_diff"]=int(np.sum(x["done"]!=base["done"]))
     print(json.dumps(out,indent=2),flush=True)
    finally:
     if env is not None:env.close()
     app.close()

def run_phase5_e1_finalize_report():
    """Run former phase5_e1_finalize_report.py stage."""
    from pathlib import Path
    import json,hashlib
    import numpy as np
    
    ROOT=Path(__file__).resolve().parents[4]
    RAW=ROOT/"runs/phase5_e1_validity_envelope/e1_report.json"
    H2A=ROOT/"runs/authority_isolated_h2a_u30_semantic_validity-2026-09-25/semantic_report.json"
    OUT=ROOT/"runs/phase5_e1_validity_envelope/e1_final_report.json"
    
    def qstats(x):
        a=np.asarray(x,float)
        return {"mean":float(a.mean()),"median":float(np.median(a)),"p10":float(np.quantile(a,.1)),
                "min":float(a.min()),"max":float(a.max())}
    
    def main():
        raw=json.load(open(RAW));h=json.load(open(H2A))
        plants=raw["plants"]
        # Rename deterministic local nominal tuple; it is not the canonical source-domain distribution.
        p0=plants.pop("nominal")
        p0["plant"]["id"]="P0_source_nominal_tuple"
        p0["plant"]["group"]="deterministic_nominal"
        plants["P0_source_nominal_tuple"]=p0
        bank=[]
        for p in raw["plant_bank"]:
            q=dict(p)
            if q["id"]=="nominal":
                q["id"]="P0_source_nominal_tuple";q["group"]="deterministic_nominal"
            bank.append(q)
    
        source_anchor={
          "provenance":"frozen exact H2a canonical source-domain evaluation",
          "report":str(H2A.relative_to(ROOT)),
          "sha256":hashlib.sha256(H2A.read_bytes()).hexdigest(),
          "endpoint":h["endpoint"],
          "endpoint_pass":h["endpoint_pass"],
          "continuum_monotonicity":h["continuum_summary"]["monotonicity_fraction"],
          "continuum_endpoint_between":h["continuum_summary"]["endpoint_between_fraction"],
          "center_compromise":h["center_compromise"]["between_heavy_envelope_fraction"],
          "critic":h["critic"],
          "global_min_survival":h["min_survival_all"],
          "endpoint_min_survival":float(min(v["min_survival"] for v in h["endpoint"].values())),
        }
    
        groups={}
        for grp in ("deterministic_nominal","in_ensemble","heldout"):
            ids=[p["id"] for p in bank if p["group"]==grp]
            vals=[plants[i] for i in ids]
            endpoint_min=[min(v["min_survival"] for v in x["endpoint"].values()) for x in vals]
            groups[grp]={
              "n":len(ids),
              "ids":ids,
              "semantic_pass_fraction":{lab:float(np.mean([x["endpoint_pass"][lab] for x in vals])) for lab in ("T","A","O","S")},
              "endpoint_survival_gate_fraction":float(np.mean(np.asarray(endpoint_min)>=.95)),
              "endpoint_min_survival":qstats(endpoint_min),
              "global_protocol_survival_gate_fraction":float(np.mean([x["engineering"]["min_survival"]>=.95 for x in vals])),
              "global_min_survival":qstats([x["engineering"]["min_survival"] for x in vals]),
              "center_pass_fraction":float(np.mean([x["center_compromise"]>=.75 for x in vals])),
              "continuum_monotonic_pass_fraction":float(np.mean([x["continuum_monotonicity"]>=.65 for x in vals])),
              "continuum_between_pass_fraction":float(np.mean([x["continuum_endpoint_between"]>=.65 for x in vals])),
              "authority_pairwise_retention_fraction":float(np.mean([x["authority"]["pairwise_retention_vs_nominal"]>=.90 for x in vals])),
              "authority_tangent_retention_fraction":float(np.mean([x["authority"]["tangent_retention_vs_nominal"]>=.90 for x in vals])),
              "critic_valid_fraction":float(np.mean([x["critic"]["h32_ev_mean"]>0 and x["critic"]["negative_fraction"]<=.25 for x in vals])),
              "min_height":qstats([x["engineering"]["min_height"] for x in vals]),
              "max_tilt_deg":qstats([x["engineering"]["max_tilt_deg"] for x in vals]),
              "max_action_saturation_fraction":qstats([x["engineering"]["max_action_saturation_fraction"] for x in vals]),
            }
    
        # Per-plant compact validity map.
        validity=[]
        for p in bank:
            x=plants[p["id"]]
            validity.append({
              "plant_id":p["id"],"group":p["group"],
              "mass_delta_kg":p["mass_delta_kg"],"passive_blend":p["passive_blend"],"contact_blend":p["contact_blend"],
              "T":bool(x["endpoint_pass"]["T"]),"A":bool(x["endpoint_pass"]["A"]),
              "O":bool(x["endpoint_pass"]["O"]),"S":bool(x["endpoint_pass"]["S"]),
              "endpoint_survival":float(min(v["min_survival"] for v in x["endpoint"].values())),
              "global_min_survival":x["engineering"]["min_survival"],
              "center":bool(x["center_compromise"]>=.75),
              "continuum_monotonic":bool(x["continuum_monotonicity"]>=.65),
              "continuum_between":bool(x["continuum_endpoint_between"]>=.65),
              "authority_pairwise_retention":x["authority"]["pairwise_retention_vs_nominal"],
              "authority_tangent_retention":x["authority"]["tangent_retention_vs_nominal"],
              "critic_valid":bool(x["critic"]["h32_ev_mean"]>0 and x["critic"]["negative_fraction"]<=.25),
            })
    
        # Interpretation flags. E1 does not authorize E2 by itself.
        inens=groups["in_ensemble"];held=groups["heldout"]
        semantic_gap=bool(
          inens["endpoint_survival_gate_fraction"]>=.75 and
          (inens["semantic_pass_fraction"]["T"]<.75 or inens["semantic_pass_fraction"]["A"]<.75 or inens["semantic_pass_fraction"]["O"]<.75)
        )
        broad_locomotion_collapse=bool(inens["endpoint_survival_gate_fraction"]<.50)
        heldout_worse={
          lab:held["semantic_pass_fraction"][lab] < inens["semantic_pass_fraction"][lab] for lab in ("T","A","O","S")
        }
        final={
          "schema":"phase5_e1_validity_envelope_final_v2",
          "measurement_only":True,"training_updates":0,
          "raw_deterministic_bank_report":str(RAW.relative_to(ROOT)),
          "raw_report_sha256":hashlib.sha256(RAW.read_bytes()).hexdigest(),
          "manifest_sha256":raw["manifest_sha256"],
          "source_domain_anchor":source_anchor,
          "deterministic_plant_bank":bank,
          "plants":plants,
          "aggregates":groups,
          "validity_map":validity,
          "plant_sensitivity":raw["plant_sensitivity"],
          "interpretation":{
            "semantic_robustness_gap_with_endpoint_viability":semantic_gap,
            "broad_locomotion_collapse":broad_locomotion_collapse,
            "heldout_semantic_pass_fraction_lower_than_inensemble":heldout_worse,
            "e2_authorized":False,
            "reason":"E1 characterization must be frozen/reviewed before any E2 training authorization."
          },
          "invalid_interpretation_note":{
            "description":"The first E1 aggregate labeled deterministic (0,0,0) as nominal source. That label is invalid because canonical H2a source uses its original source-domain randomization. Plant measurements remain valid; only source-anchor interpretation was corrected.",
            "training_or_plant_changes":0
          }
        }
        OUT.write_text(json.dumps(final,indent=2)+"\n")
        print(json.dumps({"source_domain_anchor":source_anchor,"aggregates":groups,"interpretation":final["interpretation"]},indent=2))
    
    if True:main()

STAGES = {
    "phase5_e1_capture_reset_bank": run_phase5_e1_capture_reset_bank,
    "phase5_e1_nominal_setter_probe": run_phase5_e1_nominal_setter_probe,
    "phase5_e1_finalize_report": run_phase5_e1_finalize_report,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
