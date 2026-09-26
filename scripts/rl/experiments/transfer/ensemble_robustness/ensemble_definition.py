"""Consolidated experiment stages. Use the first CLI argument to select a former script stage."""
from __future__ import annotations
import argparse

def run_phase5_e0_build_manifest():
    """Run former phase5_e0_build_manifest.py stage."""
    from pathlib import Path
    import hashlib,json,subprocess
    
    ROOT=Path(__file__).resolve().parents[4]
    OUT=ROOT/"artifacts"
    OUT.mkdir(parents=True,exist_ok=True)
    AUD=ROOT/"runs/phase3_plant_equivalence_audit"
    
    def sha256(path):
        p=Path(path)
        return hashlib.sha256(p.read_bytes()).hexdigest()
    
    def rel_or_abs(path):
        p=Path(path)
        try:return str(p.relative_to(ROOT))
        except ValueError:return str(p)
    
    def file_rec(path):
        p=Path(path)
        return {"path":rel_or_abs(p),"sha256":sha256(p)}
    
    def main():
        isaac=json.load(open(AUD/"isaac_nominal_snapshot.json"))
        mj=json.load(open(AUD/"mujoco_snapshot.json"))
        trunk_i=isaac["body_names"].index("trunk")
        trunk_m=mj["body_names"].index("trunk")
        nominal_mass=float(isaac["mass"][trunk_i])
        target_mass=float(mj["mass"][trunk_m])
        files=[
          ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd",
          ROOT/"talon_rl/assets/data/Robots/unitree_a1/mujoco/a1.xml",
          ROOT/"talon_rl/assets/data/Robots/unitree_a1/mujoco/a1_t4_passive_corrected.xml",
          ROOT/"talon_rl/assets/data/Robots/unitree_a1/mujoco/scene.xml",
          ROOT/"talon_rl/assets/data/Robots/unitree_a1/mujoco/scene_t4_passive_corrected.xml",
          ROOT/"scripts/rl/experiments/architectures/authority/isolated/authority_diagnostics.py",
          Path("/home/xero/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/unitree.py"),
          Path("/home/xero/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/a1/rough_env_cfg.py"),
          Path("/home/xero/IsaacLab/source/isaaclab/isaaclab/actuators/actuator_pd.py"),
          Path("/home/xero/IsaacLab/source/isaaclab/isaaclab/envs/mdp/events.py"),
        ]
        commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
        heldout=[
          {"id":"H01_target_like","mass_delta_kg":round(target_mass-nominal_mass,6),"passive_blend":1.0,"contact_blend":1.0},
          {"id":"H02_low_mass_passive","mass_delta_kg":-1.5,"passive_blend":1.0,"contact_blend":0.0},
          {"id":"H03_low_mass_contact","mass_delta_kg":-1.5,"passive_blend":0.0,"contact_blend":1.0},
          {"id":"H04_nominal_mass_full","mass_delta_kg":0.0,"passive_blend":1.0,"contact_blend":1.0},
          {"id":"H05_high_mass_full","mass_delta_kg":3.0,"passive_blend":1.0,"contact_blend":1.0},
          {"id":"H06_high_mass_passive","mass_delta_kg":3.0,"passive_blend":1.0,"contact_blend":0.0},
          {"id":"H07_high_mass_contact","mass_delta_kg":3.0,"passive_blend":0.0,"contact_blend":1.0},
          {"id":"H08_mix_low","mass_delta_kg":-0.75,"passive_blend":0.75,"contact_blend":0.25},
          {"id":"H09_mix_high","mass_delta_kg":1.5,"passive_blend":0.25,"contact_blend":0.75},
          {"id":"H10_mid_all","mass_delta_kg":0.75,"passive_blend":0.5,"contact_blend":0.5},
          {"id":"H11_target_mass_mid","mass_delta_kg":round(target_mass-nominal_mass,6),"passive_blend":0.5,"contact_blend":0.5},
          {"id":"H12_passive_only","mass_delta_kg":0.0,"passive_blend":1.0,"contact_blend":0.0},
        ]
        manifest={
          "schema":"phase5_e0_plant_ensemble_manifest_v1",
          "status":"PREDECLARED_PENDING_E0_SANITY",
          "date":"2026-09-26",
          "git_commit":commit,
          "policy_observes_plant_parameters":False,
          "sampling_unit":"per environment per episode/reset",
          "sampler":{"type":"scrambled_sobol","seed":26092650,"dimensions":3},
          "source_nominal":{
            "trunk_mass_kg":nominal_mass,
            "robot_static_friction":0.8,
            "robot_dynamic_friction":0.6,
            "restitution":0.0,
            "passive_dof_damping":0.0,
            "passive_dof_armature":0.0,
            "passive_dof_friction":0.0,
            "physics_dt_s":0.005,
            "policy_dt_s":0.02,
          },
          "canonical_existing_support":{
            "trunk_mass_delta_kg":{"distribution":"uniform","range":[-1.0,3.0],"cadence":"startup"},
            "contact_material":{"static":0.8,"dynamic":0.6,"restitution":0.0,"cadence":"startup_fixed"},
            "base_com_randomization":"disabled_for_A1",
            "passive_joint_randomization":"none",
          },
          "target_reference":{
            "mujoco_trunk_mass_kg":target_mass,
            "mujoco_trunk_mass_delta_from_source_kg":target_mass-nominal_mass,
            "mujoco_passive_template":{
              "hip_abduction_damping":1.0,
              "thigh_calf_damping":2.0,
              "armature":0.01,
              "frictionloss":0.2,
            },
            "mujoco_foot_sliding_friction":0.8,
            "mujoco_floor_sliding_friction":1.0,
            "contact_onset_drop_s":{"isaac":0.245,"mujoco":0.248},
          },
          "training_support":{
            "mass_delta_kg":{
              "distribution":"uniform","range":[-1.5,3.0],
              "mapping":"trunk_mass = source_nominal + mass_delta; inertia recomputed by mass ratio",
              "evidence":"extends canonical [-1,+3] only enough to include measured MuJoCo delta -1.288 kg",
            },
            "passive_blend":{
              "distribution":"uniform","range":[0.0,1.0],
              "mapping":{
                "hip_abduction_damping":"1.0 * passive_blend",
                "thigh_calf_damping":"2.0 * passive_blend",
                "all_joint_armature":"0.01 * passive_blend",
              },
              "coupling":"single latent traces measured source endpoint (0) to frozen MuJoCo baseline endpoint (1)",
              "evidence":"T4 free-space pulse: removing target passive terms reduced qdot trajectory RMSE 3.5155 -> 0.0133 rad/s",
            },
            "contact_blend":{
              "distribution":"uniform","range":[0.0,1.0],
              "mapping":{
                "robot_static_friction":0.8,
                "robot_dynamic_friction":"0.6 + 0.2 * contact_blend",
                "restitution":0.0,
              },
              "coupling":"single latent; static friction and restitution remain fixed",
              "evidence":"source dynamic friction 0.6 versus MuJoCo foot sliding friction 0.8; contact onset already nearly matched",
            },
          },
          "independence_rule":"mass_delta_kg, passive_blend, contact_blend sampled independently; mappings inside each block are coupled",
          "held_out_plants":heldout,
          "heldout_exclusion_rule":{
            "rule":"E2 must never inject the exact held-out tuples; sampler logs every realized tuple and rejects any tuple within 1e-9 L_inf of a held-out tuple",
            "scope":"E3 exact fixed realizations only; they are unseen combinations within the frozen bounded family",
          },
          "frozen_not_randomized":{
            "trunk_com":"source and MuJoCo trunk COM match to <1e-6 m in audit",
            "independent_inertia_scale":"link inertias largely match; trunk inertia changes only through mass recomputation",
            "actuator_law":"freeze source explicit DCMotor Kp=25 Kd=0.5 effort=33.5 velocity_limit=21 after T3",
            "joint_frictionloss":"MuJoCo frictionloss semantics are not directly commensurate with PhysX joint friction coefficient",
            "contact_restitution":"both source and target reference are zero",
            "contact_compliance_solver":"structurally different but no isolated equivalent per-env scalar was identified",
            "physics_timestep_solver":"global simulator settings cannot be sampled per environment while retaining the frozen 50 Hz contract",
            "foot_calf_topology":"structural representation mismatch; not a scalar randomization",
          },
          "e0_sanity":{
            "sample_count":128,
            "sampler":"same scrambled Sobol support as manifest",
            "fixed_stance_duration_s":1.28,
            "fixed_action":"zero normalized action / canonical joint target",
            "viable_definition":{"finite":True,"min_trunk_height_m":0.18,"max_tilt_deg":60.0,"no_trunk_floor_contact":True},
            "gates":{
              "finite_fraction_min":1.0,
              "reset_feasible_fraction_min":1.0,
              "fixed_stance_viable_fraction_min":0.90,
              "support_coverage_each_latent_min_normalized_max":0.95,
              "support_coverage_each_latent_max_normalized_min":0.05,
            },
          },
          "provenance_files":[file_rec(p) for p in files],
          "evidence_files":[
            "runs/phase3_plant_equivalence_audit/isaac_nominal_snapshot.json",
            "runs/phase3_plant_equivalence_audit/mujoco_snapshot.json",
            "runs/phase3_plant_equivalence_audit/isaac_physx_dof_props.json",
            "runs/phase3_plant_equivalence_audit/isaac_joint_pulse.json",
            "runs/phase3_plant_equivalence_audit/mujoco_joint_pulse.json",
            "runs/phase3_plant_equivalence_audit/isaac_drop.json",
            "runs/phase3_plant_equivalence_audit/mujoco_drop.json",
            "runs/phase3_t4c_nominal_dynamics/nominal_dynamics_report.json",
            "runs/phase4_controller_robustness/r2_divergence_report.json",
          ],
        }
        path=OUT/"phase5_e0_plant_ensemble_manifest.json"
        path.write_text(json.dumps(manifest,indent=2)+"\n")
        print(path)
        print(json.dumps({"status":manifest["status"],"train_support":manifest["training_support"],"heldout_count":len(heldout)},indent=2))
    if True:main()

def run_phase5_e0_ensemble_sanity():
    """Run former phase5_e0_ensemble_sanity.py stage."""
    from pathlib import Path
    import json,math,sys
    import numpy as np,torch
    
    ROOT=Path(__file__).resolve().parents[4]
    MAN=ROOT/"artifacts/phase5_e0_plant_ensemble_manifest.json"
    OUT=ROOT/"runs/phase5_e0_ensemble_sanity"
    OUT.mkdir(parents=True,exist_ok=True)
    N=128
    
    def main():
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]]
        app=AppLauncher({"headless":True,"enable_cameras":False}).app
        sys.argv=saved
        env=None
        try:
            import gymnasium as gym,isaaclab_tasks
            from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
            man=json.load(open(MAN))
            sob=torch.quasirandom.SobolEngine(3,scramble=True,seed=man["sampler"]["seed"]).draw(N).numpy()
            mass=-1.5+4.5*sob[:,0]
            passive=sob[:,1]
            contact=sob[:,2]
            cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=N;cfg.seed=26092650
            cfg.events.add_base_mass=None
            cfg.observations.policy.enable_corruption=False
            cfg.scene.robot.spawn.usd_path=str(ROOT/"talon_rl/assets/data/Robots/unitree_a1/a1.usd")
            env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
            u=env.unwrapped;robot=u.scene["robot"];dev=u.device
            env.reset(seed=26092650)
            env_ids=torch.arange(N,dtype=torch.int32,device="cpu")
            names=list(robot.data.joint_names);bodies=list(robot.data.body_names)
            trunk=bodies.index("trunk")
            hip_ids=[i for i,n in enumerate(names) if "_hip_joint" in n]
            flex_ids=[i for i,n in enumerate(names) if "_thigh_joint" in n or "_calf_joint" in n]
    
            # Per-environment trunk mass and inertia.
            masses=robot.root_physx_view.get_masses()
            nominal=float(robot.data.default_mass[0,trunk].cpu())
            new_mass=torch.tensor(nominal+mass,dtype=masses.dtype)
            masses[:,trunk]=new_mass
            robot.root_physx_view.set_masses(masses,env_ids)
            inertias=robot.root_physx_view.get_inertias()
            base_inertia=robot.data.default_inertia[:,trunk].cpu()
            ratio=(new_mass/nominal).unsqueeze(1)
            inertias[:,trunk]=base_inertia*ratio
            robot.root_physx_view.set_inertias(inertias,env_ids)
    
            # Passive PhysX DOF interpolation: source endpoint -> measured MuJoCo endpoint.
            damp=robot.root_physx_view.get_dof_dampings()
            arm=robot.root_physx_view.get_dof_armatures()
            damp[:]=0.0;arm[:]=0.0
            pb=torch.tensor(passive,dtype=damp.dtype)
            damp[:,hip_ids]=pb[:,None]*1.0
            damp[:,flex_ids]=pb[:,None]*2.0
            arm[:,:]=pb[:,None]*0.01
            robot.root_physx_view.set_dof_dampings(damp,indices=env_ids)
            robot.root_physx_view.set_dof_armatures(arm,indices=env_ids)
    
            # Contact block: keep static/restitution frozen; interpolate dynamic friction 0.6 -> 0.8.
            mats=robot.root_physx_view.get_material_properties()
            cb=torch.tensor(contact,dtype=mats.dtype)
            mats[:,:,0]=0.8
            mats[:,:,1]=(0.6+0.2*cb)[:,None]
            mats[:,:,2]=0.0
            robot.root_physx_view.set_material_properties(mats,env_ids)
    
            # Canonical fixed stance, zero action.
            rp=u.scene.env_origins.clone();rp[:,2]+=0.42
            rq=torch.zeros((N,4),device=dev);rq[:,0]=1.0
            robot.write_root_pose_to_sim(torch.cat([rp,rq],dim=1))
            robot.write_root_velocity_to_sim(torch.zeros((N,6),device=dev))
            q0=robot.data.default_joint_pos.clone()
            robot.write_joint_state_to_sim(q0,torch.zeros_like(q0))
            u.action_manager.reset()
            u.scene.write_data_to_sim();u.sim.forward()
    
            finite=np.ones(N,dtype=bool)
            min_h=np.full(N,np.inf);max_tilt=np.zeros(N);trunk_contact=np.zeros(N,dtype=bool)
            sensor=u.scene["contact_forces"]
            trunk_sensor_id=list(sensor.body_names).index("trunk")
            dt=float(u.cfg.sim.dt)
            action=torch.zeros((N,u.action_manager.total_action_dim),device=dev)
            for _ in range(64):
                u.action_manager.process_action(action)
                for _ in range(int(u.cfg.decimation)):
                    u.action_manager.apply_action()
                    u.scene.write_data_to_sim()
                    u.sim.step(render=False);u.scene.update(dt)
                h=robot.data.root_link_pos_w[:,2].detach().cpu().numpy()
                g=robot.data.projected_gravity_b.detach().cpu().numpy()
                tilt=np.degrees(np.arccos(np.clip(-g[:,2],-1,1)))
                f=sensor.data.net_forces_w[:,trunk_sensor_id].norm(dim=-1).detach().cpu().numpy()
                min_h=np.minimum(min_h,h);max_tilt=np.maximum(max_tilt,tilt);trunk_contact|=(f>1.0)
                finite &= np.isfinite(h)&np.isfinite(tilt)&np.isfinite(robot.data.joint_pos.detach().cpu().numpy()).all(axis=1)
            limits=robot.data.soft_joint_pos_limits.detach().cpu().numpy()
            qdef=robot.data.default_joint_pos.detach().cpu().numpy()
            reset_ok=np.isfinite(qdef).all(axis=1)&(qdef>=limits[:,:,0]).all(axis=1)&(qdef<=limits[:,:,1]).all(axis=1)
            viable=finite&(min_h>=0.18)&(max_tilt<=60.0)&(~trunk_contact)
            cov={
              "mass_norm_min":float(sob[:,0].min()),"mass_norm_max":float(sob[:,0].max()),
              "passive_norm_min":float(sob[:,1].min()),"passive_norm_max":float(sob[:,1].max()),
              "contact_norm_min":float(sob[:,2].min()),"contact_norm_max":float(sob[:,2].max()),
            }
            gates=man["e0_sanity"]["gates"]
            coverage_ok=all([
              cov["mass_norm_min"]<=gates["support_coverage_each_latent_max_normalized_min"],
              cov["passive_norm_min"]<=gates["support_coverage_each_latent_max_normalized_min"],
              cov["contact_norm_min"]<=gates["support_coverage_each_latent_max_normalized_min"],
              cov["mass_norm_max"]>=gates["support_coverage_each_latent_min_normalized_max"],
              cov["passive_norm_max"]>=gates["support_coverage_each_latent_min_normalized_max"],
              cov["contact_norm_max"]>=gates["support_coverage_each_latent_min_normalized_max"],
            ])
            summary={
              "finite_fraction":float(finite.mean()),
              "reset_feasible_fraction":float(reset_ok.mean()),
              "fixed_stance_viable_fraction":float(viable.mean()),
              "coverage":cov,
              "coverage_ok":bool(coverage_ok),
              "min_height_quantiles":{str(q):float(np.quantile(min_h,q)) for q in (0,0.1,0.5,0.9,1)},
              "max_tilt_quantiles":{str(q):float(np.quantile(max_tilt,q)) for q in (0,0.1,0.5,0.9,1)},
              "trunk_contact_fraction":float(trunk_contact.mean()),
              "mass_delta_range_realized":[float(mass.min()),float(mass.max())],
              "passive_blend_range_realized":[float(passive.min()),float(passive.max())],
              "contact_blend_range_realized":[float(contact.min()),float(contact.max())],
            }
            passed=(summary["finite_fraction"]>=gates["finite_fraction_min"]
                    and summary["reset_feasible_fraction"]>=gates["reset_feasible_fraction_min"]
                    and summary["fixed_stance_viable_fraction"]>=gates["fixed_stance_viable_fraction_min"]
                    and coverage_ok)
            rows=[{"i":i,"mass_delta_kg":float(mass[i]),"passive_blend":float(passive[i]),
                   "contact_blend":float(contact[i]),"finite":bool(finite[i]),"reset_ok":bool(reset_ok[i]),
                   "min_height":float(min_h[i]),"max_tilt_deg":float(max_tilt[i]),
                   "trunk_contact":bool(trunk_contact[i]),"viable":bool(viable[i])} for i in range(N)]
            rep={"schema":"phase5_e0_ensemble_sanity_v1","passed":bool(passed),"summary":summary,"samples":rows}
            (OUT/"sanity_report.json").write_text(json.dumps(rep,indent=2)+"\n")
            print(json.dumps({"passed":passed,"summary":summary},indent=2),flush=True)
        finally:
            if env is not None:env.close()
            app.close()
    if True:main()

def run_phase5_e0_freeze_manifest():
    """Run former phase5_e0_freeze_manifest.py stage."""
    from pathlib import Path
    import hashlib,json
    import numpy as np
    
    ROOT=Path(__file__).resolve().parents[4]
    MAN=ROOT/"artifacts/phase5_e0_plant_ensemble_manifest.json"
    SAN=ROOT/"runs/phase5_e0_ensemble_sanity/sanity_report.json"
    
    def main():
        man=json.load(open(MAN))
        san=json.load(open(SAN))
        rows=san["samples"]
        X=np.array([[r["mass_delta_kg"],r["passive_blend"],r["contact_blend"]] for r in rows],float)
        names=["mass_delta_kg","passive_blend","contact_blend"]
        summary={
          "sample_count":len(rows),
          "means":dict(zip(names,X.mean(0).tolist())),
          "stds":dict(zip(names,X.std(0).tolist())),
          "correlation_matrix":np.corrcoef(X,rowvar=False).tolist(),
          "sanity_passed":bool(san["passed"]),
          "finite_fraction":san["summary"]["finite_fraction"],
          "reset_feasible_fraction":san["summary"]["reset_feasible_fraction"],
          "fixed_stance_viable_fraction":san["summary"]["fixed_stance_viable_fraction"],
          "coverage":san["summary"]["coverage"],
          "min_height_quantiles":san["summary"]["min_height_quantiles"],
          "max_tilt_quantiles":san["summary"]["max_tilt_quantiles"],
          "sanity_report_sha256":hashlib.sha256(SAN.read_bytes()).hexdigest(),
        }
        if not summary["sanity_passed"]:
            raise SystemExit("refusing to freeze: sanity did not pass")
        man["status"]="FROZEN_PASS_P5_E0"
        man["e0_sanity_result"]=summary
        MAN.write_text(json.dumps(man,indent=2)+"\n")
        sha=hashlib.sha256(MAN.read_bytes()).hexdigest()
        (MAN.parent/"phase5_e0_plant_ensemble_manifest.sha256").write_text(
          sha+"  phase5_e0_plant_ensemble_manifest.json\n"
        )
        print(json.dumps(summary,indent=2))
        print("manifest_sha256",sha)
    
    if True:
        main()

STAGES = {
    "phase5_e0_build_manifest": run_phase5_e0_build_manifest,
    "phase5_e0_ensemble_sanity": run_phase5_e0_ensemble_sanity,
    "phase5_e0_freeze_manifest": run_phase5_e0_freeze_manifest,
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=sorted(STAGES))
    a=p.parse_args(); STAGES[a.stage]()

if __name__ == '__main__': main()
