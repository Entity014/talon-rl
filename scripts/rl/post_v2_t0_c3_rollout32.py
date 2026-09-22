#!/usr/bin/env python3
"""V2-C3 read-only short-horizon closed-loop semantic divergence audit."""
from __future__ import annotations
import argparse, hashlib, json, sys, time, traceback
from pathlib import Path
import numpy as np, torch

ROOT=Path(__file__).resolve().parents[2]; sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
PREFS={"P":[.8,.1,.1],"B":[.1,.8,.1],"E":[.1,.1,.8]}
HORIZONS=(8,16,32)
CODE_COMMIT="f77af91932b1651b531cb3f64cb3c1852ab800cb"

def obs_tensor(x):
    if isinstance(x,dict): x=x.get("policy",next(iter(x.values())))
    return x if torch.is_tensor(x) else torch.as_tensor(x)

def tilt_deg(q):
    return torch.rad2deg(torch.acos((1-2*(q[:,1]**2+q[:,2]**2)).clamp(-1,1)))

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def vecnorm(x): return torch.linalg.vector_norm(x,dim=-1)

def first_divergence(values, threshold):
    for i,v in enumerate(values):
        if v > threshold: return i
    return None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--num-envs",type=int,default=8)
    ap.add_argument("--suites",type=int,default=4)
    args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    life=args.output.with_name(args.output.stem+".lifecycle.jsonl")
    def mark(event,**extra):
        with life.open("a") as f:f.write(json.dumps({"event":event,"unix":time.time(),**extra},sort_keys=True)+"\n")
    cksha=sha(args.checkpoint)
    mark("RUN_STARTED",protocol="V2-T0-C3-ROLLOUT32",measurement_only=True,checkpoint_sha256=cksha,horizons=list(HORIZONS))
    app=env=None
    try:
        anchor_path=ROOT/"runs/post_v2_a-2026-09-23/v2a.json"
        anchor=json.loads(anchor_path.read_text());anchors=torch.tensor([anchor["anchors"][k] for k in ("P","B","E")],dtype=torch.float32)
        from isaaclab.app import AppLauncher
        saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
        import gymnasium as gym, isaaclab_tasks
        from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
        from talon_rl.v1b_s7_reward_vector import group_v1b_s7_terms
        from talon_rl.v1c_actor_critic import V1CSharedActorCritic
        from talon_rl.v2_behavior_actor_critic import V2BehaviorActorCritic

        cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.seed=0
        env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg)
        obs,_=env.reset(seed=87001);obs=obs_tensor(obs).cuda();ad=env.unwrapped.action_manager.total_action_dim
        manager=env.unwrapped.reward_manager;robot=env.unwrapped.scene["robot"]

        payload=torch.load(args.checkpoint,map_location="cuda",weights_only=False)
        v2=V2BehaviorActorCritic(obs.shape[-1],ad,anchors=anchors,film_alpha=0.5).cuda();v2.load_state_dict(payload["model"]);v2.eval()
        specs={}
        spec_paths={}
        for lab in PREFS:
            pp=ROOT/f"runs/post_v1_d1-2026-09-22/specialist_{lab}_terminal.pt";pl=torch.load(pp,map_location="cuda",weights_only=False)
            m=V1CSharedActorCritic(obs.shape[-1],ad).cuda();m.load_state_dict(pl["model"]);m.eval();specs[lab]=m;spec_paths[lab]=pp
        ws={k:torch.tensor(v,dtype=torch.float32,device="cuda").repeat(args.num_envs,1) for k,v in PREFS.items()}

        families={
          "D1_P":("spec","P"),"D1_B":("spec","B"),"D1_E":("spec","E"),
          "V2_P":("v2","P"),"V2_B":("v2","B"),"V2_E":("v2","E"),
        }
        maxH=max(HORIZONS); trajectories=[]
        for suite in range(args.suites):
            seed=87001+suite
            for fam,(kind,label) in families.items():
                cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda()
                prev=torch.zeros((args.num_envs,ad),device="cuda");cum_obj=np.zeros(3,dtype=np.float64);steps=[]
                with torch.no_grad():
                    for t in range(maxH):
                        model=specs[label] if kind=="spec" else v2
                        raw_action=model.act_inference_with_preference(cur,ws[label])
                        sat_frac=float((raw_action.abs()>1.0).float().mean())
                        action=torch.clamp(raw_action,-1,1)
                        nxt,_,term,trunc,_=env.step(action)
                        raw=manager._step_reward.detach().cpu().numpy();names=list(manager.active_terms)
                        vec=group_v1b_s7_terms({n:raw[:,i] for i,n in enumerate(names)},shape=(args.num_envs,))
                        obj_mean=vec.mean(0);cum_obj+=obj_mean
                        data=robot.data
                        # contact proxy: binary net contact force norm at feet if available
                        contact=None
                        try:
                            sensor=env.unwrapped.scene.sensors.get("contact_forces",None)
                            if sensor is not None:
                                forces=sensor.data.net_forces_w
                                contact=(torch.linalg.vector_norm(forces,dim=-1)>1.0).float()
                        except Exception:
                            contact=None
                        step={
                          "t":t,
                          "action_mean":action.mean(0).cpu().tolist(),
                          "action_l2_mean":float(vecnorm(action).mean()),
                          "action_rate_mean":float(vecnorm(action-prev).mean()),
                          "saturation_fraction":sat_frac,
                          "joint_pos_mean":data.joint_pos.mean(0).cpu().tolist(),
                          "joint_vel_mean":data.joint_vel.mean(0).cpu().tolist(),
                          "base_lin_vel_mean":data.root_lin_vel_b.mean(0).cpu().tolist(),
                          "base_ang_vel_mean":data.root_ang_vel_b.mean(0).cpu().tolist(),
                          "tilt_mean":float(tilt_deg(data.root_quat_w).mean()),
                          "torque_mean":data.applied_torque.mean(0).cpu().tolist(),
                          "torque_norm_mean":float(vecnorm(data.applied_torque).mean()),
                          "objective_step_mean":obj_mean.tolist(),
                          "objective_cumulative":cum_obj.tolist(),
                          "termination_fraction":float((term|trunc).float().mean()),
                        }
                        if contact is not None:
                            step["contact_pattern_mean"]=contact.mean(0).cpu().tolist()
                            step["contact_fraction_mean"]=float(contact.mean())
                        steps.append(step);prev=action;cur=obs_tensor(nxt).cuda()
                trajectories.append({"suite":suite,"seed":seed,"family":fam,"kind":kind,"preference":label,"steps":steps})

        # pairwise divergence over time
        pairs=[
          ("D1_P","D1_B","D1_P_vs_B"),("D1_P","D1_E","D1_P_vs_E"),
          ("V2_P","V2_B","V2_P_vs_B"),("V2_P","V2_E","V2_P_vs_E"),
        ]
        div_rows=[]
        def arr(step,key):
            return np.asarray(step[key],dtype=float)
        for suite in range(args.suites):
            for a,b,name in pairs:
                ta=next(x for x in trajectories if x["suite"]==suite and x["family"]==a)["steps"]
                tb=next(x for x in trajectories if x["suite"]==suite and x["family"]==b)["steps"]
                for H in HORIZONS:
                    ts=[]
                    for t in range(H):
                        xa,xb=ta[t],tb[t]
                        row={
                          "t":t,
                          "action":float(np.linalg.norm(arr(xa,"action_mean")-arr(xb,"action_mean"))),
                          "joint_pos":float(np.linalg.norm(arr(xa,"joint_pos_mean")-arr(xb,"joint_pos_mean"))),
                          "joint_vel":float(np.linalg.norm(arr(xa,"joint_vel_mean")-arr(xb,"joint_vel_mean"))),
                          "base_lin_vel":float(np.linalg.norm(arr(xa,"base_lin_vel_mean")-arr(xb,"base_lin_vel_mean"))),
                          "base_ang_vel":float(np.linalg.norm(arr(xa,"base_ang_vel_mean")-arr(xb,"base_ang_vel_mean"))),
                          "tilt":abs(float(xa["tilt_mean"]-xb["tilt_mean"])),
                          "torque":float(np.linalg.norm(arr(xa,"torque_mean")-arr(xb,"torque_mean"))),
                          "objective_cumulative":float(np.linalg.norm(arr(xa,"objective_cumulative")-arr(xb,"objective_cumulative"))),
                          "saturation_gap":abs(float(xa["saturation_fraction"]-xb["saturation_fraction"])),
                        }
                        if "contact_pattern_mean" in xa and "contact_pattern_mean" in xb:
                            row["contact"]=float(np.linalg.norm(arr(xa,"contact_pattern_mean")-arr(xb,"contact_pattern_mean")))
                        ts.append(row)
                    div_rows.append({"suite":suite,"pair":name,"horizon":H,"timeseries":ts})

        # thresholds derived from fixed numerical floors + relative scale for interpretability
        thresholds={"action":1e-3,"joint_pos":1e-3,"joint_vel":1e-3,"base_lin_vel":1e-3,"base_ang_vel":1e-3,"tilt":0.05,"torque":1e-2,"objective_cumulative":1e-3,"contact":1e-3,"saturation_gap":1e-3}
        summaries={}
        for _,_,name in pairs:
            summaries[name]={}
            for H in HORIZONS:
                rs=[r for r in div_rows if r["pair"]==name and r["horizon"]==H]
                metrics=list(rs[0]["timeseries"][0].keys());metrics.remove("t")
                summaries[name][str(H)]={}
                for m in metrics:
                    vals=np.asarray([[x["timeseries"][t][m] for t in range(H)] for x in rs],dtype=float)
                    mean_t=vals.mean(0)
                    firsts=[first_divergence([x["timeseries"][t][m] for t in range(H)],thresholds.get(m,1e-3)) for x in rs]
                    finite_first=[x for x in firsts if x is not None]
                    summaries[name][str(H)][m]={
                      "mean_timeseries":mean_t.tolist(),
                      "terminal_mean":float(mean_t[-1]),
                      "first_divergence_by_suite":firsts,
                      "first_divergence_mean":float(np.mean(finite_first)) if finite_first else None,
                      "diverged_fraction":float(len(finite_first)/len(firsts)),
                      "threshold":thresholds.get(m,1e-3),
                    }
        report={
          "schema":"post_v2_t0_c3_rollout32_v1","status":"MEASUREMENT_COMPLETE","measurement_only":True,
          "provenance":{"code_commit":CODE_COMMIT,"checkpoint":str(args.checkpoint),"checkpoint_sha256":cksha,
                        "v2a_anchor_sha256":sha(anchor_path),
                        "specialist_checkpoints":{k:{"path":str(spec_paths[k].relative_to(ROOT)),"sha256":sha(spec_paths[k])} for k in spec_paths}},
          "horizons":list(HORIZONS),"thresholds":thresholds,"families":families,
          "trajectories":trajectories,"divergence":div_rows,"summaries":summaries,
          "note":"Matched initial states, deterministic policies, closed-loop rollout. No parameters updated."
        }
        args.output.write_text(json.dumps(report,indent=2)+"\n");mark("ARTIFACT_WRITTEN",path=str(args.output));mark("RUN_DONE",status=report["status"])
        print(json.dumps({"status":report["status"],"trajectory_rows":len(trajectories),"divergence_rows":len(div_rows)},indent=2))
    except BaseException as exc:
        args.output.with_name(args.output.stem+".ERROR.json").write_text(json.dumps({"status":"ERROR","error":str(exc),"traceback":traceback.format_exc()},indent=2)+"\n");mark("ERROR",error=str(exc));raise
    finally:
        if env is not None:env.close()
        if app is not None:app.close()
if __name__=="__main__":main()
