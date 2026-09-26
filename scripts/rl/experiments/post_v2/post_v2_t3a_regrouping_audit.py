#!/usr/bin/env python3
"""T3-A read-only candidate reward regrouping/separability audit on matched D1 specialists."""
from __future__ import annotations
import argparse,atexit,hashlib,json,sys,time,traceback
from pathlib import Path
import numpy as np, torch

ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
LABS=("P","B","E")
GROUPS={
 "progress":["track_lin_vel_xy_exp","track_ang_vel_z_exp"],
 "stability":["lin_vel_z_l2","ang_vel_xy_l2","flat_orientation_l2"],
 "effort":["dof_torques_l2"],
 "smoothness":["dof_acc_l2","action_rate_l2"],
 "gait_contact_aux":["feet_air_time"],
 "unresolved_zero":["dof_pos_limits"],
}
CORE_GROUPS=("progress","stability","effort","smoothness","gait_contact_aux")
PHYS=("vx_error","wz_error","abs_lin_vel_z","abs_ang_vel_xy","tilt_deg","torque_l2","action_rate_l2")

def obs_tensor(v):
    if isinstance(v,dict):v=v.get("policy",next(iter(v.values())))
    return v if torch.is_tensor(v) else torch.as_tensor(v)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def safe_corr(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if x.std()==0 or y.std()==0:return 0.0
    return float(np.corrcoef(x,y)[0,1])
def corrmat(x):
    x=np.asarray(x,float);n=x.shape[1];o=np.zeros((n,n))
    for i in range(n):
      for j in range(n):o[i,j]=safe_corr(x[:,i],x[:,j]) if i!=j else 1.0
    return o

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--num-envs",type=int,default=32)
    ap.add_argument("--steps",type=int,default=250)
    ap.add_argument("--suites",type=int,default=4)
    ap.add_argument("--seed-base",type=int,default=188001)
    args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    life=args.output.with_name(args.output.stem+".lifecycle.jsonl");err=args.output.with_name(args.output.stem+".ERROR.json")
    state={"written":False,"failed":False}
    def mark(event,**kw):
        with life.open("a") as f:f.write(json.dumps({"event":event,"unix":time.time(),**kw},sort_keys=True)+"\n")
    def fail(reason,exc=None):
        state["failed"]=True;err.write_text(json.dumps({"status":"ERROR","reason":reason,"error":str(exc) if exc else None,"traceback":traceback.format_exc() if exc else None},indent=2)+"\n");mark("ERROR",reason=reason)
    def guard():
        if not state["written"] and not state["failed"]:fail("PROCESS_EXIT_BEFORE_ARTIFACT")
    atexit.register(guard);mark("RUN_STARTED",protocol="V2-T3-A",groups=GROUPS)
    from isaaclab.app import AppLauncher
    saved=sys.argv[:];sys.argv=[sys.argv[0]];app=AppLauncher({"headless":True,"enable_cameras":False}).app;sys.argv=saved
    env=None
    try:
        import gymnasium as gym,isaaclab_tasks
        from isaaclab_tasks.manager_based.locomotion.velocity.config.a1.flat_env_cfg import UnitreeA1FlatEnvCfg
        from talon_rl.v1c_actor_critic import V1CSharedActorCritic
        cfg=UnitreeA1FlatEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.seed=args.seed_base
        env=gym.make("Isaac-Velocity-Flat-Unitree-A1-v0",cfg=cfg);mark("ENV_CREATED")
        obs,_=env.reset(seed=args.seed_base);obs=obs_tensor(obs).cuda();ad=env.unwrapped.action_manager.total_action_dim
        policies={};ckpts={}
        for lab in LABS:
            p=ROOT/f"runs/post_v1_d1-2026-09-22/specialist_{lab}_terminal.pt";pl=torch.load(p,map_location="cuda",weights_only=False)
            m=V1CSharedActorCritic(obs.shape[-1],ad).cuda();m.load_state_dict(pl["model"]);m.eval();policies[lab]=m;ckpts[lab]=p
        wp={lab:torch.tensor({"P":[.8,.1,.1],"B":[.1,.8,.1],"E":[.1,.1,.8]}[lab],device="cuda").repeat(args.num_envs,1) for lab in LABS}
        suite_rows=[]; term_names=None
        for suite in range(args.suites):
            seed=args.seed_base+suite
            for lab in LABS:
                cur,_=env.reset(seed=seed);cur=obs_tensor(cur).cuda()
                terms=[];phys=[];returned=[];recon=[];done_any=np.zeros(args.num_envs,bool);finite=True
                with torch.no_grad():
                    for step in range(args.steps):
                        action=torch.clamp(policies[lab].act_inference_with_preference(cur,wp[lab]),-1,1)
                        prev=env.unwrapped.action_manager.prev_action.clone()
                        ar=(action-prev).square().mean(-1)
                        nxt,reward,term,trunc,_=env.step(action)
                        mgr=env.unwrapped.reward_manager;names=list(mgr.active_terms)
                        if term_names is None:term_names=names
                        elif names!=term_names:raise RuntimeError("reward term order changed")
                        raw=mgr._step_reward.detach().cpu().numpy().astype(np.float64)
                        terms.append(raw);returned.append(reward.detach().cpu().numpy().astype(np.float64));recon.append(raw.sum(1)*float(env.unwrapped.step_dt))
                        data=env.unwrapped.scene["robot"].data;cmd=env.unwrapped.command_manager.get_command("base_velocity");q=data.root_quat_w
                        roll=torch.atan2(2*(q[:,0]*q[:,1]+q[:,2]*q[:,3]),1-2*(q[:,1]**2+q[:,2]**2))
                        pitch=torch.asin(torch.clamp(2*(q[:,0]*q[:,2]-q[:,3]*q[:,1]),-1,1))
                        phys.append(np.stack([
                          (data.root_lin_vel_b[:,0]-cmd[:,0]).abs().cpu().numpy(),
                          (data.root_ang_vel_b[:,2]-cmd[:,2]).abs().cpu().numpy(),
                          data.root_lin_vel_b[:,2].abs().cpu().numpy(),
                          torch.linalg.vector_norm(data.root_ang_vel_b[:,:2],dim=-1).cpu().numpy(),
                          torch.rad2deg(torch.maximum(roll.abs(),pitch.abs())).cpu().numpy(),
                          torch.linalg.vector_norm(data.applied_torque,dim=-1).cpu().numpy(),
                          ar.cpu().numpy(),
                        ],1))
                        done_any|=(term|trunc).cpu().numpy();finite &= bool(torch.isfinite(action).all() and torch.isfinite(reward).all());cur=obs_tensor(nxt).cuda()
                T=np.concatenate(terms);P=np.concatenate(phys);ti={n:i for i,n in enumerate(term_names)}
                G=np.stack([T[:,[ti[t] for t in GROUPS[g]]].sum(1) for g in CORE_GROUPS],1)
                suite_rows.append({
                  "suite":suite,"policy":lab,
                  "term_mean":{n:float(T[:,i].mean()) for i,n in enumerate(term_names)},
                  "term_std":{n:float(T[:,i].std()) for i,n in enumerate(term_names)},
                  "term_nonzero":{n:float(np.mean(np.abs(T[:,i])>0)) for i,n in enumerate(term_names)},
                  "group_mean":{g:float(G[:,i].mean()) for i,g in enumerate(CORE_GROUPS)},
                  "group_std":{g:float(G[:,i].std()) for i,g in enumerate(CORE_GROUPS)},
                  "group_nonzero":{g:float(np.mean(np.abs(G[:,i])>0)) for i,g in enumerate(CORE_GROUPS)},
                  "group_corr":corrmat(G).tolist(),
                  "group_physical_corr":{g:{m:safe_corr(G[:,i],P[:,j]) for j,m in enumerate(PHYS)} for i,g in enumerate(CORE_GROUPS)},
                  "within_group_term_corr":{
                    g:(corrmat(T[:,[ti[t] for t in GROUPS[g]]]).tolist() if len(GROUPS[g])>1 else [[1.0]]) for g in CORE_GROUPS
                  },
                  "physical_mean":{m:float(P[:,j].mean()) for j,m in enumerate(PHYS)},
                  "survival":float(1-done_any.mean()),"finite":finite,
                  "scalar_reconstruction_max_abs_error":float(np.max(np.abs(np.concatenate(returned)-np.concatenate(recon))))
                })
                mark("POLICY_SUITE_DONE",suite=suite,policy=lab)
        # matched-policy separability and winner fractions
        sep={};winner_fraction={};group_scale={}
        for g in CORE_GROUPS:
            means={lab:[] for lab in LABS}
            for r in suite_rows:means[r["policy"]].append(r["group_mean"][g])
            allvals=np.array([x for lab in LABS for x in means[lab]],float)
            scale=float(np.mean(np.abs(allvals)));policy_avg={lab:float(np.mean(v)) for lab,v in means.items()}
            prange=max(policy_avg.values())-min(policy_avg.values())
            winners=[]
            for s in range(args.suites):
                vals={lab:next(r for r in suite_rows if r["suite"]==s and r["policy"]==lab)["group_mean"][g] for lab in LABS}
                winners.append(max(vals,key=vals.get))  # higher reward is better
            sep[g]={"policy_mean":policy_avg,"policy_range":float(prange),"abs_mean_scale":scale,"range_over_scale":float(prange/(scale+1e-12))}
            winner_fraction[g]={lab:float(np.mean([w==lab for w in winners])) for lab in LABS}
            group_scale[g]={"global_abs_mean":scale,"global_std_of_suite_policy_means":float(np.std(allvals))}
        # aggregate correlation / coherence over matched rows
        group_corr_mean=np.mean([np.array(r["group_corr"]) for r in suite_rows],axis=0)
        within={}
        for g in CORE_GROUPS:
            mats=[np.array(r["within_group_term_corr"][g]) for r in suite_rows]
            M=np.mean(mats,axis=0)
            if M.shape[0]>1:
                vals=[M[i,j] for i in range(M.shape[0]) for j in range(i+1,M.shape[1])]
                within[g]={"mean_pairwise_corr":float(np.mean(vals)),"mean_abs_pairwise_corr":float(np.mean(np.abs(vals))),"matrix":M.tolist()}
            else:within[g]={"mean_pairwise_corr":1.0,"mean_abs_pairwise_corr":1.0,"matrix":M.tolist()}
        # physical alignment averaged across policy/suite
        align={g:{m:float(np.mean([r["group_physical_corr"][g][m] for r in suite_rows])) for m in PHYS} for g in CORE_GROUPS}
        result={
          "schema":"v2_t3a_candidate_regrouping_audit_v1","status":"READ_ONLY_COMPLETE","measurement_only":True,
          "candidate_groups":GROUPS,"core_group_order":list(CORE_GROUPS),
          "protocol":{"matched_reset":True,"num_envs":args.num_envs,"steps":args.steps,"suites":args.suites,"seed_base":args.seed_base},
          "provenance":{"specialists":{lab:{"path":str(ckpts[lab].relative_to(ROOT)),"sha256":sha(ckpts[lab])} for lab in LABS}},
          "scalar_reconstruction":{"max_abs_error":float(max(r["scalar_reconstruction_max_abs_error"] for r in suite_rows)),"pass":bool(max(r["scalar_reconstruction_max_abs_error"] for r in suite_rows)<=1e-5)},
          "policy_separation":sep,"winner_fraction":winner_fraction,"group_scale":group_scale,
          "mean_group_pairwise_corr":group_corr_mean.tolist(),"within_group_coherence":within,
          "mean_group_physical_corr":align,"suite_policy_rows":suite_rows,
          "integrity":{"all_finite":all(r["finite"] for r in suite_rows),"min_survival":float(min(r["survival"] for r in suite_rows))}
        }
        args.output.write_text(json.dumps(result,indent=2)+"\n");state["written"]=True;mark("ARTIFACT_WRITTEN",path=str(args.output));mark("RUN_DONE")
        print(json.dumps({"status":result["status"],"separation":sep,"winners":winner_fraction,"within":within,"alignment":align},indent=2))
    except BaseException as exc:
        fail("EVALUATION_EXCEPTION",exc);raise
    finally:
        if env is not None:env.close()
        app.close()
if __name__=="__main__":main()
